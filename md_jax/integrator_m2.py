"""
integrator_m2.py — Full M2 velocity-Verlet step (NVE) with SHAKE + dummies.

Replicates the reference _vanilla_vv_step scheme exactly (no polarization):
    1. first half-kick: v += 0.5*dt*(41.84*massinv)*f   (real atoms only)
    2. drift: x += v*dt*1e-5  (real atoms), PBC wrap
    3. place dummy positions from parents
    4. compute forces at new positions (the assembled compute_all_forces)
    5. redistribute dummy forces to parents, zero dummy force
    6. SHAKE predictor-corrector: REPLACES f with combined (physical+constraint)
       force via x2 = x + 1e-5*dt*v + 1e-5*dt^2*f/m ; SHAKE(x2) ;
       f = (diff - 1e-5*dt*v)*m / (41.84e-5*dt^2)
    7. second half-kick: v += 0.5*dt*(41.84*massinv)*f

Notes:
    - Neighbor list + k-vectors are rebuilt on the host periodically (not every
      step). For M2 validation we pass them in; rebuild cadence is a tuning
      knob handled by the driver.
    - This is a Python-driver loop calling jit'd kernels. The force eval (the
      expensive part) is jit'd; the cheap vector ops are plain jax. We avoid
      the M1 recompile bug by NOT putting changing ints into pytree aux.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .forces.all_forces import compute_all_forces
from .forces.dummy import update_dummy_coords, dummy_force_redistribution
from .constraints.shake import shake_constrain

UNIT = 41.84
VSCALE = 1.0e-5


def _mi(d, box):
    return d - box * jnp.round(d / box)


@partial(jax.jit, static_argnames=('nat',))
def _half_kick(v, f, massinv, ldummy, dt, nat):
    rmass = UNIT * massinv
    factor = 0.5 * dt * rmass
    nd = (~ldummy)[:, None]
    return jnp.where(nd, v + factor[:, None] * f, v)


@partial(jax.jit, static_argnames=('nat',))
def _drift(x, v, ldummy, dt, box, nat):
    delta = dt * VSCALE * v
    nd = (~ldummy)[:, None]
    x_new = jnp.where(nd, x + delta, x)
    x_new = x_new - box * jnp.floor(x_new / box)
    return x_new


@partial(jax.jit, static_argnames=('nat',))
def _shake_combined_force(x_con, x_old, v, mass, ldummy, dt, box, nat):
    """Combined-force formula: f = (diff - 1e-5*dt*v)*m / (41.84e-5*dt^2).
    Applied to real atoms; dummy forces left as-is (zeroed earlier)."""
    diff = _mi(x_con - x_old, box)             # (nat,3)
    denom = UNIT * VSCALE * dt * dt
    num = (diff - VSCALE * dt * v) * mass[:, None]
    f_comb = num / denom
    nd = (~ldummy)[:, None]
    return jnp.where(nd, f_comb, 0.0)


@partial(jax.jit, static_argnames=('nat',))
def _kinetic(v, mass, ldummy, nat):
    nd = (~ldummy)[:, None]
    vr = jnp.where(nd, v, 0.0)
    return 0.5 * jnp.sum(mass[:, None] * vr * vr) / (4184.0 * 1000.0)


def make_force_fn(Dj, kx, ky, kz, ewself):
    """Bind compute_all_forces to a function of (x, idx_i, idx_j).

    The neighbor pairs are ARGUMENTS, not closed-over constants, so the jit'd
    function is compiled ONCE and reused across neighbor rebuilds (as long as
    the padded pair shape is constant). This is the key recompile-free path.

    Also applies dummy placement + dummy force redistribution.
    """
    nat = Dj['nat']
    jik = Dj['dummy']['jik']
    adummy = Dj['dummy']['adummy']; bdummy = Dj['dummy']['bdummy']; cdummy = Dj['dummy']['cdummy']
    box = Dj['box']
    has_dummy = Dj['dummy']['ndummy'] > 0

    @jax.jit
    def force_fn(x, idx_i, idx_j):
        if has_dummy:
            x = update_dummy_coords(x, jik, adummy, bdummy, cdummy, box, nat)
        pe, f, comps = compute_all_forces(x, Dj, idx_i, idx_j, kx, ky, kz, ewself)
        if has_dummy:
            f = dummy_force_redistribution(f, x, jik, adummy, bdummy, cdummy, box, nat)
        return x, pe, f, comps
    return force_fn


def vv_step(state, Dj, force_fn, idx_i, idx_j):
    """One NVE velocity-Verlet step with SHAKE + dummies.

    state: dict with x, v, f (jax arrays), plus scalars.
    Dj: jax data dict (constraints etc).
    force_fn: from make_force_fn.
    Returns updated state dict.
    """
    nat = Dj['nat']; box = Dj['box']; dt = Dj['dt']
    mass = Dj['mass']; massinv = Dj['massinv']; ldummy = Dj['ldummy']
    con = Dj['con']

    x = state['x']; v = state['v']; f = state['f']

    # 1. first half-kick
    v = _half_kick(v, f, massinv, ldummy, dt, nat)
    # 2. drift
    x_old = x
    x = _drift(x, v, ldummy, dt, box, nat)
    # 3+4+5. place dummies, compute forces, redistribute (inside force_fn)
    x, pe, f, comps = force_fn(x, idx_i, idx_j)
    # 6. SHAKE predictor-corrector (if constraints)
    if con['constrain'] and con['nbcon'] > 0:
        # predictor x2 = x + 1e-5*dt*v + 1e-5*dt^2*f/m  (real atoms)
        rmass = UNIT * massinv
        nd = (~ldummy)[:, None]
        x2 = jnp.where(nd, x + VSCALE*dt*v + VSCALE*dt*dt*f*rmass[:, None], x)
        x2 = x2 - box * jnp.floor(x2 / box)
        x2c, sigma, niter = shake_constrain(
            x2, x, con['con_i'], con['con_j'], con['d2'], con['massred'],
            massinv, box, nat, con['tol'], 100)
        f = _shake_combined_force(x2c, x, v, mass, ldummy, dt, box, nat)
    # 7. second half-kick
    v = _half_kick(v, f, massinv, ldummy, dt, nat)

    ke = _kinetic(v, mass, ldummy, nat)
    return {'x': x, 'v': v, 'f': f, 'pe': pe, 'ke': ke, 'comps': comps}
