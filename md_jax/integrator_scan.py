"""
integrator_scan.py — on-device NVE loop (M4, Fix 2).

Runs the ENTIRE trajectory inside a single lax.scan: the velocity-Verlet step
(SHAKE + dummies + full force eval) AND a periodic neighbor rebuild (via
lax.cond), with NO Python between steps and NO host<->device transfers. One
compile, then the GPU runs free.

This is only possible because (a) every force/constraint kernel is already a
pure jit-clean function, and (b) the neighbor build now runs on-device
(neighbor_ondevice.build_neighbor_ondevice). The host-loop driver
(main_m2.py) had to sync x to host every rebuild and re-upload the pairs; that
sync is what exposed the ~4.5 s host build and serialized the GPU. Here pairs
live on device and are carried in the scan state.

Physics is byte-for-byte the same arithmetic as integrator_m2.vv_step — only
the *driving* (Python loop -> lax.scan) and the *scatter* (in the segsum pair
kernel) change.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from .forces.all_forces import compute_all_forces
from .forces.dummy import update_dummy_coords, dummy_force_redistribution
from .constraints.shake import shake_constrain
from .forces.neighbor_ondevice import build_neighbor_ondevice

UNIT = 41.84
VSCALE = 1.0e-5


def _mi(d, box):
    return d - box * jnp.round(d / box)


def make_scan_step(Dj, kx, ky, kz, ewself, excl_flat, capacity,
                   rebuild_every, rlist2, polarization_fn=None, use_aspc=False):
    """Build a jit-able scan body that does one VV step + maybe a rebuild.

    Carry = (x, v, f, idx_i, idx_j, step, mu1, mu2, mu3). The neighbor pairs and
    (when polarizable) the dipole history travel in the carry so a rebuild just
    overwrites the pairs — no recompile, no host sync.

    If polarization_fn is given, the full polarizable force field is used.

    use_aspc: if True, warm-start the SCF with the ASPC predictor
        mu_pred = 3*(mu1 - mu2) + mu3 (fewer SCF iterations). DEFAULT False
        because the reference convergence test (max|Δμ|² ≤ tolpol) can stop
        early when the warm start changes little per iteration but is not yet
        at the true fixed point — a realistic 1% predictor error was observed
        to converge to dipoles ~24 kcal/mol off in E_pol. Cold-starting from
        mu=0 each step converges to the reference dipoles at 8e-16. Enable
        ASPC only together with a residual-based convergence test (see
        polarization_full / solve_scf TODO).

    Returns (scan_body, force_fn).
    """
    nat = Dj['nat']
    box = Dj['box']
    dt = Dj['dt']
    mass = Dj['mass']
    massinv = Dj['massinv']
    ldummy = Dj['ldummy']
    con = Dj['con']

    jik = Dj['dummy']['jik']
    adummy = Dj['dummy']['adummy']
    bdummy = Dj['dummy']['bdummy']
    cdummy = Dj['dummy']['cdummy']
    has_dummy = Dj['dummy']['ndummy'] > 0
    do_shake = con['constrain'] and con['nbcon'] > 0
    has_pol = polarization_fn is not None

    rmass = UNIT * massinv
    nd = (~ldummy)[:, None]

    def force_fn(x, idx_i, idx_j, mu_init=None):
        if has_dummy:
            x = update_dummy_coords(x, jik, adummy, bdummy, cdummy, box, nat)
        pe, f, comps = compute_all_forces(x, Dj, idx_i, idx_j, kx, ky, kz, ewself)
        mu_out = None
        if has_pol:
            e_pol, f_pol, mu_out = polarization_fn(x, idx_i, idx_j, mu_init)
            pe = pe + e_pol
            f = f + f_pol
            comps = dict(comps); comps['epol'] = e_pol
        if has_dummy:
            f = dummy_force_redistribution(f, x, jik, adummy, bdummy, cdummy, box, nat)
        return x, pe, f, comps, mu_out

    def _half_kick(v, f):
        return jnp.where(nd, v + (0.5 * dt * rmass)[:, None] * f, v)

    def _drift(x, v):
        x_new = jnp.where(nd, x + dt * VSCALE * v, x)
        return x_new - box * jnp.floor(x_new / box)

    def scan_body(carry, _):
        x, v, f, idx_i, idx_j, step, mu1, mu2, mu3 = carry

        # --- periodic neighbor rebuild, fully on device ---
        def do_rebuild(_):
            return build_neighbor_ondevice(x, box, rlist2, excl_flat,
                                           nat=nat, capacity=capacity)
        def keep(_):
            return idx_i, idx_j
        need = (step % rebuild_every) == 0
        idx_i, idx_j = jax.lax.cond(need, do_rebuild, keep, operand=None)

        # --- velocity-Verlet (identical arithmetic to vv_step) ---
        v = _half_kick(v, f)
        x = _drift(x, v)

        # ASPC dipole predictor: mu_pred = 3*(mu1 - mu2) + mu3, warm-starting
        # the SCF from a linear extrapolation of the last three steps' converged
        # dipoles (matches the reference scf_iteration for kount>0). For the
        # first few steps history is zero so this is a cold start, as in the ref.
        if has_pol:
            if use_aspc:
                mu_pred = 3.0 * (mu1 - mu2) + mu3
            else:
                mu_pred = None        # cold start each step (robust default)
            x, pe, f, comps, mu_new = force_fn(x, idx_i, idx_j, mu_pred)
            # rotate history regardless (so ASPC can be toggled without gaps)
            mu3 = mu2; mu2 = mu1; mu1 = mu_new
        else:
            x, pe, f, comps, _ = force_fn(x, idx_i, idx_j)

        if do_shake:
            # SHAKE reference position is the post-drift x (matches vv_step),
            # NOT the pre-drift position.
            x_ref = x
            x2 = jnp.where(nd, x + VSCALE*dt*v + VSCALE*dt*dt*f*rmass[:, None], x)
            x2 = x2 - box * jnp.floor(x2 / box)
            x2c, sigma, niter = shake_constrain(
                x2, x_ref, con['con_i'], con['con_j'], con['d2'], con['massred'],
                massinv, box, nat, con['tol'], 100)
            diff = _mi(x2c - x_ref, box)
            denom = UNIT * VSCALE * dt * dt
            f = jnp.where(nd, (diff - VSCALE*dt*v) * mass[:, None] / denom, 0.0)

        v = _half_kick(v, f)

        vr = jnp.where(nd, v, 0.0)
        ke = 0.5 * jnp.sum(mass[:, None] * vr * vr) / (4184.0 * 1000.0)

        out = (pe, ke)                       # logged per step (cheap)
        return (x, v, f, idx_i, idx_j, step + 1, mu1, mu2, mu3), out

    return scan_body, force_fn


def make_runner(scan_body):
    """Wrap scan_body into a jit'd fixed-length runner.

    nsteps is static (closed over per chunk) so the scan compiles once per
    chunk length. The driver runs in chunks (e.g. print_every) so it can log
    a (pe, ke) sample between chunks without breaking the on-device loop.
    """
    @partial(jax.jit, static_argnames=('nsteps',))
    def run(x0, v0, f0, idx_i0, idx_j0, step0, mu1, mu2, mu3, nsteps):
        carry0 = (x0, v0, f0, idx_i0, idx_j0, step0, mu1, mu2, mu3)
        carry, hist = jax.lax.scan(scan_body, carry0, None, length=nsteps)
        return carry, hist
    return run
