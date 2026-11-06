"""
integrator.py — Time integration.

For M1: vanilla velocity Verlet, NVE only, no thermostat, no constraints.
M2 adds: SHAKE constraints, dummy atom force redistribution.
M3 adds: NHC thermostat, NPT barostat.

Velocity Verlet (Tuckerman split):
    v(t + dt/2) = v(t) + (dt/2) * a(t)
    x(t + dt) = x(t) + dt * v(t + dt/2)
    [recompute forces at x(t + dt)]
    v(t + dt) = v(t + dt/2) + (dt/2) * a(t + dt)

Where a = f / m. APPLE&P uses kcal·mol⁻¹·Å⁻¹ for force, amu for mass,
fs for time. The unit conversion factor (41.84 in the reference Python)
arises from: 1 kcal·mol⁻¹·Å⁻¹ / amu * 1 fs² = 1/41.84 Å — so we divide
the (delt² * f / m) term by 41.84 when used as displacement, or
equivalently scale velocities consistently.

For now I match the reference Python's convention: velocities are stored
in (Å/fs) × (1e-5 internal scale factor), and the unit factor 41.84
appears in the kick step. This means a quick port is bit-equivalent.

Once we validate, we may switch to a cleaner unit system where
v has plain Å/fs units and avoid the 1e-5 / 41.84 internal factor.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


# APPLE&P / reference Python unit-conversion constants
# These mirror the reference Python so we can validate bit-equivalently.
# Eventually we want to move to a cleaner unit system.
KCAL_PER_AMU_PER_ANGSTROM_TO_VELOCITY = 41.84   # see docstring above
VELOCITY_INTERNAL_SCALE = 1.0e-5                 # internal velocity scaling


def velocity_kick(v: jnp.ndarray, f: jnp.ndarray, massinv: jnp.ndarray,
                  dt: float, ldummy: jnp.ndarray) -> jnp.ndarray:
    """v += 0.5 * dt * f * (massinv * 41.84) on real atoms (skip dummies).
    
    Matches reference Python's velocity update exactly.
    """
    rmass = KCAL_PER_AMU_PER_ANGSTROM_TO_VELOCITY * massinv  # (nat,)
    factor = 0.5 * dt * rmass                                # (nat,)
    factor = factor[:, None]                                 # (nat, 1) for broadcast
    not_dummy = (~ldummy)[:, None]                           # (nat, 1) bool mask
    return jnp.where(not_dummy, v + factor * f, v)


def position_drift(x: jnp.ndarray, v: jnp.ndarray, dt: float,
                   box: jnp.ndarray, ldummy: jnp.ndarray) -> jnp.ndarray:
    """x += dt * v * 1e-5 (internal scale), then PBC wrap. Skip dummies.
    
    Matches reference Python's drift step. The 1e-5 factor is part of the
    reference Python's internal unit convention — we preserve it for
    bit-equivalence during validation.
    """
    delta = dt * VELOCITY_INTERNAL_SCALE * v                 # (nat, 3)
    not_dummy = (~ldummy)[:, None]
    new_x = jnp.where(not_dummy, x + delta, x)
    
    # PBC wrap: keep all atoms in [0, box). Reference Python uses
    # `if (x > box) x -= box; else if (x < 0) x += box` which assumes
    # at most one box-length drift per step. We use a more robust modulo.
    new_x = new_x - box * jnp.floor(new_x / box)
    return new_x


def velocity_verlet_step(state, force_fn) -> 'SystemState':
    """One velocity Verlet step. NVE only, no constraints.
    
    Args:
        state: SystemState with current x, v, f.
        force_fn: callable(state) -> (energy, forces). Must be a pure
                  function (no side effects). Already jit-compiled is fine.
    
    Returns:
        New SystemState with updated x, v, f, pe, step.
    """
    # First half-kick: v += 0.5*dt*f/m
    v_half = velocity_kick(state.v, state.f, state.massinv, state.dt, state.ldummy)
    
    # Drift: x += dt * v_half
    x_new = position_drift(state.x, v_half, state.dt, state.box, state.ldummy)
    
    # Build a new state to recompute forces with — preserve everything else
    # We can't mutate state in jit, so we make a copy with x updated.
    # Easiest pattern: use dataclasses.replace.
    from dataclasses import replace
    state_drifted = replace(state, x=x_new, v=v_half)
    
    # Recompute forces at new positions
    pe_new, f_new = force_fn(state_drifted)
    
    # Second half-kick: v += 0.5*dt*f_new/m
    v_new = velocity_kick(v_half, f_new, state.massinv, state.dt, state.ldummy)
    
    # Compute kinetic energy (real atoms only)
    not_dummy = (~state.ldummy)[:, None]
    v_real = jnp.where(not_dummy, v_new, 0.0)
    # KE = 0.5 * sum_i m_i * v_i² in matching units to PE (kcal/mol)
    # Reference Python: KE = 0.5 * 1e-5 * sum(m_i * v_i²) / 4.184e-4 (or similar)
    # For M1 we just compute it dimensionlessly; cross-check with reference
    # Python in the validation test.
    ke_new = 0.5 * jnp.sum(state.mass[:, None] * v_real * v_real) * VELOCITY_INTERNAL_SCALE * VELOCITY_INTERNAL_SCALE / (4.184e-4)
    
    new_state = replace(
        state,
        x=x_new,
        v=v_new,
        f=f_new,
        pe=pe_new,
        ke=ke_new,
    )
    return new_state


# ============================================================================
# Fully-jit'd stepper and multi-step runner
# ============================================================================
# The naive pattern — calling velocity_verlet_step in a Python for-loop with
# only force_fn jit'd — pays per-op dispatch overhead on every JAX call and,
# worse, recompiles if any aux field (e.g. the old `step` int) changes each
# step. The fixes:
#   1. `step` removed from the state pytree (see state.py).
#   2. jit the ENTIRE step (kick+drift+force+kick) as one fused function.
#   3. Use lax.scan or lax.fori_loop to run N steps without returning to
#      Python between steps — no dispatch overhead, single compile.

from functools import partial


def make_jit_step(force_fn):
    """Build a fully-jit'd single step function for a given force_fn.

    force_fn must be a pure function (SystemState) -> (pe, forces).
    Returns a jit'd fn (SystemState) -> SystemState that does one full
    velocity-Verlet step with all ops fused into one compiled kernel.
    """
    @jax.jit
    def _step(state):
        return velocity_verlet_step(state, force_fn)
    return _step


def run_nve(state, force_fn, n_steps: int):
    """Run n_steps of velocity Verlet, fully on device via lax.fori_loop.

    This compiles ONCE and runs all n_steps on the GPU without returning to
    Python between steps — the correct pattern for throughput. Energy/force
    bookkeeping (pe, ke) is carried in the state and available at the end.

    For per-step logging, use run_nve_logged instead (it returns a stacked
    history but compiles once via lax.scan).

    Args:
        state: initial SystemState (after .to_jax()).
        force_fn: pure (state) -> (pe, forces).
        n_steps: number of steps to run.
    Returns:
        final SystemState after n_steps.
    """
    def body(i, st):
        return velocity_verlet_step(st, force_fn)

    return jax.lax.fori_loop(0, n_steps, body, state)


# jit the runner with n_steps static (changing n_steps recompiles, which is
# fine — you rarely change it, and each distinct value compiles once).
run_nve_jit = jax.jit(run_nve, static_argnums=(1, 2))


def run_nve_logged(state, force_fn, n_steps: int):
    """Run n_steps, returning (final_state, pe_history, ke_history).

    Uses lax.scan so it compiles once and runs on device, but also collects
    per-step PE and KE for monitoring. The histories come back as device
    arrays of shape (n_steps,).
    """
    def body(st, _):
        st = velocity_verlet_step(st, force_fn)
        return st, (st.pe, st.ke)

    final, (pe_hist, ke_hist) = jax.lax.scan(body, state, None, length=n_steps)
    return final, pe_hist, ke_hist


run_nve_logged_jit = jax.jit(run_nve_logged, static_argnums=(1, 2))
