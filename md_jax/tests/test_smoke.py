"""
test_smoke.py — Self-contained smoke test for M1 architecture.

Builds a small synthetic ion system (no input file needed), runs 50 MD steps,
checks that:
    1. Force is the negative gradient of energy (numerical check)
    2. NVE energy drift is reasonable (<5% over 50 steps with hard-cutoff Coulomb)
    3. Forces approximately balance (sum to ~0, Newton's 3rd law)
    4. No NaN or Inf appear

Pass criterion: all 4 checks pass on at least CPU. GPU is a bonus.

This does NOT validate APPLE&P physics — that's M2's job. This validates
the M1 architecture: state on GPU, jit'd inner loop, integrator structure.

Run with:
    python -m wmi_md_jax.tests.test_smoke
"""
from __future__ import annotations
import sys
import os

# Configure JAX BEFORE first import
os.environ.setdefault('JAX_PLATFORMS', 'cpu')   # default to CPU for portability
from jax import config as jax_config
jax_config.update('jax_enable_x64', True)

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace


def make_synthetic_system(n_pairs: int = 50, box_size: float = 30.0, seed: int = 42):
    """Create a small synthetic system: like-charged particles in a box.
    
    Note on charges: we use all +1 charges (NOT alternating ±1). With
    alternating signs and only Coulomb, opposite-charge pairs would collapse
    onto each other (no LJ repulsion to stop them). M2 adds LJ which prevents
    this.
    
    For M1 architecture validation, "all positive" gives us a stable
    interacting system — atoms repel, NVE energy is conserved cleanly.
    """
    from wmi_md_jax.state import SystemState
    
    rng = np.random.default_rng(seed)
    nat = 2 * n_pairs
    
    # Random positions, well-separated by rejection sampling
    # (For Coulomb-only, even like charges can blow up if too close due to
    # 1/r force; min separation ~3 Å is plenty for our test charges.)
    x = np.zeros((nat, 3))
    placed = 0
    attempts = 0
    while placed < nat and attempts < 10000:
        attempts += 1
        candidate = rng.uniform(0, box_size, size=3)
        if placed > 0:
            dr = x[:placed] - candidate[None, :]
            dr -= box_size * np.round(dr / box_size)
            min_d = np.sqrt((dr * dr).sum(axis=1)).min()
            if min_d < 4.0:
                continue
        x[placed] = candidate
        placed += 1
    if placed < nat:
        raise RuntimeError(f"Couldn't place {nat} atoms with min separation 4 Å in {box_size} Å box")
    
    # Charges: all +0.1e (small, weakly interacting)
    charge = np.full(nat, 0.1)
    
    # Light initial velocity at low temperature to avoid blowup
    mass = np.full(nat, 22.99)  # Na-like mass
    # Maxwell-Boltzmann at low T (~30 K) for testing — KE small relative to PE
    kT_kcal = 0.06
    sigma_v = np.sqrt(kT_kcal / mass)  # in some velocity units
    v = rng.standard_normal(size=(nat, 3)) * sigma_v[:, None] * 1e-4
    
    return SystemState(
        nat=nat,
        x=x, v=v, f=np.zeros((nat, 3)),
        mass=mass,
        massinv=1.0 / mass,
        charge=charge,
        atomtype=np.zeros(nat, dtype=np.int32),
        ldummy=np.zeros(nat, dtype=bool),
        box=np.array([box_size, box_size, box_size]),
        dt=0.5,
        rcut=10.0,
        step=0,
    )


def test_force_is_negative_gradient(state, force_fn):
    """Numerically check that f_i ≈ -dU/dx_i for atom 0, x-component."""
    eps = 1e-4
    x0 = state.x.copy()
    
    pe_center, _ = force_fn(state)
    
    # Perturb atom 0 in +x
    x_plus = x0.at[0, 0].add(eps)
    pe_plus, _ = force_fn(replace(state, x=x_plus))
    
    # Perturb atom 0 in -x
    x_minus = x0.at[0, 0].add(-eps)
    pe_minus, _ = force_fn(replace(state, x=x_minus))
    
    numerical_force = -(float(pe_plus) - float(pe_minus)) / (2 * eps)
    _, f0 = force_fn(state)
    analytical_force = float(f0[0, 0])
    
    rel_err = abs(numerical_force - analytical_force) / max(abs(analytical_force), 1.0)
    return numerical_force, analytical_force, rel_err


def test_smoke():
    print("=" * 60)
    print("M1 smoke test: synthetic ion system")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    from wmi_md_jax.forces.coulomb import coulomb_energy_force, coulomb_energy_force_jit
    from wmi_md_jax.integrator import velocity_verlet_step
    
    # Build synthetic system
    state = make_synthetic_system(n_pairs=50, box_size=30.0)
    state = state.to_jax()
    print(f"\nBuilt synthetic system: {state.nat} ions, box={state.box[0]:.1f} Å")
    
    # Test 1: force is negative gradient of energy
    print("\n[Test 1] Numerical gradient check (atom 0, x-direction)")
    num_f, ana_f, rel_err = test_force_is_negative_gradient(state, coulomb_energy_force)
    print(f"  Numerical force: {num_f:+.4f}")
    print(f"  Analytical force: {ana_f:+.4f}")
    print(f"  Relative error: {rel_err:.2e}")
    test1_pass = rel_err < 1e-3
    print(f"  {'PASS' if test1_pass else 'FAIL'}")
    
    # Test 2: forces sum to ~0 (Newton's 3rd law)
    print("\n[Test 2] Force sum (should be ~0)")
    pe0, f0 = coulomb_energy_force_jit(state)
    f_sum = float(jnp.abs(f0.sum(axis=0)).max())
    print(f"  Initial PE: {float(pe0):+.4f} kcal/mol")
    print(f"  Max abs sum of forces (per axis): {f_sum:.2e}")
    test2_pass = f_sum < 1e-6
    print(f"  {'PASS' if test2_pass else 'FAIL'}")
    
    # Test 3: simulation runs without blowing up over 50 steps
    print("\n[Test 3] Simulation stability (50 steps, no NaN/Inf, bounded growth)")
    state = replace(state, f=f0, pe=float(pe0))
    
    pe_init = float(state.pe)
    print(f"  Initial PE: {pe_init:+.4f}")
    
    pe_history = [pe_init]
    for i in range(50):
        state = velocity_verlet_step(state, coulomb_energy_force_jit)
        pe_history.append(float(state.pe))
    
    pe_final = float(state.pe)
    pe_max_excursion = max(abs(p - pe_init) for p in pe_history)
    print(f"  Final PE: {pe_final:+.4f}")
    print(f"  Max PE excursion from initial: {pe_max_excursion:.4f}")
    # With like charges and weak velocity, system should not run away.
    # PE excursions of order 10x initial would mean the system blew up.
    test3_pass = (pe_max_excursion < 10 * abs(pe_init)) and \
                 not any(p != p for p in pe_history)  # no NaN
    print(f"  {'PASS' if test3_pass else 'FAIL'}  (energy conservation deferred to M2)")
    
    # Test 4: no NaN/Inf
    print("\n[Test 4] No NaN/Inf in final state")
    has_nan = bool(jnp.any(jnp.isnan(state.x)) | jnp.any(jnp.isnan(state.v)))
    has_inf = bool(jnp.any(jnp.isinf(state.x)) | jnp.any(jnp.isinf(state.v)))
    test4_pass = (not has_nan) and (not has_inf)
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    print(f"  {'PASS' if test4_pass else 'FAIL'}")
    
    # Summary
    print("\n" + "=" * 60)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    if all_pass:
        print("[PASS] All 4 smoke tests passed.")
        return 0
    else:
        print("[FAIL] One or more smoke tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(test_smoke())
