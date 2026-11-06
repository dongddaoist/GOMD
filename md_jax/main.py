"""
main.py — M1 simulation driver.

Reads input from current directory (mdrun.params, ff.dat, etc.), runs N steps
of pure Coulomb + velocity Verlet, prints energies, writes nothing.

This is M1 only — meant to prove the architecture works on GPU. M2+ adds
all the real APPLE&P physics. A simulation result from this code will NOT
match the reference Python's APPLE&P results because:
    - No Lennard-Jones / vdW (M2)
    - No bonded forces (M2)
    - No SHAKE constraints (M2)
    - No dummy atom redistribution (M2)
    - No PME — Coulomb has hard cutoff (M2)
    - No polarization (M3)

What this WILL show:
    - State on GPU works
    - jit'd inner loop compiles and runs
    - Forces are well-defined gradients of energy (NVE energy roughly conserved)
    - Per-step time gives a baseline measurement on V100

Usage:
    cd /path/to/system
    python -m wmi_md_jax.main --steps 100
"""
from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="M1 MD driver — Coulomb + velocity Verlet")
    parser.add_argument('--input-dir', default='.',
                        help="Directory with mdrun.params, ff.dat, etc.")
    parser.add_argument('--steps', type=int, default=100,
                        help="Number of MD steps")
    parser.add_argument('--print-every', type=int, default=10,
                        help="Print energies every N steps")
    parser.add_argument('--cpu', action='store_true',
                        help="Force CPU (default uses GPU if available)")
    args = parser.parse_args()
    
    # JAX config — must come BEFORE any jax import
    import os
    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Enable x64 (fp64) for validation. M4 will switch to mixed precision.
    from jax import config as jax_config
    jax_config.update('jax_enable_x64', True)
    
    import jax
    print(f"\nJAX devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")
    
    # Now load the rest
    from .io import load_simulation_state
    from .forces.coulomb import coulomb_energy_force_jit, coulomb_energy_force
    from .integrator import velocity_verlet_step, run_nve_logged_jit
    
    print(f"\nLoading from {args.input_dir}...")
    state = load_simulation_state(args.input_dir)
    print(state.summary())
    
    # Convert to jax arrays
    state = state.to_jax()
    print(f"\nState moved to JAX. Total charge: {float(state.charge.sum()):+.6f} e")
    
    # Initial force evaluation
    print("\nComputing initial Coulomb energy + force...")
    t0 = time.time()
    pe0, f0 = coulomb_energy_force_jit(state)
    pe0.block_until_ready()
    t_compile = time.time() - t0
    print(f"  First call (compile + run): {t_compile:.2f} s")
    print(f"  Initial PE (Coulomb only): {float(pe0):+.4f} kcal/mol")
    print(f"  Initial |F| RMS: {float((f0 ** 2).mean() ** 0.5):.4f}")
    
    # Warmup: second call should be much faster (already compiled)
    t0 = time.time()
    pe0, f0 = coulomb_energy_force_jit(state)
    pe0.block_until_ready()
    t_warm = time.time() - t0
    print(f"  Second call (cached compile): {t_warm * 1000:.1f} ms")
    
    # Set initial forces in state
    from dataclasses import replace
    state = replace(state, f=f0, pe=float(pe0))
    
    # MD loop — run on device in chunks of `print_every` steps.
    # Each chunk compiles ONCE (lax.scan) and runs entirely on the GPU with
    # no Python-between-steps overhead and no per-step recompilation.
    # The first chunk pays the compile cost; subsequent chunks are fast.
    print(f"\nRunning {args.steps} MD steps (on-device, chunked)...")
    print(f"{'step':>6}  {'PE':>14}  {'KE':>14}  {'TE':>14}  {'ms/step':>10}")
    
    chunk = max(1, args.print_every)
    done = 0
    step_counter = 0
    t_start = time.time()
    # Use the unjitted force for scan body; run_nve_logged_jit jits the whole
    # scan (force included) as one fused compile.
    while done < args.steps:
        this_chunk = min(chunk, args.steps - done)
        t_chunk0 = time.time()
        state, pe_hist, ke_hist = run_nve_logged_jit(
            state, coulomb_energy_force, this_chunk
        )
        state.x.block_until_ready()
        t_chunk = time.time() - t_chunk0
        done += this_chunk
        step_counter += this_chunk
        ms_per_step = t_chunk / this_chunk * 1000
        pe = float(pe_hist[-1]); ke = float(ke_hist[-1])
        te = pe + ke
        note = "  (incl. 1st-chunk compile)" if done == this_chunk else ""
        print(f"{step_counter:>6}  {pe:>+14.4f}  {ke:>14.4f}"
              f"  {te:>+14.4f}  {ms_per_step:>10.2f}{note}")
    
    state.x.block_until_ready()
    t_total = time.time() - t_start
    
    # Report both raw average and compile-excluded average. The first chunk
    # includes the one-time compile; the steady-state number is what matters.
    print(f"\n=== Run complete ===")
    print(f"  Total wall time: {t_total:.2f} s")
    print(f"  Average ms/step (incl. compile): {t_total / args.steps * 1000:.2f}")
    if args.steps > chunk:
        # Re-time a steady-state chunk now that everything is compiled
        t_ss0 = time.time()
        state, _, _ = run_nve_logged_jit(state, coulomb_energy_force, chunk)
        state.x.block_until_ready()
        t_ss = time.time() - t_ss0
        print(f"  Steady-state ms/step (compiled): {t_ss / chunk * 1000:.2f}")
    print(f"  Steps per second (steady): see above")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
