"""
verify_spline_fix.py
====================
Quick sanity check after applying the patched splines.py.

Drop this in your python-current/ directory and run:
    python verify_spline_fix.py

It does three things:
  1. Loads the system and builds splines.
  2. Verifies that deltaspline = 0.3 (was 2.245).
  3. Compares spline values vs analytical at 5 random pair distances per
     interaction type, prints the worst error.
  4. Computes the energy components and reports whether PE went negative
     (the expected sign for a thermalised condensed-phase system).

Expected: with the patch, spline relative error should drop from ~5e2 to <1e-3,
and PE should become negative (~-thousands kcal/mol).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def main():
    from input_parser import load_simulation
    from main import setup_system
    from splines import funct

    cwd = os.getcwd()
    print(f"Loading system from {cwd}...")
    state = load_simulation(cwd)
    setup_system(state)

    sp = state.splines
    delta = sp.deltaspline
    nz = sp.zs.shape[0] - 1
    print(f"\n  deltaspline = {delta:.4f}    (expected 0.3000)")
    print(f"  numpoints   = {nz}            (expected ~750)")
    if abs(delta - 0.3) > 0.01:
        print("  WARNING: deltaspline is not 0.3 — patch was not applied")

    # Pick 5 random r in [1.5, rcut-0.5], evaluate spline vs analytical
    rng = np.random.default_rng(0)
    rcut = state.params.rcut
    test_r = rng.uniform(1.5, rcut - 0.5, size=5)

    typee_flat = state.ff.typee.flatten()
    valid_innb = sorted(set(int(t) for t in typee_flat if t >= 0))[:10]

    alpha = state.params.alphai
    map_arr = state.ff.map if hasattr(state.ff, 'map') else state.ff.map_types

    print("\n  Spline vs analytical (10 types x 5 r):")
    print(f"  {'innb':>5} {'r':>6} {'analytical':>14} {'spline':>14} {'rel.err':>10}")
    print("  " + "-" * 55)
    max_err = 0.0
    for innb in valid_innb:
        for r in test_r:
            z = r * r
            if z < sp.zs[0] or z > sp.zs[-1]:
                continue
            ipiece = int((z - sp.zs[0]) * sp.deltainv) + 1
            if ipiece < 1: ipiece = 1
            if ipiece >= nz: ipiece = nz - 1
            zp = z - sp.zs[ipiece - 1]
            zp2 = zp*zp; zp3 = zp2*zp; zp4 = zp3*zp; zp5 = zp4*zp
            spl = (sp.coefft[0, ipiece, innb] +
                   sp.coefft[1, ipiece, innb]*zp +
                   sp.coefft[2, ipiece, innb]*zp2 +
                   sp.coefft[3, ipiece, innb]*zp3 +
                   sp.coefft[4, ipiece, innb]*zp4 +
                   sp.coefft[5, ipiece, innb]*zp5)
            ana = (funct(z, innb, 1, state.ff.electrostatic, state.ff.nonbonded,
                         map_arr, alpha, False)
                 + funct(z, innb, 3, state.ff.electrostatic, state.ff.nonbonded,
                         map_arr, alpha, False))
            err = abs(spl - ana) / max(abs(ana), 1.0)
            if err > max_err: max_err = err
            print(f"  {innb:>5d} {r:>6.2f} {ana:>+14.4e} {spl:>+14.4e} {err:>10.2e}")

    print(f"\n  Worst relative error: {max_err:.2e}    (was ~5e+02 before patch)")

    # Run the energy decomposition too
    from forces import compute_all_forces
    energy = compute_all_forces(state)

    print(f"\n  Energy decomposition (step 0):")
    print(f"    Bond     = {energy.ebond:+10.2f}    (was: +0.003)")
    print(f"    Bend     = {energy.ebend:+10.2f}    (was: +5582)")
    print(f"    Torsion  = {energy.etort + energy.eopbs:+10.2f}    (was: -180)")
    print(f"    vdW      = {getattr(energy, 'evdw', 0.0):+10.2f}    (was: +8898 -- expect to drop sharply)")
    print(f"    Electro  = {getattr(energy, 'eelec', 0.0):+10.2f}    (was: -2301)")
    print(f"    Polariz  = {getattr(energy, 'epol', 0.0):+10.2f}    (was: -7044)")
    print(f"    PE total = {energy.poten:+10.2f}    (was: +4955  -- expect NEGATIVE now)")

    if energy.poten < 0:
        print("\n  [GOOD] PE is negative — spline fix appears to have worked")
    else:
        print("\n  [BAD] PE still positive — investigate other components")


if __name__ == "__main__":
    main()
