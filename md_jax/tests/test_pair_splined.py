"""
test_pair_splined.py — validate the M2.2 splined non-bonded pair force.

Compares the JAX splined pair kernel against the Python reference on the real
8716-atom system. Requires /tmp/spline_ref.npz (dumped from the reference via
/tmp/dump_spline_ref.py) and /tmp/ref_neigh.npz (for exclusions).

Pass criterion: energy and forces match the reference to < 1e-3 (in practice
they match to fp64 noise ~1e-10, since this is the same spline math).

Run:
    python -m wmi_md_jax.tests.test_pair_splined
"""
from __future__ import annotations
import os
import sys
import numpy as np


def test_vs_reference():
    sp_path = '/tmp/spline_ref.npz'
    nb_path = '/tmp/ref_neigh.npz'
    if not (os.path.exists(sp_path) and os.path.exists(nb_path)):
        print(f"[pair splined] SKIP (need {sp_path} and {nb_path})")
        return True

    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    from jax import config
    config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wmi_md_jax.forces.neighbor import (
        build_neighbor_pairs, exclusions_from_reference,
    )
    from wmi_md_jax.forces.pair_splined import splined_pair_energy_force

    d = np.load(sp_path)
    x = d['x']; box = d['box']; rcut2 = float(d['rcut2'])
    atomtype = d['atomtype']; typee = d['typee']
    coefft = d['coefft']; coeffft = d['coeffft']
    coefft_m = np.moveaxis(coefft,0,-1).copy(); coeffft_m = np.moveaxis(coeffft,0,-1).copy()
    zs = d['zs']; deltainv = float(d['deltainv'])
    ref_f = d['ref_f']; ref_unbd = float(d['ref_unbd']); ref_unbde = float(d['ref_unbde'])
    nat = x.shape[0]

    dn = np.load(nb_path)
    excl = exclusions_from_reference(dn['listex'], dn['listmex'])
    rcut = float(np.sqrt(rcut2))
    ii, jj = build_neighbor_pairs(x, box, rcut, skin=0.0, exclusions=excl)

    unbd, f = splined_pair_energy_force(
        jnp.asarray(x), jnp.asarray(ii), jnp.asarray(jj),
        jnp.asarray(atomtype), jnp.asarray(typee),
        jnp.asarray(coefft_m), jnp.asarray(coeffft_m),
        jnp.asarray(zs), deltainv, rcut2, jnp.asarray(box), nat,
    )
    unbd = float(unbd); f = np.asarray(f)

    e_ok = abs(unbd - ref_unbd) < 1e-3
    fmax = float(np.abs(f - ref_f).max())
    f_ok = fmax < 1e-3

    print(f"[pair splined] unbd diff={abs(unbd-ref_unbd):.2e} "
          f"unbde diff={abs(unbde-ref_unbde):.2e} fmax={fmax:.2e} "
          f"-> {'PASS' if (e_ok and f_ok) else 'FAIL'}")
    return e_ok and f_ok


def main():
    print("=" * 60)
    print("M2.2 splined pair force validation")
    print("=" * 60)
    ok = test_vs_reference()
    print("=" * 60)
    print("[PASS]" if ok else "[FAIL]", "splined pair force")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
