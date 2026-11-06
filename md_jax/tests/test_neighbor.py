"""
test_neighbor.py — validate the M2.1 neighbor list.

Two levels:
  1. Cell list vs brute force on a small synthetic system (exact pair set).
  2. (If /tmp/ref_neigh.npz exists) JAX neighbor list vs the Python reference
     on the real 8716-atom system — must be bit-identical.

The reference dump is produced by /tmp/dump_ref_neigh.py against the Python
package. On a machine without that, only the synthetic test runs.

Run:
    python -m wmi_md_jax.tests.test_neighbor
"""
from __future__ import annotations
import sys
import os
import numpy as np

from wmi_md_jax.forces.neighbor import (
    build_neighbor_pairs, _brute_pairs, exclusions_from_reference,
)


def test_cell_vs_brute():
    """Cell list must produce exactly the brute-force pair set."""
    rng = np.random.default_rng(1)
    nat = 400
    box = np.array([30., 30., 30.])
    x = rng.uniform(0, 30, (nat, 3))
    rcut, skin = 8.0, 1.0
    rlist2 = (rcut + skin) ** 2

    ii, jj = build_neighbor_pairs(x, box, rcut, skin=skin, exclusions=None)
    cell_set = set(zip(ii.tolist(), jj.tolist()))

    bi, bj = _brute_pairs(x, box, rlist2, exclusions=None)
    brute_set = set(zip(bi.tolist(), bj.tolist()))

    ok = cell_set == brute_set
    print(f"[cell vs brute] cell={len(cell_set)} brute={len(brute_set)} "
          f"identical={ok}")
    return ok


def test_vs_reference():
    """JAX neighbor list must match the Python reference on the real system."""
    path = '/tmp/ref_neigh.npz'
    if not os.path.exists(path):
        print(f"[vs reference] SKIP (no {path})")
        return True

    d = np.load(path)
    x = d['x']; box = d['box']; rcut = float(d['rcut'])
    listex = d['listex']; listmex = d['listmex']
    ref_set = set(zip(d['ref_i'].tolist(), d['ref_j'].tolist()))

    excl = exclusions_from_reference(listex, listmex)
    ii, jj = build_neighbor_pairs(x, box, rcut, skin=0.0, exclusions=excl)
    jax_set = set(zip(ii.tolist(), jj.tolist()))

    ok = jax_set == ref_set
    print(f"[vs reference] ref={len(ref_set)} jax={len(jax_set)} "
          f"identical={ok}")
    if not ok:
        print(f"  only in jax: {len(jax_set - ref_set)}")
        print(f"  only in ref: {len(ref_set - jax_set)}")
    return ok


def main():
    print("=" * 60)
    print("M2.1 neighbor list validation")
    print("=" * 60)
    t1 = test_cell_vs_brute()
    t2 = test_vs_reference()
    print("=" * 60)
    if t1 and t2:
        print("[PASS] neighbor list validated.")
        return 0
    print("[FAIL] neighbor list mismatch.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
