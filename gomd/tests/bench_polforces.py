"""
Induced-force pair term (polarization.pol_forces): exact fp64 erfc/exp vs the
reference's 16,384-knot tables vs fp32 transcendentals. Accuracy + speed.

    python -m gomd.tests.bench_polforces --input-dir ./md-input --ref-dir ./pkg
"""
from __future__ import annotations
import argparse, statistics, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--reps', type=int, default=10)
    args = ap.parse_args(argv)
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np
    from gomd.main_scan import build_Dj
    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.neighbor_ondevice import (exclusions_to_flat, build_neighbor_ondevice_count,
                                               compact_pairs)
    from gomd.forces.ewald import build_kvectors
    from gomd.forces.dummy import update_dummy_coords
    from gomd.forces.polarization_full import make_polarization_parts
    from gomd.forces.polarization import pol_forces, COULOMB_CONST
    from gomd.forces.pol_field import build_ewald_tables

    def block(o):
        for leaf in jax.tree_util.tree_leaves(o):
            leaf.block_until_ready()
        return o

    def bench(fn, *a):
        out = block(fn(*a)); ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); block(fn(*a)); ts.append(time.perf_counter() - t0)
        return out, statistics.median(ts) * 1e3

    D = load_m2_system(args.input_dir, args.ref_dir)
    nat = D['nat']; Dj = build_Dj(D); box = Dj['box']; dm = Dj['dummy']
    x = update_dummy_coords(Dj['x'], dm['jik'], dm['adummy'], dm['bdummy'], dm['cdummy'], box, nat)
    rcut = float(D['rcut'])
    ii_h, _ = build_neighbor_pairs(np.asarray(D['x']), D['box'], rcut, skin=1.0, exclusions=D['excl_set'])
    ii, jj, _ = build_neighbor_ondevice_count(x, box, (rcut + 1.0) ** 2,
                                              exclusions_to_flat(D['excl_set'], nat),
                                              nat=nat, capacity=int(ii_h.size * 1.25))
    ci, cj, nc = compact_pairs(x, ii, jj, box, (rcut + 0.05) ** 2, int(ii_h.size))
    kx, ky, kz = [jnp.asarray(a) for a in build_kvectors(D['box'], *D['klimit'])]
    D['pol_data']['tolpol'] = 1e-10
    pp = make_polarization_parts(D, kx, ky, kz, scf_max_iter=300, solver='pcg')
    G = jax.jit(pp['geometry'])(x, ci, cj)
    E = jax.jit(pp['field'])(x, ci, cj)
    mu = block(jax.jit(pp['scf'])(E, G, ci, cj)[0])
    q = jnp.asarray(D['q']); alpha = float(D['alpha'])
    erfc_tab, exp_tab, dx_inv = build_ewald_tables()
    print(f"device {jax.devices()[0]}  {int(nc)} pairs (compacted)", flush=True)

    def make(**kw):
        @jax.jit
        def f(m, Gg, a, b):
            good = Gg['good']
            f5, _ = pol_forces(m, q, alpha, a, b,
                               jnp.where(good, Gg['rx'], 0.0), jnp.where(good, Gg['ry'], 0.0),
                               jnp.where(good, Gg['rz'], 0.0), Gg['r'], Gg['rinv'],
                               jnp.where(good, Gg['c1ij'], 0.0), jnp.where(good, Gg['expc1r3'], 0.0),
                               Gg['tau'], Gg['dtau'], nat,
                               q_i_override=jnp.where(good, q[a], 0.0),
                               q_j_override=jnp.where(good, q[b], 0.0), **kw)
            return COULOMB_CONST * f5
        return f
    f_ex, t_ex = bench(make(), mu, G, ci, cj)
    f_tab, t_tab = bench(make(erfc_tab=erfc_tab, exp_tab=exp_tab, dx_inv=dx_inv), mu, G, ci, cj)
    f_32, t_32 = bench(make(fp32_special=True), mu, G, ci, cj)
    scale = float(jnp.max(jnp.abs(f_ex)))
    rms = float(jnp.sqrt(jnp.mean(f_ex ** 2)))
    for name, f, t in (('exact fp64 erfc/exp', f_ex, t_ex), ('reference tables', f_tab, t_tab),
                       ('fp32 erfc/exp', f_32, t_32)):
        d = f - f_ex
        print(f"  {name:<22} {t:7.2f} ms   max|dF| {float(jnp.max(jnp.abs(d))):.2e} kcal/mol/A "
              f"(rel to max {float(jnp.max(jnp.abs(d))) / scale:.1e}, rms dF {float(jnp.sqrt(jnp.mean(d * d))):.1e}"
              f" vs rms F {rms:.2f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
