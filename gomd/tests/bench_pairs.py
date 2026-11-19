"""
Pair-layout experiment for the polarization kernels: one T.mu product.

    python -m gomd.tests.bench_pairs --input-dir ./md-input --ref-dir ./pkg

Same physics, five data layouts, each checked against the current kernel:
  A  current: half list (i<j) over the padded capacity, two .at[].add scatters
  B  compacted: the half list cut to pairs inside rcut (+0.05 A), per step
  C  sorted segment_sum: i-side on the i-sorted list, j-side through a
     j-sorting permutation made at rebuild time (indices_are_sorted=True)
  D  prefix sums: same orders as C, row sums as differences of cumulative sums
     (no scatter at all)
  E  dense full list: (nat, max_nb) neighbour table with both directions,
     row reduction (each pair computed twice, no scatter)
Reports the per-step / per-rebuild preparation cost of each layout too.
"""
from __future__ import annotations
import argparse, os, statistics, sys, time


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
    from gomd.forces.neighbor_ondevice import exclusions_to_flat, build_neighbor_ondevice_count
    from gomd.forces.ewald import build_kvectors
    from gomd.forces.dummy import update_dummy_coords
    from gomd.forces.polarization_full import make_polarization_parts
    from gomd.forces.pol_field import _min_image_ref

    def block(o):
        for leaf in jax.tree_util.tree_leaves(o):
            leaf.block_until_ready()
        return o

    def bench(fn, *a):
        out = block(fn(*a))
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); block(fn(*a)); ts.append(time.perf_counter() - t0)
        return out, statistics.median(ts) * 1e3

    D = load_m2_system(args.input_dir, args.ref_dir)
    nat = D['nat']
    Dj = build_Dj(D)
    box = Dj['box']
    dm = Dj['dummy']
    x = update_dummy_coords(Dj['x'], dm['jik'], dm['adummy'], dm['bdummy'], dm['cdummy'], box, nat)
    rcut = float(D['rcut']); rcut2 = rcut * rcut
    ii_h, _ = build_neighbor_pairs(np.asarray(D['x']), D['box'], rcut, skin=1.0, exclusions=D['excl_set'])
    cap = int(ii_h.size * 1.25)
    ii, jj, cnt = build_neighbor_ondevice_count(x, box, (rcut + 1.0) ** 2,
                                                exclusions_to_flat(D['excl_set'], nat),
                                                nat=nat, capacity=cap)
    kx, ky, kz = [jnp.asarray(a) for a in build_kvectors(D['box'], *D['klimit'])]
    pp = make_polarization_parts(D, kx, ky, kz, solver='pcg')
    G = jax.jit(pp['geometry'])(x, ii, jj)
    E = jax.jit(pp['field'])(x, ii, jj)
    mu = block(pp['alpha_atom'][:, None] * E * 3.0)          # a realistic dipole vector
    el = jnp.asarray(D['pol_data']['el'])
    print(f"device {jax.devices()[0]}  nat {nat}  list {int(cnt)} pairs, capacity {cap}", flush=True)

    rows = []
    def report(name, t_prod, t_prep, prep_what, out):
        err = float(jnp.max(jnp.abs(out - ref)) / jnp.max(jnp.abs(ref)))
        rows.append((name, t_prod, t_prep, prep_what, err))
        print(f"  {name:<34} T.mu {t_prod:7.2f} ms   prep {t_prep:7.2f} ms ({prep_what})   "
              f"max rel diff {err:.1e}", flush=True)

    # ---------------- A: current ----------------
    town = jax.jit(pp['town'])
    ref, tA = bench(town, mu, G, ii, jj)
    _, tGA = bench(jax.jit(pp['geometry']), x, ii, jj)
    report('A current (half list, scatter)', tA, tGA, 'geometry, per step', ref)

    # ---------------- B: compacted per step ----------------
    d_all = _min_image_ref(x[jj] - x[ii], box, el)
    ncut = int(jnp.sum((jnp.sum(d_all * d_all, axis=1) <= (rcut + 0.05) ** 2) & (ii != jj)))
    capc = int(ncut * 1.10)

    @jax.jit
    def compact(y, a, b):
        d = _min_image_ref(y[b] - y[a], box, el)
        keep = (jnp.sum(d * d, axis=1) <= (rcut + 0.05) ** 2) & (a != b)
        dest = jnp.where(keep, jnp.cumsum(keep, dtype=jnp.int32) - 1, capc + 1)
        ca = jnp.zeros((capc,), jnp.int32).at[dest].set(a, mode='drop')
        cb = jnp.zeros((capc,), jnp.int32).at[dest].set(b, mode='drop')
        return ca, cb
    (ci, cj), tC0 = bench(compact, x, ii, jj)
    Gc, tGc = bench(jax.jit(pp['geometry']), x, ci, cj)
    outB, tB = bench(town, mu, Gc, ci, cj)
    report(f'B compacted ({capc} slots)', tB, tC0 + tGc, 'compaction + geometry, per step', outB)

    # ---------------- A2 / B2: padding (0,0) sent to a dropped index ----------------
    def contribs0(m, Gg, a, b):
        rij = jnp.stack([Gg['rx'], Gg['ry'], Gg['rz']], axis=1)
        mu_i = m[a]; mu_j = m[b]
        pjr = Gg['three_zf_l5'] * jnp.sum(mu_j * rij, axis=1)
        pir = Gg['three_zf_l5'] * jnp.sum(mu_i * rij, axis=1)
        return pjr[:, None] * rij + mu_j * Gg['tfact'][:, None], \
               pir[:, None] * rij + mu_i * Gg['tfact'][:, None]

    @jax.jit
    def town_drop(m, Gg, a, b):
        v = a != b
        ci_, cj_ = contribs0(m, Gg, a, b)
        t = jnp.zeros((nat, 3)).at[jnp.where(v, a, nat)].add(ci_, mode='drop')
        return t.at[jnp.where(v, b, nat)].add(cj_, mode='drop')
    outA2, tA2 = bench(town_drop, mu, G, ii, jj)
    report('A2 current + padding dropped', tA2, tGA, 'geometry, per step', outA2)
    outB2, tB2 = bench(town_drop, mu, Gc, ci, cj)
    report('B2 compacted + padding dropped', tB2, tC0 + tGc, 'compaction + geometry, per step', outB2)

    @jax.jit
    def town_seg_drop(m, Gg, a, b):
        v = a != b
        ci_, cj_ = contribs0(m, Gg, a, b)
        seg = jnp.concatenate([jnp.where(v, a, nat), jnp.where(v, b, nat)])
        return jax.ops.segment_sum(jnp.concatenate([ci_, cj_]), seg, num_segments=nat)
    outA3, tA3 = bench(town_seg_drop, mu, G, ii, jj)
    report('A3 one segment_sum, padding dropped', tA3, tGA, 'geometry, per step', outA3)

    # ---------------- C / D: sorted reductions ----------------
    @jax.jit
    def sort_prep(a, b):
        valid = a != b
        a_s = jnp.where(valid, a, nat)                    # i-sorted already; padding -> sentinel
        b_s = jnp.where(valid, b, nat)
        perm = jnp.argsort(b_s, stable=True)
        b_sorted = b_s[perm]
        bounds = jnp.arange(nat + 1)
        si = jnp.searchsorted(a_s, bounds)
        sj = jnp.searchsorted(b_sorted, bounds)
        return a_s, perm, b_sorted, si, sj
    (a_s, perm, b_sorted, si, sj), tSort = bench(sort_prep, ii, jj)

    def contribs(m, Gg, a, b):
        rij = jnp.stack([Gg['rx'], Gg['ry'], Gg['rz']], axis=1)
        mu_i = m[a]; mu_j = m[b]
        pjr = Gg['three_zf_l5'] * jnp.sum(mu_j * rij, axis=1)
        pir = Gg['three_zf_l5'] * jnp.sum(mu_i * rij, axis=1)
        ci_ = pjr[:, None] * rij + mu_j * Gg['tfact'][:, None]
        cj_ = pir[:, None] * rij + mu_i * Gg['tfact'][:, None]
        return ci_, cj_

    @jax.jit
    def town_segsorted(m, Gg, a, b, a_s, perm, b_sorted):
        ci_, cj_ = contribs(m, Gg, a, b)
        ti = jax.ops.segment_sum(ci_, a_s, num_segments=nat + 1, indices_are_sorted=True)
        tj = jax.ops.segment_sum(cj_[perm], b_sorted, num_segments=nat + 1, indices_are_sorted=True)
        return (ti + tj)[:nat]
    outC, tCs = bench(town_segsorted, mu, G, ii, jj, a_s, perm, b_sorted)
    report('C sorted segment_sum', tCs, tSort, 'argsort + bounds, per rebuild', outC)

    @jax.jit
    def town_cumsum(m, Gg, a, b, perm, si, sj):
        ci_, cj_ = contribs(m, Gg, a, b)
        z = jnp.zeros((1, 3))
        csi = jnp.concatenate([z, jnp.cumsum(ci_, axis=0)])
        csj = jnp.concatenate([z, jnp.cumsum(cj_[perm], axis=0)])
        return (csi[si[1:]] - csi[si[:-1]]) + (csj[sj[1:]] - csj[sj[:-1]])
    outD, tD = bench(town_cumsum, mu, G, ii, jj, perm, si, sj)
    report('D prefix sums (no scatter)', tD, tSort, 'argsort + bounds, per rebuild', outD)

    # ---------------- E: dense full list ----------------
    valid = ii != jj
    counts = np.bincount(np.concatenate([np.asarray(ii)[np.asarray(valid)],
                                         np.asarray(jj)[np.asarray(valid)]]), minlength=nat)
    max_nb = int(counts.max() * 1.10) + 1

    @jax.jit
    def dense_prep(a, b):
        v = a != b
        src = jnp.where(jnp.concatenate([v, v]), jnp.concatenate([a, b]), nat)
        dst = jnp.concatenate([b, a])
        order = jnp.argsort(src, stable=True)
        so = src[order]; do = dst[order]
        start = jnp.searchsorted(so, jnp.arange(nat + 1))
        slot = jnp.arange(so.shape[0]) - start[jnp.minimum(so, nat)]
        slot = jnp.where(so < nat, slot, max_nb)          # sentinel rows dropped
        return jnp.full((nat, max_nb), nat, jnp.int32).at[so, slot].set(do, mode='drop')
    nbr, tDp = bench(dense_prep, ii, jj)

    ncharge = D['pol_data']['ncharge']; c1 = jnp.asarray(D['pol_data']['c1'])
    atomtype = jnp.asarray(D['atomtype']); arf = float(D['pol_data']['arf'])

    @jax.jit
    def dense_geometry(y, nb):
        ypad = jnp.concatenate([y, y[:1]])                   # sentinel row (masked)
        d = _min_image_ref(ypad[nb] - y[:, None, :], box, el)  # (nat, max_nb, 3)
        z = jnp.sum(d * d, axis=-1)
        good = (z <= rcut2) & (z > 1e-10) & (nb < nat)
        r = jnp.sqrt(jnp.where(good, z, 1.0)); rinv = 1.0 / r; zinv = rinv * rinv
        tpad = jnp.concatenate([atomtype, atomtype[:1]])
        ti = atomtype[:, None]; tj = tpad[nb]
        both = (ti < ncharge) & (tj < ncharge)
        c1ij = jnp.where(both, c1[jnp.minimum(ti, ncharge - 1), jnp.minimum(tj, ncharge - 1)], 1e6)
        e3 = jnp.where(both, jnp.exp(-c1ij * r ** 3), 0.0)
        l3 = 1.0 - e3; l5 = 1.0 - e3 * (1.0 + c1ij * r ** 3)
        fact = rinv * zinv
        tfact = jnp.where(good, arf - fact * l3, 0.0)
        t5 = jnp.where(good, 3.0 * zinv * fact * l5, 0.0)
        return d, tfact, t5
    (dd, tf, t5), tGd = bench(dense_geometry, x, nbr)

    @jax.jit
    def town_dense(m, nb, d, tf, t5):
        mpad = jnp.concatenate([m, jnp.zeros((1, 3))])
        mn = mpad[nb]                                         # (nat, max_nb, 3)
        mjr = jnp.sum(mn * d, axis=-1)
        return jnp.sum((t5 * mjr)[..., None] * d + tf[..., None] * mn, axis=1)
    outE, tE = bench(town_dense, mu, nbr, dd, tf, t5)
    report(f'E dense full list ({nat}x{max_nb})', tE, tGd, 'geometry, per step', outE)
    print(f"  E: table build per rebuild {tDp:.2f} ms (argsort of {2 * cap} directed pairs)")

    print("\nsummary: per-step cost at 2.35 T.mu products/step (PCG + predict), excluding rebuild prep")
    for name, tp, tprep, what, err in rows:
        per_step = 2.35 * tp + (tprep if 'per step' in what else 0.0)
        print(f"  {name:<34} {per_step:8.2f} ms/step  (rel diff {err:.1e})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
