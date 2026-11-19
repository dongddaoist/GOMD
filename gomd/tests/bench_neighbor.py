"""
Benchmark + equivalence check of the on-device neighbor build.

    python -m gomd.tests.bench_neighbor --input-dir ./md-input --ref-dir ./pkg

Compares the current build_neighbor_ondevice_count with the original
one-pass version (kept below as _build_v1, the 2026-11-16 code), and both
with the host cell-list build:
  * pair arrays identical (old vs new, element by element, padding included)
  * pair set identical to the host build, except pairs within fp32 rounding
    of rlist (the device tests r^2 < rlist^2 in fp32; those pairs lie in the
    skin and are masked by rcut in every kernel)
  * time per build (median), for several row-tile sizes
"""
from __future__ import annotations
import argparse, math, os, statistics, sys, time
from functools import partial


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--skin', type=float, default=1.0)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--tiles', default='256,512,1024,2048')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args(argv)
    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np
    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.neighbor_ondevice import exclusions_to_flat, build_neighbor_ondevice_count

    @partial(jax.jit, static_argnames=('nat', 'capacity', 'tile'))
    def _build_v1(x, box, rlist2, excl_flat, nat, capacity, tile=512):
        xf = x.astype(jnp.float32); boxf = box.astype(jnp.float32)
        ntile = (nat + tile - 1) // tile
        jrange = jnp.arange(nat)

        def body(t, state):
            out, count = state
            idx_i = t * tile + jnp.arange(tile)
            xi = xf[jnp.clip(idx_i, 0, nat - 1)]
            d = xi[:, None, :] - xf[None, :, :]
            d = d - boxf * jnp.round(d / boxf)
            r2 = jnp.sum(d * d, axis=-1)
            ii = idx_i[:, None]; jj = jrange[None, :]
            m = (r2 < rlist2) & (ii < jj) & (ii < nat)
            key = jnp.where(m, ii * nat + jj, -1).ravel().astype(jnp.int64)
            if excl_flat.shape[0] > 0:
                pos = jnp.clip(jnp.searchsorted(excl_flat, key), 0, excl_flat.shape[0] - 1)
                key = jnp.where(excl_flat[pos] == key, -1, key)
            valid = key >= 0
            dest = jnp.clip(jnp.where(valid, count + jnp.cumsum(valid) - 1, capacity), 0, capacity)
            return out.at[dest].set(key), count + jnp.sum(valid)

        out, count = jax.lax.fori_loop(0, ntile, body,
                                       (jnp.zeros((capacity + 1,), jnp.int64), jnp.int64(0)))
        sel = jnp.where(out[:capacity] >= 0, out[:capacity], 0)
        return (sel // nat).astype(jnp.int32), (sel % nat).astype(jnp.int32), count

    def block(o):
        for leaf in jax.tree_util.tree_leaves(o):
            leaf.block_until_ready()
        return o

    def bench(fn, *a):
        block(fn(*a))
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); block(fn(*a)); ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e3

    D = load_m2_system(args.input_dir, args.ref_dir)
    nat = D['nat']
    print(f"device {jax.devices()[0]}  nat {nat}  skin {args.skin}", flush=True)
    x = jnp.asarray(D['x']); box = jnp.asarray(D['box'])
    rlist2 = float((D['rcut'] + args.skin) ** 2)
    ii_h, jj_h = build_neighbor_pairs(np.asarray(D['x']), D['box'], D['rcut'], skin=args.skin,
                                      exclusions=D['excl_set'])
    capacity = int(ii_h.size * 1.25)
    ex = exclusions_to_flat(D['excl_set'], nat)
    host_keys = np.sort(ii_h.astype(np.int64) * nat + jj_h)
    print(f"host build: {ii_h.size} pairs, capacity {capacity}, {ex.shape[0]} exclusions", flush=True)

    ok_all = True
    for tile in [int(t) for t in args.tiles.split(',')]:
        f_old = lambda y: _build_v1(y, box, rlist2, ex, nat=nat, capacity=capacity, tile=tile)
        f_new = lambda y: build_neighbor_ondevice_count(y, box, rlist2, ex, nat=nat,
                                                        capacity=capacity, tile=tile)
        a = block(f_old(x)); b = block(f_new(x))
        same = (int(a[2]) == int(b[2]) and bool(jnp.all(a[0] == b[0])) and bool(jnp.all(a[1] == b[1])))
        n = int(b[2])
        kb = np.asarray(b[0][:n]).astype(np.int64) * nat + np.asarray(b[1][:n])
        # the device build tests r^2 < rlist^2 in fp32, the host in fp64: pairs within
        # rounding of rlist may differ, and they lie in the skin (masked by rcut)
        diff = np.setxor1d(kb, host_keys)
        xh = np.asarray(D['x'])
        dv = xh[diff % nat] - xh[diff // nat]
        dv -= D['box'] * np.round(dv / D['box'])
        rd = np.sqrt(np.sum(dv * dv, axis=1))
        rl = math.sqrt(rlist2)
        host_same = bool(np.all(np.abs(rd - rl) < 1e-4))
        if diff.size:
            print(f"  {diff.size} pair(s) differ from the host build, at r - rlist = "
                  + ', '.join(f'{r - rl:+.2e}' for r in rd[:5]) + ' A', flush=True)
        ordered = bool(np.all(np.diff(kb) > 0))
        pad_ok = bool(jnp.all(b[0][n:] == 0) & jnp.all(b[1][n:] == 0))
        t_old = bench(f_old, x); t_new = bench(f_new, x)
        ok = same and host_same and ordered and pad_ok
        ok_all &= ok
        print(f"tile {tile:>5}: old {t_old:8.2f} ms  new {t_new:8.2f} ms  ({t_old / t_new:5.1f}x)  "
              f"pairs {n}  identical to old: {same}  = host set: {host_same}  "
              f"(i,j)-ordered: {ordered}  padding (0,0): {pad_ok}", flush=True)

    # overflow: a capacity below the pair count must be reported, not hidden
    small = int(ii_h.size * 0.9)
    c = int(build_neighbor_ondevice_count(x, box, rlist2, ex, nat=nat, capacity=small, tile=512)[2])
    print(f"overflow check: capacity {small} -> count {c} ({'reported' if c > small else 'NOT reported'})")
    ok_all &= c > small
    print('ALL OK' if ok_all else 'MISMATCH')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
