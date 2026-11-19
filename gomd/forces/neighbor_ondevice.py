"""
forces/neighbor_ondevice.py — GPU neighbor build (replaces the host build).

Why this exists
---------------
The host cell-list build (`neighbor.py`) takes ~4.5 s for this 8,716-atom box
and runs on the CPU regardless of the GPU. At 47 A with rlist=13 A the box has
only 3 cells/dim, so the cell list barely prunes — the build is effectively
all-pairs work executed in Python. With `--rebuild-every 10` that ~4.5 s lands
inside every 10-step window and *is* the ~800 ms/step you measured.

This module does the same build on-device. For this box, all-pairs r^2 is the
right primitive: 8716^2 ~ 76M distances is a few ms on a V100 in fp32, fully
parallel, with NO host<->device sync. It returns the SAME fixed-capacity padded
pair arrays the force kernels already consume (padding = self-pairs (0,0),
masked downstream by i != j), so nothing else changes.

Validated: produces the bit-identical reference pair set (i<j, within cutoff,
exclusions removed).

Usage
-----
    excl_flat = exclusions_to_flat(excl_set, nat)      # once, host-side
    idx_i, idx_j = build_neighbor_ondevice(
        x, box, rlist2, excl_flat, nat=nat, capacity=capacity)
    idx_i, idx_j, count = build_neighbor_ondevice_count(...)   # + true pair count
    missed = count_pairs_within(x, ii_new, jj_new, box, rcut2) \
           - count_pairs_within(x, ii_old, jj_old, box, rcut2)

`x` is a JAX array already on device; the MD loop (integrator_scan) calls it
inside lax.scan, on a fixed interval and whenever a site has moved more than
half the skin since the last build. count > capacity flags an overflow.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


def exclusions_to_flat(excl_set, nat):
    """Pack an (i<j) exclusion set into a sorted int64 array of flat keys i*nat+j."""
    if not excl_set:
        return jnp.zeros((0,), dtype=jnp.int64)
    arr = np.fromiter((i * nat + j for (i, j) in excl_set), dtype=np.int64,
                      count=len(excl_set))
    arr.sort()
    return jnp.asarray(arr)


def build_neighbor_ondevice(x, box, rlist2, excl_flat, nat, capacity, tile=512, skip=None):
    """All-pairs (i<j) neighbor build on device, fixed-capacity padded output.

    Args:
        x: (nat,3) positions (JAX array, on device).
        box: (3,) orthorhombic box lengths.
        rlist2: (rcut+skin)^2.
        excl_flat: sorted int64 flat keys (i*nat+j) of excluded pairs, or a
            length-0 array for none. Build with exclusions_to_flat().
        nat, capacity, tile: static ints. capacity must exceed the real pair
            count (size it ~1.25x the first build, exactly as today).

    Returns:
        idx_i, idx_j: (capacity,) int32, real pairs first, padded with (0,0).
        Pairs beyond `capacity` are dropped; use build_neighbor_ondevice_count
        to detect that.
    """
    idx_i, idx_j, _ = build_neighbor_ondevice_count(x, box, rlist2, excl_flat,
                                                    nat=nat, capacity=capacity, tile=tile,
                                                    skip=skip)
    return idx_i, idx_j


@partial(jax.jit, static_argnames=('nat', 'capacity', 'tile'))
def build_neighbor_ondevice_count(x, box, rlist2, excl_flat, nat, capacity, tile=512, skip=None):
    """As build_neighbor_ondevice, plus the number of pairs found.

    skip: optional (nat,) bool; pairs with both atoms marked are left out (the
    frozen electrodes: their mutual interactions are constant).

    Returns idx_i, idx_j (capacity,) int32 and count (int64 scalar). count is
    the number of pairs inside rlist; count > capacity means the list
    overflowed and pairs were dropped.

    Two passes (2026-11-17; same output as the original one-pass build, which
    took 186 ms on an RTX 5070):
      1. per row tile, every candidate pair inside rlist is compacted into the
         list in (i, j) order. Rejected candidates (about 95% of the 76M) get
         an out-of-range slot and are dropped (mode='drop'), not all written to
         one dummy slot, which serialised the scatter.
      2. the excluded pairs are removed once, on the compacted list (searchsorted
         over ~3.4M keys instead of 76M), and the list is compacted again, so
         the real pairs come first in (i, j) order and the padding is (0,0).
    Keys i*nat+j are int32 when nat^2 < 2^31 (nat <= 46340), int64 otherwise.
    While pass 1 holds the candidates, the list needs room for the excluded
    pairs too (count > capacity is reported if it does not have it).
    """
    kdt = jnp.int32 if nat * nat < 2**31 - 1 else jnp.int64
    xf = x.astype(jnp.float32)
    boxf = box.astype(jnp.float32)
    ntile = (nat + tile - 1) // tile
    jrange = jnp.arange(nat, dtype=kdt)
    oob = capacity + 1                                   # dropped scatter slot

    def body(t, state):
        out, count = state                               # out: (capacity,), count: scalar
        idx_i = (t * tile + jnp.arange(tile)).astype(kdt)
        xi = xf[jnp.clip(idx_i, 0, nat - 1)]             # (tile,3)
        d = xi[:, None, :] - xf[None, :, :]              # (tile,nat,3)
        d = d - boxf * jnp.round(d / boxf)
        r2 = jnp.sum(d * d, axis=-1)                      # (tile,nat)
        ii = idx_i[:, None]
        jj = jrange[None, :]
        m = (r2 < rlist2) & (ii < jj) & (ii < nat)
        if skip is not None:
            m = m & ~(skip[jnp.clip(idx_i, 0, nat - 1)][:, None] & skip[None, :])
        m = m.ravel()
        key = (ii * nat + jj).ravel()                     # (tile*nat,) kdt
        dest = count + jnp.cumsum(m, dtype=kdt) - 1
        dest = jnp.where(m & (dest < capacity), dest, oob)
        out = out.at[dest].set(key, mode='drop')
        return out, count + jnp.sum(m, dtype=kdt)

    out0 = jnp.zeros((capacity,), dtype=kdt)
    out, ncand = jax.lax.fori_loop(0, ntile, body, (out0, jnp.zeros((), kdt)))

    stored = jnp.arange(capacity) < jnp.minimum(ncand, capacity)
    if excl_flat.shape[0] > 0:
        ex = excl_flat.astype(kdt)
        pos = jnp.clip(jnp.searchsorted(ex, out), 0, ex.shape[0] - 1)
        keep = stored & (ex[pos] != out)
        dest = jnp.where(keep, jnp.cumsum(keep, dtype=kdt) - 1, oob)
        out = jnp.zeros((capacity,), dtype=kdt).at[dest].set(out, mode='drop')
        nreal = jnp.sum(keep, dtype=jnp.int64)
    else:
        nreal = jnp.minimum(ncand, capacity).astype(jnp.int64)
    count = jnp.where(ncand > capacity, ncand.astype(jnp.int64), nreal)
    return ((out // nat).astype(jnp.int32), (out % nat).astype(jnp.int32), count)


@partial(jax.jit, static_argnames=('capacity',))
def compact_pairs(x, idx_i, idx_j, box, rc2, capacity):
    """The listed pairs with r^2 <= rc2 (fp32, round-based min image), in list
    order, compacted into `capacity` slots padded with (0,0).

    The MD step calls this once per step with rc = rcut + 0.05 A, so every pair
    kernel runs over ~2.9M slots instead of the 4.19M of the rlist capacity.
    Returns ci, cj (capacity,) int32 and n (int64); n > capacity = overflow.
    """
    xf = x.astype(jnp.float32)
    boxf = box.astype(jnp.float32)
    d = xf[idx_j] - xf[idx_i]
    d = d - boxf * jnp.round(d / boxf)
    keep = (jnp.sum(d * d, axis=1) <= rc2) & (idx_i != idx_j)
    pos = jnp.cumsum(keep, dtype=jnp.int32) - 1
    dest = jnp.where(keep & (pos < capacity), pos, capacity + 1)
    ci = jnp.zeros((capacity,), jnp.int32).at[dest].set(idx_i, mode='drop')
    cj = jnp.zeros((capacity,), jnp.int32).at[dest].set(idx_j, mode='drop')
    return ci, cj, jnp.sum(keep, dtype=jnp.int64)


@jax.jit
def count_pairs_within(x, idx_i, idx_j, box, rc2):
    """Number of listed pairs with r^2 <= rc2 (padding (0,0) excluded).

    Same fp32 round-based min image as the build. Comparing it on a stale list
    and on a fresh build at the same positions counts the pairs the stale list
    misses: missed = count(fresh) - count(stale). Both lists hold every pair
    inside rcut they know about, so the difference is exact.
    """
    xf = x.astype(jnp.float32)
    boxf = box.astype(jnp.float32)
    d = xf[idx_j] - xf[idx_i]
    d = d - boxf * jnp.round(d / boxf)
    r2 = jnp.sum(d * d, axis=1)
    return jnp.sum((r2 <= rc2) & (idx_i != idx_j)).astype(jnp.int64)
