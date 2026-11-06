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

`x` is a JAX array already on device, so this can be called every k steps from
the driver — or, better, embedded inside a lax.scan step loop (see notes).
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


@partial(jax.jit, static_argnames=('nat', 'capacity', 'tile'))
def build_neighbor_ondevice(x, box, rlist2, excl_flat, nat, capacity, tile=512):
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
    """
    xf = x.astype(jnp.float32)
    boxf = box.astype(jnp.float32)
    ntile = (nat + tile - 1) // tile
    jrange = jnp.arange(nat)

    def body(t, state):
        out, count = state                              # out: (capacity+1,), count: scalar
        i0 = t * tile
        idx_i = i0 + jnp.arange(tile)
        xi = xf[jnp.clip(idx_i, 0, nat - 1)]            # (tile,3)
        d = xi[:, None, :] - xf[None, :, :]             # (tile,nat,3)  <- only tile rows
        d = d - boxf * jnp.round(d / boxf)
        r2 = jnp.sum(d * d, axis=-1)                     # (tile,nat)
        ii = idx_i[:, None]
        jj = jrange[None, :]
        m = (r2 < rlist2) & (ii < jj) & (ii < nat)
        key = jnp.where(m, ii * nat + jj, -1).ravel().astype(jnp.int64)  # (tile*nat,)

        # exclusions (per-tile, small)
        if excl_flat.shape[0] > 0:
            pos = jnp.clip(jnp.searchsorted(excl_flat, key), 0, excl_flat.shape[0]-1)
            key = jnp.where(excl_flat[pos] == key, -1, key)

        # compact this tile's valid keys directly into `out` at running offset
        valid = key >= 0
        dest = count + jnp.cumsum(valid) - 1
        dest = jnp.where(valid, dest, capacity)
        dest = jnp.clip(dest, 0, capacity)
        out = out.at[dest].set(key)
        count = count + jnp.sum(valid)
        return (out, count)

    out0 = jnp.zeros((capacity + 1,), dtype=jnp.int64)
    out, count = jax.lax.fori_loop(0, ntile, body, (out0, jnp.int64(0)))

    sel = out[:capacity]
    sel = jnp.where(sel >= 0, sel, 0)
    return (sel // nat).astype(jnp.int32), (sel % nat).astype(jnp.int32)
