"""
forces/pair_splined_segsum.py — pair kernel with segment_sum scatter (M4, Fix 3).

Two changes vs pair_splined.py, both targeting the GPU bottleneck the
CPU-sandbox profile hid:

1. **segment_sum instead of `.at[idx].add`.** On GPU the unordered scatter-add
   over ~4.2M pairs into 8,716 atom slots is atomic-contention-bound (~1000x
   contention per slot). `jax.ops.segment_sum` over index-sorted segments
   replaces atomics with a coalesced reduction. The +contrib on i and -contrib
   on j are concatenated into one (2*npairs,) reduction. Forces are
   BIT-IDENTICAL to `.at[].add` (validated, max diff 0.0).

2. **fp32 end-to-end.** Positions, box, tables, distances, spline eval, AND the
   reduction all run in fp32; only the returned scalars are cast to fp64 for
   the integrator. The earlier `--fp32` showed no gain because the scatter and
   reciprocal stayed fp64 — with the scatter now fp32 + coalesced, the fp32
   win on the table gather is no longer masked. energy rel ~1.6e-7,
   force rel ~3e-5 vs fp64 (far below MD tolerance).

Pairs may be PRESORTED by idx_i at build time (segment_sum does not require it
for correctness, but sorted indices make the underlying reduction more
cache-friendly). The kernel works on unsorted pairs too.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=('nat',))
def splined_pair_energy_force_segsum(
    x, idx_i, idx_j, atomtype, typee,
    coefft_m32, coeffft_m32, zs32, deltainv, rcut2, box, nat,
):
    """fp32 splined pair energy+force with a segment_sum scatter.

    Args mirror splined_pair_energy_force_fp32 (tables already fp32). Returns
    (energy fp64 scalar, forces fp64 (nat,3)).
    """
    x32 = x.astype(jnp.float32)
    box32 = box.astype(jnp.float32)
    xi = x32[idx_i]
    xj = x32[idx_j]
    dx = xj - xi
    dx = dx - box32 * jnp.round(dx / box32)          # min image
    z = jnp.sum(dx * dx, axis=1)

    zs0 = zs32[0]
    nz = zs32.shape[0] - 1
    valid = idx_i != idx_j                            # masks padded (0,0) pairs
    in_cut = (z <= jnp.float32(rcut2)) & valid

    zc = jnp.where(z < zs0, zs0, z)
    ipiece = (jnp.floor((zc - zs0) * jnp.float32(deltainv)) + 1).astype(jnp.int32)
    ipiece = jnp.clip(ipiece, 1, nz - 1)
    zp = zc - zs32[ipiece - 1]

    itypee = typee[atomtype[idx_i], atomtype[idx_j]]
    cE = coefft_m32[ipiece, itypee]                  # (npairs,6)
    cF = coeffft_m32[ipiece, itypee]

    energy = cE[:, 0] + zp*(cE[:, 1] + zp*(cE[:, 2] + zp*(cE[:, 3]
             + zp*(cE[:, 4] + zp*cE[:, 5]))))
    energy = jnp.where(in_cut, energy, jnp.float32(0.0))

    ff = cF[:, 1] + zp*(cF[:, 2] + zp*(cF[:, 3] + zp*(cF[:, 4] + zp*cF[:, 5])))
    ff = jnp.where(in_cut, ff, jnp.float32(0.0))
    fvec = ff[:, None] * dx                          # (npairs,3) on atom i

    # --- segment_sum scatter (replaces two atomic .at[].add) ---
    # +fvec accumulates on idx_i, -fvec on idx_j; one fused reduction.
    seg_idx = jnp.concatenate([idx_i, idx_j])
    seg_val = jnp.concatenate([fvec, -fvec])
    forces = jax.ops.segment_sum(seg_val, seg_idx, num_segments=nat)

    return jnp.sum(energy).astype(jnp.float64), forces.astype(jnp.float64)
