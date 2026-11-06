"""
forces/pair_splined.py — APPLE&P splined non-bonded pair force (M2.2).

This is the short-range force that stabilizes MD: exp-6 / 12-6 van der Waals
(the repulsive wall that stops Coulomb collapse) plus erfc-damped real-space
electrostatics. Both are pre-tabulated as quintic splines in z = r², exactly
as in the reference Python/Fortran.

Strategy:
    - The spline COEFFICIENT TABLES are built once at setup (numpy, reuse the
      reference's compute_spline_coefficients). They are static input here.
    - This module does the PER-PAIR EVALUATION over the neighbor pairs from
      M2.1, vectorized in JAX, with scatter-add to per-atom forces. This is
      the jit'd hot path.

Tables (from reference state.splines):
    coefft  (6, npiece, ninter)  total energy spline (vdW + real-space elec)
    coeffft (6, npiece, ninter)  total force spline (dE/dz pre-scaled)
    coeffee (6, npiece, ninter)  electrostatic-only energy spline
    zs      (npiece,)            grid of z=r^2 knot points (spacing deltaspline)
    deltainv = 1/deltaspline

Per-pair interaction index:
    itypee = typee[atomtype[i], atomtype[j]]   (symmetric lookup table)

Evaluation (matches nonbonded_splined.py exactly):
    z = r_ij^2  (minimum image)
    if z > rcut2: skip
    if z < zs[0]: clamp z = zs[0]
    ipiece = int((z - zs[0]) * deltainv) + 1, clamped to [1, nz-1]
    zp = z - zs[ipiece-1]
    energy = sum_{p=0..5} coefft[p, ipiece, itypee] * zp^p
    ff     = sum_{p=1..5} coeffft[p, ipiece, itypee] * zp^(p-1)
    f_i += ff * dx ;  f_j -= ff * dx   (dx = x_j - x_i)
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial


def _minimum_image(dx, box):
    return dx - box * jnp.round(dx / box)


@partial(jax.jit, static_argnames=('nat',))
def splined_pair_energy_force(
    x, idx_i, idx_j, atomtype, typee,
    coefft_m, coeffft_m, zs, deltainv, rcut2, box, nat,
):
    """Evaluate splined non-bonded energy + forces over neighbor pairs.

    Args:
        x: (nat, 3) positions.
        idx_i, idx_j: (npairs,) neighbor pair indices (i<j), from M2.1.
        atomtype: (nat,) int atom types.
        typee: (ntype, ntype) int -> interaction index.
        coefft_m, coeffft_m: (npiece, ninter, 6) spline tables, moved-axis from
            the reference (6, npiece, ninter) so one gather yields all 6 coeffs.
            Build with jnp.moveaxis(coefft, 0, -1) once at setup.
        zs: (npiece,) z-grid knots.
        deltainv: 1/deltaspline.
        rcut2: cutoff^2.
        box: (3,) box lengths.
        nat: number of atoms (static int).

    Returns:
        unbd: total non-bonded spline energy (scalar).
        forces: (nat, 3) force array.
    """
    xi = x[idx_i]                     # (npairs, 3)
    xj = x[idx_j]
    dx = _minimum_image(xj - xi, box)  # (npairs, 3), = x_j - x_i
    z = jnp.sum(dx * dx, axis=1)       # (npairs,)

    zs0 = zs[0]
    nz = zs.shape[0] - 1

    # in_cut also excludes padded pairs. Fixed-capacity padding uses
    # idx_i == idx_j (a self-pair sentinel) so the force fn compiles ONCE
    # regardless of how many real pairs a given rebuild produces. Real pairs
    # always have i != j, so this cleanly zeroes padded contributions.
    valid = idx_i != idx_j
    in_cut = (z <= rcut2) & valid

    # Clamp z into spline domain
    zc = jnp.where(z < zs0, zs0, z)

    # Spline piece index
    ipiece = (jnp.floor((zc - zs0) * deltainv) + 1).astype(jnp.int32)
    ipiece = jnp.clip(ipiece, 1, nz - 1)

    zp = zc - zs[ipiece - 1]

    # Interaction index per pair
    itypee = typee[atomtype[idx_i], atomtype[idx_j]]   # (npairs,)
    # Gather all 6 coefficients per pair in ONE indexing op per table.
    # coefft is (6, npiece, ninter); moveaxis to (npiece, ninter, 6) so
    # coefft_m[ipiece, itypee] -> (npairs, 6). One gather instead of six.
    cE = coefft_m[ipiece, itypee]    # (npairs, 6)  total energy coeffs
    cF = coeffft_m[ipiece, itypee]   # (npairs, 6)  force coeffs

    # Horner's method in zp for energy: c0 + zp*(c1 + zp*(c2 + ...))
    energy = cE[:, 0] + zp*(cE[:, 1] + zp*(cE[:, 2] + zp*(cE[:, 3]
             + zp*(cE[:, 4] + zp*cE[:, 5]))))
    energy = jnp.where(in_cut, energy, 0.0)

    # Force scalar uses coeffft indices 1..5 (index 0 unused), Horner:
    ff = cF[:, 1] + zp*(cF[:, 2] + zp*(cF[:, 3] + zp*(cF[:, 4] + zp*cF[:, 5])))
    ff = jnp.where(in_cut, ff, 0.0)

    # Force vectors: f_i += ff*dx, f_j -= ff*dx
    fvec = ff[:, None] * dx            # (npairs, 3)

    forces = jnp.zeros((nat, 3))
    forces = forces.at[idx_i].add(fvec)
    forces = forces.at[idx_j].add(-fvec)

    unbd = jnp.sum(energy)
    return unbd, forces


@partial(jax.jit, static_argnames=('nat',))
def splined_pair_energy_force_fp32(
    x, idx_i, idx_j, atomtype, typee,
    coefft_m32, coeffft_m32, zs32, deltainv, rcut2, box, nat,
):
    """Mixed-precision splined pair force.

    Positions/box come in fp64 (the integrator state stays fp64); they are
    cast to fp32 INSIDE for the heavy per-pair math (3.3M pairs, memory-bound,
    ~2x faster in fp32 on GPU). Coefficient tables are passed pre-cast to fp32
    (coefft_m32, coeffft_m32, zs32). The returned force is cast back to fp64 so
    the rest of the step (accumulation, SHAKE, integration) stays fp64.

    Validated: vs fp64 this gives energy rel diff ~1.6e-7, force rel ~3e-5 —
    far below MD tolerance.
    """
    x32 = x.astype(jnp.float32)
    box32 = box.astype(jnp.float32)
    xi = x32[idx_i]; xj = x32[idx_j]
    dx = xj - xi
    dx = dx - box32 * jnp.round(dx / box32)
    z = jnp.sum(dx * dx, axis=1)

    zs0 = zs32[0]
    nz = zs32.shape[0] - 1
    valid = idx_i != idx_j
    in_cut = (z <= jnp.float32(rcut2)) & valid
    zc = jnp.where(z < zs0, zs0, z)
    ipiece = (jnp.floor((zc - zs0) * jnp.float32(deltainv)) + 1).astype(jnp.int32)
    ipiece = jnp.clip(ipiece, 1, nz - 1)
    zp = zc - zs32[ipiece - 1]

    itypee = typee[atomtype[idx_i], atomtype[idx_j]]
    cE = coefft_m32[ipiece, itypee]
    cF = coeffft_m32[ipiece, itypee]

    energy = cE[:, 0] + zp*(cE[:, 1] + zp*(cE[:, 2] + zp*(cE[:, 3]
             + zp*(cE[:, 4] + zp*cE[:, 5]))))
    energy = jnp.where(in_cut, energy, jnp.float32(0.0))
    ff = cF[:, 1] + zp*(cF[:, 2] + zp*(cF[:, 3] + zp*(cF[:, 4] + zp*cF[:, 5])))
    ff = jnp.where(in_cut, ff, jnp.float32(0.0))

    fvec = ff[:, None] * dx
    forces = jnp.zeros((nat, 3), dtype=jnp.float32)
    forces = forces.at[idx_i].add(fvec)
    forces = forces.at[idx_j].add(-fvec)

    # cast back to fp64 for the rest of the pipeline
    return jnp.sum(energy).astype(jnp.float64), forces.astype(jnp.float64)
