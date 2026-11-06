"""
coulomb.py — Direct (non-Ewald) Coulomb sum for M1 architecture proof.

This is a **placeholder**: it computes pairwise Coulomb energy/force with
a hard cutoff and minimum-image PBC. It does NOT use Ewald and is NOT
suitable for production polar systems — it's just a vehicle to prove the
architecture works on GPU.

In M2 this gets replaced by:
    - Real-space splined pair potential (12-6-1 + erfc) for short-range
    - PME for long-range Coulomb
    - All the APPLE&P-specific physics

What this does demonstrate:
    - State on GPU (jax arrays in SystemState)
    - O(N²) pairwise computation with PBC
    - jax.jit compatibility
    - Correct energy/force pair (gradient of energy gives force, sign right)

Algorithm:
    For each pair i<j with r_ij < rcut:
        u_ij = k_e * q_i * q_j / r_ij
        f_ij = (k_e * q_i * q_j / r_ij^3) * r_ij_vec   (on i, opposite on j)
    
    This is O(N²) because we compute all pairs. Fine for M1 testing on
    small systems (~1000 atoms). M2 introduces a neighbor list to make
    it O(N) with reasonable cutoff.

Constants:
    Coulomb constant in kcal/mol/Å with charges in e:
        k_e = 332.0716 (matches APPLE&P / TINKER / OpenMM convention)
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial


# kcal·Å/mol/e² — the standard Coulomb constant used in APPLE&P
COULOMB_K = 332.0716


def _pbc_minimum_image(dr: jnp.ndarray, box: jnp.ndarray) -> jnp.ndarray:
    """Apply PBC minimum image convention to displacement vectors.
    
    dr: (..., 3) displacements
    box: (3,) box dimensions
    
    Returns dr - box * round(dr / box), elementwise per axis.
    """
    return dr - box * jnp.round(dr / box)


def coulomb_energy_force(state) -> tuple:
    """Compute total Coulomb energy and per-atom forces, O(N²) direct sum.
    
    Args:
        state: SystemState with .x (nat, 3), .charge (nat,), .box (3,), .rcut.
    
    Returns:
        (energy: scalar, forces: (nat, 3)). Forces in kcal/mol/Å.
    
    Notes:
        - O(N²) — fine for ≤2000 atoms on GPU; for large systems M2's
          neighbor list will reduce to O(N).
        - Hard cutoff at rcut (no smoothing) — produces minor energy
          discontinuities. Smoothed cutoff comes in M2.
        - Newton's third law enforced by symmetric construction.
    """
    x = state.x                               # (nat, 3)
    q = state.charge                          # (nat,)
    box = state.box                           # (3,)
    rcut2 = state.rcut * state.rcut
    nat = x.shape[0]
    
    # All pairs displacement (nat, nat, 3) — uses memory N² * 3 * 8 bytes.
    # For nat=8716 this is ~1.8 GB. For M1 we test with smaller systems.
    # M2's neighbor list eliminates this entirely.
    dr = x[None, :, :] - x[:, None, :]        # (nat, nat, 3)
    dr = _pbc_minimum_image(dr, box)
    r2 = jnp.sum(dr * dr, axis=-1)            # (nat, nat)
    
    # Mask: i < j, within cutoff, exclude self (r2 == 0)
    iu, ju = jnp.triu_indices(nat, k=1)       # upper triangle, off-diagonal
    r2_pairs = r2[iu, ju]                     # (npairs,)
    dr_pairs = dr[iu, ju]                     # (npairs, 3)
    qi_qj = q[iu] * q[ju]                     # (npairs,)
    
    in_cutoff = r2_pairs < rcut2              # (npairs,) bool
    r2_safe = jnp.where(in_cutoff, r2_pairs, 1.0)   # avoid div-by-zero
    r_inv = jnp.where(in_cutoff, 1.0 / jnp.sqrt(r2_safe), 0.0)
    r3_inv = r_inv * r_inv * r_inv
    
    # Energy: sum over pairs of k_e * q_i * q_j / r
    e_pairs = COULOMB_K * qi_qj * r_inv       # (npairs,)
    energy = jnp.sum(e_pairs)
    
    # Force on atom i from j: k_e * q_i * q_j / r^3 * r_ij_vec, where
    # r_ij_vec = x_i - x_j = -dr_pairs (because dr_ij = x_j - x_i).
    # So force_on_i = -COULOMB_K * qi*qj/r^3 * dr_pairs
    f_pair_mag = COULOMB_K * qi_qj * r3_inv   # (npairs,) — scalar per pair
    f_pair_vec = -f_pair_mag[:, None] * dr_pairs  # (npairs, 3) — force on atom i
    
    # Scatter forces: f[i] += f_pair_vec, f[j] -= f_pair_vec
    forces = jnp.zeros((nat, 3))
    forces = forces.at[iu].add(f_pair_vec)
    forces = forces.at[ju].add(-f_pair_vec)
    
    return energy, forces


# JIT-compiled version. State is passed as a pytree; the int/bool fields
# in state are static (don't change between calls), the arrays are dynamic.
# In M2+ we'll need to be careful about which fields are static (donate them
# as `static_argnames`) to avoid recompilation.
coulomb_energy_force_jit = jax.jit(coulomb_energy_force)
