"""
forces/all_forces.py — Unified M2 force evaluation.

Assembles the six validated pieces into one force/energy call:
    splined pair (vdW + real-space elec)   [M2.2]
    bonded (bond/bend/torsion)             [M2.3]
    Ewald reciprocal + self + exclusion    [M2.4]
    (neighbor list                          [M2.1] built on host, passed in)
    (SHAKE                                  [M2.5] applied in integrator)
    (dummy coords/forces                    [M2.6] applied in integrator)

This returns total potential energy and the per-atom force from the
position-dependent interactions. SHAKE and dummy redistribution are applied
in the integrator (they are not simple additive forces).

No polarization (that's M3). So energies here = classical APPLE&P:
    PE = E_bond + E_bend + E_tort + E_vdw + E_realelec + E_recip - E_self + E_corr
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from .pair_splined import splined_pair_energy_force, splined_pair_energy_force_fp32
from .pair_splined_segsum import splined_pair_energy_force_segsum
from .bonded import bond_energy_force, bend_energy_force, torsion_energy_force
from .ewald import (
    reciprocal_energy_force, exclusion_correction,
)


def compute_all_forces(x, D, idx_i, idx_j, kx, ky, kz, ewself):
    """Full classical (non-polarizable) force + energy.

    Args:
        x: (nat,3) positions (dummies already placed).
        D: the dict of jax arrays from the assembly loader (see below).
        idx_i, idx_j: neighbor pairs (host-built, jax arrays).
        kx, ky, kz: Ewald k-vectors (host-built, jax arrays).
        ewself: scalar Ewald self-energy (precomputed).

    Returns:
        pe (scalar), forces (nat,3), components (dict of scalars).
    """
    nat = D['nat']
    box = D['box']

    # --- Splined pair (vdW + real-space electrostatics) ---
    # Mixed precision: if fp32 tables are present, use the fp32 pair kernel
    # (~2x faster on GPU, energy rel ~1e-7, force rel ~3e-5). Falls back to
    # fp64 otherwise.
    if D.get('use_segsum_pair', False):
        unbd, f_pair = splined_pair_energy_force_segsum(
            x, idx_i, idx_j, D['atomtype'], D['typee'],
            D['coefft_m32'], D['coeffft_m32'],
            D['zs32'], D['deltainv'], D['rcut'] * D['rcut'], box, nat,
        )
    elif D.get('use_fp32_pair', False):
        unbd, f_pair = splined_pair_energy_force_fp32(
            x, idx_i, idx_j, D['atomtype'], D['typee'],
            D['coefft_m32'], D['coeffft_m32'],
            D['zs32'], D['deltainv'], D['rcut'] * D['rcut'], box, nat,
        )
    else:
        unbd, f_pair = splined_pair_energy_force(
            x, idx_i, idx_j, D['atomtype'], D['typee'],
            D['coefft_m'], D['coeffft_m'],
            D['zs'], D['deltainv'], D['rcut'] * D['rcut'], box, nat,
        )

    # --- Bonded ---
    eb, fb = bond_energy_force(x, D['bonds'], D['stretch'], box, nat)
    ebe, fbe = bend_energy_force(x, D['bends'], D['bend'], box, nat)
    et, ft = torsion_energy_force(x, D['torts'], D['twist'], D['nprms'],
                                  box, nat, D['max_torsion_terms'])

    # --- Ewald reciprocal + self + exclusion correction ---
    erecip_raw, f_recip = reciprocal_energy_force(
        x, D['q'], box, kx, ky, kz, D['alpha'])
    erecip = erecip_raw - ewself
    ecorr, f_corr = exclusion_correction(
        x, D['excl_qq'], D['excl_i'], D['excl_j'], box, D['alpha'], nat)

    forces = f_pair + fb + fbe + ft + f_recip + f_corr
    pe = unbd + eb + ebe + et + erecip + ecorr

    comps = {
        'evdw_realelec': unbd,
        'ebond': eb, 'ebend': ebe, 'etort': et,
        'erecip': erecip, 'ecorr': ecorr,
    }
    return pe, forces, comps
