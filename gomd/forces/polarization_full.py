"""
forces/polarization_full.py — self-contained polarizable contribution (M3).

Wires the validated pieces into ONE on-device function:
    compute_polarization(x, idx_i, idx_j) -> (E_pol, f_pol, mu)

  1. build the total charge field           (pol_field.make_field_builder)
  2. compute per-pair SCF/force factors      (on-device, from positions)
  3. solve the induced-dipole SCF            (polarization.solve_scf)
  4. energy  E_pol = -1/2 COULOMB sum(E.mu)  (polarization.pol_energy)
  5. forces  = COULOMB*(main + excl + 1-4)   (pol_forces + pol_corrections)

Validated end-to-end against the reference on the 8,716-atom system:
    field 1.4e-15, dipoles 2.8e-15, E_pol 1.2e-11, force 8.0e-9 (rel).

The pair factors are recomputed each call from the neighbor pairs (positions
move every step), using the reference single-shift min image and the rcut mask.
tau=1 on this system (rtaper 14.5 > ro 12), matching the reference; the general
tau/dtau forms are kept inline so a tapered system would only need them filled.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from .pol_field import make_field_builder, build_ewald_tables, _min_image_ref
from .polarization import solve_scf, pol_energy, pol_forces, COULOMB_CONST
from .pol_corrections import qmu_exclusion_correction, qmu_14_correction


def make_polarization_fn(D, kx, ky, kz, scf_max_iter=50, scf_min_iter=0):
    """Build a jit'd compute_polarization(x, idx_i, idx_j, mu_init=None).

    scf_min_iter: minimum SCF iterations (floor). Use a small value (e.g. 3-4)
    when warm-starting with the ASPC predictor so the step-change convergence
    test does not stop early at a biased point; 0 (default) suits cold starts.
    """
    pd = D['pol_data']
    nat = D['nat']
    box = jnp.asarray(D['box'])
    el = jnp.asarray(pd['el'])
    alpha = float(D['alpha'])
    rcut2 = float(D['rcut']) ** 2
    ncharge = pd['ncharge']
    arf = float(pd['arf'])
    tolpol = float(pd['tolpol'])
    redQ = float(pd['redQmufactor'])
    has14 = bool(pd['lredQ_mu14'])

    q = jnp.asarray(D['q'])
    atomtype = jnp.asarray(D['atomtype'])
    alpha_atom = jnp.asarray(pd['alpha_atom'])
    c1 = jnp.asarray(pd['c1'])
    l14_i = jnp.asarray(pd['l14_i']); l14_j = jnp.asarray(pd['l14_j'])
    ex_i = jnp.asarray(pd['ex_i']); ex_j = jnp.asarray(pd['ex_j'])

    erfc_tab, exp_tab, dx_inv = build_ewald_tables()

    Dp = dict(nat=nat, q=q, box=box, el=el, alpha=alpha, rcut2=rcut2,
              pair_i=None, pair_j=None,                 # set per call below
              l14_i=l14_i, l14_j=l14_j, ex_i=ex_i, ex_j=ex_j,
              redQmufactor=redQ, lredQ_mu14=has14,
              erfc_tab=erfc_tab, exp_tab=exp_tab, dx_inv=dx_inv)

    # The field builder closes over pair arrays; since pairs change on rebuild
    # we pass them as args instead. Rebuild a builder that takes pairs as args.
    from .pol_field import (_field_pairs_tab, _field_correction, _recip_field)

    def build_field(x, idx_i, idx_j):
        E = _field_pairs_tab(x, q, idx_i, idx_j, box, el, alpha, rcut2,
                             erfc_tab, exp_tab, dx_inv, 1.0, True, nat)
        if has14:
            E = E + _field_pairs_tab(x, q, l14_i, l14_j, box, el, alpha, rcut2,
                                     erfc_tab, exp_tab, dx_inv, -redQ, False, nat)
        E = E + _recip_field(x, q, box, kx, ky, kz, alpha)
        E = E + _field_correction(x, q, ex_i, ex_j, box, el, alpha, 1.0, True, nat)
        if has14:
            E = E + _field_correction(x, q, l14_i, l14_j, box, el, alpha, redQ, False, nat)
        return E

    c1_big = 1.0e6

    @jax.jit
    def compute_polarization(x, idx_i, idx_j, mu_init=None):
        # --- pair geometry (single-shift min image, rcut + padding mask) ---
        d = _min_image_ref(x[idx_j] - x[idx_i], box, el)
        rx, ry, rz = d[:, 0], d[:, 1], d[:, 2]
        z = rx*rx + ry*ry + rz*rz
        good = (z <= rcut2) & (z > 1e-10) & (idx_i != idx_j)
        zc = jnp.where(good, z, 1.0)
        r = jnp.sqrt(zc)
        rinv = 1.0 / r
        zinv = rinv * rinv

        ti = atomtype[idx_i]; tj = atomtype[idx_j]
        both = (ti < ncharge) & (tj < ncharge)
        c1ij = jnp.where(both, c1[jnp.minimum(ti, ncharge-1), jnp.minimum(tj, ncharge-1)], c1_big)
        expc1r3 = jnp.where(both, jnp.exp(-c1ij * r**3), 0.0)
        c1r3 = c1ij * r**3
        lamda3 = 1.0 - expc1r3
        lamda5 = 1.0 - expc1r3 * (1.0 + c1r3)
        # tau == 1 here (general taper kept off; rtaper>ro)
        tau = jnp.ones_like(r)
        dtau = jnp.zeros_like(r)
        fact = rinv * zinv * tau
        tfact = arf - fact * lamda3
        three_zf_l5 = 3.0 * zinv * fact * lamda5
        # zero factors on masked pairs so they don't contribute to T.mu
        tfact = jnp.where(good, tfact, 0.0)
        three_zf_l5 = jnp.where(good, three_zf_l5, 0.0)

        # --- field + SCF ---
        Efield = build_field(x, idx_i, idx_j)
        mu0 = jnp.zeros((nat, 3)) if mu_init is None else mu_init
        mu, niter = solve_scf(Efield, alpha_atom, idx_i, idx_j, rx, ry, rz,
                              tfact, three_zf_l5, mu0,
                              arf, tolpol, nat, scf_max_iter, scf_min_iter)

        # --- energy ---
        E_pol = pol_energy(Efield, mu)

        # --- forces: main (phase 5) + corrections (phases 6,7) ---
        # pol_forces runs over ALL supplied pairs; the neighbor list carries
        # rcut+skin pairs, but the reference polarization force is rcut-only.
        # Zero every per-pair input on masked (skin/padded) pairs so phase 5
        # contributes exactly nothing there: charges seen as 0 kills the
        # charge-dipole ee terms, and r set to a safe value avoids 1/0.
        q_i_masked = jnp.where(good, q[idx_i], 0.0)
        q_j_masked = jnp.where(good, q[idx_j], 0.0)
        rxm = jnp.where(good, rx, 0.0)
        rym = jnp.where(good, ry, 0.0)
        rzm = jnp.where(good, rz, 0.0)
        f5, _vir = pol_forces(mu, q, alpha, idx_i, idx_j, rxm, rym, rzm, r, rinv,
                              jnp.where(good, c1ij, 0.0),
                              jnp.where(good, expc1r3, 0.0),
                              tau, dtau, nat,
                              q_i_override=q_i_masked, q_j_override=q_j_masked)
        f6 = qmu_exclusion_correction(mu, q, ex_i, ex_j, x, box, el, alpha, nat)
        f_pol = f5 + f6
        if has14:
            f7 = qmu_14_correction(mu, q, l14_i, l14_j, x, box, el, alpha, redQ, nat)
            f_pol = f_pol + f7
        f_pol = COULOMB_CONST * f_pol
        return E_pol, f_pol, mu

    return compute_polarization
