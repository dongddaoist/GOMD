"""
forces/polarization_full.py — self-contained polarizable contribution (M3).

Wires the validated pieces into ONE on-device function:
    compute_polarization(x, idx_i, idx_j, mu_init=None)
        -> (E_pol, f_pol, mu, stats)          stats = {'niter': int32}

  1. per-pair SCF/force factors from positions (geometry)
  2. the total charge field                    (pol_field kernels)
  3. the induced-dipole SCF                    (polarization.solve_scf, Jacobi,
                                                or solve_scf_pcg)
  4. energy  E_pol = -1/2 COULOMB sum(E.mu)    (polarization.pol_energy)
                     [- 1/2 COULOMB sum(mu.r)  with PCG: variational form]
  5. forces  = COULOMB*(main + excl + 1-4 + k-space Q-mu)

make_polarization_parts() returns each stage as a separate function (the
profiler times them one by one); make_polarization_fn() returns the jit'd whole.

Validated end-to-end against the reference on the 8,716-atom system (Jacobi):
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

from .pol_field import (build_ewald_tables, _min_image_ref, _field_pairs_tab,
                        _field_correction, _recip_field)
from .polarization import (solve_scf, solve_scf_pcg, town_product, pol_energy,
                           pol_forces, COULOMB_CONST)
from .pol_corrections import qmu_exclusion_correction, qmu_14_correction
from .recip_shared import make_recip

SOLVERS = ('jacobi', 'pcg')


def make_polarization_parts(D, kx, ky, kz, scf_max_iter=50, scf_min_iter=0,
                            recip_qmu_force=True, solver='jacobi', recip=None):
    """The polarizable term as separate (un-jit'd) stages; see the module doc.

    scf_min_iter: minimum SCF iterations (floor). 0 suits cold starts.
    recip_qmu_force: include the reciprocal-space charge-dipole force
        F = COULOMB * d/dx [ sum_i mu_i . E0_recip,i(x) ]  (mu fixed: at the SCF
        solution E_pol is stationary in mu). E_pol contains the k-space charge
        field, so this term is part of -dE_pol/dx; the reference pkg leaves it
        out (compute_recipQmu is a TODO there), which makes its polarizable
        forces non-conservative: about +0.1 kcal/mol/fs of NVE drift on the
        8,716-site system. False reproduces the reference forces.
    solver: 'jacobi' (the reference iteration, validated bit-for-bit) or
        'pcg' (preconditioned conjugate gradient with a residual test, safe to
        warm-start). With 'pcg' the energy is the variational
        -1/2 E.mu - 1/2 mu.r, which equals -1/2 E.mu at convergence and is
        accurate to second order in the dipole error before it.
    recip: the shared k-space pass (recip_shared.make_recip); None builds one.
        The reciprocal field and the k-space Q-mu force come from its tables
        (the Q-mu force in closed form, formerly jax.grad of the field pass);
        compute(..., rt=tables) reuses tables the caller already built for
        the classical reciprocal force.

    Returns a dict:
        geometry(x, idx_i, idx_j) -> G       per-pair factors (dict)
        field(x, idx_i, idx_j) -> E          total charge field (nat,3)
        field_parts: real(x,i,j), real14(x), recip(x), excl(x), corr14(x)
        town(mu, G, idx_i, idx_j) -> T.mu    one SCF matrix-vector product
        scf(E, G, idx_i, idx_j, mu_init=None) -> (mu, niter, r or None)
        energy(E, mu, r) -> E_pol
        forces(x, mu, G, idx_i, idx_j) -> f_pol (COULOMB applied)
        force_parts: main(x,mu,G,i,j), excl(x,mu), c14(x,mu), recip_qmu(x,mu)
                     (unscaled, as summed inside forces)
        compute(x, idx_i, idx_j, mu_init=None, rt=None) -> (E_pol, f_pol, mu, stats)
        has14, solver, tolpol, alpha_atom, arf, nat, recip
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
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
    c1_big = 1.0e6
    rtaper = float(pd['rtaper']); ro = float(pd['ro'])
    taper_on = rtaper < ro
    tauden = (ro - rtaper) ** 3 / 6.0 if taper_on else 1.0
    if recip is None:
        recip = make_recip(D['box'], D['q'], kx, ky, kz, alpha)
    tables = recip['tables']

    # ---------------- 1. charge field (pairs passed as arguments) ----------------
    def field_real(x, idx_i, idx_j):
        return _field_pairs_tab(x, q, idx_i, idx_j, box, el, alpha, rcut2,
                                erfc_tab, exp_tab, dx_inv, 1.0, True, nat)

    def field_real14(x):
        return _field_pairs_tab(x, q, l14_i, l14_j, box, el, alpha, rcut2,
                                erfc_tab, exp_tab, dx_inv, -redQ, False, nat)

    def field_recip(x, rt=None):
        return (tables(x) if rt is None else rt)['field']

    def field_excl(x):
        return _field_correction(x, q, ex_i, ex_j, box, el, alpha, 1.0, True, nat)

    def field_corr14(x):
        return _field_correction(x, q, l14_i, l14_j, box, el, alpha, redQ, False, nat)

    def build_field(x, idx_i, idx_j, rt=None):
        E = field_real(x, idx_i, idx_j)
        if has14:
            E = E + field_real14(x)
        E = E + field_recip(x, rt)
        E = E + field_excl(x)
        if has14:
            E = E + field_corr14(x)
        return E

    # ---------------- 2. pair geometry + Thole factors ----------------
    def geometry(x, idx_i, idx_j):
        # single-shift min image, rcut + padding mask
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
        # dipole-dipole taper (reference tau_func): 1 below rtaper, a cubic
        # switch to 0 at ro, tauden = (ro - rtaper)^3 / 6. Off when rtaper >= ro
        # (this input: 14.5 > 12, so the mu-mu terms are cut hard at ro).
        # dtau is the true derivative (r - rtaper)(r - ro)/tauden; the
        # reference dtau_func has an extra factor (r - rtaper), which is not
        # the derivative of its own tau_func.
        if taper_on:
            w = r - rtaper
            mid = (r > rtaper) & (r < ro)
            tau = jnp.where(r <= rtaper, 1.0,
                            jnp.where(r >= ro, 0.0,
                                      1.0 + w * w * (w / 3.0 + 0.5 * (rtaper - ro)) / tauden))
            dtau = jnp.where(mid, w * (r - ro) / tauden, 0.0)
        else:
            tau = jnp.ones_like(r)
            dtau = jnp.zeros_like(r)
        fact = rinv * zinv * tau
        tfact = arf - fact * lamda3
        three_zf_l5 = 3.0 * zinv * fact * lamda5
        # zero factors on masked pairs so they don't contribute to T.mu
        tfact = jnp.where(good, tfact, 0.0)
        three_zf_l5 = jnp.where(good, three_zf_l5, 0.0)
        return {'rx': rx, 'ry': ry, 'rz': rz, 'r': r, 'rinv': rinv, 'good': good,
                'c1ij': c1ij, 'expc1r3': expc1r3, 'tau': tau, 'dtau': dtau,
                'tfact': tfact, 'three_zf_l5': three_zf_l5}

    # ---------------- 3. SCF ----------------
    def town(mu, G, idx_i, idx_j):
        return town_product(mu, idx_i, idx_j, G['rx'], G['ry'], G['rz'],
                            G['tfact'], G['three_zf_l5'], nat)

    def scf(E, G, idx_i, idx_j, mu_init=None):
        if solver == 'pcg':
            return solve_scf_pcg(E, alpha_atom, idx_i, idx_j, G['rx'], G['ry'], G['rz'],
                                 G['tfact'], G['three_zf_l5'], mu_init,
                                 arf, tolpol, nat, scf_max_iter, scf_min_iter)
        mu0 = jnp.zeros((nat, 3)) if mu_init is None else mu_init
        mu, niter = solve_scf(E, alpha_atom, idx_i, idx_j, G['rx'], G['ry'], G['rz'],
                              G['tfact'], G['three_zf_l5'], mu0,
                              arf, tolpol, nat, scf_max_iter, scf_min_iter)
        return mu, niter, None

    # ---------------- 4. energy ----------------
    def energy(E, mu, r=None):
        e = pol_energy(E, mu)
        if r is not None:
            e = e - 0.5 * COULOMB_CONST * jnp.sum(mu * r)
        return e

    # ---------------- 5. forces: main (phase 5) + corrections (6, 7, 8) ----------------
    def force_main(x, mu, G, idx_i, idx_j):
        # pol_forces runs over ALL supplied pairs; the neighbor list carries
        # rcut+skin pairs, but the reference polarization force is rcut-only.
        # Zero every per-pair input on masked (skin/padded) pairs so phase 5
        # contributes exactly nothing there: charges seen as 0 kills the
        # charge-dipole ee terms, and r set to a safe value avoids 1/0.
        good = G['good']
        q_i_masked = jnp.where(good, q[idx_i], 0.0)
        q_j_masked = jnp.where(good, q[idx_j], 0.0)
        rxm = jnp.where(good, G['rx'], 0.0)
        rym = jnp.where(good, G['ry'], 0.0)
        rzm = jnp.where(good, G['rz'], 0.0)
        f5, _vir = pol_forces(mu, q, alpha, idx_i, idx_j, rxm, rym, rzm, G['r'], G['rinv'],
                              jnp.where(good, G['c1ij'], 0.0),
                              jnp.where(good, G['expc1r3'], 0.0),
                              G['tau'], G['dtau'], nat,
                              q_i_override=q_i_masked, q_j_override=q_j_masked,
                              erfc_tab=erfc_tab, exp_tab=exp_tab, dx_inv=dx_inv)
        return f5

    def force_excl(x, mu):
        return qmu_exclusion_correction(mu, q, ex_i, ex_j, x, box, el, alpha, nat)

    def force_c14(x, mu):
        return qmu_14_correction(mu, q, l14_i, l14_j, x, box, el, alpha, redQ, nat)

    def force_recip_qmu(x, mu, rt=None):
        # closed form on the shared k-space tables; equals jax.grad of
        # sum(mu . _recip_field) at fixed mu to 1.4e-15 (tests/bench_recip.py)
        return recip['qmu_grad'](tables(x) if rt is None else rt, mu)

    def forces(x, mu, G, idx_i, idx_j, rt=None):
        f_pol = force_main(x, mu, G, idx_i, idx_j) + force_excl(x, mu)
        if has14:
            f_pol = f_pol + force_c14(x, mu)
        if recip_qmu_force:
            f_pol = f_pol + force_recip_qmu(x, mu, rt)
        return COULOMB_CONST * f_pol

    # ---------------- all together ----------------
    def compute(x, idx_i, idx_j, mu_init=None, rt=None):
        # gomd_* named scopes label the stages in profiler traces (no run-time cost)
        if rt is None:
            with jax.named_scope('gomd_kspace'):
                rt = tables(x)
        with jax.named_scope('gomd_pol_geometry'):
            G = geometry(x, idx_i, idx_j)
        with jax.named_scope('gomd_pol_field'):
            E = build_field(x, idx_i, idx_j, rt)
        with jax.named_scope('gomd_pol_scf'):
            mu, niter, r = scf(E, G, idx_i, idx_j, mu_init)
            E_pol = energy(E, mu, r)
        with jax.named_scope('gomd_pol_forces'):
            f_pol = forces(x, mu, G, idx_i, idx_j, rt)
        return E_pol, f_pol, mu, {'niter': jnp.asarray(niter).astype(jnp.int32)}

    return {
        'geometry': geometry, 'field': build_field,
        'field_parts': {'real': field_real, 'real14': field_real14, 'recip': field_recip,
                        'excl': field_excl, 'corr14': field_corr14},
        'town': town, 'scf': scf, 'energy': energy, 'forces': forces,
        'force_parts': {'main': force_main, 'excl': force_excl, 'c14': force_c14,
                        'recip_qmu': force_recip_qmu},
        'compute': compute,
        'has14': has14, 'solver': solver, 'tolpol': tolpol,
        'alpha_atom': alpha_atom, 'arf': arf, 'nat': nat, 'recip': recip,
    }


def make_polarization_fn(D, kx, ky, kz, scf_max_iter=50, scf_min_iter=0,
                         recip_qmu_force=True, solver='jacobi', recip=None):
    """jit'd compute_polarization(x, idx_i, idx_j, mu_init=None, rt=None)
    -> (E_pol, f_pol, mu, {'niter'}). Arguments: see make_polarization_parts."""
    parts = make_polarization_parts(D, kx, ky, kz, scf_max_iter=scf_max_iter,
                                    scf_min_iter=scf_min_iter,
                                    recip_qmu_force=recip_qmu_force, solver=solver,
                                    recip=recip)
    return jax.jit(parts['compute'])
