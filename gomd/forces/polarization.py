"""
forces/polarization.py - Induced-dipole polarization SCF (M3).

This is the heart of the engine and the piece that roadmap step 3 will replace
with ML. It is written modularly so the SCF solver is a clean, swappable
function: solve_scf(field, pair_factors, alpha) -> dipoles.

Physics (matches reference polarization.py):
    Each polarizable atom i gets an induced dipole mu_i = alpha_i * E_total,i
    where E_total = E_charge (from fixed charges) + sum_j T_ij . mu_j  (the
    dipole-dipole field), solved self-consistently.

    Dipole-dipole tensor with Thole damping (lamda3, lamda5) and tau taper:
        per pair (i,j), with r = |r_ij|, rinv=1/r, zinv=rinv^2:
        c1r3   = c1[ti,tj] * r^3
        lamda3 = 1 - exp(-c1r3)
        lamda5 = 1 - exp(-c1r3)*(1 + c1r3)
        tau    = tau_func(r)   (==1 here since rtaper>ro)
        fact   = rinv*zinv*tau
        T.mu contribution to atom i from mu_j:
            pjr = 3*zinv*fact*(mu_j . r_ij)*lamda5
            tfact = arf - fact*lamda3
            txown_i += pjr*rx + mu_j_x * tfact     (and y,z)
        symmetric contribution to j from mu_i.

    Update: mu_i = alpha_i * (E_charge_i + Town_i + arf*mu_i)
    Converged when max per-atom |dmu|^2 <= tolpol.

    Energy: E_pol = -0.5 * COULOMB * sum_i E_charge_i . mu_i

This module consumes:
    - the precomputed total charge field E_charge (incl recip + corrections),
      built host-side or by the field kernel, and
    - per-pair geometric factors (rx,ry,rz, rinv, zinv, lamda3, lamda5, tau)
      computed once per neighbor build.
The SCF itself (solve_scf) is pure JAX and jit/while_loop friendly.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

COULOMB_CONST = 332.08


def tau_func_np(r, rtaper, ro, tauden):
    """Tapering function (numpy, host-side precompute)."""
    out = np.ones_like(r)
    mid = (r > rtaper) & (r < ro)
    w = r - rtaper
    out = np.where(mid, 1.0 + w*w*(w/3.0 + 0.5*(rtaper-ro))/tauden, out)
    out = np.where(r >= ro, 0.0, out)
    return out


def build_pair_factors(rx, ry, rz, rr, rrinv, atomtype, c1, ncharge,
                       arf, rtaper, ro, tauden):
    """Precompute per-pair geometric factors for the SCF (host-side numpy).

    Inputs are OrderedSparse pair arrays (npairs,) for i<j pairs:
        rx,ry,rz : displacement x_j - x_i components
        rr       : |r_ij|
        rrinv    : 1/r
        (these come from the neighbor build / charge-field kernel)
    Returns a dict of (npairs,) factor arrays used by solve_scf.
    """
    r = rr
    rinv = rrinv
    zinv = rinv * rinv
    ti = atomtype  # already per-pair? No: caller passes per-pair types
    # caller supplies per-pair c1r3 etc. Here c1 indexing done by caller.
    return None  # (kept for reference; real packing done in pack_pairs)


def pack_pairs(idx_i, idx_j, x, box, atomtype, c1, ncharge,
               arf, rtaper, ro, tauden):
    """Build per-pair factor arrays for the SCF from positions (numpy host).

    Returns dict with arrays of shape (npairs,):
        rx,ry,rz, fact, lamda3, lamda5, tfact, three_zinv_fact_lamda5
    plus idx_i, idx_j. All static during the SCF (positions fixed during solve).
    """
    xi = x[idx_i]; xj = x[idx_j]
    dr = xj - xi
    dr = dr - box * np.round(dr / box)         # min image
    rx, ry, rz = dr[:, 0], dr[:, 1], dr[:, 2]
    r2 = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r2)
    rinv = 1.0 / r
    zinv = rinv * rinv

    ti = atomtype[idx_i]; tj = atomtype[idx_j]
    both = (ti < ncharge) & (tj < ncharge)
    c1ij = np.where(both, c1[ti, tj], 1.0e6)
    c1r3 = c1ij * r**3
    expc1r3 = np.where(both, np.exp(-c1[np.minimum(ti,ncharge-1), np.minimum(tj,ncharge-1)] * r**3), 0.0)
    # match reference: expc1r3 stored only when both<ncharge, else 0
    expc1r3 = np.where(both, np.exp(-c1ij * r**3), 0.0)
    lamda3 = 1.0 - expc1r3
    lamda5 = 1.0 - expc1r3 * (1.0 + c1r3)

    tau = tau_func_np(r, rtaper, ro, tauden)
    fact = rinv * zinv * tau
    tfact = arf - fact * lamda3
    three_zf_l5 = 3.0 * zinv * fact * lamda5

    return {
        'idx_i': idx_i.astype(np.int32), 'idx_j': idx_j.astype(np.int32),
        'rx': rx, 'ry': ry, 'rz': rz,
        'fact': fact, 'tfact': tfact, 'three_zf_l5': three_zf_l5,
    }


def town_product(mu, idx_i, idx_j, rx, ry, rz, tfact, three_zf_l5, nat):
    """Dipole field T.mu at every site: one matrix-vector product of the SCF.

    Per pair (i, j) with r_ij = x_j - x_i: the field at i from mu_j is
    three_zf_l5*(mu_j.r)*r + tfact*mu_j, and symmetrically at j from mu_i
    (T is symmetric). Masked pairs carry zero factors.
    """
    mu_i = mu[idx_i]; mu_j = mu[idx_j]              # (npairs,3)
    rij = jnp.stack([rx, ry, rz], axis=1)           # (npairs,3)
    mjr = jnp.sum(mu_j * rij, axis=1)               # (npairs,)
    mir = jnp.sum(mu_i * rij, axis=1)
    pjr = three_zf_l5 * mjr                          # to atom i from mu_j
    pir = three_zf_l5 * mir                          # to atom j from mu_i
    contrib_i = pjr[:, None] * rij + mu_j * tfact[:, None]
    contrib_j = pir[:, None] * rij + mu_i * tfact[:, None]
    # padded (0,0) slots go to a dropped index: adding their zeros into atom 0
    # serialised the scatter (6.8 -> 3.6 ms per product on an RTX 5070)
    v = idx_i != idx_j
    t = jnp.zeros((nat, 3))
    t = t.at[jnp.where(v, idx_i, nat)].add(contrib_i, mode='drop')
    t = t.at[jnp.where(v, idx_j, nat)].add(contrib_j, mode='drop')
    return t


@partial(jax.jit, static_argnames=('nat', 'max_iter', 'min_iter'))
def solve_scf(Efield, alpha_atom, idx_i, idx_j, rx, ry, rz, tfact, three_zf_l5,
              mu_init, arf, tolpol, nat, max_iter, min_iter=0):
    """Self-consistent induced-dipole solve (pure JAX, lax.while_loop).

    Args:
        Efield: (nat,3) total charge field at each atom (E_charge).
        alpha_atom: (nat,) atomic polarizability (0 for non-polarizable).
        idx_i, idx_j: (npairs,) pair indices.
        rx,ry,rz, tfact, three_zf_l5: (npairs,) precomputed factors.
        mu_init: (nat,3) initial dipole guess (ASPC predictor, host-built).
        arf, tolpol: scalars. nat,max_iter,min_iter static.
        min_iter: minimum iterations before the tolerance test can stop the
            loop. The Jacobi step-change test (max|Δμ|² ≤ tolpol) can trip
            early when warm-started near a biased point that barely moves per
            iteration; a small floor (e.g. 3) forces the dipoles to actually
            settle. Cold starts (mu=0) don't need it. Default 0 preserves the
            original behavior exactly.

    Returns:
        mu: (nat,3) converged dipoles, niter (= number of T.mu products,
        including the first one on mu_init).
    """
    def town(mu):
        return town_product(mu, idx_i, idx_j, rx, ry, rz, tfact, three_zf_l5, nat)

    def cond(state):
        mu, diff2max, it = state
        return ((diff2max > tolpol) | (it < min_iter)) & (it < max_iter)

    def body(state):
        mu, _d, it = state
        t = town(mu)
        mu_new = alpha_atom[:, None] * (Efield + t + arf * mu)
        dmu = mu_new - mu
        diff2 = jnp.sum(dmu * dmu, axis=1)              # (nat,)
        diff2max = jnp.max(diff2)
        return (mu_new, diff2max, it + 1)

    mu0 = mu_init
    t0 = town(mu0)
    mu1 = alpha_atom[:, None] * (Efield + t0 + arf * mu0)
    d0 = jnp.max(jnp.sum((mu1 - mu0) ** 2, axis=1))
    mu, diff2max, niter = jax.lax.while_loop(
        cond, body, (mu1, d0, jnp.array(1)))
    return mu, niter


@partial(jax.jit, static_argnames=('nat', 'max_iter', 'min_iter'))
def solve_scf_pcg(Efield, alpha_atom, idx_i, idx_j, rx, ry, rz, tfact, three_zf_l5,
                  mu_init, arf, tolpol, nat, max_iter, min_iter=0):
    """Induced dipoles by preconditioned conjugate gradient.

    The SCF condition mu = alpha*(E + T.mu + arf*mu) is the linear system
        A mu = E,   A = diag(1/alpha - arf) - T     (polarizable sites only),
    with T the symmetric Thole-damped dipole tensor, so A is symmetric and,
    without a polarization catastrophe, positive definite. CG minimises
        U(mu) = 1/2 mu.A.mu - E.mu
    whose minimum is E_pol/COULOMB = -1/2 E.mu. Preconditioner: 1/A_ii.

    Convergence uses the reference metric on the current residual
    r = E - A.mu: stop when max_i |alpha_i r_i|^2 <= tolpol, i.e. when the
    next Jacobi step would change no dipole by more than sqrt(tolpol). Unlike
    the Jacobi step-change test this measures the equation itself, so a warm
    start (mu_init from a predictor) is safe.

    Same arguments as solve_scf. mu_init=None is a cold start (mu = 0) and
    skips the product A.0. Sites with alpha = 0 keep mu = 0.

    Returns:
        mu (nat,3), niter (number of T.mu products), r (nat,3) the final
        residual E - A.mu (zero on non-polarizable sites). With r, the energy
        U = -1/2 E.mu - 1/2 mu.r is correct to second order in the error of mu.
    """
    pol = alpha_atom > 0.0
    m = pol[:, None]
    a_safe = jnp.where(pol, alpha_atom, 1.0)
    diag = jnp.where(pol, 1.0 / a_safe - arf, 0.0)               # A_ii
    minv = jnp.where(pol, a_safe / (1.0 - arf * a_safe), 0.0)    # 1 / A_ii
    jw = jnp.where(pol, alpha_atom, 0.0)                         # Jacobi step = alpha*r

    def amul(p):
        t = town_product(p, idx_i, idx_j, rx, ry, rz, tfact, three_zf_l5, nat)
        return jnp.where(m, diag[:, None] * p - t, 0.0)

    def jstep2(r):
        s = jw[:, None] * r
        return jnp.max(jnp.sum(s * s, axis=1))

    b = jnp.where(m, Efield, 0.0)
    if mu_init is None:
        x0 = jnp.zeros_like(b)
        r0 = b
        it0 = 0
    else:
        x0 = jnp.where(m, mu_init, 0.0)
        r0 = b - amul(x0)
        it0 = 1
    z0 = minv[:, None] * r0
    rz0 = jnp.sum(r0 * z0)

    def cond(state):
        _x, _r, _p, _rz, d2, it = state
        return ((d2 > tolpol) | (it < min_iter)) & (it < max_iter)

    def body(state):
        x, r, p, rz, _d2, it = state
        ap = amul(p)
        pap = jnp.sum(p * ap)
        ok = pap > 0.0
        a = jnp.where(ok, rz / jnp.where(ok, pap, 1.0), 0.0)
        x = x + a * p
        r = r - a * ap
        z = minv[:, None] * r
        rz_new = jnp.sum(r * z)
        okb = rz > 0.0
        beta = jnp.where(okb, rz_new / jnp.where(okb, rz, 1.0), 0.0)
        p = z + beta * p
        # p.A.p <= 0: A is not positive definite (polarization catastrophe) or
        # CG broke down. Stop at once and report niter = max_iter, which the
        # driver flags as an unconverged SCF.
        it_next = jnp.where(ok, it + 1, jnp.asarray(max_iter, dtype=it.dtype))
        return (x, r, p, rz_new, jstep2(r), it_next)

    state0 = (x0, r0, z0, rz0, jstep2(r0), jnp.asarray(it0, dtype=jnp.int32))
    mu, r, _p, _rz, _d2, niter = jax.lax.while_loop(cond, body, state0)
    return mu, niter, r


def pol_energy(Efield, mu):
    """E_pol = -0.5 * COULOMB * sum_i E_charge_i . mu_i."""
    return -0.5 * COULOMB_CONST * jnp.sum(Efield * mu)


# ===========================================================================
# Polarization forces (M3)
# ===========================================================================
@partial(jax.jit, static_argnames=('nat', 'fp32_special'))
def pol_forces(mu, q_atom, alpha_ew, idx_i, idx_j,
               rx, ry, rz, rr, rrinv, c1ij_pair, expc1r3_pair,
               tau_pair, dtau_pair, nat,
               q_i_override=None, q_j_override=None,
               erfc_tab=None, exp_tab=None, dx_inv=None, fp32_special=False):
    """Induced-dipole forces (before COULOMB scaling). Vectorized over pairs.

    Ported from reference compute_pol_forces. erfc(alpha r) and exp(-(alpha r)^2):
      * erfc_tab/exp_tab/dx_inv given (pol_field.build_ewald_tables): the
        reference's own 16,384-knot tables with linear interpolation, exactly
        as compute_pol_forces and the SCF field use them (default in
        polarization_full since 2026-11-17: the exact fp64 erfc made this
        the most expensive kernel of the step);
      * fp32_special: both evaluated in fp32 (benchmark variant);
      * otherwise exact fp64 (agrees with the tables to ~1e-7).

    Args (all (npairs,) unless noted):
        mu: (nat,3) converged dipoles.
        q_atom: (nat,) per-atom charge.
        alpha_ew: Ewald alpha (scalar).
        idx_i, idx_j: pair indices.
        rx,ry,rz: displacement components (x_j - x_i).
        rr, rrinv: r and 1/r.
        c1ij_pair: Thole c1[ti,tj] per pair.
        expc1r3_pair: exp(-c1*r^3) per pair.
        tau_pair, dtau_pair: tau and dtau per pair.
        q_i_override, q_j_override: optional (npairs,) per-pair charges. When
            given they replace q_atom[idx_i]/[idx_j]; used to zero masked
            (skin/padded) pairs so they contribute no force. The dipole terms
            on those pairs are killed separately by zeroed rx/ry/rz + c1/expc1r3.
    Returns:
        f: (nat,3) induced forces (unscaled), virial scalar.
    """
    from jax.scipy.special import erfc as erfc_exact
    rootpi = jnp.sqrt(jnp.pi)
    twoalphapi = 2.0 * alpha_ew / rootpi

    ci = q_atom[idx_i] if q_i_override is None else q_i_override
    cj = q_atom[idx_j] if q_j_override is None else q_j_override
    r = rr; rinv = rrinv
    z = r * r
    zinv = rinv * rinv
    r3inv = zinv * rinv
    r5inv = r3inv * zinv

    c1r3 = c1ij_pair * r**3
    lamda3 = 1.0 - expc1r3_pair
    lamda5 = 1.0 - expc1r3_pair * (1.0 + c1r3)
    three_c1r2 = 3.0 * c1ij_pair * z
    dlamda3 = three_c1r2 * expc1r3_pair
    dlamda5 = three_c1r2 * c1r3 * expc1r3_pair

    mu_i = mu[idx_i]; mu_j = mu[idx_j]
    rij = jnp.stack([rx, ry, rz], axis=1)
    pjr = jnp.sum(mu_j * rij, axis=1)
    pir = jnp.sum(mu_i * rij, axis=1)
    pipj = jnp.sum(mu_i * mu_j, axis=1)
    qq = ci * pjr - cj * pir

    p1 = 5.0 * zinv * pjr * pir * lamda5 - pipj * lamda3
    p2 = dtau_pair * rinv * (3.0 * r5inv * pir * pjr * lamda5 - pipj * r3inv * lamda3)
    p3 = 3.0 * r5inv * tau_pair
    p4 = p3 * p1 - p2
    p6 = (3.0 * r5inv * pir * pjr * dlamda5 - pipj * r3inv * dlamda3) * rinv

    alphar = alpha_ew * r
    if erfc_tab is not None:
        from .pol_field import _tab_lookup
        erfc_v = _tab_lookup(alphar, erfc_tab, dx_inv)
        exp_v = _tab_lookup(alphar, exp_tab, dx_inv)
    elif fp32_special:
        a32 = alphar.astype(jnp.float32)
        erfc_v = erfc_exact(a32).astype(alphar.dtype)
        exp_v = jnp.exp(-a32 * a32).astype(alphar.dtype)
    else:
        erfc_v = erfc_exact(alphar)
        exp_v = jnp.exp(-alphar * alphar)
    ee1 = erfc_v * r3inv
    ee2 = zinv * twoalphapi * exp_v
    ee3 = ee2 * (3.0 + 2.0 * alphar * alphar)
    ee5 = qq * zinv * (3.0 * ee1 + ee3)
    ee6 = ee1 + ee2

    f1 = p4 + ee5 - p6 * tau_pair
    f2 = -(ee6 * ci + p3 * pir * lamda5)
    f3 = ee6 * cj - p3 * pjr * lamda5

    ffx = f1 * rx + f2 * mu_j[:, 0] + f3 * mu_i[:, 0]
    ffy = f1 * ry + f2 * mu_j[:, 1] + f3 * mu_i[:, 1]
    ffz = f1 * rz + f2 * mu_j[:, 2] + f3 * mu_i[:, 2]
    ff = jnp.stack([ffx, ffy, ffz], axis=1)

    v = idx_i != idx_j                     # padded (0,0) slots: dropped, not added to atom 0
    f = jnp.zeros((nat, 3))
    f = f.at[jnp.where(v, idx_i, nat)].add(ff, mode='drop')
    f = f.at[jnp.where(v, idx_j, nat)].add(-ff, mode='drop')
    vir = -jnp.sum(ffx * rx + ffy * ry + ffz * rz)
    return f, vir


# ===========================================================================
# Charge electric field - real-space part (M3)
# ===========================================================================
@partial(jax.jit, static_argnames=('nat',))
def charge_field_realspace(q_atom, alpha_ew, idx_i, idx_j, rx, ry, rz, rr, rrinv, nat):
    """Real-space Ewald electric field at each atom from fixed charges.

    E_i += -q_j * (erfc(a r)/r + 2a/sqrt(pi) exp(-(a r)^2))/r^2 * r_ij   (on i)
    E_j += +q_i * (...) * r_ij                                            (on j)
    where r_ij = x_j - x_i.

    Matches reference compute_charge_efield_and_pairs (field part), using exact
    erfc (matches the reference erfc-table to ~1e-7).
    Returns E: (nat,3). (Reciprocal + exclusion + 1-4 corrections are added
    separately; see build_total_field.)

    NOTE (M3 status): this real-space field kernel is NOT yet bit-validated
    standalone - a discrepancy vs the reference dumped field remains under
    investigation (likely erfc-table interpolation or the 1-4 Q-mu field term
    that the reference folds in here). The SCF/energy/forces ARE validated
    using the reference total field. Use the reference field-builder until this
    is resolved; see M2_BUILD_PLAN.md "REMAINING for fully self-contained M3".
    """
    from jax.scipy.special import erfc as erfc_exact
    rootpi = jnp.sqrt(jnp.pi)
    twoalphapi = 2.0 * alpha_ew / rootpi

    r = rr; rinv = rrinv
    zinv = rinv * rinv
    alphar = alpha_ew * r
    erfc_v = erfc_exact(alphar)
    exp_v = jnp.exp(-alphar * alphar)
    fact = (erfc_v * rinv + twoalphapi * exp_v) * zinv     # (npairs,)

    ci = q_atom[idx_i]; cj = q_atom[idx_j]
    rij = jnp.stack([rx, ry, rz], axis=1)
    contrib_i = -(fact * cj)[:, None] * rij     # field at i from charge j
    contrib_j = (fact * ci)[:, None] * rij      # field at j from charge i

    E = jnp.zeros((nat, 3))
    E = E.at[idx_i].add(contrib_i)
    E = E.at[idx_j].add(contrib_j)
    return E
