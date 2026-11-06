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
        mu: (nat,3) converged dipoles, niter.
    """
    def town(mu):
        mu_i = mu[idx_i]; mu_j = mu[idx_j]              # (npairs,3)
        rij = jnp.stack([rx, ry, rz], axis=1)           # (npairs,3)
        mjr = jnp.sum(mu_j * rij, axis=1)               # (npairs,)
        mir = jnp.sum(mu_i * rij, axis=1)
        pjr = three_zf_l5 * mjr                          # to atom i from mu_j
        pir = three_zf_l5 * mir                          # to atom j from mu_i
        contrib_i = pjr[:, None] * rij + mu_j * tfact[:, None]
        contrib_j = pir[:, None] * rij + mu_i * tfact[:, None]
        t = jnp.zeros((nat, 3))
        t = t.at[idx_i].add(contrib_i)
        t = t.at[idx_j].add(contrib_j)
        return t

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


def pol_energy(Efield, mu):
    """E_pol = -0.5 * COULOMB * sum_i E_charge_i . mu_i."""
    return -0.5 * COULOMB_CONST * jnp.sum(Efield * mu)


# ===========================================================================
# Polarization forces (M3)
# ===========================================================================
@partial(jax.jit, static_argnames=('nat',))
def pol_forces(mu, q_atom, alpha_ew, idx_i, idx_j,
               rx, ry, rz, rr, rrinv, c1ij_pair, expc1r3_pair,
               tau_pair, dtau_pair, nat,
               q_i_override=None, q_j_override=None):
    """Induced-dipole forces (before COULOMB scaling). Vectorized over pairs.

    Ported from reference compute_pol_forces. Uses exact erfc (matches the
    reference's fine erfc table to ~1e-7).

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

    f = jnp.zeros((nat, 3))
    f = f.at[idx_i].add(ff)
    f = f.at[idx_j].add(-ff)
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
