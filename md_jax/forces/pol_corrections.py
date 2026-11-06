"""
forces/pol_corrections.py — induced-dipole force corrections (M3 phases 6, 7).

The main induced force (compute_pol_forces) is the validated `pol_forces` in
polarization.py. The reference adds two correction passes before scaling by
COULOMB:

  Phase 6  compute_qmu_correction       — erf charge-dipole correction over the
           excluded (bonded) pair list, /2 (symmetric list).
  Phase 7  compute_qmu_14_corrections   — Q-mu correction over the 1-4 list.
           Two sub-branches: lredonefour (off here, redfactor=0) and lredQ_mu14
           (on). Only the active branch is ported; the structure mirrors the
           reference so the other can be added if a system needs it.

Phase 8 (recipQmu, k-space Q-mu) is a TODO in the reference itself and is NOT
applied to the reference forces, so parity does not require it.

All use the A&S erf/erfc polynomials (erf_approx/erfc_approx) exactly as the
reference correction routines do, and the single-shift min image (el = box/2).
Returns forces in the (nat,3) layout, BEFORE the COULOMB (332.08) scale — the
caller applies it, matching `state.atoms.f += COULOMB_CONST * findown`.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

PI = np.pi


def _erfc_approx_jax(x):
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p = -1.453152027, 1.061405429, 0.3275911
    ax = jnp.abs(x)
    t = 1.0 / (1.0 + p * ax)
    val = (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*jnp.exp(-ax*ax)
    return jnp.where(x < 0, 2.0 - val, val)


def _erf_approx_jax(x):
    return 1.0 - _erfc_approx_jax(x)


def _min_image_ref(d, box, el):
    return d - jnp.where(jnp.abs(d) > el, jnp.sign(d) * box, 0.0)


@partial(jax.jit, static_argnames=('nat',))
def qmu_exclusion_correction(mu, q, ex_i, ex_j, x, box, el, alpha, nat):
    """Phase 6: erf charge-dipole correction over excluded pairs (/2)."""
    i = ex_i; j = ex_j
    ci = q[i]; cj = q[j]
    d = _min_image_ref(x[j] - x[i], box, el)
    rx, ry, rz = d[:, 0], d[:, 1], d[:, 2]
    z = jnp.sum(d * d, axis=1)
    good = z > 1e-10
    r = jnp.sqrt(jnp.where(good, z, 1.0))
    rinv = 1.0 / r
    zinv = rinv * rinv
    r3inv = zinv * rinv
    alphar = alpha * r
    twoalphapi = 2.0 * alpha / jnp.sqrt(PI)

    mui = mu[i]; muj = mu[j]
    pir = jnp.sum(mui * d, axis=1)
    pjr = jnp.sum(muj * d, axis=1)
    qq = ci * pjr - cj * pir

    ee1 = -_erf_approx_jax(alphar) * r3inv
    ee2 = zinv * twoalphapi * jnp.exp(-alphar * alphar)
    ee3 = ee2 * (3.0 + 2.0 * alphar * alphar)
    ee5 = qq * zinv * (3.0 * ee1 + ee3)
    ee6 = ee1 + ee2

    # f = (ee5*r - ee6*(ci*mu_j - cj*mu_i)) / 2
    cross = ci[:, None] * muj - cj[:, None] * mui
    ff = (ee5[:, None] * d - ee6[:, None] * cross) * 0.5
    ff = jnp.where(good[:, None], ff, 0.0)

    f = jnp.zeros((nat, 3))
    f = f.at[i].add(ff)
    f = f.at[j].add(-ff)
    return f


@partial(jax.jit, static_argnames=('nat',))
def qmu_14_correction(mu, q, l14_i, l14_j, x, box, el, alpha, redQmufactor, nat):
    """Phase 7 (lredQ_mu14 branch): Q-mu correction over 1-4 pairs.

    Two contributions per pair (matching the reference exactly):
      (a) erfc Q-mu term scaled by -redQmufactor
      (b) erf correction term scaled by +redQmufactor
    """
    i = l14_i; j = l14_j
    ci = q[i]; cj = q[j]
    d = _min_image_ref(x[j] - x[i], box, el)
    z = jnp.sum(d * d, axis=1)
    good = z > 1e-10
    r = jnp.sqrt(jnp.where(good, z, 1.0))
    rinv = 1.0 / r
    zinv = rinv * rinv
    r3inv = zinv * rinv
    alphar = alpha * r
    twoalphapi = 2.0 * alpha / jnp.sqrt(PI)

    mui = mu[i]; muj = mu[j]
    pir = jnp.sum(mui * d, axis=1)
    pjr = jnp.sum(muj * d, axis=1)
    qq = ci * pjr - cj * pir

    ee2 = zinv * twoalphapi * jnp.exp(-alphar * alphar)
    ee3 = ee2 * (3.0 + 2.0 * alphar * alphar)

    # (a) erfc term, scale -redQmufactor
    ee1 = _erfc_approx_jax(alphar) * r3inv
    ee5 = qq * zinv * (3.0 * ee1 + ee3)
    ee6 = ee1 + ee2
    crossA = ci[:, None] * muj - cj[:, None] * mui     # ee6*ci*mu_j - ee6*cj*mu_i
    ffA = (ee5[:, None] * d - ee6[:, None] * crossA) * (-redQmufactor)

    # (b) erf correction term, scale +redQmufactor
    ee1c = -_erf_approx_jax(alphar) * r3inv
    ee5c = qq * zinv * (3.0 * ee1c + ee3)
    ee6c = ee1c + ee2
    crossB = ci[:, None] * muj - cj[:, None] * mui
    ffB = (ee5c[:, None] * d - ee6c[:, None] * crossB) * redQmufactor

    ff = ffA + ffB
    ff = jnp.where(good[:, None], ff, 0.0)

    f = jnp.zeros((nat, 3))
    f = f.at[i].add(ff)
    f = f.at[j].add(-ff)
    return f
