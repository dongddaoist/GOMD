"""
forces/ewald.py — Ewald reciprocal-space electrostatics (M2.4).

Full long-range electrostatics = reciprocal-space sum + self-energy
+ exclusion correction. The real-space part is already in the M2.2 splines
(erfc-damped), so this module supplies the rest.

NOTE on PME vs explicit-k:
    The reference uses an EXPLICIT k-vector Ewald sum (not FFT-based PME).
    klimit = 7x7x7 -> a few hundred k-vectors with a spherical cutoff, so it
    is cheap. We reproduce it EXACTLY here for bit-identical validation.
    Switching to FFT-based PME (O(N log N)) is a PERFORMANCE optimization
    deferred to M4 — it changes the approximation and must be re-validated.
    For M2 the bar is "match the reference", which explicit-k does exactly.

Constants:
    COULOMB_CONST = 332.0716  (kcal*A/mol/e^2)

erfc:
    The reference uses the Abramowitz-Stegun 7.1.26 polynomial approximation
    (max err ~1.5e-7). We replicate it EXACTLY (not exact erfc) so the
    exclusion correction matches the reference bit-for-bit and is consistent
    with the spline tables (which were built with the same approximation).
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

COULOMB_CONST = 332.08    # MUST match reference constants.py exactly (not 332.0716)
PI = np.pi


# ===========================================================================
# erfc approximation (Abramowitz-Stegun 7.1.26), JAX version
# ===========================================================================
def erfc_approx_jax(x):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    absx = jnp.abs(x)
    t = 1.0 / (1.0 + p * absx)
    erfc_val = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * jnp.exp(-absx * absx)
    return jnp.where(x < 0, 2.0 - erfc_val, erfc_val)


# ===========================================================================
# k-vector enumeration (host-side, done once per box; small)
# ===========================================================================
def build_kvectors(box, klimitx, klimity, klimitz):
    """Enumerate the half-space k-vectors within the spherical cutoff.

    Matches the reference loop ordering and spherical cutoff:
        kx in [0, klimitx]
        ky in [kymin, klimity]  (kymin starts 0, then -klimity after first kx)
        kz in [kzmin, klimitz]  (kzmin starts 1, then -klimitz after first ky)
        include if (kx/klimitx)^2 + (ky/klimity)^2 + (kz/klimitz)^2 <= 1
    Returns integer arrays (kx, ky, kz) for the retained vectors.
    """
    kxs, kys, kzs = [], [], []
    kxmax2inv = 1.0 / float(klimitx * klimitx)
    kymax2inv = 1.0 / float(klimity * klimity)
    kzmax2inv = 1.0 / float(klimitz * klimitz)

    kymin = 0
    kzmin = 1
    for kx in range(0, klimitx + 1):
        for ky in range(kymin, klimity + 1):
            for kz in range(kzmin, klimitz + 1):
                tmp2 = (kx*kx) * kxmax2inv + (ky*ky) * kymax2inv + (kz*kz) * kzmax2inv
                if tmp2 > 1.0:
                    continue
                kxs.append(kx); kys.append(ky); kzs.append(kz)
            kzmin = -klimitz
        kymin = -klimity
    return (np.asarray(kxs, dtype=np.int32),
            np.asarray(kys, dtype=np.int32),
            np.asarray(kzs, dtype=np.int32))


# ===========================================================================
# Reciprocal-space energy + forces (JAX, vectorized over atoms x k-vectors)
# ===========================================================================
@jax.jit
def reciprocal_energy_force(x, q, box, kx, ky, kz, alpha):
    """Ewald reciprocal-space energy and forces.

    Args:
        x: (nat, 3) positions.
        q: (nat,) per-atom charges.
        box: (3,) box lengths.
        kx, ky, kz: (nk,) integer k-vector components (from build_kvectors).
        alpha: Ewald splitting parameter.

    Returns:
        erecip_raw: reciprocal energy BEFORE subtracting self-energy.
        forces: (nat, 3).
    (Caller subtracts self-energy from the energy.)
    """
    vol = box[0] * box[1] * box[2]
    twopifact = 2.0 * PI / vol
    fouralpha2inv = 1.0 / (4.0 * alpha * alpha)
    elfact = 2.0 * twopifact

    front = 2.0 * PI / box                    # (3,)
    # k-vectors in reciprocal length units: h = k * front
    hx = kx * front[0]                          # (nk,)
    hy = ky * front[1]
    hz = kz * front[2]
    rkmag2 = hx*hx + hy*hy + hz*hz             # (nk,)
    akvec = 2.0 * jnp.exp(-rkmag2 * fouralpha2inv) / rkmag2   # (nk,)

    # Phase k.r for every (atom, k): (nat, nk)
    # k.r = hx*rx + hy*ry + hz*rz
    kr = (x[:, 0:1] * hx[None, :]
          + x[:, 1:2] * hy[None, :]
          + x[:, 2:3] * hz[None, :])          # (nat, nk)
    cos_kr = jnp.cos(kr)
    sin_kr = jnp.sin(kr)

    # Structure factor S(k) = sum_i q_i exp(i k.r_i)
    qcol = q[:, None]                          # (nat,1)
    cossum = jnp.sum(qcol * cos_kr, axis=0)    # (nk,)
    sinsum = jnp.sum(qcol * sin_kr, axis=0)    # (nk,)

    # Energy: sum_k |S(k)|^2 * akvec
    sfh = (cossum * cossum + sinsum * sinsum) * akvec   # (nk,)
    recip = jnp.sum(sfh)
    erecip_raw = recip * twopifact * COULOMB_CONST

    # Forces, fused as matmuls (GPU-friendly) instead of a (nat,nk) tmp.
    # f_i = q_i*C * sum_k prefact_k*(cossum_k*sin_kr_ik - sinsum_k*cos_kr_ik)*h_k
    #     = q_i*C * [ sin_kr @ (prefact*cossum*h)  -  cos_kr @ (prefact*sinsum*h) ]
    prefact = akvec * elfact                   # (nk,)
    a = prefact * cossum                        # (nk,)
    b = prefact * sinsum                        # (nk,)
    # per-axis k-space coefficient vectors
    cax = jnp.stack([a * hx, a * hy, a * hz], axis=1)   # (nk,3)
    cbx = jnp.stack([b * hx, b * hy, b * hz], axis=1)   # (nk,3)
    # (nat,nk) @ (nk,3) -> (nat,3)
    forces = (sin_kr @ cax - cos_kr @ cbx) * (q[:, None] * COULOMB_CONST)

    return erecip_raw, forces


def self_energy(q, alpha):
    """Ewald self-energy = COULOMB * alpha * sum(q^2) / sqrt(pi). Scalar."""
    return COULOMB_CONST * alpha * float(np.sum(np.asarray(q) ** 2)) / np.sqrt(PI)


# ===========================================================================
# Exclusion correction (JAX, vectorized over excluded pairs)
# ===========================================================================
def _minimum_image(dx, box):
    return dx - box * jnp.round(dx / box)


@partial(jax.jit, static_argnames=('nat',))
def exclusion_correction(x, qq_pairs, excl_i, excl_j, box, alpha, nat):
    """Ewald exclusion correction over excluded (bonded) pairs.

    Args:
        x: (nat,3) positions.
        qq_pairs: (npair,) = electrostatic[typee[i,j],0] = 332.08*q_i*q_j
                  for each excluded pair (precomputed host-side to match the
                  reference's per-type electrostatic table exactly).
        excl_i, excl_j: (npair,) excluded pair indices.
        box: (3,), alpha: float, nat: static int.

    Returns:
        e_correct (scalar), f_correct (nat,3).
    """
    rootpi = jnp.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi

    xi = x[excl_i]; xj = x[excl_j]
    dx = _minimum_image(xj - xi, box)          # (npair,3), = x_j - x_i
    z = jnp.sum(dx * dx, axis=1)
    good = z >= 1e-10
    zc = jnp.where(good, z, 1.0)
    r = jnp.sqrt(zc)
    alphar = alpha * r
    zinv = 1.0 / zc
    rinv = 1.0 / r
    r3inv = zinv * rinv

    erf_val = 1.0 - erfc_approx_jax(alphar)

    # Energy: -qq * erf(alphar)/r / 2
    e = jnp.where(good, -qq_pairs * erf_val * rinv / 2.0, 0.0)

    # Force scalar
    ffc = (erf_val * r3inv - twoalphapi * jnp.exp(-alphar * alphar) * zinv) / 2.0
    ffc = jnp.where(good, ffc, 0.0)
    fvec = (ffc * qq_pairs)[:, None] * dx       # (npair,3) on iat

    f = jnp.zeros((nat, 3))
    f = f.at[excl_i].add(fvec)
    f = f.at[excl_j].add(-fvec)
    return jnp.sum(e), f
