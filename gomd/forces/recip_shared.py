"""
forces/recip_shared.py — one reciprocal-space pass for every k-space user.

The explicit-k Ewald sum (same k-vectors, prefactors and formulas as
ewald.reciprocal_energy_force and pol_field._recip_field) is needed three
times per polarizable step:
  * the classical reciprocal energy and force,
  * the reciprocal charge field E_recip that drives the SCF,
  * the k-space charge-dipole force C * d/dx [sum_i mu_i . E_recip,i(x)]
    at fixed mu (formerly jax.grad through the whole field pass).
They all start from cos(k.r_i), sin(k.r_i) (nat x nk) and the structure
factors C_k = sum_j q_j cos(k.r_j), S_k = sum_j q_j sin(k.r_j). Here that
table is built once per step and every output is a small matmul on it:

    E_recip,i  = sum_k P_k h_k (C_k sin_ik - S_k cos_ik)        P_k = akvec_k elfact
    F_recip,i  = COULOMB q_i E_recip,i                           (same prefactors)
    E_recip    = COULOMB twopifact sum_k akvec_k (C_k^2 + S_k^2)
    dW/dr_l    = q_l sum_k P_k h_k (-A_k sin_lk - B_k cos_lk)
               +     sum_k P_k h_k m_lk (C_k cos_lk + S_k sin_lk)
      with W = sum_i mu_i . E_recip,i, m_lk = mu_l . h_k,
      A_k = sum_i m_ik sin_ik, B_k = sum_i m_ik cos_ik.

The table itself is factorised: exp(i k.r) = exp(i kx tx) exp(i ky ty)
exp(i kz tz), t = 2 pi r / L, so each site needs cos/sin of only
(kx range + ky range + kz range) = 8 + 15 + 15 angles instead of one per
k-vector (709 here); the rest is two complex multiplications per (site, k).

Slab (slab=True, constant-potential task P1, 2026-11-18): the box is taller than
the electrode | electrolyte | electrode stack, with vacuum above and below, and
the planar dipole term of Yeh & Berkowitz / Hu (JCTC 2014 eq 19) is added,
    E_planar = 2 pi M_z^2 / V,   M_z = sum_i q_i z_i,
so the sum is the 2D Ewald sum done with the trapezoid rule in k_z (validated
against the exact 2D sum in gomd/tests/check_ewald2d.py). Its uniform field
-4 pi M_z / V (z) is added to 'field', so the classical force, the SCF field and
hence the charge-dipole energy carry it; qmu_grad gets the matching
-4 pi q_l sum_i mu_i,z / V. The induced dipoles do not enter M_z: in APPLE&P the
dipole-dipole interaction is real-space only (RF/taper), so it has no k = 0 term.
z must be continuous across the stack (no atom wraps in z), which the slab
layout guarantees.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

COULOMB_CONST = 332.08      # as ewald.py / polarization.py
PI = np.pi


def make_recip(box, q, kx, ky, kz, alpha, slab=False):
    """Build the shared k-space pass for a fixed box, charges and k-set.

    box (3,), q (nat,), kx/ky/kz integer k-vectors from ewald.build_kvectors
    (numpy or JAX; their ranges set static table sizes), alpha the Ewald
    splitting parameter. slab: add the planar dipole term (module docstring).

    Returns a dict of functions (pure, jit-able):
        tables(x) -> rt   {'cos','sin' (nat,nk), 'C','S' (nk,), 'field' (nat,3)}
        energy(rt) -> reciprocal energy before the self term (COULOMB applied)
        force(rt)  -> classical reciprocal force (nat,3) (COULOMB applied)
        qmu_grad(rt, mu) -> d/dx sum_i mu_i . E_recip,i at fixed mu (no COULOMB)
    """
    kxn = np.asarray(kx, dtype=np.int64); kyn = np.asarray(ky, dtype=np.int64)
    kzn = np.asarray(kz, dtype=np.int64)
    box = jnp.asarray(box, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    vol = box[0] * box[1] * box[2]
    twopifact = 2.0 * PI / vol
    elfact = 2.0 * twopifact
    front = 2.0 * PI / box
    H = jnp.stack([jnp.asarray(kxn) * front[0], jnp.asarray(kyn) * front[1],
                   jnp.asarray(kzn) * front[2]], axis=1)                 # (nk,3)
    rk2 = jnp.sum(H * H, axis=1)
    akvec = 2.0 * jnp.exp(-rk2 / (4.0 * alpha * alpha)) / rk2
    P = akvec * elfact

    # per-axis integer ranges and gather indices into the per-axis tables
    rng = []
    for kn in (kxn, kyn, kzn):
        lo, hi = int(kn.min()), int(kn.max())
        rng.append((jnp.arange(lo, hi + 1, dtype=jnp.float64), jnp.asarray(kn - lo, dtype=jnp.int32)))

    def tables(x):
        t = x * (2.0 * PI / box)                                          # (nat,3)
        cs = []
        for a, (m, idx) in enumerate(rng):
            ang = t[:, a:a + 1] * m[None, :]                              # (nat, n_a)
            cs.append((jnp.cos(ang)[:, idx], jnp.sin(ang)[:, idx]))       # (nat, nk) each
        (cx, sx), (cy, sy), (cz, sz) = cs
        c1 = cx * cy - sx * sy
        s1 = sx * cy + cx * sy
        cos = c1 * cz - s1 * sz
        sin = s1 * cz + c1 * sz
        C = q @ cos
        S = q @ sin
        field = sin @ ((P * C)[:, None] * H) - cos @ ((P * S)[:, None] * H)
        if slab:
            Mz = q @ x[:, 2]
            field = field.at[:, 2].add(-elfact * Mz)
        else:
            Mz = jnp.zeros((), x.dtype)
        return {'cos': cos, 'sin': sin, 'C': C, 'S': S, 'field': field, 'Mz': Mz}

    def energy(rt):
        e = jnp.sum((rt['C'] ** 2 + rt['S'] ** 2) * akvec) * twopifact
        if slab:
            e = e + twopifact * rt['Mz'] ** 2
        return e * COULOMB_CONST

    def force(rt):
        return rt['field'] * (q[:, None] * COULOMB_CONST)

    def qmu_grad(rt, mu):
        mu = jax.lax.stop_gradient(mu)
        cos, sin, C, S = rt['cos'], rt['sin'], rt['C'], rt['S']
        m = mu @ H.T                                                      # (nat,nk)
        A = jnp.sum(m * sin, axis=0)
        B = jnp.sum(m * cos, axis=0)
        term_q = -(sin @ ((P * A)[:, None] * H) + cos @ ((P * B)[:, None] * H)) * q[:, None]
        term_m = (m * (cos * C + sin * S) * P) @ H
        g = term_q + term_m
        if slab:
            g = g.at[:, 2].add(-elfact * q * jnp.sum(mu[:, 2]))
        return g

    return {'tables': tables, 'energy': energy, 'force': force, 'qmu_grad': qmu_grad,
            'nk': int(kxn.size), 'slab': bool(slab)}
