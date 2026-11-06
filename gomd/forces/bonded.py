"""
forces/bonded.py — Bonded forces (M2.3): bonds, angles, torsions, impropers.

Ported from the reference forces.py. Each term is vectorized with vmap over
its connectivity array and scattered into the per-atom force array.

Conventions (match reference exactly):
    Bond:    E = 0.5 * k * (r - r0)^2 ;  fmag = k*(r-r0)/r ; f_i += fmag*dx,
             f_j -= fmag*dx  (dx = x_j - x_i)
    Bend:    E = 0.5 * k * (theta - theta0)^2 ; theta0 in degrees -> radians.
    Torsion: E = V0 - sum_{n>=1} 0.5*V_n*cos(n*phi) ; dE/dphi accumulated.
    Improper:E = k * chi^2  (out-of-plane angle). This system has ndeforms=0
             so it's a no-op here, but ported for completeness.

Connectivity layouts (column = one interaction):
    bonds (3, nbond): [iat, jat, itype]
    bends (>=4, nbend): [iat, jat, kat, itype, ...]
    torts (>=5, ntort): [iat, jat, kat, lat, itype, ...]
    stretch (ntype, >=2): [k, r0, ...]
    bend_params (ntype, >=2): [k, theta0_deg, ...]
    twist (nterm_max, ntype): Fourier coeffs ; nprms (ntype,): #terms
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

PI = jnp.pi


def _mi(dx, box):
    """Minimum image (vectorized)."""
    return dx - box * jnp.round(dx / box)


# ===========================================================================
# Bonds
# ===========================================================================
@partial(jax.jit, static_argnames=('nat',))
def bond_energy_force(x, bonds, stretch, box, nat):
    """bonds: (3, nbond) int [iat,jat,itype]. stretch: (ntype,>=2) [k,r0]."""
    iat = bonds[0]; jat = bonds[1]; itype = bonds[2]
    k = stretch[itype, 0]
    r0 = stretch[itype, 1]

    dx = _mi(x[jat] - x[iat], box)            # (nbond, 3)
    r2 = jnp.sum(dx * dx, axis=1)
    r = jnp.sqrt(r2)
    safe = r > 1e-10
    rinv = jnp.where(safe, 1.0 / jnp.where(safe, r, 1.0), 0.0)
    dr = r - r0

    e = jnp.where(safe, 0.5 * k * dr * dr, 0.0)
    fmag = jnp.where(safe, k * dr * rinv, 0.0)   # (nbond,)
    fvec = fmag[:, None] * dx                     # (nbond,3) on iat

    f = jnp.zeros((nat, 3))
    f = f.at[iat].add(fvec)
    f = f.at[jat].add(-fvec)
    return jnp.sum(e), f


# ===========================================================================
# Bends (angles)
# ===========================================================================
@partial(jax.jit, static_argnames=('nat',))
def bend_energy_force(x, bends, bend_params, box, nat):
    """bends: (>=4, nbend) [iat,jat,kat,itype]. bend_params: (ntype,>=2) [k,theta0_deg]."""
    iat = bends[0]; jat = bends[1]; kat = bends[2]; itype = bends[3]
    k = bend_params[itype, 0]
    theta0 = bend_params[itype, 1] * PI / 180.0

    rji = _mi(x[iat] - x[jat], box)           # (nbend,3)
    rjk = _mi(x[kat] - x[jat], box)
    rji2 = jnp.sum(rji * rji, axis=1)
    rjk2 = jnp.sum(rjk * rjk, axis=1)
    good = (rji2 > 1e-20) & (rjk2 > 1e-20)
    rjiinv = jnp.where(good, 1.0 / jnp.sqrt(jnp.where(good, rji2, 1.0)), 0.0)
    rjkinv = jnp.where(good, 1.0 / jnp.sqrt(jnp.where(good, rjk2, 1.0)), 0.0)

    dot = jnp.sum(rji * rjk, axis=1)
    costheta = jnp.clip(dot * rjiinv * rjkinv, -1.0, 1.0)
    theta = jnp.arccos(costheta)
    sintheta = jnp.sin(theta)
    good2 = good & (jnp.abs(sintheta) > 1e-10)

    dtheta = theta - theta0
    e = jnp.where(good2, 0.5 * k * dtheta * dtheta, 0.0)
    fmag = jnp.where(good2, k * dtheta / jnp.where(good2, sintheta, 1.0), 0.0)  # (nbend,)

    # d(costheta)/d position, per Cartesian component
    # dcos_di = rjk*rjiinv*rjkinv - costheta*rji*rjiinv^2
    dcos_di = (rjk * (rjiinv * rjkinv)[:, None]
               - (costheta * rjiinv * rjiinv)[:, None] * rji)   # (nbend,3)
    dcos_dk = (rji * (rjiinv * rjkinv)[:, None]
               - (costheta * rjkinv * rjkinv)[:, None] * rjk)

    fmagb = fmag[:, None]
    f_i = fmagb * dcos_di
    f_k = fmagb * dcos_dk
    f_j = -(f_i + f_k)

    f = jnp.zeros((nat, 3))
    f = f.at[iat].add(f_i)
    f = f.at[kat].add(f_k)
    f = f.at[jat].add(f_j)
    return jnp.sum(e), f


# ===========================================================================
# Torsions (dihedrals)
# ===========================================================================
@partial(jax.jit, static_argnames=('nat', 'max_terms'))
def torsion_energy_force(x, torts, twist, nprms, box, nat, max_terms):
    """torts: (>=5, ntort) [iat,jat,kat,lat,itype].
       twist: (nterm_max, ntype) Fourier coeffs. nprms: (ntype,) #terms.
       max_terms: static int >= max(nprms), used to bound the term loop.
    """
    iat = torts[0]; jat = torts[1]; kat = torts[2]; lat = torts[3]; itype = torts[4]

    rij = _mi(x[iat] - x[jat], box)           # (ntort,3)
    rjk = _mi(x[jat] - x[kat], box)
    rkl = _mi(x[kat] - x[lat], box)

    # m = rij x rjk ; n = rjk x rkl
    m = jnp.cross(rij, rjk)
    n = jnp.cross(rjk, rkl)
    mm = jnp.sum(m * m, axis=1)
    nn = jnp.sum(n * n, axis=1)
    rjk2 = jnp.sum(rjk * rjk, axis=1)
    good = (mm > 1e-10) & (nn > 1e-10) & (rjk2 > 1e-20)

    rjk_mag = jnp.sqrt(jnp.where(good, rjk2, 1.0))
    mm_inv = jnp.where(good, 1.0 / jnp.where(good, mm, 1.0), 0.0)
    nn_inv = jnp.where(good, 1.0 / jnp.where(good, nn, 1.0), 0.0)

    cosphi = jnp.clip(jnp.sum(m * n, axis=1) / jnp.sqrt(jnp.where(good, mm * nn, 1.0)),
                      -1.0, 1.0)
    sign = jnp.sum(rij * n, axis=1)
    sinphi = jnp.sqrt(jnp.maximum(1.0 - cosphi * cosphi, 0.0))
    sinphi = jnp.where(sign < 0, -sinphi, sinphi)
    phi = jnp.arctan2(sinphi, cosphi)         # (ntort,)

    # Energy + dE/dphi over Fourier terms. Term 0 is the constant V0.
    nterms = nprms[itype]                       # (ntort,)
    e = jnp.zeros_like(phi)
    dedphi = jnp.zeros_like(phi)
    # term 0: constant
    e = e + jnp.where(nterms > 0, twist[0, itype], 0.0)
    for t in range(1, max_terms):
        active = (t < nterms)
        vn = jnp.where(active, twist[t, itype], 0.0)
        e = e - 0.5 * vn * jnp.cos(t * phi)
        dedphi = dedphi + 0.5 * vn * t * jnp.sin(t * phi)

    # Force derivatives
    rijdotrjk = jnp.sum(rij * rjk, axis=1)
    rjkdotrkl = jnp.sum(rjk * rkl, axis=1)

    # dphi_di = -rjk_mag * mm_inv * m ; dphi_dl = rjk_mag * nn_inv * n
    dphi_di = -(rjk_mag * mm_inv)[:, None] * m       # (ntort,3)
    dphi_dl = (rjk_mag * nn_inv)[:, None] * n
    a = (rijdotrjk / jnp.where(good, rjk2, 1.0))[:, None]
    b = (rjkdotrkl / jnp.where(good, rjk2, 1.0))[:, None]
    dphi_dj = (a - 1.0) * dphi_di - b * dphi_dl
    dphi_dk = (b - 1.0) * dphi_dl - a * dphi_di

    ded = jnp.where(good & (jnp.abs(dedphi) > 1e-12), dedphi, 0.0)[:, None]
    f = jnp.zeros((nat, 3))
    f = f.at[iat].add(-ded * dphi_di)
    f = f.at[jat].add(-ded * dphi_dj)
    f = f.at[kat].add(-ded * dphi_dk)
    f = f.at[lat].add(-ded * dphi_dl)

    e = jnp.where(good, e, 0.0)
    return jnp.sum(e), f


def compute_bonded(x, conn, ff, box, nat, max_terms):
    """Convenience: sum bond + bend + torsion energies and forces.

    conn: dict-like with bonds, bends, torts arrays.
    ff:   dict-like with stretch, bend, twist, nprms arrays.
    Returns (e_total, f_total, components_dict).
    """
    eb, fb = bond_energy_force(x, conn['bonds'], ff['stretch'], box, nat)
    ebe, fbe = bend_energy_force(x, conn['bends'], ff['bend'], box, nat)
    et, ft = torsion_energy_force(x, conn['torts'], ff['twist'], ff['nprms'],
                                  box, nat, max_terms)
    e = eb + ebe + et
    f = fb + fbe + ft
    return e, f, {'ebond': eb, 'ebend': ebe, 'etort': et}
