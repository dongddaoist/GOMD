"""
forces/ewald2d_ref.py — reference electrostatics for slab systems (numpy, fp64, small N).

Constant-potential task, phase P1 (2026-11-18). Point charges and Gaussian charges,
periodic in x and y only (a slab), in Gaussian units (charges in e, lengths in A,
energy in e^2/A; multiply by COULOMB = 332.08 for kcal/mol). Two independent
routes to the same energy, so each checks the other:

  ewald2d_exact   the 2D Ewald sum (Parry; Heyes, Barber & Clarke; de Leeuw &
                  Perram; Hu, JCTC 2014 eqs 37-39), generalised to Gaussian
                  charges as in Reed, Lanning & Madden, JCP 2007 (eq A31).
  ewald3dc        3D Ewald in a box taller than the slab plus the planar dipole
                  term 2 pi M_z^2 / V (Yeh & Berkowitz 1999; Hu 2014 eq 19), i.e.
                  the trapezoid rule for the k_z integral of the 2D sum. Its error
                  is the slab's interaction with its z images (it falls roughly as
                  exp(-2 pi gap / L_x)).

Both return (U, phi, F): the energy, the potential at every site phi_i = dU/dq_i
(what a constant-potential electrode needs), and the forces -dU/dr_i.

Charge model. Site i is a point charge (eta_i = inf) or a Gaussian
rho_i(r) = q_i (eta_i / sqrt(pi))^3 exp(-eta_i^2 |r - r_i|^2) (Reed's convention;
eta = 1/width, e.g. 1.979 1/A for a width of 0.505 A). A pair then interacts through
    erf(s_ij r) / r,     1/s_ij^2 = 1/eta_i^2 + 1/eta_j^2   (s = inf: 1/r)
and the Ewald split with parameter alpha uses for the pair
    1/sig_ij^2 = 1/alpha^2 + 1/eta_i^2 + 1/eta_j^2
(sig = alpha for two points; Reed's gamma for point-Gaussian, beta for two
Gaussians). Unlike Reed's eq A31 the real-space kernel keeps the full form
erf(s r)/r - erf(sig r)/r: for graphite (C-C 1.41 A, eta 1.979) the dropped
erfc(eta r / sqrt 2) is 0.6 % at nearest neighbours.

Energy terms (sums over ordered pairs i, j including i = j where stated):
  real      1/2 sum_{i,j,n}' q_i q_j [erf(s r)/r - erf(sig r)/r],  r = |r_ij + n|,
            n over lateral images, prime: no i = j at n = 0
  k != 0    pi/(2A) sum_{i,j} q_i q_j sum_{K != 0} cos(K . rho_ij)/K F_K(z_ij)
            F_K(z) = e^{K z} erfc(K/2sig + sig z) + e^{-K z} erfc(K/2sig - sig z)
  k = 0     -pi/A sum_{i,j} q_i q_j [z_ij erf(sig z_ij) + exp(-sig^2 z_ij^2)/(sig sqrt pi)]
  self      -sum_i q_i^2 sig_ii / sqrt(pi)  +  sum_{Gaussian i} q_i^2 eta_i / sqrt(2 pi)
The system must be neutral.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erf, erfc, erfcx

SQPI = np.sqrt(np.pi)
COULOMB = 332.08


def _inv2(eta):
    eta = np.asarray(eta, dtype=np.float64)
    return np.where(np.isinf(eta), 0.0, 1.0 / np.where(np.isinf(eta), 1.0, eta) ** 2)


def _pair_s_sig(inv2_i, inv2_j, alpha):
    """s (inf for two points) and sig for broadcast site pairs."""
    tot = inv2_i + inv2_j
    s = np.where(tot > 0, 1.0 / np.sqrt(np.where(tot > 0, tot, 1.0)), np.inf)
    sig = 1.0 / np.sqrt(1.0 / alpha ** 2 + tot)
    return s, sig


def _kernel(r, s, sig):
    """phi(r) = erf(s r)/r - erf(sig r)/r and dphi/dr (s = inf: erfc(sig r)/r)."""
    point = np.isinf(s)
    sfin = np.where(point, 1.0, s)
    phi = np.where(point, erfc(sig * r), erf(sfin * r) - erf(sig * r)) / r
    g = (2.0 / SQPI) * (np.where(point, 0.0, sfin * np.exp(-(sfin * r) ** 2))
                        - sig * np.exp(-(sig * r) ** 2))
    dphi = (g * r - phi * r) / (r * r)          # d/dr [erf(.)/r]: (g r - erf)/r^2, erf = phi r
    return phi, dphi


def real_space(x, q, eta, box_xy, alpha, rcut):
    """Real-space part with explicit lateral images (no z images): U, phi, F."""
    x = np.asarray(x, float); q = np.asarray(q, float)
    n = len(q)
    inv2 = _inv2(eta)
    s, sig = _pair_s_sig(inv2[:, None], inv2[None, :], alpha)
    L = np.asarray(box_xy, float)
    nmax = np.ceil((rcut + L) / L).astype(int)
    U = 0.0; phi = np.zeros(n); F = np.zeros((n, 3))
    d0 = x[:, None, :] - x[None, :, :]                      # r_i - r_j
    for ix in range(-nmax[0], nmax[0] + 1):
        for iy in range(-nmax[1], nmax[1] + 1):
            d = d0 + np.array([ix * L[0], iy * L[1], 0.0])
            r = np.sqrt((d * d).sum(-1))
            m = r < rcut
            if ix == 0 and iy == 0:
                np.fill_diagonal(m, False)
            if not m.any():
                continue
            rr = np.where(m, r, 1.0)
            ph, dph = _kernel(rr, s, sig)
            ph = np.where(m, ph, 0.0); dph = np.where(m, dph, 0.0)
            U += 0.5 * np.sum(q[:, None] * q[None, :] * ph)
            phi += ph @ q
            F -= ((q[:, None] * q[None, :] * dph / rr)[:, :, None] * d).sum(1)
    return U, phi, F


def _self(q, eta, alpha):
    inv2 = _inv2(eta)
    sig_ii = 1.0 / np.sqrt(1.0 / alpha ** 2 + 2.0 * inv2)
    gauss = ~np.isinf(np.asarray(eta, float))
    s_ii = np.where(gauss, 1.0 / np.sqrt(np.where(gauss, 2.0 * inv2, 1.0)), 0.0)
    U = -np.sum(q * q * sig_ii) / SQPI + np.sum(np.where(gauss, q * q * s_ii, 0.0)) / SQPI
    phi = -2.0 * q * sig_ii / SQPI + np.where(gauss, 2.0 * q * s_ii / SQPI, 0.0)
    return U, phi


def _FK(K, z, sig):
    """F_K(z) and dF_K/dz, overflow-safe (erfcx for non-negative arguments)."""
    a = K / (2.0 * sig)
    def E(sgn):
        u = a + sgn * sig * z
        pos = erfcx(np.maximum(u, 0.0)) * np.exp(-a * a - (sig * z) ** 2)
        neg = erfc(np.minimum(u, 0.0)) * np.exp(np.minimum(sgn * K * z, 700.0))
        return np.where(u >= 0, pos, neg)
    Ep, Em = E(1.0), E(-1.0)
    return Ep + Em, K * (Ep - Em)


def ewald2d_exact(x, q, eta, box_xy, alpha, rcut, kmax, chunk=32):
    """Exact 2D Ewald sum (module docstring). kmax: lateral integer limits (kx, ky);
    every K != 0 with |kx| <= kmax[0], |ky| <= kmax[1] is used."""
    x = np.asarray(x, float); q = np.asarray(q, float); eta = np.asarray(eta, float)
    n = len(q)
    if abs(q.sum()) > 1e-10:
        raise ValueError(f"2D Ewald needs a neutral system (sum q = {q.sum():.3e})")
    Lx, Ly = map(float, box_xy); A = Lx * Ly
    U, phi, F = real_space(x, q, eta, (Lx, Ly), alpha, rcut)
    Us, ps = _self(q, eta, alpha)
    U += Us; phi += ps
    inv2 = _inv2(eta)
    _, sig = _pair_s_sig(inv2[:, None], inv2[None, :], alpha)
    rho = x[:, None, :2] - x[None, :, :2]
    z = x[:, 2][:, None] - x[:, 2][None, :]
    qq = q[:, None] * q[None, :]
    # ---- K != 0 ----
    kx, ky = np.meshgrid(np.arange(-kmax[0], kmax[0] + 1), np.arange(-kmax[1], kmax[1] + 1),
                         indexing='ij')
    Kv = np.stack([kx.ravel() * 2 * np.pi / Lx, ky.ravel() * 2 * np.pi / Ly], 1)
    Kv = Kv[np.any(Kv != 0, axis=1)]
    for c0 in range(0, len(Kv), chunk):
        Kc = Kv[c0:c0 + chunk]                                   # (m,2)
        K = np.sqrt((Kc ** 2).sum(1))                            # (m,)
        ph = np.einsum('ijd,md->ijm', rho, Kc)                    # (n,n,m)
        cs, sn = np.cos(ph), np.sin(ph)
        Fk, dFk = _FK(K[None, None, :], z[:, :, None], sig[:, :, None])
        w = (np.pi / (2.0 * A)) / K
        U += np.sum(qq[:, :, None] * cs * Fk * w)
        phi += 2.0 * np.einsum('ijm,j->i', cs * Fk * w, q)
        # lateral: F_i = (pi/A) q_i sum_j q_j sum_K Kvec sin(K.rho_ij)/K F_K
        F[:, :2] += 2.0 * np.einsum('ijm,md->id', qq[:, :, None] * sn * Fk * w, Kc)
        F[:, 2] += -2.0 * np.sum(qq[:, :, None] * cs * dFk * w, axis=(1, 2))
    # ---- K = 0 ----
    G = z * erf(sig * z) + np.exp(-(sig * z) ** 2) / (sig * SQPI)
    U += -(np.pi / A) * np.sum(qq * G)
    phi += -(2.0 * np.pi / A) * (G @ q)
    F[:, 2] += (2.0 * np.pi / A) * q * (erf(sig * z) @ q)
    return U, phi, F


def kgrid(kmax):
    """Integer k-vectors k != 0 of the full grid inside the ellipsoid
    (kx/kmax_x)^2 + (ky/kmax_y)^2 + (kz/kmax_z)^2 <= 1 (both signs)."""
    g = np.meshgrid(*[np.arange(-k, k + 1) for k in kmax], indexing='ij')
    kint = np.stack([a.ravel() for a in g], 1)
    keep = ((kint / np.asarray(kmax, float)) ** 2).sum(1) <= 1.0
    return kint[keep & np.any(kint != 0, axis=1)]


def recip3d(x, q, eta, box, alpha, kint, weight=1.0, planar=True, chunk=256):
    """k-space part of 3D Ewald (+ planar term): U, phi, F and the field E (= F/q for
    a site with charge q). kint: integer k-vectors; weight 2 for a half-space set
    (GOMD's ewald.build_kvectors), 1 for a full set (kgrid)."""
    x = np.asarray(x, float); q = np.asarray(q, float)
    n = len(q)
    box = np.asarray(box, float); V = float(np.prod(box))
    inv2 = _inv2(eta)
    U = 0.0; phi = np.zeros(n); E = np.zeros((n, 3))
    Hall = np.asarray(kint, float) * (2 * np.pi / box)
    for c0 in range(0, len(Hall), chunk):
        H = Hall[c0:c0 + chunk]
        k2 = (H ** 2).sum(1)
        w = weight * np.exp(-k2 / (4 * alpha ** 2)) / k2         # (m,)
        gfac = np.exp(-k2[None, :] * inv2[:, None] / 4.0)        # (n,m) Gaussian factors
        ph = x @ H.T
        c, s = np.cos(ph) * gfac, np.sin(ph) * gfac
        C = q @ c; S = q @ s
        U += (2 * np.pi / V) * np.sum(w * (C * C + S * S))
        phi += (4 * np.pi / V) * (c @ (w * C) + s @ (w * S))
        E += (4 * np.pi / V) * ((s * C[None, :] - c * S[None, :]) * w) @ H
    if planar:
        Mz = float(q @ x[:, 2])
        U += (2 * np.pi / V) * Mz * Mz
        phi += (4 * np.pi / V) * Mz * x[:, 2]
        E[:, 2] += -(4 * np.pi / V) * Mz
    return U, phi, q[:, None] * E, E


def ewald3dc(x, q, eta, box, alpha, rcut, kmax, planar=True, chunk=256):
    """3D Ewald in `box` (taller than the slab, vacuum >= rcut) + the planar dipole
    term: the trapezoid form of the 2D sum. kmax: integer limits (kx, ky, kz); the
    full grid within the ellipsoid (kx/kmax_x)^2 + ... <= 1 is used, k != 0."""
    x = np.asarray(x, float); q = np.asarray(q, float); eta = np.asarray(eta, float)
    box = np.asarray(box, float)
    U, phi, F = real_space(x, q, eta, box[:2], alpha, rcut)
    Us, ps = _self(q, eta, alpha)
    Ur, pr, Fr, _ = recip3d(x, q, eta, box, alpha, kgrid(kmax), 1.0, planar, chunk)
    return U + Us + Ur, phi + ps + pr, F + Fr
