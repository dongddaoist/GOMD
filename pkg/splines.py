"""
Spline Setup and Non-Bonded Potential Functions for MD Simulation
This module handles quintic spline interpolation for non-bonded interactions

PATCH (2026-05-04): Fixed grid-spacing bug.
  Fortran (spline.f90 line 26): deltaspline = 0.3 fixed; numpoints = maxpoints (2000).
  Old Python: numpoints capped at 100 -> deltaspline recalculated to ~2.245.
  Fix: keep deltaspline = 0.3 and let numpoints follow from (rcut2-r2min)/0.3.
  Effect: ~7.5x finer grid; quintic interpolation accuracy in the steep
          repulsive region improves by ~(7.5)^6 ~ 1.8e5.
"""

import numpy as np
from numba import jit
from typing import Tuple
from constants import PI


@jit(nopython=True, cache=True)
def erfc_approx(x: float) -> float:
    """erfc(x) via Abramowitz and Stegun (formula 7.1.26)."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    absx = abs(x)
    t = 1.0 / (1.0 + p * absx)
    erfc_val = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-absx * absx)
    if x < 0:
        return 2.0 - erfc_val
    return erfc_val


@jit(nopython=True, cache=True)
def derfc(x: float) -> float:
    """Derivative of erfc: -2/sqrt(pi) * exp(-x^2)"""
    rootpi = np.sqrt(PI)
    return -2.0 / rootpi * np.exp(-x * x)


@jit(nopython=True, cache=True)
def funct(z, innb, ifunct, electrostatic, nonbonded, map_array, alpha, tbpol):
    tolz = 1.0e-8
    r = np.sqrt(z)
    if ifunct == 2:
        return 0.0
    if ifunct == 3:
        imap = int(map_array[innb])
        a = nonbonded[0, imap]; b = nonbonded[1, imap]; c = nonbonded[2, imap]
        if b < 1.0e-6:
            return a / (z**6) - c / (z**3)
        aa = 0.5e-4 * (12.0 / b)**12
        return a * np.exp(-b * np.sqrt(z)) - c / (z**3) + aa / (z**6)
    if ifunct == 1:
        return erfc_approx(alpha * r) * electrostatic[innb, 0] / r
    return 0.0


@jit(nopython=True, cache=True)
def functd(z, innb, ifunct, electrostatic, nonbonded, map_array, alpha, tbpol):
    r = np.sqrt(z)
    if ifunct == 2:
        return 0.0
    if ifunct == 3:
        imap = int(map_array[innb])
        a = nonbonded[0, imap]; b = nonbonded[1, imap]; c = nonbonded[2, imap]
        if b < 1.0e-6:
            return -6.0 * a / (z**7) + 3.0 * c / (z**4)
        aa = 0.5e-4 * (12.0 / b)**12
        return (-a * b * np.exp(-b * np.sqrt(z)) / (2.0 * np.sqrt(z))
                + 3.0 * c / (z**4) - 6.0 * aa / (z**7))
    if ifunct == 1:
        rootpi = np.sqrt(PI)
        erfc_val = erfc_approx(alpha * r)
        return (-electrostatic[innb, 0] *
                (erfc_val / (2.0 * r**3)
                 + alpha * np.exp(-(alpha * r)**2) / (r * r * rootpi)))
    return 0.0


@jit(nopython=True, cache=True)
def functdd(z, innb, ifunct, electrostatic, nonbonded, map_array, alpha, tbpol):
    r = np.sqrt(z)
    if ifunct == 2:
        return 0.0
    if ifunct == 3:
        imap = int(map_array[innb])
        a = nonbonded[0, imap]; b = nonbonded[1, imap]; c = nonbonded[2, imap]
        if b < 1.0e-6:
            return 42.0 * a / (z**8) - 12.0 * c / (z**5)
        aa = 0.5e-4 * (12.0 / b)**12
        return (0.25 * a * b * np.exp(-b * np.sqrt(z))
                * (z**(-1.5) + b * z**(-1.0))
                - 12.0 * c / (z**5) + 42.0 * aa / (z**8))
    if ifunct == 1:
        rootpi = np.sqrt(PI)
        erfc_val = erfc_approx(alpha * r)
        eff = erfc_val / (r * r) + 2.0 * alpha / rootpi * np.exp(-(alpha * r)**2) / r
        edf = (-2.0 * erfc_val / (r**3)
               - 4.0 * alpha / rootpi * np.exp(-(alpha * r)**2)
                 * (1.0 / (r**2) + alpha**2))
        endder = (1.0 / (4.0 * r)) * (eff / (r * r) - edf / r)
        return electrostatic[innb, 0] * endder
    return 0.0


def cload(deltaspline: float) -> np.ndarray:
    """6x6 matrix of quintic spline boundary conditions."""
    cc = np.zeros((6, 6))
    cc[0, 0] = 1.0
    for j in range(6):
        cc[1, j] = deltaspline**j
    cc[2, 1] = 1.0
    for j in range(1, 6):
        cc[3, j] = j * deltaspline**(j-1)
    cc[4, 2] = 2.0
    for j in range(2, 6):
        cc[5, j] = j * (j-1) * deltaspline**(j-2)
    return cc


def gauss_elimin(A, n):
    LU = A.copy().astype(np.float64)
    pivot = np.arange(n)
    for k in range(n - 1):
        max_idx = k + np.argmax(np.abs(LU[k:, k]))
        if max_idx != k:
            LU[[k, max_idx]] = LU[[max_idx, k]]
            pivot[[k, max_idx]] = pivot[[max_idx, k]]
        if abs(LU[k, k]) < 1e-14:
            continue
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            for j in range(k + 1, n):
                LU[i, j] -= LU[i, k] * LU[k, j]
    return LU, pivot


def gsolve(LU, n, b, pivot):
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[pivot[i]]
        for j in range(i):
            y[i] -= LU[i, j] * y[j]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= LU[i, j] * x[j]
        x[i] /= LU[i, i]
    return x


def compute_spline_coefficients(state):
    """Build quintic-spline coefficients for the nonbonded potential surface.

    Match Fortran: deltaspline = 0.3 (FIXED), numpoints follows from rcut2.
    """
    print("\n  Computing spline coefficients for non-bonded interactions...")

    nfunct = 3
    r2min = 0.50

    # =====================================================================
    # CRITICAL FIX (was the cause of test_01 failures and likely the
    # +PE-instead-of-negative regression): match Fortran spline.f90:23-26
    #   nfunct = 3
    #   r2min = 0.50d0
    #   numpoints = maxpoints   ! 2000 in params.h
    #   deltaspline = 0.3d0     ! FIXED, not derived
    # The previous Python recalculated deltaspline = (rcut2-r2min)/numpoints
    # with numpoints capped at 100, giving deltaspline ~ 2.245 — 7.5x too coarse.
    # =====================================================================
    deltaspline = 0.3                        # MATCHES FORTRAN
    rcut = state.params.rcut
    rcut2 = rcut * rcut
    numpoints = int(np.ceil((rcut2 - r2min) / deltaspline))
    # Cap by an explicit user-provided override if present, else use 2000 (Fortran)
    maxpoints_user = getattr(state.params, 'maxpoints', None)
    if maxpoints_user is None:
        maxpoints_user = 2000
    if numpoints > maxpoints_user:
        numpoints = maxpoints_user

    ros = state.params.ros
    ros2 = ros * ros
    driftmax = state.params.driftmax
    rshort2 = (ros + driftmax) ** 2
    alpha = state.params.alphai

    # Determine number of interaction types from typee (unchanged logic)
    print(f"\n    === Determining number of interaction types ===")
    maxnnb_determined = False
    if hasattr(state.ff, 'typee'):
        typee = state.ff.typee
        print(f"    Found typee array: shape={typee.shape}")
        print(f"    typee range: min={np.min(typee)}, max={np.max(typee)}")
        positive_vals = typee[typee >= 0]
        if len(positive_vals) > 0:
            max_typee = int(np.max(np.unique(positive_vals)))
            maxnnb = max_typee + 1
            print(f"    Will compute splines for types 0 to {max_typee} ({maxnnb} total)")
            maxnnb_determined = True
    if not maxnnb_determined:
        maxnnb = state.ff.electrostatic.shape[0] if hasattr(state.ff, 'electrostatic') else 50
        print(f"    FALLBACK: maxnnb = {maxnnb}")
    print(f"    === Final maxnnb = {maxnnb} ===\n")

    zs = np.zeros(numpoints + 1)
    for ipoint in range(numpoints + 1):
        zs[ipoint] = r2min + ipoint * deltaspline

    print(f"    Grid points: {numpoints}        (Fortran uses ~{int((rcut2-r2min)/0.3)})")
    print(f"    r2 range:    {r2min:.3f} to {zs[-1]:.3f}")
    print(f"    deltaspline: {deltaspline:.4f}  (FIXED, matches Fortran)")

    cc = cload(deltaspline)
    LU, pivot = gauss_elimin(cc, 6)

    coeff1 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coeff2 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coeff3 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coeff4 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coeff5 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coeff6 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coefff2 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coefff3 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coefff4 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coefff5 = np.zeros((nfunct, numpoints + 1, maxnnb))
    coefff6 = np.zeros((nfunct, numpoints + 1, maxnnb))

    electrostatic = state.ff.electrostatic if hasattr(state.ff, 'electrostatic') else np.zeros((maxnnb, 3))
    nonbonded = state.ff.nonbonded if hasattr(state.ff, 'nonbonded') else np.zeros((3, maxnnb))
    map_array = (state.ff.map if hasattr(state.ff, 'map')
                 else (state.ff.map_types if hasattr(state.ff, 'map_types')
                       else np.arange(maxnnb)))
    tbpol = getattr(state.params, 'tbpol', False)

    for ifunct in range(1, nfunct + 1):
        for innb in range(maxnnb):
            fs = np.zeros(numpoints + 1)
            fsd = np.zeros(numpoints + 1)
            fsdd = np.zeros(numpoints + 1)
            for ipoint in range(numpoints + 1):
                z = zs[ipoint]
                fs[ipoint] = funct(z, innb, ifunct, electrostatic, nonbonded, map_array, alpha, tbpol)
                fsd[ipoint] = functd(z, innb, ifunct, electrostatic, nonbonded, map_array, alpha, tbpol)
                fsdd[ipoint] = functdd(z, innb, ifunct, electrostatic, nonbonded, map_array, alpha, tbpol)
            for ipoint in range(1, numpoints + 1):
                aa = np.array([
                    fs[ipoint-1], fs[ipoint],
                    fsd[ipoint-1], fsd[ipoint],
                    fsdd[ipoint-1], fsdd[ipoint]
                ])
                xtemp = gsolve(LU, 6, aa, pivot)
                coeff1[ifunct-1, ipoint, innb] = xtemp[0]
                coeff2[ifunct-1, ipoint, innb] = xtemp[1]
                coeff3[ifunct-1, ipoint, innb] = xtemp[2]
                coeff4[ifunct-1, ipoint, innb] = xtemp[3]
                coeff5[ifunct-1, ipoint, innb] = xtemp[4]
                coeff6[ifunct-1, ipoint, innb] = xtemp[5]
                coefff2[ifunct-1, ipoint, innb] = xtemp[1] * 2.0
                coefff3[ifunct-1, ipoint, innb] = xtemp[2] * 4.0
                coefff4[ifunct-1, ipoint, innb] = xtemp[3] * 6.0
                coefff5[ifunct-1, ipoint, innb] = xtemp[4] * 8.0
                coefff6[ifunct-1, ipoint, innb] = xtemp[5] * 10.0

    coefft = np.zeros((6, numpoints + 1, maxnnb))
    coeffft = np.zeros((6, numpoints + 1, maxnnb))
    coeffft1 = np.zeros((6, numpoints + 1, maxnnb))
    for innb in range(maxnnb):
        for ipoint in range(1, numpoints + 1):
            z = zs[ipoint]
            gtaper = (np.sqrt(z) - ros) / driftmax
            staper = 1.0 + gtaper * gtaper * (2.0 * gtaper - 3.0)
            for ifunct in range(nfunct):
                coefft[0, ipoint, innb] += coeff1[ifunct, ipoint, innb]
                coefft[1, ipoint, innb] += coeff2[ifunct, ipoint, innb]
                coefft[2, ipoint, innb] += coeff3[ifunct, ipoint, innb]
                coefft[3, ipoint, innb] += coeff4[ifunct, ipoint, innb]
                coefft[4, ipoint, innb] += coeff5[ifunct, ipoint, innb]
                coefft[5, ipoint, innb] += coeff6[ifunct, ipoint, innb]
                coeffft[1, ipoint, innb] += coefff2[ifunct, ipoint, innb]
                coeffft[2, ipoint, innb] += coefff3[ifunct, ipoint, innb]
                coeffft[3, ipoint, innb] += coefff4[ifunct, ipoint, innb]
                coeffft[4, ipoint, innb] += coefff5[ifunct, ipoint, innb]
                coeffft[5, ipoint, innb] += coefff6[ifunct, ipoint, innb]
            if z < rshort2:
                for icoeff in range(1, 6):
                    coeffft1[icoeff, ipoint, innb] = coeffft[icoeff, ipoint, innb]
                    if z > ros2:
                        coeffft1[icoeff, ipoint, innb] *= staper

    coeffee = np.zeros((6, numpoints + 1, maxnnb))
    coefffee = np.zeros((6, numpoints + 1, maxnnb))
    for innb in range(maxnnb):
        for ipoint in range(1, numpoints + 1):
            coeffee[0, ipoint, innb] = coeff1[0, ipoint, innb]
            coeffee[1, ipoint, innb] = coeff2[0, ipoint, innb]
            coeffee[2, ipoint, innb] = coeff3[0, ipoint, innb]
            coeffee[3, ipoint, innb] = coeff4[0, ipoint, innb]
            coeffee[4, ipoint, innb] = coeff5[0, ipoint, innb]
            coeffee[5, ipoint, innb] = coeff6[0, ipoint, innb]
            coefffee[1, ipoint, innb] = coefff2[0, ipoint, innb]
            coefffee[2, ipoint, innb] = coefff3[0, ipoint, innb]
            coefffee[3, ipoint, innb] = coefff4[0, ipoint, innb]
            coefffee[4, ipoint, innb] = coefff5[0, ipoint, innb]
            coefffee[5, ipoint, innb] = coefff6[0, ipoint, innb]

    from datatypes import SplineCoeffs
    state.splines = SplineCoeffs()
    state.splines.zs = zs
    state.splines.deltaspline = deltaspline
    state.splines.deltainv = 1.0 / deltaspline
    state.splines.intzs0 = int(zs[0])
    state.splines.coefft = coefft
    state.splines.coeffft = coeffft
    state.splines.coeffft1 = coeffft1
    state.splines.coeffee = coeffee
    state.splines.coefffee = coefffee

    print(f"    Spline coefficients computed: shape {coefft.shape}")
    print(f"  ========================================\n")
