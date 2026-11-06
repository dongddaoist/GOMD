"""
Many-Body Polarization for APPLE&P Force Field
Implements the iterative self-consistent field (SCF) induced dipole solver.

Ported from Fortran interch() lines 2845-3756 and recipQmu() lines 8476-8863.

Algorithm:
  1. Compute electric field at each atom from charges (Ewald real-space)
  2. Predict induced dipoles from 3 previous timesteps
  3. SCF iteration: mu_i = alpha_i * (E_charge + T_dipole + arf*mu_i)
     with Thole damping (lambda3, lambda5) and tau tapering
  4. Compute polarization energy: E_pol = -0.5 * 332.08 * sum(E_charge * mu)
  5. Compute induced forces (charge-dipole + dipole-dipole gradients)
  6. Ewald correction for excluded Q-mu pairs
  7. 1-4 Q-mu scaling (if lredQ_mu14)
  8. recipQmu: k-space charge-dipole interaction
"""

import numpy as np
from numba import jit
from constants import PI, COULOMB_CONST

# Thole damping constant (from params.h)
A_THOLE = 0.20


# =========================================================================
# TAPERING FUNCTIONS (Fortran tau/dtau, lines 6582-6624)
# =========================================================================

@jit(nopython=True, cache=True)
def tau_func(r, rtaper, ro, tauden):
    """Tapering function: smooth cutoff from 1 at rtaper to 0 at ro."""
    if r <= rtaper:
        return 1.0
    elif r >= ro:
        return 0.0
    else:
        w = r - rtaper
        return 1.0 + w * w * (w / 3.0 + 0.5 * (rtaper - ro)) / tauden


@jit(nopython=True, cache=True)
def dtau_func(r, rtaper, ro, tauden):
    """Derivative of tapering function."""
    if r > rtaper and r < ro:
        return ((r - ro) * (r - rtaper)**2) / tauden
    return 0.0


# =========================================================================
# ERFC APPROXIMATION (same as in splines.py)
# =========================================================================

@jit(nopython=True, cache=True)
def erfc_approx(x):
    """Complementary error function (Abramowitz & Stegun)."""
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * abs(x))
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    result = poly * np.exp(-x * x)
    if x < 0.0:
        result = 2.0 - result
    return result


@jit(nopython=True, cache=True)
def erf_approx(x):
    """Error function."""
    return 1.0 - erfc_approx(x)


# =========================================================================
# C1 ARRAY (Thole damping constants)
# =========================================================================

def build_c1_array(pol, ncharge):
    """
    Build Thole damping constant matrix.
    c1(i,j) = a_thole / sqrt(pol_i * pol_j) if both polarizable
    c1(i,j) = 1e6 otherwise (effectively no damping)
    """
    c1 = np.full((ncharge, ncharge), 1.0e6)
    for i in range(ncharge):
        for j in range(i, ncharge):
            if pol[i] > 1.0e-5 and pol[j] > 1.0e-5:
                c1[i, j] = A_THOLE / np.sqrt(pol[i] * pol[j])
                c1[j, i] = c1[i, j]
    return c1


# =========================================================================
# ARF: REACTION FIELD CONSTANT (Fortran setup lines 6185-6192)
# =========================================================================

def compute_arf(rtaper, ro, epsrf, tauden):
    """Reaction field constant: arf = (epsrf-1)/(rrf^3*(epsrf+0.5))"""
    if abs(epsrf - 1.0) < 1.0e-10:
        return 0.0  # Vacuum boundary

    # Numerical integration for rrf^3
    rsum = 0.0
    nsteps = 1000
    dr = (ro - rtaper) / nsteps
    for isum in range(nsteps):
        rint = rtaper + (ro - rtaper) * (0.5 + float(isum)) / float(nsteps)
        rsum += rint * rint * tau_func(rint, rtaper, ro, tauden)
    rsum *= 3.0 * (ro - rtaper) / float(nsteps)
    rrf3 = rsum + rtaper**3
    return (epsrf - 1.0) / (rrf3 * (epsrf + 0.5))


# =========================================================================
# PHASE 1: ELECTRIC FIELD FROM CHARGES + STORE PAIR DATA
# (Fortran interch lines 2845-2864, 2914-2988, 3107-3137)
# =========================================================================

@jit(nopython=True, cache=True)
def compute_charge_efield_and_pairs(
    x, nat, atomtype, list_neighbors, listm,
    q, ncharge, alpha, box, el,
    c1, rcut2,
    # 1-4 lists
    list14, listm14, lredQ_mu14, redQmufactor,
    # 1-4 vdW (lredonefour)
    lredonefour, redfactor,
    # Tabulated Ewald functions (saves 2x exp + erfc per pair)
    erfc_tab, exp_negsq_tab
):
    """
    Compute electric field at each atom from charges (Ewald real-space).
    Also store pair data needed for SCF and force computation.

    Returns:
        elfown: (3, nat) electric field at each atom from charges
        listpol: (maxneigh, nat) polarization neighbor list
        listmpol: (nat,) count per atom
        rx, ry, rz: (maxneigh, nat) pair displacement vectors
        rr: (maxneigh, nat) pair distances
        rrinv: (maxneigh, nat) 1/r
        expc1r3: (maxneigh, nat) exp(-c1*r^3) Thole damping
    """
    rootpi = np.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi

    elfown = np.zeros((3, nat))

    # Estimate max neighbors per atom
    max_neigh = 0
    for iat in range(nat):
        if listm[iat] > max_neigh:
            max_neigh = int(listm[iat])
    max_neigh = max_neigh + 10  # small safety margin

    listpol = np.zeros((max_neigh, nat), dtype=np.int32)
    listmpol = np.zeros(nat, dtype=np.int32)
    rx_arr = np.zeros((max_neigh, nat))
    ry_arr = np.zeros((max_neigh, nat))
    rz_arr = np.zeros((max_neigh, nat))
    rr_arr = np.zeros((max_neigh, nat))
    rrinv_arr = np.zeros((max_neigh, nat))
    expc1r3_arr = np.zeros((max_neigh, nat))

    # Main neighbor loop (same pairs as nonbonded spline evaluation)
    for iat in range(nat):
        itype = atomtype[iat]
        nneigh = 0
        for k in range(int(listm[iat])):
            jat = int(list_neighbors[k, iat])
            jtype = atomtype[jat]

            xij1 = x[0, jat] - x[0, iat]
            xij2 = x[1, jat] - x[1, iat]
            xij3 = x[2, jat] - x[2, iat]
            if abs(xij1) > el[0]: xij1 -= np.sign(xij1) * box[0]
            if abs(xij2) > el[1]: xij2 -= np.sign(xij2) * box[1]
            if abs(xij3) > el[2]: xij3 -= np.sign(xij3) * box[2]

            z = xij1*xij1 + xij2*xij2 + xij3*xij3
            if z > rcut2:
                continue

            if z < 1.0e-10:
                continue
            r = np.sqrt(z)
            rinv = 1.0 / r
            zinv = rinv * rinv

            # Store pair data for polarization
            if nneigh < max_neigh:
                listpol[nneigh, iat] = jat
                rx_arr[nneigh, iat] = xij1
                ry_arr[nneigh, iat] = xij2
                rz_arr[nneigh, iat] = xij3
                rr_arr[nneigh, iat] = r
                rrinv_arr[nneigh, iat] = rinv
                # Thole damping: exp(-c1(i,j) * r^3)
                if itype < ncharge and jtype < ncharge:
                    expc1r3_arr[nneigh, iat] = np.exp(-c1[itype, jtype] * r**3)
                else:
                    expc1r3_arr[nneigh, iat] = 0.0
                nneigh += 1

            # Electric field: E_i += q_j * (erfc(alpha*r)/r + 2*alpha/sqrt(pi)*exp(-(alpha*r)^2)) / r^2 * r_hat
            # Use tabulated erfc and exp(-x^2) (~2x faster than inline)
            alphar = alpha * r
            if alphar < 6.0:
                fpos = alphar * 2730.6666666666665   # TABLE_DX_INV (16384/6)
                idx = int(fpos)
                frac = fpos - idx
                erfc_val = erfc_tab[idx] + frac * (erfc_tab[idx+1] - erfc_tab[idx])
                exp_val  = exp_negsq_tab[idx] + frac * (exp_negsq_tab[idx+1] - exp_negsq_tab[idx])
            else:
                erfc_val = 0.0
                exp_val  = 0.0
            fact = (erfc_val * rinv + twoalphapi * exp_val) * zinv

            qi = q[itype] if itype < ncharge else 0.0
            qj = q[jtype] if jtype < ncharge else 0.0

            facti = fact * qi
            factj = fact * qj
            # Field at iat from charge qj: -qj * gradient(1/r) = -qj * r_hat/r^2
            elfown[0, iat] -= factj * xij1
            elfown[1, iat] -= factj * xij2
            elfown[2, iat] -= factj * xij3
            # Field at jat from charge qi (Newton's third law)
            elfown[0, jat] += facti * xij1
            elfown[1, jat] += facti * xij2
            elfown[2, jat] += facti * xij3

        listmpol[iat] = nneigh

    # 1-4 Q-mu field correction (Fortran lines 3107-3137)
    if lredQ_mu14:
        for iat in range(nat):
            itype = atomtype[iat]
            for k in range(int(listm14[iat])):
                jat = int(list14[k, iat])
                jtype = atomtype[jat]

                xij1 = x[0, jat] - x[0, iat]
                xij2 = x[1, jat] - x[1, iat]
                xij3 = x[2, jat] - x[2, iat]
                if abs(xij1) > el[0]: xij1 -= np.sign(xij1) * box[0]
                if abs(xij2) > el[1]: xij2 -= np.sign(xij2) * box[1]
                if abs(xij3) > el[2]: xij3 -= np.sign(xij3) * box[2]

                z = xij1*xij1 + xij2*xij2 + xij3*xij3
                if z < 1.0e-10:
                    continue
                r = np.sqrt(z)
                rinv = 1.0 / r
                zinv = rinv * rinv
                alphar = alpha * r
                # Tabulated Ewald (same approach as main loop)
                if alphar < 6.0:
                    fpos = alphar * 2730.6666666666665
                    idx = int(fpos)
                    frac = fpos - idx
                    erfc_v = erfc_tab[idx] + frac * (erfc_tab[idx+1] - erfc_tab[idx])
                    exp_v  = exp_negsq_tab[idx] + frac * (exp_negsq_tab[idx+1] - exp_negsq_tab[idx])
                else:
                    erfc_v = 0.0
                    exp_v  = 0.0
                fact = (erfc_v * rinv + twoalphapi * exp_v) * zinv
                fact *= (-redQmufactor)

                qi = q[itype] if itype < ncharge else 0.0
                qj = q[jtype] if jtype < ncharge else 0.0
                facti = fact * qi
                factj = fact * qj
                elfown[0, iat] -= factj * xij1
                elfown[1, iat] -= factj * xij2
                elfown[2, iat] -= factj * xij3
                elfown[0, jat] += facti * xij1
                elfown[1, jat] += facti * xij2
                elfown[2, jat] += facti * xij3

    return (elfown, listpol, listmpol, rx_arr, ry_arr, rz_arr,
            rr_arr, rrinv_arr, expc1r3_arr)


# =========================================================================
# PHASE 2-3: DIPOLE PREDICTION + SCF ITERATION
# (Fortran interch lines 3141-3274)
# =========================================================================

@jit(nopython=True, cache=True)
def scf_iteration(
    elfown, nat, atomtype, pol, ncharge,
    listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3,
    c1, arf, rtaper, ro, tauden, tolpol,
    # Previous dipoles for prediction
    px1old, py1old, pz1old,
    px2old, py2old, pz2old,
    px3old, py3old, pz3old,
    kount
):
    """
    Self-consistent field iteration for induced dipoles.

    Returns:
        px, py, pz: (nat,) converged induced dipole components (e*Angstrom)
        niter: number of SCF iterations
    """
    # Initialize / predict dipoles
    px = np.zeros(nat)
    py = np.zeros(nat)
    pz = np.zeros(nat)

    if kount > 0:
        # Linear extrapolation from 3 previous timesteps
        for iat in range(nat):
            px[iat] = 3.0*(px1old[iat] - px2old[iat]) + px3old[iat]
            py[iat] = 3.0*(py1old[iat] - py2old[iat]) + py3old[iat]
            pz[iat] = 3.0*(pz1old[iat] - pz2old[iat]) + pz3old[iat]

    # SCF loop
    niter = 0
    converged = False

    # Pre-allocate scratch buffers ONCE for the whole SCF loop.
    # Reusing them and zeroing in-place is faster than re-allocating
    # 3 * 8716 * 8 bytes = 200 KB on each of ~11 iterations.
    txown = np.zeros(nat)
    tyown = np.zeros(nat)
    tzown = np.zeros(nat)

    while not converged:
        niter += 1
        if niter > 1000:
            break

        # Zero in place (faster than re-allocating)
        for iat in range(nat):
            txown[iat] = 0.0
            tyown[iat] = 0.0
            tzown[iat] = 0.0

        for iat in range(nat):
            itype = atomtype[iat]
            for k in range(listmpol[iat]):
                jat = listpol[k, iat]
                jtype = atomtype[jat]
                rinv_val = rrinv[k, iat]
                r_val = rr[k, iat]
                zinv = rinv_val * rinv_val
                rxij = rx[k, iat]
                ryij = ry[k, iat]
                rzij = rz[k, iat]

                # Thole damping
                c1r3 = c1[itype, jtype] * r_val**3 if (itype < ncharge and jtype < ncharge) else 1.0e6 * r_val**3
                expc1r3val = expc1r3[k, iat]
                lamda3 = 1.0 - expc1r3val
                lamda5 = 1.0 - expc1r3val * (1.0 + c1r3)

                # Tau tapering
                tauval = tau_func(r_val, rtaper, ro, tauden)
                fact = rinv_val * zinv * tauval

                # Dipole-dipole interaction tensor: T_ij * mu_j
                pjr = 3.0 * zinv * fact * (px[jat]*rxij + py[jat]*ryij + pz[jat]*rzij) * lamda5
                pir = 3.0 * zinv * fact * (px[iat]*rxij + py[iat]*ryij + pz[iat]*rzij) * lamda5

                tfact = arf - fact * lamda3

                # Own atom contribution (from jat dipole)
                txown[iat] += pjr * rxij + px[jat] * tfact
                tyown[iat] += pjr * ryij + py[jat] * tfact
                tzown[iat] += pjr * rzij + pz[jat] * tfact

                # Other atom contribution (from iat dipole)
                txown[jat] += pir * rxij + px[iat] * tfact
                tyown[jat] += pir * ryij + py[iat] * tfact
                tzown[jat] += pir * rzij + pz[iat] * tfact

        # Update dipoles: mu_i = alpha_i * (E_charge + T_dipole + arf*mu)
        converged = True
        for iat in range(nat):
            itype = atomtype[iat]
            alpha_i = pol[itype] if itype < ncharge else 0.0

            pxnew = alpha_i * (elfown[0, iat] + txown[iat] + arf * px[iat])
            pynew = alpha_i * (elfown[1, iat] + tyown[iat] + arf * py[iat])
            pznew = alpha_i * (elfown[2, iat] + tzown[iat] + arf * pz[iat])

            dx = pxnew - px[iat]
            dy = pynew - py[iat]
            dz = pznew - pz[iat]
            diff2 = dx*dx + dy*dy + dz*dz

            px[iat] = pxnew
            py[iat] = pynew
            pz[iat] = pznew

            if diff2 > tolpol:
                converged = False

    return px, py, pz, niter


# =========================================================================
# PHASE 4: POLARIZATION ENERGY
# (Fortran interch lines 3311-3318)
# =========================================================================

@jit(nopython=True, cache=True)
def compute_pol_energy(elfown, px, py, pz, nat):
    """
    Induced dipole energy: E_pol = -0.5 * 332.08 * sum(E_charge . mu)
    """
    uind = 0.0
    for iat in range(nat):
        uind -= (elfown[0, iat]*px[iat] +
                 elfown[1, iat]*py[iat] +
                 elfown[2, iat]*pz[iat])
    return uind * 0.5 * COULOMB_CONST


# =========================================================================
# PHASE 5: POLARIZATION FORCES
# (Fortran interch lines 3348-3449)
# =========================================================================

@jit(nopython=True, cache=True)
def compute_pol_forces(
    nat, atomtype, q, ncharge, alpha,
    px, py, pz,
    listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3,
    c1, rtaper, ro, tauden,
    erfc_tab, exp_negsq_tab
):
    """
    Compute forces from induced dipoles:
      - p.grad(T).p (dipole-dipole gradient with Thole damping + tau tapering)
      - p.grad(E) (charge-dipole gradient with Ewald real-space)

    Returns:
        findown: (3, nat) induced forces (before 332.08 scaling)
        virind: scalar induced virial
    """
    rootpi = np.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi

    findown = np.zeros((3, nat))
    virind = 0.0

    for iat in range(nat):
        itype = atomtype[iat]
        ci = q[itype] if itype < ncharge else 0.0

        for k in range(listmpol[iat]):
            jat = listpol[k, iat]
            jtype = atomtype[jat]
            cj = q[jtype] if jtype < ncharge else 0.0

            r_val = rr[k, iat]
            rinv_val = rrinv[k, iat]
            z = r_val * r_val
            zinv = rinv_val * rinv_val
            r3inv = zinv * rinv_val
            r5inv = r3inv * zinv
            rxij = rx[k, iat]
            ryij = ry[k, iat]
            rzij = rz[k, iat]

            # Thole damping
            if itype < ncharge and jtype < ncharge:
                c1val = c1[itype, jtype]
            else:
                c1val = 1.0e6
            c1r3 = c1val * r_val**3
            expc1r3val = expc1r3[k, iat]
            lamda3 = 1.0 - expc1r3val
            lamda5 = 1.0 - expc1r3val * (1.0 + c1r3)
            three_c1r2 = 3.0 * c1val * z
            dlamda3 = three_c1r2 * expc1r3val
            dlamda5 = three_c1r2 * c1r3 * expc1r3val

            # Dot products
            pjr = px[jat]*rxij + py[jat]*ryij + pz[jat]*rzij
            pir = px[iat]*rxij + py[iat]*ryij + pz[iat]*rzij
            pipj = px[jat]*px[iat] + py[jat]*py[iat] + pz[jat]*pz[iat]
            qq = ci*pjr - cj*pir

            # p.grad(T).p with tau tapering
            tauval = tau_func(r_val, rtaper, ro, tauden)
            dtauval = dtau_func(r_val, rtaper, ro, tauden)

            p1 = 5.0*zinv*pjr*pir*lamda5 - pipj*lamda3
            p2 = dtauval*rinv_val*(3.0*r5inv*pir*pjr*lamda5 - pipj*r3inv*lamda3)
            p3 = 3.0 * r5inv * tauval
            p4 = p3 * p1 - p2

            # Derivative of damped terms
            p6 = (3.0*r5inv*pir*pjr*dlamda5 - pipj*r3inv*dlamda3) * rinv_val

            # p.grad(E) (charge-dipole with Ewald real-space, tabulated)
            alphar = alpha * r_val
            if alphar < 6.0:
                fpos = alphar * 2730.6666666666665
                idx = int(fpos)
                frac = fpos - idx
                erfc_v = erfc_tab[idx] + frac * (erfc_tab[idx+1] - erfc_tab[idx])
                exp_v  = exp_negsq_tab[idx] + frac * (exp_negsq_tab[idx+1] - exp_negsq_tab[idx])
            else:
                erfc_v = 0.0
                exp_v  = 0.0
            ee1 = erfc_v * r3inv
            ee2 = zinv * twoalphapi * exp_v
            ee3 = ee2 * (3.0 + 2.0*alphar*alphar)
            ee5 = qq * zinv * (3.0*ee1 + ee3)
            ee6 = ee1 + ee2

            # Total force on this pair
            f1 = p4 + ee5 - p6*tauval
            f2 = -(ee6*ci + p3*pir*lamda5)
            f3 = ee6*cj - p3*pjr*lamda5

            ffx = f1*rxij + f2*px[jat] + f3*px[iat]
            ffy = f1*ryij + f2*py[jat] + f3*py[iat]
            ffz = f1*rzij + f2*pz[jat] + f3*pz[iat]

            findown[0, iat] += ffx
            findown[1, iat] += ffy
            findown[2, iat] += ffz
            findown[0, jat] -= ffx
            findown[1, jat] -= ffy
            findown[2, jat] -= ffz

            virind -= (ffx*rxij + ffy*ryij + ffz*rzij)

    return findown, virind


# =========================================================================
# PHASE 6: Q-MU EWALD CORRECTION FOR EXCLUDED PAIRS
# (Fortran interch lines 3456-3515)
# =========================================================================

@jit(nopython=True, cache=True)
def compute_qmu_correction(
    nat, atomtype, q, ncharge, alpha,
    px, py, pz,
    listex, listmex, x, box, el
):
    """
    Ewald correction for charge-dipole: subtract erf(alpha*r)/r^3 terms
    for excluded (bonded) pairs. Uses /2 factor because exclusion list
    is symmetric (each pair counted twice).

    Returns:
        findown: (3, nat) correction forces (before 332.08 scaling)
        virind: scalar correction virial
    """
    rootpi = np.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi

    findown = np.zeros((3, nat))
    virind = 0.0

    for iat in range(nat):
        itype = atomtype[iat]
        ci = q[itype] if itype < ncharge else 0.0
        for k in range(int(listmex[iat])):
            jat = int(listex[k, iat])
            jtype = atomtype[jat]
            cj = q[jtype] if jtype < ncharge else 0.0

            xij1 = x[0, jat] - x[0, iat]
            xij2 = x[1, jat] - x[1, iat]
            xij3 = x[2, jat] - x[2, iat]
            if abs(xij1) > el[0]: xij1 -= np.sign(xij1) * box[0]
            if abs(xij2) > el[1]: xij2 -= np.sign(xij2) * box[1]
            if abs(xij3) > el[2]: xij3 -= np.sign(xij3) * box[2]

            z = xij1*xij1 + xij2*xij2 + xij3*xij3
            if z < 1.0e-10:
                continue
            r = np.sqrt(z)
            rinv = 1.0 / r
            zinv = rinv * rinv
            r3inv = zinv * rinv

            alphar = alpha * r
            pir = px[iat]*xij1 + py[iat]*xij2 + pz[iat]*xij3
            pjr = px[jat]*xij1 + py[jat]*xij2 + pz[jat]*xij3
            qq = ci*pjr - cj*pir

            # erf-based correction (opposite sign from erfc in real-space)
            ee1 = -erf_approx(alphar) * r3inv
            ee2 = zinv * twoalphapi * np.exp(-alphar*alphar)
            ee3 = ee2 * (3.0 + 2.0*alphar*alphar)
            ee5 = qq * zinv * (3.0*ee1 + ee3)
            ee6 = ee1 + ee2

            # Factor 1/2 for symmetric exclusion list
            f1 = (ee5*xij1 - ee6*(ci*px[jat] - cj*px[iat])) / 2.0
            f2 = (ee5*xij2 - ee6*(ci*py[jat] - cj*py[iat])) / 2.0
            f3 = (ee5*xij3 - ee6*(ci*pz[jat] - cj*pz[iat])) / 2.0

            virind += (-ee5*z + ee6*qq) / 2.0

            findown[0, iat] += f1
            findown[1, iat] += f2
            findown[2, iat] += f3
            findown[0, jat] -= f1
            findown[1, jat] -= f2
            findown[2, jat] -= f3

    return findown, virind


# =========================================================================
# PHASE 7: 1-4 Q-MU CORRECTIONS
# (Fortran interch lines 3526-3735)
# =========================================================================

@jit(nopython=True, cache=True)
def compute_qmu_14_corrections(
    nat, atomtype, q, ncharge, alpha,
    px, py, pz,
    list14, listm14, x, box, el,
    lredonefour, redfactor, lredQ_mu14, redQmufactor
):
    """
    1-4 pair corrections for charge-dipole interactions.
    Handles both lredonefour and lredQ_mu14 scaling.

    Returns:
        findown: (3, nat) correction forces (before 332.08 scaling)
        virind: scalar correction virial
    """
    rootpi = np.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi

    findown = np.zeros((3, nat))
    virind = 0.0

    for iat in range(nat):
        itype = atomtype[iat]
        ci = q[itype] if itype < ncharge else 0.0
        for k in range(int(listm14[iat])):
            jat = int(list14[k, iat])
            jtype = atomtype[jat]
            cj = q[jtype] if jtype < ncharge else 0.0

            xij1 = x[0, jat] - x[0, iat]
            xij2 = x[1, jat] - x[1, iat]
            xij3 = x[2, jat] - x[2, iat]
            if abs(xij1) > el[0]: xij1 -= np.sign(xij1) * box[0]
            if abs(xij2) > el[1]: xij2 -= np.sign(xij2) * box[1]
            if abs(xij3) > el[2]: xij3 -= np.sign(xij3) * box[2]

            z = xij1*xij1 + xij2*xij2 + xij3*xij3
            if z < 1.0e-10:
                continue
            r = np.sqrt(z)
            rinv = 1.0 / r
            zinv = rinv * rinv
            r3inv = zinv * rinv

            pjr = px[jat]*xij1 + py[jat]*xij2 + pz[jat]*xij3
            pir = px[iat]*xij1 + py[iat]*xij2 + pz[iat]*xij3
            qq = ci*pjr - cj*pir
            alphar = alpha * r

            # erfc-based Q-mu for 1-4 (with reduction factor)
            ee1 = erfc_approx(alphar) * r3inv
            ee2 = zinv * twoalphapi * np.exp(-alphar*alphar)
            ee3 = ee2 * (3.0 + 2.0*alphar*alphar)
            ee5 = qq * zinv * (3.0*ee1 + ee3)
            ee6 = ee1 + ee2

            if lredonefour:
                ffx = (ee5*xij1 - ee6*ci*px[jat] + ee6*cj*px[iat]) * (-redfactor)
                ffy = (ee5*xij2 - ee6*ci*py[jat] + ee6*cj*py[iat]) * (-redfactor)
                ffz = (ee5*xij3 - ee6*ci*pz[jat] + ee6*cj*pz[iat]) * (-redfactor)
                findown[0, iat] += ffx; findown[1, iat] += ffy; findown[2, iat] += ffz
                findown[0, jat] -= ffx; findown[1, jat] -= ffy; findown[2, jat] -= ffz
                virind -= (ffx*xij1 + ffy*xij2 + ffz*xij3)

                # erf correction for lredonefour
                ee1c = -erf_approx(alphar) * r3inv
                ee5c = qq * zinv * (3.0*ee1c + ee3)
                ee6c = ee1c + ee2
                f1 = (ee5c*xij1 - ee6c*(ci*px[jat]-cj*px[iat])) * redfactor
                f2 = (ee5c*xij2 - ee6c*(ci*py[jat]-cj*py[iat])) * redfactor
                f3 = (ee5c*xij3 - ee6c*(ci*pz[jat]-cj*pz[iat])) * redfactor
                virind -= redfactor*(ee5c*z - ee6c*qq)
                findown[0, iat] += f1; findown[1, iat] += f2; findown[2, iat] += f3
                findown[0, jat] -= f1; findown[1, jat] -= f2; findown[2, jat] -= f3

            if lredQ_mu14:
                ffx = (ee5*xij1 - ee6*ci*px[jat] + ee6*cj*px[iat]) * (-redQmufactor)
                ffy = (ee5*xij2 - ee6*ci*py[jat] + ee6*cj*py[iat]) * (-redQmufactor)
                ffz = (ee5*xij3 - ee6*ci*pz[jat] + ee6*cj*pz[iat]) * (-redQmufactor)
                findown[0, iat] += ffx; findown[1, iat] += ffy; findown[2, iat] += ffz
                findown[0, jat] -= ffx; findown[1, jat] -= ffy; findown[2, jat] -= ffz
                virind -= (ffx*xij1 + ffy*xij2 + ffz*xij3)

                # erf correction for lredQ_mu14
                ee1c = -erf_approx(alphar) * r3inv
                ee5c = qq * zinv * (3.0*ee1c + ee3)
                ee6c = ee1c + ee2
                f1 = (ee5c*xij1 - ee6c*(ci*px[jat]-cj*px[iat])) * redQmufactor
                f2 = (ee5c*xij2 - ee6c*(ci*py[jat]-cj*py[iat])) * redQmufactor
                f3 = (ee5c*xij3 - ee6c*(ci*pz[jat]-cj*pz[iat])) * redQmufactor
                virind -= redQmufactor*(ee5c*z - ee6c*qq)
                findown[0, iat] += f1; findown[1, iat] += f2; findown[2, iat] += f3
                findown[0, jat] -= f1; findown[1, jat] -= f2; findown[2, jat] -= f3

    return findown, virind


# =========================================================================
# PHASE 8: RECIPROCAL Q-MU (k-space charge-dipole)
# (Fortran recipQmu lines 8476-8863)
# =========================================================================

def compute_recipQmu(state, px, py, pz):
    """
    Reciprocal-space charge-dipole interaction.
    Uses the same k-vectors and structure factors as the Ewald sum,
    but with dipole structure factors S_p(k) = sum_i mu_i . k * exp(ik.ri).

    Returns:
        f_recip: (3, nat) reciprocal Q-mu forces
        virind_recip: scalar virial
    """
    nat = state.atoms.nat
    x = state.atoms.x
    box = state.atoms.box
    atomtype = state.atoms.atomtype
    q = state.ff.q
    ncharge = state.ff.ncharge
    alpha = state.params.alphai

    klimitx = state.params.klimitx
    klimity = state.params.klimity
    klimitz = state.params.klimitz

    front = 2.0 * PI / box
    twopifact = 2.0 * PI / np.prod(box)
    fouralpha2inv = 1.0 / (4.0 * alpha * alpha)
    elfact = 2.0 * twopifact

    kxmax2inv = 1.0 / float(klimitx**2)
    kymax2inv = 1.0 / float(klimity**2)
    kzmax2inv = 1.0 / float(klimitz**2)

    # Pre-compute cos/sin for each atom and k-index (from ewald.py)
    cos_x = np.zeros((nat, klimitx + 1))
    sin_x = np.zeros((nat, klimitx + 1))
    cos_y = np.zeros((nat, klimity + 1))
    sin_y = np.zeros((nat, klimity + 1))
    cos_z = np.zeros((nat, klimitz + 1))
    sin_z = np.zeros((nat, klimitz + 1))

    for iat in range(nat):
        cos_x[iat, 0] = 1.0; sin_x[iat, 0] = 0.0
        cos_y[iat, 0] = 1.0; sin_y[iat, 0] = 0.0
        cos_z[iat, 0] = 1.0; sin_z[iat, 0] = 0.0
        cos_x[iat, 1] = np.cos(front[0] * x[0, iat])
        sin_x[iat, 1] = np.sin(front[0] * x[0, iat])
        cos_y[iat, 1] = np.cos(front[1] * x[1, iat])
        sin_y[iat, 1] = np.sin(front[1] * x[1, iat])
        cos_z[iat, 1] = np.cos(front[2] * x[2, iat])
        sin_z[iat, 1] = np.sin(front[2] * x[2, iat])
        for kn in range(2, klimitx + 1):
            cos_x[iat, kn] = cos_x[iat, kn-1]*cos_x[iat, 1] - sin_x[iat, kn-1]*sin_x[iat, 1]
            sin_x[iat, kn] = sin_x[iat, kn-1]*cos_x[iat, 1] + cos_x[iat, kn-1]*sin_x[iat, 1]
        for kn in range(2, klimity + 1):
            cos_y[iat, kn] = cos_y[iat, kn-1]*cos_y[iat, 1] - sin_y[iat, kn-1]*sin_y[iat, 1]
            sin_y[iat, kn] = sin_y[iat, kn-1]*cos_y[iat, 1] + cos_y[iat, kn-1]*sin_y[iat, 1]
        for kn in range(2, klimitz + 1):
            cos_z[iat, kn] = cos_z[iat, kn-1]*cos_z[iat, 1] - sin_z[iat, kn-1]*sin_z[iat, 1]
            sin_z[iat, kn] = sin_z[iat, kn-1]*cos_z[iat, 1] + cos_z[iat, kn-1]*sin_z[iat, 1]

    f_recip = np.zeros((3, nat))
    virind = 0.0

    # Loop over k-vectors (half-space)
    for kx in range(0, klimitx + 1):
        hx = float(kx) * front[0]
        kymin = -klimity if kx > 0 else 0
        for ky in range(kymin, klimity + 1):
            hy = float(ky) * front[1]
            kzmin = 1 if (kx == 0 and ky == 0) else -klimitz
            for kz in range(kzmin, klimitz + 1):
                hz = float(kz) * front[2]

                tmp2 = float(kx*kx)*kxmax2inv + float(ky*ky)*kymax2inv + float(kz*kz)*kzmax2inv
                if tmp2 > 1.0:
                    continue

                rkmag2 = hx*hx + hy*hy + hz*hz
                if rkmag2 < 1.0e-12:
                    continue
                akvec = 2.0 * np.exp(-rkmag2 * fouralpha2inv) / rkmag2

                # Compute cos/sin of k.r for each atom
                cos_xyz = np.zeros(nat)
                sin_xyz = np.zeros(nat)
                kya = abs(ky)
                kza = abs(kz)
                for iat in range(nat):
                    # cos(kx*x + ky*y + kz*z) using recurrence
                    cx = cos_x[iat, kx]
                    sx = sin_x[iat, kx]
                    if ky >= 0:
                        cxy = cx * cos_y[iat, kya] - sx * sin_y[iat, kya]
                        sxy = sx * cos_y[iat, kya] + cx * sin_y[iat, kya]
                    else:
                        cxy = cx * cos_y[iat, kya] + sx * sin_y[iat, kya]
                        sxy = sx * cos_y[iat, kya] - cx * sin_y[iat, kya]
                    if kz >= 0:
                        cos_xyz[iat] = cxy * cos_z[iat, kza] - sxy * sin_z[iat, kza]
                        sin_xyz[iat] = sxy * cos_z[iat, kza] + cxy * sin_z[iat, kza]
                    else:
                        cos_xyz[iat] = cxy * cos_z[iat, kza] + sxy * sin_z[iat, kza]
                        sin_xyz[iat] = sxy * cos_z[iat, kza] - cxy * sin_z[iat, kza]

                # Charge structure factor: S_q(k) = sum qi * exp(ik.ri)
                cossum_q = 0.0
                sinsum_q = 0.0
                # Dipole structure factor: S_p(k) = sum (mu_i . k) * exp(ik.ri)
                cossum_p = 0.0
                sinsum_p = 0.0
                for iat in range(nat):
                    itype = atomtype[iat]
                    ci = q[itype] if itype < ncharge else 0.0
                    pk = px[iat]*hx + py[iat]*hy + pz[iat]*hz

                    cossum_q += ci * cos_xyz[iat]
                    sinsum_q += ci * sin_xyz[iat]
                    cossum_p += pk * cos_xyz[iat]
                    sinsum_p += pk * sin_xyz[iat]

                # Cross term: 2 * Re(S_q* . S_p) = 2*(cosQ*sinP - sinQ*cosP)
                # Energy contribution: elfact * akvec * (cosQ*sinP - sinQ*cosP) per k
                # Force on atom i: elfact * akvec * ...
                prefact1 = elfact * akvec * COULOMB_CONST

                for iat in range(nat):
                    itype = atomtype[iat]
                    ci = q[itype] if itype < ncharge else 0.0
                    pk = px[iat]*hx + py[iat]*hy + pz[iat]*hz
                    cost = cos_xyz[iat]
                    sint = sin_xyz[iat]

                    # Force from Q-mu reciprocal interaction
                    # Charge part: ci * (sinP*cost - cosP*sint)
                    fq = ci * (sinsum_p * cost - cossum_p * sint)
                    # Dipole part: pk * (cosQ*cost + sinQ*sint)  [different sign convention]
                    fp = pk * (cossum_q * cost + sinsum_q * sint)
                    # mu_i contribution to field: k * (cosQ*sint - sinQ*cost)
                    fmu = cossum_q * sint - sinsum_q * cost

                    ftotal = prefact1 * (fq - fp)
                    f_recip[0, iat] += ftotal * hx + prefact1 * fmu * hx * ci
                    # Simplified: the full expression includes charge and dipole cross-terms
                    # For forces on charge part:
                    charge_force = prefact1 * (sinsum_p * cost - cossum_p * sint)
                    f_recip[0, iat] += ci * charge_force * hx  # placeholder - needs proper derivation
                    # This is approximate - full recipQmu force is complex

                # Note: The full recipQmu force derivation is lengthy.
                # For now, accumulate energy contribution for validation.

    # Simplified version - full implementation would follow Fortran exactly
    # The energy contribution is typically 5-10% of the total polarization
    return f_recip, virind


# =========================================================================
# MASTER FUNCTION: COMPUTE ALL POLARIZATION
# =========================================================================

# =========================================================================
# PHASE 1c-1d: ELFOWN CORRECTIONS (Fortran correct() lines 643-786)
# =========================================================================

@jit(nopython=True, cache=True)
def _apply_elfown_correction(
    elfown, x, nat, atomtype, q, ncharge, alpha, box, el,
    listex, listmex
):
    """
    Ewald correction to electric field for excluded pairs.
    Fortran correct() lines 643-653: adds erf-based field correction.
    Factor /2 for symmetric exclusion list.
    """
    rootpi = np.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi

    for iat in range(nat):
        itype = atomtype[iat]
        ci = q[itype] if itype < ncharge else 0.0
        for k in range(int(listmex[iat])):
            jat = int(listex[k, iat])
            jtype = atomtype[jat]
            cj = q[jtype] if jtype < ncharge else 0.0

            xij1 = x[0, jat] - x[0, iat]
            xij2 = x[1, jat] - x[1, iat]
            xij3 = x[2, jat] - x[2, iat]
            if abs(xij1) > el[0]: xij1 -= np.sign(xij1) * box[0]
            if abs(xij2) > el[1]: xij2 -= np.sign(xij2) * box[1]
            if abs(xij3) > el[2]: xij3 -= np.sign(xij3) * box[2]

            z = xij1*xij1 + xij2*xij2 + xij3*xij3
            if z < 1.0e-10:
                continue
            r = np.sqrt(z)
            alphar = alpha * r
            rinv = 1.0 / r
            zinv = rinv * rinv
            r3inv = zinv * rinv
            erf_val = 1.0 - erfc_approx(alphar)

            # Fortran: ffcorrect = (erf/r³ - 2α/√π*exp/r²) / 2
            ffcorrect = (erf_val * r3inv - twoalphapi * np.exp(-alphar*alphar) * zinv) / 2.0

            f1 = ffcorrect * xij1
            f2 = ffcorrect * xij2
            f3 = ffcorrect * xij3
            elfown[0, iat] += f1 * cj
            elfown[1, iat] += f2 * cj
            elfown[2, iat] += f3 * cj
            elfown[0, jat] -= f1 * ci
            elfown[1, jat] -= f2 * ci
            elfown[2, jat] -= f3 * ci


@jit(nopython=True, cache=True)
def _apply_elfown_14_correction(
    elfown, x, nat, atomtype, q, ncharge, alpha, box, el,
    list14, listm14, redQmufactor
):
    """
    Q-mu 1-4 correction to electric field.
    Fortran correct() lines 776-786.
    NO /2 factor (1-4 list is not symmetric like exclusion list).
    """
    rootpi = np.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi

    for iat in range(nat):
        itype = atomtype[iat]
        ci = q[itype] if itype < ncharge else 0.0
        for k in range(int(listm14[iat])):
            jat = int(list14[k, iat])
            jtype = atomtype[jat]
            cj = q[jtype] if jtype < ncharge else 0.0

            xij1 = x[0, jat] - x[0, iat]
            xij2 = x[1, jat] - x[1, iat]
            xij3 = x[2, jat] - x[2, iat]
            if abs(xij1) > el[0]: xij1 -= np.sign(xij1) * box[0]
            if abs(xij2) > el[1]: xij2 -= np.sign(xij2) * box[1]
            if abs(xij3) > el[2]: xij3 -= np.sign(xij3) * box[2]

            z = xij1*xij1 + xij2*xij2 + xij3*xij3
            if z < 1.0e-10:
                continue
            r = np.sqrt(z)
            alphar = alpha * r
            rinv = 1.0 / r
            zinv = rinv * rinv
            r3inv = zinv * rinv
            erf_val = 1.0 - erfc_approx(alphar)

            ffcorrect = redQmufactor * (erf_val * r3inv - twoalphapi * np.exp(-alphar*alphar) * zinv)

            f1 = ffcorrect * xij1
            f2 = ffcorrect * xij2
            f3 = ffcorrect * xij3
            elfown[0, iat] += f1 * cj
            elfown[1, iat] += f2 * cj
            elfown[2, iat] += f3 * cj
            elfown[0, jat] -= f1 * ci
            elfown[1, jat] -= f2 * ci
            elfown[2, jat] -= f3 * ci


class PolarizationState:
    """State for the polarization module, persists between timesteps."""
    def __init__(self, nat):
        self.px = np.zeros(nat)
        self.py = np.zeros(nat)
        self.pz = np.zeros(nat)
        # Use float64 for dipole history. The previous float32 storage gave
        # quantization error ~1e-7 in mu, exceeding tolpol=1e-9 noise floor
        # and degrading the warm-start predictor. Memory cost: ~600 KB
        # extra for an 8716-atom system; trivial.
        self.px1old = np.zeros(nat, dtype=np.float64)
        self.py1old = np.zeros(nat, dtype=np.float64)
        self.pz1old = np.zeros(nat, dtype=np.float64)
        self.px2old = np.zeros(nat, dtype=np.float64)
        self.py2old = np.zeros(nat, dtype=np.float64)
        self.pz2old = np.zeros(nat, dtype=np.float64)
        self.px3old = np.zeros(nat, dtype=np.float64)
        self.py3old = np.zeros(nat, dtype=np.float64)
        self.pz3old = np.zeros(nat, dtype=np.float64)
        self.c1 = None
        self.arf = 0.0
        self.tauden = 1.0
        self.initialized = False


def init_polarization(state):
    """Initialize polarization data structures. Call once during setup."""
    nat = state.atoms.nat
    ncharge = state.ff.ncharge

    if not hasattr(state, 'pol_state') or state.pol_state is None:
        state.pol_state = PolarizationState(nat)

    ps = state.pol_state

    # Build c1 (Thole damping) array
    ps.c1 = build_c1_array(state.ff.pol, ncharge)

    # Compute tauden
    ro = state.params.rcut
    rtaper = state.params.rtaper
    if state.params.tapering:
        ps.tauden = ((ro - rtaper)**3) / 6.0
    else:
        ps.tauden = 1.0
        state.params.rtaper = ro

    # Compute arf (reaction field constant)
    ps.arf = compute_arf(rtaper, ro, state.params.epsrf, ps.tauden)

    ps.initialized = True
    print(f"  Polarization initialized:")
    print(f"    a_thole = {A_THOLE}, arf = {ps.arf:.6f}")
    print(f"    rtaper = {rtaper}, ro = {ro}, tauden = {ps.tauden:.6e}")
    print(f"    tolpol = {state.params.tolpol}")


def compute_polarization(state):
    """
    Master function: compute all polarization contributions.
    Call after nonbonded forces and Ewald are computed.

    Modifies:
        state.atoms.f: adds induced forces
        state.energy.unbd: adds polarization energy

    Returns:
        epol: polarization energy (kcal/mol)
    """
    if not state.ff.lpolarizable:
        return 0.0

    ps = state.pol_state
    if not ps.initialized:
        init_polarization(state)

    nat = state.atoms.nat
    ncharge = state.ff.ncharge
    alpha = state.params.alphai
    rcut2 = state.params.rcut**2
    ro = state.params.rcut
    rtaper = state.params.rtaper

    # Ensure 1-4 list exists
    if not hasattr(state.neighbors, 'list14') or state.neighbors.list14 is None:
        list14 = np.zeros((1, nat), dtype=np.int32)
        listm14 = np.zeros(nat, dtype=np.int32)
    else:
        list14 = state.neighbors.list14
        listm14 = state.neighbors.listm14

    # Phase 1: Electric field from charges + pair data  (timing instrumented)
    import time as _ttime
    import os as _os
    _t_phase = _ttime.time()
    # Get tabulated Ewald functions (cached at module import)
    from ewald_tables import ERFC_TABLE, EXP_NEGSQ_TABLE

    # Route: GPU pair builder (if available + enabled) or CPU JIT path
    use_gpu_pair = (
        hasattr(state, 'gpu_pair_builder')
        and state.gpu_pair_builder is not None
        and _os.environ.get('GOMD_GPU_PAIR', '1') != '0'
    )
    if use_gpu_pair:
        (elfown, listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3_arr
        ) = state.gpu_pair_builder.compute(state)
        # Optional verification mode: compare against CPU
        if _os.environ.get('GOMD_GPU_PAIR_VERIFY', '0') == '1':
            cpu_out = compute_charge_efield_and_pairs(
                state.atoms.x, nat, state.atoms.atomtype,
                state.neighbors.list_neighbors, state.neighbors.listm,
                state.ff.q, ncharge, alpha, state.atoms.box, state.atoms.el,
                ps.c1, rcut2,
                list14, listm14, state.params.lredQ_mu14, state.params.redQmufactor,
                state.params.lredonefour, state.params.redfactor,
                ERFC_TABLE, EXP_NEGSQ_TABLE
            )
            cpu_elfown = cpu_out[0]
            err = float(np.max(np.abs(elfown - cpu_elfown)))
            relmag = float(np.max(np.abs(cpu_elfown)))
            rel = err / max(relmag, 1e-30)
            print(f"  [GPU verify] elfown max abs diff: {err:.2e}  rel: {rel:.2e}")
            if rel > 1e-6:
                print(f"  [GPU verify] WARNING: GPU/CPU diverged at step {state.kount}")
    else:
        (elfown, listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3_arr
        ) = compute_charge_efield_and_pairs(
            state.atoms.x, nat, state.atoms.atomtype,
            state.neighbors.list_neighbors, state.neighbors.listm,
            state.ff.q, ncharge, alpha, state.atoms.box, state.atoms.el,
            ps.c1, rcut2,
            list14, listm14, state.params.lredQ_mu14, state.params.redQmufactor,
            state.params.lredonefour, state.params.redfactor,
            ERFC_TABLE, EXP_NEGSQ_TABLE
        )
    state._t_pair_build = _ttime.time() - _t_phase

    # Phase 1b: Add reciprocal-space electric field from charges
    # (Fortran: reciprocal() zeros elfown then adds k-space field BEFORE interch)
    if state.params.lewald:
        # Prefer GPU > CPU-JIT-fast > CPU-original (each falls back if absent)
        if hasattr(state, 'gpu_ewald_recip') and state.gpu_ewald_recip is not None:
            elfown_recip = state.gpu_ewald_recip.compute(state.atoms.x)
        else:
            try:
                from ewald_fast import compute_recip_efield_fast as compute_recip_efield
            except ImportError:
                from ewald import compute_recip_efield
            elfown_recip = compute_recip_efield(state)
        elfown += elfown_recip

    # Phase 1c: Ewald correction to elfown for excluded pairs
    # (Fortran correct() lines 643-653: adds erf-based field correction)
    _apply_elfown_correction(
        elfown, state.atoms.x, nat, state.atoms.atomtype,
        state.ff.q, ncharge, alpha, state.atoms.box, state.atoms.el,
        state.neighbors.listex, state.neighbors.listmex
    )

    # Phase 1d: Q-mu 1-4 correction to elfown (Fortran correct() lines 776-786)
    if state.params.lredQ_mu14:
        _apply_elfown_14_correction(
            elfown, state.atoms.x, nat, state.atoms.atomtype,
            state.ff.q, ncharge, alpha, state.atoms.box, state.atoms.el,
            list14, listm14, state.params.redQmufactor
        )

    # Phase 2-3: SCF iteration  (timing-instrumented)
    import time as _time
    _t = _time.time()
    px, py, pz, niter = scf_iteration(
        elfown, nat, state.atoms.atomtype, state.ff.pol, ncharge,
        listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3_arr,
        ps.c1, ps.arf, rtaper, ro, ps.tauden, state.params.tolpol,
        ps.px1old, ps.py1old, ps.pz1old,
        ps.px2old, ps.py2old, ps.pz2old,
        ps.px3old, ps.py3old, ps.pz3old,
        state.kount
    )
    state._t_scf_iteration = _time.time() - _t

    # Track SCF iteration count on state for the per-step timer
    state._last_scf_niter = niter
    if state.kount <= 5 or state.kount % 100 == 0:
        mu_rms = np.sqrt(np.mean(px*px + py*py + pz*pz))
        print(f"  Polarization SCF: {niter} iterations, RMS(mu) = {mu_rms:.6e}")

    # Expose elfown for test_05_polarization.py self-consistency check.
    # No effect on production runs beyond a small memory copy.
    ps.elfown_last = elfown.copy()

    # Phase 4: Polarization energy
    epol = compute_pol_energy(elfown, px, py, pz, nat)

    # Phase 5: Induced forces  (timing-instrumented)
    _t = _time.time()
    # (capture _t_pol_forces after the call)
    findown, virind = compute_pol_forces(
        nat, state.atoms.atomtype, state.ff.q, ncharge, alpha,
        px, py, pz,
        listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3_arr,
        ps.c1, rtaper, ro, ps.tauden,
        ERFC_TABLE, EXP_NEGSQ_TABLE
    )
    state._t_pol_forces = _time.time() - _t

    # Phase 6: Q-mu Ewald correction for excluded pairs
    fcorr, vircorr = compute_qmu_correction(
        nat, state.atoms.atomtype, state.ff.q, ncharge, alpha,
        px, py, pz,
        state.neighbors.listex, state.neighbors.listmex,
        state.atoms.x, state.atoms.box, state.atoms.el
    )
    findown += fcorr
    virind += vircorr

    # Phase 7: 1-4 corrections
    if state.params.lredonefour or state.params.lredQ_mu14:
        f14, vir14 = compute_qmu_14_corrections(
            nat, state.atoms.atomtype, state.ff.q, ncharge, alpha,
            px, py, pz,
            list14, listm14, state.atoms.x, state.atoms.box, state.atoms.el,
            state.params.lredonefour, state.params.redfactor,
            state.params.lredQ_mu14, state.params.redQmufactor
        )
        findown += f14
        virind += vir14

    # Scale by 332.08 and add to total forces
    state.atoms.f += COULOMB_CONST * findown

    # Return energy and virial (caller adds to energy components)
    pol_virial = virind * COULOMB_CONST

    # Phase 8: recipQmu (k-space Q-mu)
    # TODO: Full implementation - typically 5-10% of polarization energy
    # For now, skip k-space Q-mu (add later for full accuracy)

    # Save dipole history for next step prediction
    ps.px3old[:] = ps.px2old
    ps.py3old[:] = ps.py2old
    ps.pz3old[:] = ps.pz2old
    ps.px2old[:] = ps.px1old
    ps.py2old[:] = ps.py1old
    ps.pz2old[:] = ps.pz1old
    # Keep at float64 to preserve full precision for the next-step predictor
    ps.px1old[:] = px
    ps.py1old[:] = py
    ps.pz1old[:] = pz

    # Store current dipoles in state
    ps.px[:] = px
    ps.py[:] = py
    ps.pz[:] = pz

    return epol, pol_virial
