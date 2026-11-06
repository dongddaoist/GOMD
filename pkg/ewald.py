"""
Ewald Summation for MD Simulation
Implements reciprocal space, self-energy, and exclusion corrections.
Matches Fortran subroutines: reciprocal(), correct(), and ewself computation.

Key equations (all in kcal/mol with 332.08 Coulomb constant):
  E_recip = (2π/V) * 332.08 * Σ_k [|S(k)|² * 2·exp(-k²/4α²)/k²] - E_self
  E_self  = 332.08 * α/√π * Σ_i q_i²
  E_correct = -Σ_{excluded} 332.08·q_i·q_j·erf(α·r)/r / 2

  CRITICAL: alpha = alphai directly (Fortran line 1297: alpha = alphai)
            NOT 1/alphai as previously coded!
"""

import numpy as np
from numba import jit, prange
from constants import PI, COULOMB_CONST


@jit(nopython=True, cache=True)
def erfc_approx(x):
    """Complementary error function erfc(x) = 1 - erf(x)."""
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


def compute_ewald_self_energy(state):
    """
    Compute Ewald self-energy correction (computed once, constant during NVT).
    
    Fortran (lines 2166-2172):
        ewself = 332.08 * alpha * sum(q_i^2) / sqrt(pi)
    """
    alpha = state.params.alphai  # FIXED: alpha = alphai directly
    nat = state.atoms.nat
    atomtype = state.atoms.atomtype
    q = state.ff.q
    
    sum_q2 = 0.0
    for iat in range(nat):
        itype = atomtype[iat]
        qi = q[itype]
        sum_q2 += qi * qi
    
    ewself = COULOMB_CONST * alpha * sum_q2 / np.sqrt(PI)
    return ewself


@jit(nopython=True, cache=True)
def _reciprocal_space(
    x, nat, atomtype, q, alpha, box,
    klimitx, klimity, klimitz, newpress
):
    """
    Ewald reciprocal-space energy and forces.
    
    Matches Fortran reciprocal() subroutine (lines 8029-8343).
    Uses the recurrence relation for cos/sin(k·r) to avoid
    recomputing trig functions.
    
    Returns: (erecip_raw, forces, virial_tensor)
    where erecip_raw = twopifact * 332.08 * Σ_k [|S(k)|² * akvec]
    (ewself is subtracted outside this function)
    """
    f = np.zeros((3, nat))
    virial_tensor = np.zeros((3, 3))
    
    front = np.zeros(3)
    for kk in range(3):
        front[kk] = 2.0 * PI / box[kk]
    
    vol = box[0] * box[1] * box[2]
    twopifact = 2.0 * PI / vol
    fouralpha2inv = 1.0 / (4.0 * alpha * alpha)
    elfact = 2.0 * twopifact
    
    kxmax2inv = 1.0 / float(klimitx * klimitx)
    kymax2inv = 1.0 / float(klimity * klimity)
    kzmax2inv = 1.0 / float(klimitz * klimitz)
    
    # ===================================================================
    # Pre-compute cos/sin arrays using recurrence relations
    # cos(n*theta) = cos((n-1)*theta)*cos(theta) - sin((n-1)*theta)*sin(theta)
    # sin(n*theta) = sin((n-1)*theta)*cos(theta) + cos((n-1)*theta)*sin(theta)
    # ===================================================================
    cos_x = np.zeros((nat, klimitx + 1))
    sin_x = np.zeros((nat, klimitx + 1))
    cos_y = np.zeros((nat, klimity + 1))
    sin_y = np.zeros((nat, klimity + 1))
    cos_z = np.zeros((nat, klimitz + 1))
    sin_z = np.zeros((nat, klimitz + 1))
    
    for iat in range(nat):
        cos_x[iat, 0] = 1.0
        sin_x[iat, 0] = 0.0
        cos_y[iat, 0] = 1.0
        sin_y[iat, 0] = 0.0
        cos_z[iat, 0] = 1.0
        sin_z[iat, 0] = 0.0
        
        cos_x[iat, 1] = np.cos(x[0, iat] * front[0])
        sin_x[iat, 1] = np.sin(x[0, iat] * front[0])
        cos_y[iat, 1] = np.cos(x[1, iat] * front[1])
        sin_y[iat, 1] = np.sin(x[1, iat] * front[1])
        cos_z[iat, 1] = np.cos(x[2, iat] * front[2])
        sin_z[iat, 1] = np.sin(x[2, iat] * front[2])
    
    for kk in range(2, klimitx + 1):
        for iat in range(nat):
            cos_x[iat, kk] = cos_x[iat, kk-1] * cos_x[iat, 1] - sin_x[iat, kk-1] * sin_x[iat, 1]
            sin_x[iat, kk] = sin_x[iat, kk-1] * cos_x[iat, 1] + cos_x[iat, kk-1] * sin_x[iat, 1]
    for kk in range(2, klimity + 1):
        for iat in range(nat):
            cos_y[iat, kk] = cos_y[iat, kk-1] * cos_y[iat, 1] - sin_y[iat, kk-1] * sin_y[iat, 1]
            sin_y[iat, kk] = sin_y[iat, kk-1] * cos_y[iat, 1] + cos_y[iat, kk-1] * sin_y[iat, 1]
    for kk in range(2, klimitz + 1):
        for iat in range(nat):
            cos_z[iat, kk] = cos_z[iat, kk-1] * cos_z[iat, 1] - sin_z[iat, kk-1] * sin_z[iat, 1]
            sin_z[iat, kk] = sin_z[iat, kk-1] * cos_z[iat, 1] + cos_z[iat, kk-1] * sin_z[iat, 1]
    
    # ===================================================================
    # Pass 1: Compute structure factors S(k) = Σ q_i exp(ik·r_i)
    # Only half of k-space needed due to S(-k) = S*(k) symmetry;
    # factor of 2 in akvec accounts for this.
    # ===================================================================
    # Allocate structure factor arrays for the half-space
    # We'll store (cossum, sinsum) indexed by (kx, ky, kz) in flat lists
    # But for simplicity, compute them on-the-fly in pass 2.
    # Fortran does two passes; we combine them for clarity.
    
    recip = 0.0
    cos_xy = np.zeros(nat)
    sin_xy = np.zeros(nat)
    cos_xyz = np.zeros(nat)
    sin_xyz = np.zeros(nat)
    
    kymin = 0
    kzmin = 1
    
    for kx in range(0, klimitx + 1):
        hx = float(kx) * front[0]
        
        for ky in range(kymin, klimity + 1):
            hy = float(ky) * front[1]
            
            # Compute cos/sin(kx*rx + ky*ry) using addition theorem
            if ky >= 0:
                for iat in range(nat):
                    cos_xy[iat] = cos_x[iat, kx] * cos_y[iat, ky] - sin_x[iat, kx] * sin_y[iat, ky]
                    sin_xy[iat] = sin_x[iat, kx] * cos_y[iat, ky] + cos_x[iat, kx] * sin_y[iat, ky]
            else:
                kyabs = -ky
                for iat in range(nat):
                    cos_xy[iat] = cos_x[iat, kx] * cos_y[iat, kyabs] + sin_x[iat, kx] * sin_y[iat, kyabs]
                    sin_xy[iat] = sin_x[iat, kx] * cos_y[iat, kyabs] - cos_x[iat, kx] * sin_y[iat, kyabs]
            
            for kz in range(kzmin, klimitz + 1):
                hz = float(kz) * front[2]
                
                # Spherical cutoff in k-space
                tmp2 = float(kx*kx) * kxmax2inv + float(ky*ky) * kymax2inv + float(kz*kz) * kzmax2inv
                if tmp2 > 1.0:
                    continue
                
                rkmag2 = hx*hx + hy*hy + hz*hz
                akvec = 2.0 * np.exp(-rkmag2 * fouralpha2inv) / rkmag2
                btens = 2.0 * (1.0 + rkmag2 * fouralpha2inv) / rkmag2
                
                # Compute cos/sin(k·r) = cos/sin(kx*rx + ky*ry + kz*rz)
                if kz >= 0:
                    for iat in range(nat):
                        cos_xyz[iat] = cos_xy[iat] * cos_z[iat, kz] - sin_xy[iat] * sin_z[iat, kz]
                        sin_xyz[iat] = sin_xy[iat] * cos_z[iat, kz] + cos_xy[iat] * sin_z[iat, kz]
                else:
                    kzabs = -kz
                    for iat in range(nat):
                        cos_xyz[iat] = cos_xy[iat] * cos_z[iat, kzabs] + sin_xy[iat] * sin_z[iat, kzabs]
                        sin_xyz[iat] = sin_xy[iat] * cos_z[iat, kzabs] - cos_xy[iat] * sin_z[iat, kzabs]
                
                # Structure factor: S(k) = Σ q_i * exp(ik·r_i)
                cossum = 0.0
                sinsum = 0.0
                for iat in range(nat):
                    itype = atomtype[iat]
                    ci = q[itype]
                    cossum += ci * cos_xyz[iat]
                    sinsum += ci * sin_xyz[iat]
                
                # Energy: |S(k)|² * akvec
                sfh = (cossum * cossum + sinsum * sinsum) * akvec
                recip += sfh
                
                # Forces: F_i = -dE/dr_i
                # Fortran (lines 8295-8304)
                prefact = akvec * elfact
                for iat in range(nat):
                    ci = q[atomtype[iat]]
                    tmp = prefact * (cossum * sin_xyz[iat] - sinsum * cos_xyz[iat])
                    f1 = ci * tmp * COULOMB_CONST
                    f[0, iat] += f1 * hx
                    f[1, iat] += f1 * hy
                    f[2, iat] += f1 * hz
                
                # Virial tensor (Fortran lines 8306-8314)
                if newpress:
                    virial_tensor[0, 0] += sfh * (1.0 - btens * hx * hx)
                    virial_tensor[0, 1] += sfh * (0.0 - btens * hx * hy)
                    virial_tensor[0, 2] += sfh * (0.0 - btens * hx * hz)
                    virial_tensor[1, 0] += sfh * (0.0 - btens * hy * hx)
                    virial_tensor[1, 1] += sfh * (1.0 - btens * hy * hy)
                    virial_tensor[1, 2] += sfh * (0.0 - btens * hy * hz)
                    virial_tensor[2, 0] += sfh * (0.0 - btens * hz * hx)
                    virial_tensor[2, 1] += sfh * (0.0 - btens * hz * hy)
                    virial_tensor[2, 2] += sfh * (1.0 - btens * hz * hz)
            
            kzmin = -klimitz
        kymin = -klimity
    
    # Convert energy: Fortran line 8322
    erecip_raw = recip * twopifact * COULOMB_CONST
    
    # Virial tensor: Fortran line 8326-8329
    for i in range(3):
        for j in range(3):
            virial_tensor[i, j] *= COULOMB_CONST * twopifact
    
    return erecip_raw, f, virial_tensor


@jit(nopython=True, cache=True)
def _ewald_correction(
    x, nat, atomtype, q, alpha, box, el,
    listex, listmex, electrostatic, typee
):
    """
    Ewald exclusion correction for bonded pairs.
    
    Matches Fortran correct() subroutine (lines 584-802).
    Removes the Ewald (erf) contribution for pairs in the exclusion list,
    since these pairs are handled by explicit bonded interactions.
    
    E_correct = -Σ_{excluded} qq * erf(α·r) / r / 2
    F_correct = -Σ_{excluded} qq * [erf(α·r)/r³ - 2α/√π·exp(-α²r²)/r²] / 2 · r_vec
    """
    rootpi = np.sqrt(PI)
    twoalphapi = 2.0 * alpha / rootpi
    unbdcorrect = 0.0
    f_correct = np.zeros((3, nat))
    vir_correct = np.zeros((3, 3))
    
    for iat in range(nat):
        itype = atomtype[iat]
        ci = q[itype]
        neightot = int(listmex[iat])
        
        for k in range(neightot):
            jat = int(listex[k, iat])
            if jat == iat:
                continue  # Skip self-pairs
            jtype = atomtype[jat]
            cj = q[jtype]
            itypee = typee[itype, jtype]
            qq = electrostatic[itypee, 0]  # = 332.08 * q_i * q_j
            
            # Distance with PBC
            xij1 = x[0, jat] - x[0, iat]
            xij2 = x[1, jat] - x[1, iat]
            xij3 = x[2, jat] - x[2, iat]
            if abs(xij1) > el[0]: xij1 -= np.sign(xij1) * box[0]
            if abs(xij2) > el[1]: xij2 -= np.sign(xij2) * box[1]
            if abs(xij3) > el[2]: xij3 -= np.sign(xij3) * box[2]
            
            z = xij1*xij1 + xij2*xij2 + xij3*xij3
            if z < 1.0e-10:
                continue  # Skip degenerate pair (overlapping atoms)
            r = np.sqrt(z)
            alphar = alpha * r
            zinv = 1.0 / z
            rinv = 1.0 / r
            r3inv = zinv * rinv
            
            # erf(alpha*r) = 1 - erfc(alpha*r)
            erf_val = 1.0 - erfc_approx(alphar)
            
            # Energy correction: Fortran line 639
            # unbdcorrect = unbdcorrect - qq*dalphar*rinv/2.0
            # where dalphar = 1 - derfc(alphar) = erf(alphar)
            unbdcorrect -= qq * erf_val * rinv / 2.0
            
            # Force correction: Fortran lines 643-647
            ffcorrect = (erf_val * r3inv - twoalphapi * np.exp(-alphar*alphar) * zinv) / 2.0
            
            f1 = ffcorrect * qq * xij1
            f2 = ffcorrect * qq * xij2
            f3 = ffcorrect * qq * xij3
            
            f_correct[0, iat] += f1
            f_correct[1, iat] += f2
            f_correct[2, iat] += f3
            f_correct[0, jat] -= f1
            f_correct[1, jat] -= f2
            f_correct[2, jat] -= f3
            
            # Virial correction (Fortran lines 669-677)
            vir_correct[0, 0] -= f1 * xij1
            vir_correct[0, 1] -= f1 * xij2
            vir_correct[0, 2] -= f1 * xij3
            vir_correct[1, 0] -= f2 * xij1
            vir_correct[1, 1] -= f2 * xij2
            vir_correct[1, 2] -= f2 * xij3
            vir_correct[2, 0] -= f3 * xij1
            vir_correct[2, 1] -= f3 * xij2
            vir_correct[2, 2] -= f3 * xij3
    
    return unbdcorrect, f_correct, vir_correct


def compute_ewald(state, newpress=True):
    """
    Complete Ewald summation: reciprocal + self-energy + exclusion correction.
    
    Calling sequence matches Fortran (initialize lines 2175-2180):
        CALL reciprocal()   -> reciprocal space energy/forces
        CALL correct()      -> exclusion corrections
    
    The self-energy (ewself) is subtracted inside reciprocal().
    
    Returns:
        erecip: Reciprocal space energy (minus self-energy)
        ecorrect: Exclusion correction energy
        f_ewald: Total Ewald forces (reciprocal + correction)
        vir_ewald: Virial tensor contribution
    """
    nat = state.atoms.nat
    x = state.atoms.x
    atomtype = state.atoms.atomtype
    q = state.ff.q
    alpha = state.params.alphai  # FIXED: alpha = alphai (Fortran line 1297)
    box = state.atoms.box
    el = state.atoms.el
    klimitx = state.params.klimitx
    klimity = state.params.klimity
    klimitz = state.params.klimitz
    
    # Self-energy (constant, could be cached)
    ewself = compute_ewald_self_energy(state)
    
    # Reciprocal space
    erecip_raw, f_recip, vir_recip = _reciprocal_space(
        x, nat, atomtype, q, alpha, box,
        klimitx, klimity, klimitz, newpress
    )
    
    # Subtract self-energy (Fortran line 8322)
    erecip = erecip_raw - ewself
    
    # Exclusion correction
    ecorrect, f_correct, vir_correct = _ewald_correction(
        x, nat, atomtype, q, alpha, box, el,
        state.neighbors.listex, state.neighbors.listmex,
        state.ff.electrostatic, state.ff.typee
    )
    
    # Combine forces and virial
    f_ewald = f_recip + f_correct
    vir_ewald = vir_recip + vir_correct
    
    return erecip, ecorrect, ewself, f_ewald, vir_ewald


def compute_recip_efield(state):
    """
    Compute reciprocal-space electric field at each atom from charges.
    This is the k-space contribution to elfown used in polarization SCF.
    
    Fortran: reciprocal() lines 8294-8300:
      prefact = akvec * elfact
      tmp = prefact * (cossum*sin(k.ri) - sinsum*cos(k.ri))
      elfown(kk,iat) += tmp * h_kk
    
    Returns:
        elfown_recip: (3, nat) reciprocal electric field at each atom
    """
    nat = state.atoms.nat
    x = state.atoms.x
    atomtype = state.atoms.atomtype
    q = state.ff.q
    ncharge = state.ff.ncharge
    alpha = state.params.alphai
    box = state.atoms.box
    klimitx = state.params.klimitx
    klimity = state.params.klimity
    klimitz = state.params.klimitz
    
    front = 2.0 * np.pi / box
    twopifact = 2.0 * np.pi / np.prod(box)
    fouralpha2inv = 1.0 / (4.0 * alpha * alpha)
    elfact = 2.0 * twopifact  # NO 332.08 — matches Fortran line 8093
    
    kxmax2inv = 1.0 / float(klimitx**2)
    kymax2inv = 1.0 / float(klimity**2)
    kzmax2inv = 1.0 / float(klimitz**2)
    
    elfown_recip = np.zeros((3, nat))
    
    # Pre-compute cos/sin for each atom
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
        if klimitx > 0:
            cos_x[iat, 1] = np.cos(front[0] * x[0, iat])
            sin_x[iat, 1] = np.sin(front[0] * x[0, iat])
        if klimity > 0:
            cos_y[iat, 1] = np.cos(front[1] * x[1, iat])
            sin_y[iat, 1] = np.sin(front[1] * x[1, iat])
        if klimitz > 0:
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
    
    # k-vector loop (half-space with factor of 2 in elfact)
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
                
                # Compute cos/sin(k·r) for each atom
                kya = abs(ky); kza = abs(kz)
                cos_xyz = np.zeros(nat)
                sin_xyz = np.zeros(nat)
                for iat in range(nat):
                    cx = cos_x[iat, kx]; sx = sin_x[iat, kx]
                    if ky >= 0:
                        cxy = cx*cos_y[iat, kya] - sx*sin_y[iat, kya]
                        sxy = sx*cos_y[iat, kya] + cx*sin_y[iat, kya]
                    else:
                        cxy = cx*cos_y[iat, kya] + sx*sin_y[iat, kya]
                        sxy = sx*cos_y[iat, kya] - cx*sin_y[iat, kya]
                    if kz >= 0:
                        cos_xyz[iat] = cxy*cos_z[iat, kza] - sxy*sin_z[iat, kza]
                        sin_xyz[iat] = sxy*cos_z[iat, kza] + cxy*sin_z[iat, kza]
                    else:
                        cos_xyz[iat] = cxy*cos_z[iat, kza] + sxy*sin_z[iat, kza]
                        sin_xyz[iat] = sxy*cos_z[iat, kza] - cxy*sin_z[iat, kza]
                
                # Charge structure factor: S(k) = sum qi * exp(ik·ri)
                cossum = 0.0
                sinsum = 0.0
                for iat in range(nat):
                    itype = atomtype[iat]
                    ci = q[itype] if itype < ncharge else 0.0
                    cossum += ci * cos_xyz[iat]
                    sinsum += ci * sin_xyz[iat]
                
                # Electric field: prefact * (cossum*sin(k·ri) - sinsum*cos(k·ri)) * k_vec
                prefact = akvec * elfact
                for iat in range(nat):
                    tmp = prefact * (cossum * sin_xyz[iat] - sinsum * cos_xyz[iat])
                    elfown_recip[0, iat] += tmp * hx
                    elfown_recip[1, iat] += tmp * hy
                    elfown_recip[2, iat] += tmp * hz
    
    return elfown_recip
