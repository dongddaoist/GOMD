"""
Time Integration Methods for MD Simulation
Includes NVE, NVT (Nose-Hoover chains), and NPT integrators
"""

import numpy as np
from numba import jit
from typing import Tuple
from datatypes import SimulationState, ThermostatState, BarostatState
from constants import GASK_KCAL


def init_thermostat(state: SimulationState) -> None:
    """
    Initialize Nose-Hoover chain thermostat parameters.
    """
    Tset = state.Tset
    ndof = state.ndof
    nnos = state.params.nnos
    qfreq = state.params.qfreq
    wfreq = state.params.wfreq
    
    # Temperature factors
    gkt = GASK_KCAL * Tset
    gnkt = ndof * gkt
    gn3kt = (ndof + 3) * gkt
    
    state.thermostat.gkt = gkt
    state.thermostat.gnkt = gnkt
    state.thermostat.gn3kt = gn3kt
    
    # Thermostat masses
    state.thermostat.qtmass[0] = gnkt / (qfreq * qfreq)
    state.barostat.wtmass[0] = gn3kt / (wfreq * wfreq)
    
    state.thermostat.qtmass[1] = gkt / (qfreq * qfreq)
    state.barostat.wtmass[1] = state.barostat.wtmass[0] / 3.0
    
    for i in range(2, 10):
        state.thermostat.qtmass[i] = state.thermostat.qtmass[1]
        state.barostat.wtmass[i] = state.barostat.wtmass[1]


def init_integration(state: SimulationState) -> None:
    """
    Initialize integration parameters (Yoshida-Suzuki weights).
    """
    stepout = state.params.delt * state.params.multibig
    
    nhcstep1 = 5
    nhcstep2 = 5
    
    # Yoshida-Suzuki weights for 5-step integration
    wys = np.zeros(5)
    w0 = 1.0 / (4.0 - 4.0**(1.0/3.0))
    wys[0] = w0
    wys[1] = w0
    wys[2] = -4.0**(1.0/3.0) * w0
    wys[3] = w0
    wys[4] = w0
    
    for i in range(nhcstep2):
        state.thermostat.wdt1[i] = wys[i] * stepout / nhcstep1
        state.thermostat.wdt2[i] = state.thermostat.wdt1[i] / 2.0
        state.thermostat.wdt4[i] = state.thermostat.wdt1[i] / 4.0
        state.thermostat.wdt8[i] = state.thermostat.wdt1[i] / 8.0
    
    # Polynomial coefficients for NPT
    state.e2 = 1.0 / 6.0
    state.e4 = state.e2 / 20.0
    state.e6 = state.e4 / 42.0
    state.e8 = state.e6 / 72.0


@jit(nopython=True, cache=True)
def compute_kinetic(v: np.ndarray, mass: np.ndarray, nat: int, ldummy: np.ndarray) -> float:
    """Compute kinetic energy from velocities."""
    kinen = 0.0
    for iat in range(nat):
        if not ldummy[iat]:
            kinen += mass[iat] * (v[0, iat]**2 + v[1, iat]**2 + v[2, iat]**2)
    return 0.5 * kinen / (4184.0 * 1000.0)  # kcal/mol


@jit(nopython=True, cache=True)
def apply_pbc(x: np.ndarray, box: np.ndarray, nat: int) -> np.ndarray:
    """Apply periodic boundary conditions."""
    for iat in range(nat):
        for k in range(3):
            if x[k, iat] > box[k]:
                x[k, iat] -= box[k]
            elif x[k, iat] < 0.0:
                x[k, iat] += box[k]
    return x


def nhc_integrate(state: SimulationState) -> None:
    """
    Nose-Hoover chain thermostat integration.
    Updates velocities based on thermostat variables.
    
    Includes overflow protection for numerical stability.
    """
    nnos = state.params.nnos
    nhcstep1 = 5
    nhcstep2 = 5
    
    # Maximum exponent argument to prevent overflow
    # exp(700) ≈ 1e304, exp(-700) ≈ 0 (safe range)
    MAX_EXP_ARG = 700.0
    
    # Get kinetic energy
    kinen = compute_kinetic(state.atoms.v, state.atoms.mass, 
                           state.atoms.nat, state.atoms.ldummy)
    state.energy.kinen = kinen
    
    ekin2 = 2.0 * kinen
    scale = 1.0
    
    gnkt = state.thermostat.gnkt
    gkt = state.thermostat.gkt
    qtmass = state.thermostat.qtmass
    xlogs = state.thermostat.xlogs
    vlogs = state.thermostat.vlogs
    glogs = state.thermostat.glogs
    wdt2 = state.thermostat.wdt2
    wdt4 = state.thermostat.wdt4
    wdt8 = state.thermostat.wdt8
    
    # DEBUG: Print thermostat parameters
    
    # Check for thermostat instability BEFORE integration
    vlogs_max = np.max(np.abs(vlogs))
    if vlogs_max > 1e10:
        print(f"WARNING: Thermostat velocity very large (max |vlogs| = {vlogs_max:.2e})")
        print(f"         This may indicate timestep too large or poor initialization.")
        print(f"         Clamping thermostat velocities to prevent overflow.")
        vlogs[:] = np.clip(vlogs, -1e10, 1e10)
    
    glogs[0] = (ekin2 - gnkt) / qtmass[0]
    
    overflow_detected = False
    
    for i in range(nhcstep1):
        for j in range(nhcstep2):
            # Update last thermostat velocity
            vlogs[nnos-1] += glogs[nnos-1] * wdt4[j]
            
            # Chain propagation backward
            for inos in range(nnos-2, -1, -1):
                exp_arg = -wdt8[j] * vlogs[inos+1]
                
                # Clamp exponential argument to prevent overflow
                if abs(exp_arg) > MAX_EXP_ARG:
                    if not overflow_detected and state.kount <= 100:
                        print(f"           exp_arg = {exp_arg:.2e}, vlogs[{inos+1}] = {vlogs[inos+1]:.2e}")
                        overflow_detected = True
                    exp_arg = np.sign(exp_arg) * MAX_EXP_ARG
                
                aa = np.exp(exp_arg)
                vlogs[inos] = vlogs[inos] * aa * aa + wdt4[j] * glogs[inos] * aa
            
            # Scale factor for velocities
            exp_arg = -wdt2[j] * vlogs[0]
            
            # Clamp exponential argument to prevent overflow
            if abs(exp_arg) > MAX_EXP_ARG:
                if not overflow_detected and state.kount <= 100:
                    print(f"           exp_arg = {exp_arg:.2e}, vlogs[0] = {vlogs[0]:.2e}")
                    overflow_detected = True
                exp_arg = np.sign(exp_arg) * MAX_EXP_ARG
            
            aa = np.exp(exp_arg)
            scale *= aa
            
            # Check for NaN or Inf in scale
            if not np.isfinite(scale):
                print(f"         scale = {scale}, aa = {aa}, exp_arg = {exp_arg}")
                scale = 1.0
            
            glogs[0] = (scale * scale * ekin2 - gnkt) / qtmass[0]
            
            # Update thermostat positions
            for inos in range(nnos):
                xlogs[inos] += vlogs[inos] * wdt2[j]
            
            # Chain propagation forward
            for inos in range(nnos-1):
                exp_arg = -wdt8[j] * vlogs[inos+1]
                
                # Clamp exponential argument
                if abs(exp_arg) > MAX_EXP_ARG:
                    exp_arg = np.sign(exp_arg) * MAX_EXP_ARG
                
                aa = np.exp(exp_arg)
                vlogs[inos] = vlogs[inos] * aa * aa + wdt4[j] * glogs[inos] * aa
                glogs[inos+1] = (qtmass[inos] * vlogs[inos]**2 - gkt) / qtmass[inos+1]
            
            vlogs[nnos-1] += glogs[nnos-1] * wdt4[j]
    
    # Check for NaN or Inf in vlogs
    if not np.all(np.isfinite(vlogs)):
        print(f"         vlogs = {vlogs}")
        vlogs[:] = 0.0
    
    # DEBUG: Print scale factor
    
    # Additional check for extreme scale values
    if abs(scale - 1.0) > 10.0:
        print(f"           scale = {scale:.6f}, this indicates thermostat instability")
        # Limit scale to reasonable range
        scale = np.clip(scale, 0.1, 10.0)
    
    # Scale velocities - SKIP DUMMY ATOMS (vectorized)
    nd_mask = ~state.atoms.ldummy
    state.atoms.v[:, nd_mask] *= scale
    
    # Store updated thermostat state
    state.thermostat.xlogs = xlogs
    state.thermostat.vlogs = vlogs
    state.thermostat.glogs = glogs


def npt_integrate(state: SimulationState) -> None:
    """
    NPT integration with Nose-Hoover chains and Parrinello-Rahman barostat.
    """
    nnos = state.params.nnos
    nhcstep1 = 5
    nhcstep2 = 5
    
    # Get kinetic energy
    kinen = compute_kinetic(state.atoms.v, state.atoms.mass,
                           state.atoms.nat, state.atoms.ldummy)
    state.energy.kinen = kinen
    
    ekin2 = 2.0 * kinen
    box3 = state.atoms.box[0] * state.atoms.box[1] * state.atoms.box[2]
    
    gnkt = state.thermostat.gnkt
    gn3kt = state.thermostat.gn3kt
    gkt = state.thermostat.gkt
    qtmass = state.thermostat.qtmass
    wtmass = state.barostat.wtmass
    xlogs = state.thermostat.xlogs.copy()
    vlogs = state.thermostat.vlogs.copy()
    glogs = state.thermostat.glogs.copy()
    vlogv = state.barostat.vlogv.copy()
    glogv = state.barostat.glogv.copy()
    wdt2 = state.thermostat.wdt2
    wdt4 = state.thermostat.wdt4
    wdt8 = state.thermostat.wdt8
    
    pfix = state.params.pfix
    prt = state.pressure.prt.copy()
    akin2 = state.pressure.akin2.copy()
    box_mult = state.params.box_mult
    avg_mult = state.params.avg_mult
    ndof = state.ndof
    onf = 1.0 / ndof
    
    cons = 1.4582454e-5  # Pressure conversion
    
    # Debug: Check for problematic values
    
    scalebar = np.ones(3)
    
    # Pressure calculation
    str_avg = 0.0
    akin2avg = 0.0
    avgterms = 0
    for kk in range(3):
        str_avg += avg_mult[kk] * prt[kk]
        akin2avg += avg_mult[kk] * akin2[kk]
        avgterms += avg_mult[kk]
    
    if avgterms > 0:
        str_avg /= avgterms
        akin2avg /= avgterms
    
    prta = np.zeros(3)
    ak2 = np.zeros(3)
    for kk in range(3):
        prta[kk] = (1 - avg_mult[kk]) * prt[kk] + avg_mult[kk] * str_avg
        ak2[kk] = (1 - avg_mult[kk]) * akin2[kk] + avg_mult[kk] * akin2avg
        
        if not state.params.bUsePressure:
            prta[kk] = (prta[kk] - pfix[kk]) * box3 * cons
        else:
            pke = state.pressure.pke
            prta[kk] = ((state.pressure.pres - pke) - pfix[kk]) * box3 * cons
    
    boxmv2 = wtmass[0] * (vlogv[0]**2 + vlogv[1]**2 + vlogv[2]**2)
    glogs[0] = (ekin2 + boxmv2 - gn3kt) / qtmass[0]
    
    for kk in range(3):
        glogv[kk] = box_mult[kk] * (onf * ekin2 + ak2[kk] + prta[kk]) / wtmass[0]
    
    # Yoshida-Suzuki integration
    for i in range(nhcstep1):
        for j in range(nhcstep2):
            vlogs[nnos-1] += glogs[nnos-1] * wdt4[j]
            
            for inos in range(nnos-2, -1, -1):
                aa = np.exp(-wdt8[j] * vlogs[inos+1])
                vlogs[inos] = vlogs[inos] * aa * aa + wdt4[j] * glogs[inos] * aa
            
            aa = np.exp(-wdt8[j] * vlogs[0])
            for kk in range(3):
                vlogv[kk] = box_mult[kk] * (vlogv[kk] * aa * aa + wdt4[j] * glogv[kk] * aa)
            
            trvlogv = onf * (vlogv[0] + vlogv[1] + vlogv[2])
            
            for kk in range(3):
                aa = np.exp(-wdt2[j] * (vlogs[0] + trvlogv + vlogv[kk]))
                akin2[kk] *= aa * aa
                scalebar[kk] *= aa
            
            ekin2 = akin2[0] + akin2[1] + akin2[2]
            
            for kk in range(3):
                ak2[kk] = (1 - avg_mult[kk]) * akin2[kk] + avg_mult[kk] * akin2avg
            
            for kk in range(3):
                glogv[kk] = box_mult[kk] * (onf * ekin2 + ak2[kk] + prta[kk]) / wtmass[0]
            
            for inos in range(nnos):
                xlogs[inos] += vlogs[inos] * wdt2[j]
            
            aa = np.exp(-wdt8[j] * vlogs[0])
            for kk in range(3):
                vlogv[kk] = box_mult[kk] * (vlogv[kk] * aa * aa + wdt4[j] * glogv[kk] * aa)
            
            boxmv2 = wtmass[0] * (vlogv[0]**2 + vlogv[1]**2 + vlogv[2]**2)
            glogs[0] = (ekin2 + boxmv2 - gn3kt) / qtmass[0]
            
            for inos in range(nnos-1):
                aa = np.exp(-wdt8[j] * vlogs[inos+1])
                vlogs[inos] = vlogs[inos] * aa * aa + wdt4[j] * glogs[inos] * aa
                glogs[inos+1] = (qtmass[inos] * vlogs[inos]**2 - gkt) / qtmass[inos+1]
            
            vlogs[nnos-1] += glogs[nnos-1] * wdt4[j]
    
    # Scale velocities - SKIP DUMMY ATOMS (vectorized)
    nd_mask = ~state.atoms.ldummy
    state.atoms.v[:, nd_mask] *= scalebar[:, None]
    
    # Update states
    state.thermostat.xlogs = xlogs
    state.thermostat.vlogs = vlogs
    state.thermostat.glogs = glogs
    state.barostat.vlogv = vlogv
    state.barostat.glogv = glogv
    state.pressure.akin2 = akin2


def update_dummy_coords(state: SimulationState) -> None:
    """
    Update positions of dummy (lone pair) atoms based on parent atoms.
    
    This is equivalent to DoDummyCoords() in Fortran.
    Dummy atoms have their positions computed from parent atoms (j, i, k)
    using parameters adummy, bdummy, cdummy.
    
    NOTE: This requires jikdummy connectivity data which stores:
    - jikdummy[0, idummy] = jat (parent atom 1)
    - jikdummy[1, idummy] = iat (central parent atom)  
    - jikdummy[2, idummy] = kat (parent atom 2)
    - jikdummy[3, idummy] = idat (dummy atom index)
    - jikdummy[4, idummy] = itype (dummy type)
    """
    ndummy = state.atoms.ndummy
    if ndummy == 0:
        return
    
    # Check if we have the connectivity data
    if state.atoms.jikdummy is None:
        # First time warning
        if not hasattr(state, '_dummy_warning_shown'):
            print("WARNING: update_dummy_coords called but jikdummy not available!")
            print("         Dummy atom positions may be incorrect.")
            state._dummy_warning_shown = True
        return
    
    # DEBUG
    
    box = state.atoms.box
    el = state.atoms.el
    x = state.atoms.x
    
    adummy = state.ff.adummy
    bdummy = state.ff.bdummy
    cdummy = state.ff.cdummy
    LpGeomType = state.ff.LpGeomType
    
    zero = 1.0e-12
    
    for idummy in range(ndummy):
        jat = state.atoms.jikdummy[0, idummy]
        iat = state.atoms.jikdummy[1, idummy]
        kat = state.atoms.jikdummy[2, idummy]
        idat = state.atoms.jikdummy[3, idummy]
        itype = state.atoms.jikdummy[4, idummy]
        
        aa = adummy[itype]
        bb = bdummy[itype]
        cc = cdummy[itype]
        
        # Compute vectors with PBC
        rij = np.zeros(3)
        rjk = np.zeros(3)
        rik = np.zeros(3)
        
        rijabs2 = 0.0
        rjkabs2 = 0.0
        rikabs2 = 0.0
        absrij_a_rjkinv = 0.0
        rijdotrik = 0.0
        rij_a_rjk = np.zeros(3)
        
        for kk in range(3):
            rij[kk] = x[kk, jat] - x[kk, iat]
            if abs(rij[kk]) > el[kk]:
                rij[kk] -= np.sign(rij[kk]) * box[kk]
            rijabs2 += rij[kk] * rij[kk]
            
            rjk[kk] = x[kk, kat] - x[kk, jat]
            if abs(rjk[kk]) > el[kk]:
                rjk[kk] -= np.sign(rjk[kk]) * box[kk]
            rjkabs2 += rjk[kk] * rjk[kk]
            
            rij_a_rjk[kk] = rij[kk] + aa * rjk[kk]
            absrij_a_rjkinv += rij_a_rjk[kk] * rij_a_rjk[kk]
            
            rik[kk] = x[kk, kat] - x[kk, iat]
            if abs(rik[kk]) > el[kk]:
                rik[kk] -= np.sign(rik[kk]) * box[kk]
            rikabs2 += rik[kk] * rik[kk]
            rijdotrik += rij[kk] * rik[kk]
        
        rjkabs = np.sqrt(rjkabs2)
        crossabs = np.sqrt(max(0.0, rijabs2 * rikabs2 - rijdotrik**2))
        if abs(crossabs) < zero:
            crossabs = zero
        if absrij_a_rjkinv < 1.0e-20:
            continue  # Skip degenerate dummy geometry
        absrij_a_rjkinv = 1.0 / np.sqrt(absrij_a_rjkinv)
        
        factor = -bb * absrij_a_rjkinv
        rid = factor * rij_a_rjk
        
        if LpGeomType[itype] <= 2:
            # Cross product [rij x rik]
            cross = np.zeros(3)
            cross[0] = rij[1] * rik[2] - rik[1] * rij[2]
            cross[1] = -rij[0] * rik[2] + rik[0] * rij[2]
            cross[2] = rij[0] * rik[1] - rik[0] * rij[1]
            
            for kk in range(3):
                rid[kk] += cc * cross[kk] / crossabs
                x[kk, idat] = x[kk, iat] + rid[kk]
                if x[kk, idat] > box[kk]:
                    x[kk, idat] -= box[kk]
                if x[kk, idat] < 0:
                    x[kk, idat] += box[kk]
        
        elif LpGeomType[itype] == 3:
            for kk in range(3):
                rid[kk] += cc * rjk[kk] / rjkabs
                x[kk, idat] = x[kk, iat] + rid[kk]
                if x[kk, idat] > box[kk]:
                    x[kk, idat] -= box[kk]
                if x[kk, idat] < 0:
                    x[kk, idat] += box[kk]


def velocity_verlet_step(state: SimulationState) -> None:
    """
    Perform position update for Velocity Verlet integrator.
    
    This function ONLY updates positions. The velocity kicks happen
    in integrate_step() before and after this call.
    
    Position update:
    - NVE/NVT: x = x + v*dt
    - NPT: x = x*scale + v*dt*scale (with box scaling)
    
    Then applies PBC and updates dummy atom positions.
    
    NOTE: Velocity kicks are NOT done here - they're in integrate_step()!
    """
    delt = state.params.delt
    nat = state.atoms.nat
    # Save reference positions for SHAKE position constraints (Fortran: CALL shake(xold,x1,x1))
    x_ref = state.atoms.x.copy()

    ldummy = state.atoms.ldummy
    
    # =========================================================================
    # POSITION UPDATE - SKIP DUMMY ATOMS
    # =========================================================================
    if state.params.nvt or state.params.nve:
        # Simple position update: x = x + v*dt (vectorized)
        nd_mask = ~ldummy
        delta = state.atoms.v[:, nd_mask] * (delt * 1.0e-5)
        state.atoms.x[:, nd_mask] += delta
        max_disp = float(np.abs(delta).max()) if delta.size else 0.0
        
    
    # NPT position update with box scaling - SKIP DUMMY ATOMS
    if state.params.npt:
        vlogv = state.barostat.vlogv
        e2 = state.e2
        e4 = state.e4
        e6 = state.e6
        e8 = state.e8
        
        for kk in range(3):
            aa = np.exp(delt * 0.5 * vlogv[kk])
            aa2 = aa * aa
            arg2 = (vlogv[kk] * delt * 0.5) ** 2
            poly = ((e8 * arg2 + e6) * arg2 + e4) * arg2 + e2 * arg2 + 1.0
            bb = aa * poly * delt
            
            nd_mask = ~ldummy
            state.atoms.x[kk, nd_mask] = (state.atoms.x[kk, nd_mask] * aa2
                                          + state.atoms.v[kk, nd_mask] * bb * 1.0e-5)
            
            # Update barostat variables
            state.barostat.xlogv[kk] += vlogv[kk] * delt
            scale = np.exp(vlogv[kk] * delt)
            state.atoms.box[kk] *= scale
    
    # =========================================================================
    # APPLY PERIODIC BOUNDARY CONDITIONS - REAL ATOMS ONLY
    # =========================================================================
    # Vectorized PBC wrap
    nd_mask = ~ldummy
    box = state.atoms.box
    for kk in range(3):
        col = state.atoms.x[kk, nd_mask]
        col = np.where(col > box[kk], col - box[kk], col)
        col = np.where(col < 0.0,    col + box[kk], col)
        state.atoms.x[kk, nd_mask] = col

    # Update half-box lengths
    state.atoms.el = state.atoms.box * 0.5
    

    # =========================================================================
    # APPLY SHAKE TO THE UPDATED POSITIONS (missing in prior Python version)
    # Fortran reference: after x1 update -> CALL shake(xold, x1, x1)
    # =========================================================================
    if state.params.constrain and state.constraints.nbcon > 0:
        from utils import apply_shake
        x_constrained, converged = apply_shake(state, state.atoms.x, x_ref, state.constraints.max_iter)
        state.atoms.x = x_constrained
        # Re-wrap real atoms (vectorized)
        nd_mask = ~ldummy
        box_v = state.atoms.box
        for kk in range(3):
            col = state.atoms.x[kk, nd_mask]
            col = np.where(col > box_v[kk], col - box_v[kk], col)
            col = np.where(col < 0.0,       col + box_v[kk], col)
            state.atoms.x[kk, nd_mask] = col
    # =========================================================================
    # UPDATE DUMMY ATOM POSITIONS
    # =========================================================================
    update_dummy_coords(state)


def apply_shake_predictor_corrector(state: SimulationState) -> None:
    """
    Apply SHAKE predictor-corrector and calculate COMBINED forces.
    
    CORRECTED VERSION - Matches Fortran lines 2547-2578 EXACTLY
    
    Fortran algorithm:
    1. Calculate predictor: x2 = x + 1e-5*dt*v + 1e-5*dt^2*f/m
    2. Apply SHAKE: CALL shake(x, x2, x2)
    3. Calculate COMBINED forces:
           diff = x2 - x  (with PBC!)
           f = (diff - 1e-5*dt*v) * m / (41.84e-5*dt^2)
    
    KEY POINTS:
    - Use 1.0e-5 factor (NOT 0.5!)
    - This is NOT standard velocity Verlet
    - Formula gives COMBINED forces (physical + constraint)
    - MUST apply PBC to diff
    - This REPLACES forces, doesn't add
    
    Args:
        state: Simulation state
    """
    if not state.params.constrain or state.constraints.nbcon == 0:
        return
    
    delt = state.params.delt  # Inner timestep in fs
    nat = state.atoms.nat
    mass = state.atoms.mass
    massinv = state.atoms.massinv
    ldummy = state.atoms.ldummy
    v = state.atoms.v
    f = state.atoms.f
    x = state.atoms.x
    box = state.atoms.box
    el = state.atoms.el
    
    # =========================================================================
    # STEP 1: Calculate predictor positions (Fortran lines 2550-2559)
    # =========================================================================
    # Formula: x2 = x + 1.0e-5*dt*v + 1.0e-5*dt^2*f*rmass
    # The 1e-5 factor is deliberate - makes predictor a small perturbation
    
    # Vectorized SHAKE predictor
    x2_pred = x.copy()  # default: dummies stay at their current position
    nd_mask = ~ldummy
    rmass_v = 41.84 * massinv[nd_mask]  # (nreal,)
    f_real = f[:, nd_mask]                # (3, nreal)
    v_real = v[:, nd_mask]                # (3, nreal)
    x_real = x[:, nd_mask]                # (3, nreal)
    pred_real = (x_real
                 + 1.0e-5 * delt * v_real
                 + 1.0e-5 * delt * delt * f_real * rmass_v)
    # PBC
    for kk in range(3):
        col = pred_real[kk]
        col = np.where(col > box[kk], col - box[kk], col)
        col = np.where(col < 0.0,    col + box[kk], col)
        pred_real[kk] = col
    x2_pred[:, nd_mask] = pred_real
    
    # =========================================================================
    # STEP 2: Apply SHAKE to predictor (Fortran line 2564)
    # =========================================================================
    from utils import apply_shake
    
    x2_constrained, converged = apply_shake(
        state, 
        x2_pred,  # Predictor positions (to be constrained)
        x,        # Current positions (reference)
        state.constraints.max_iter
    )
    
    # =========================================================================
    # STEP 3: Calculate COMBINED forces (Fortran lines 2569-2577)
    # =========================================================================
    # This is the CRITICAL formula that gives physical + constraint forces!
    # 
    # Formula: f = (diff - 1.0e-5*dt*v) * m / (41.84e-5*dt^2)
    # where diff = x2_constrained - x_current (with PBC!)
    #
    # This automatically combines:
    # - Physical forces (from interactions)
    # - Constraint forces (from SHAKE corrections)
    
    # Vectorized SHAKE combined-force update
    nd_mask2 = ~ldummy
    diff = x2_constrained[:, nd_mask2] - x[:, nd_mask2]   # (3, nreal)
    # Minimum image
    for kk in range(3):
        col = diff[kk]
        col = np.where(np.abs(col) > el[kk], col - np.sign(col) * box[kk], col)
        diff[kk] = col

    v_real2 = v[:, nd_mask2]
    mass_real = mass[nd_mask2]
    denominator = 41.84e-5 * delt * delt
    numerator = (diff - 1.0e-5 * delt * v_real2) * mass_real[None, :]
    state.atoms.f[:, nd_mask2] = numerator / denominator
    
    # =========================================================================
    # DEBUG OUTPUT
    # =========================================================================
    if state.kount <= 10:
        f_max = np.max(np.abs(state.atoms.f))
        f_rms = np.sqrt(np.mean(state.atoms.f**2))


def integrate_step(state: SimulationState) -> None:
    """
    Perform one complete integration step.
    
    DISPATCHER: chooses between vanilla velocity-Verlet (K=1) and
    r-RESPA multi-timestep (K=multibig>1).
    
    With multibig=1 (default), this is identical to the original code path.
    With multibig>1, calls respa_integrate_step which uses force splitting:
        - fast forces (bonded + NB_splined): every inner step
        - slow forces (Ewald reciprocal + polarization): every K-th inner step,
          impulse-delivered with magnitude K
    
    See respa_integrate_step for r-RESPA implementation details.
    """
    K = state.params.multibig
    if K > 1:
        respa_integrate_step(state, K)
    else:
        _vanilla_vv_step(state)


def _vanilla_vv_step(state: SimulationState) -> None:
    """Original velocity-Verlet integrator (bit-equivalent to pre-RESPA code).
    
    Used when multibig=1. Preserved exactly to avoid regressions.
    """
    delt = state.params.delt
    nat = state.atoms.nat
    ldummy = state.atoms.ldummy
    massinv = state.atoms.massinv
    
    # =========================================================================
    # STEP 1: Thermostat/Barostat (before velocity Verlet)
    # =========================================================================
    if state.params.nvt:
        nhc_integrate(state)
    elif state.params.npt:
        npt_integrate(state)
    else:  # NVE
        state.energy.kinen = compute_kinetic(
            state.atoms.v, state.atoms.mass, nat, ldummy
        )
    
    # Increment step counter
    state.kount += 1
    
    # =========================================================================
    # STEP 2: First velocity kick (v = v + 0.5*dt*f/m)
    # =========================================================================
    
    # Vectorized first velocity kick (was 50k Python iter, now ~0.5ms)
    not_dummy_mask = ~ldummy
    rmass_vec = 41.84 * massinv  # (nat,)
    factor = 0.5 * delt * rmass_vec[not_dummy_mask]   # (n_real,)
    state.atoms.v[:, not_dummy_mask] += factor * state.atoms.f[:, not_dummy_mask]


    # =========================================================================
    # STEP 3: Position update (x = x + v*dt, PBC, dummy coords)
    # =========================================================================
    velocity_verlet_step(state)
    
    # =========================================================================
    # STEP 4: Calculate forces at new positions  (PER-STEP TIMING POINT)
    # =========================================================================
    import time as _time
    _t_force = _time.time()
    from forces import compute_all_forces
    state.energy = compute_all_forces(state)
    state._t_forces_last = _time.time() - _t_force

    # =========================================================================
    # STEP 5: Redistribute dummy atom forces to parent atoms
    # =========================================================================
    do_dummy_forces(state)
    
    # =========================================================================
    # STEP 6: SHAKE predictor-corrector (if constraints enabled)
    # =========================================================================
    # This REPLACES state.atoms.f with combined forces (physical + constraint)!
    # After this, state.atoms.f contains everything needed for velocity update.
    if state.params.constrain and state.constraints.nbcon > 0:
        apply_shake_predictor_corrector(state)
    
    # =========================================================================
    # STEP 7: Second velocity kick (v = v + 0.5*dt*f/m)
    # =========================================================================
    # Vectorized second velocity kick
    not_dummy_mask2 = ~ldummy
    rmass_vec2 = 41.84 * massinv
    factor2 = 0.5 * delt * rmass_vec2[not_dummy_mask2]
    state.atoms.v[:, not_dummy_mask2] += factor2 * state.atoms.f[:, not_dummy_mask2]
    
    
    # =========================================================================
    # STEP 8: Thermostat/Barostat (after velocity Verlet)
    # =========================================================================
    if state.params.nvt:
        nhc_integrate(state)
    elif state.params.npt:
        npt_integrate(state)
    else:  # NVE
        state.energy.kinen = compute_kinetic(
            state.atoms.v, state.atoms.mass, nat, ldummy
        )



def respa_integrate_step(state, K: int) -> None:
    """
    r-RESPA multi-timestep integrator (2-level: fast + slow forces).
    
    Mirrors Fortran MD_serial.f90 lines 2336-2588 with multimed=1, multibig=K.
    
    Force splitting:
        - fast = bonded + nonbonded splined real-space (computed every inner step)
        - slow = Ewald reciprocal + polarization SCF + pol_forces (every K-th step)
    
    The slow forces are accumulated over K timesteps and impulse-delivered as
    a single multiplied kick at the outer step boundary. This is mathematically
    equivalent to applying them every step in the limit K -> 1.
    
    Algorithm:
        Pre-loop: thermostat/barostat (if NVT/NPT)
        For istep = 1..K:
            v += 0.5*dt*f_now / m
            x += dt*v ; PBC ; dummies
            zero f
            compute_fast_forces()       (bonded + NB_splined, into state.atoms.f)
            f_ref = state.atoms.f.copy()  (= bonded + NB)
            f_now = f_ref.copy()        (default: full force is fast)
            
            IF istep < K (inner step):
                # f_now stays as f_ref (= fast forces only)
                # SHAKE
                # v += 0.5*dt*f_now / m
            
            IF istep == K (outer step):
                f_short = f_ref.copy()  (= bonded + NB)
                compute_slow_forces(state, ...)   (adds Ewald + pol to state.atoms.f)
                f_full = state.atoms.f.copy()  (= bonded + NB + slow)
                
                # Impulse-delivered slow kick:
                f_now = f_short + K * (f_full - f_short)
                       = f_short + K * (slow forces)
                
                # SHAKE with the new f_now
                # v += 0.5*dt*f_now / m
        
        Post-loop: thermostat/barostat (if NVT/NPT)
    
    State requirements at entry:
        state.atoms.f must contain f_now from previous outer step
        (= bonded + NB + K*slow). On first call (kount=0), this is the
        full force from setup_system, which IS the slow forces too at K=1
        equivalent. Caveat: for K>1, f at kount=0 should ideally be just
        f_short + K*slow instead of f_full = f_short + slow. For first-step
        accuracy this doesn't matter (transient), but document it.
    """
    import time as _time
    delt = state.params.delt
    nat = state.atoms.nat
    ldummy = state.atoms.ldummy
    massinv = state.atoms.massinv
    not_dummy_mask = ~ldummy
    rmass_vec = 41.84 * massinv  # (nat,)
    
    # Pre-loop thermostat/barostat
    if state.params.nvt:
        nhc_integrate(state)
    elif state.params.npt:
        npt_integrate(state)
    else:  # NVE
        state.energy.kinen = compute_kinetic(
            state.atoms.v, state.atoms.mass, nat, ldummy
        )
    
    state.kount += 1
    
    # f_now starts as whatever's in state.atoms.f from previous outer step
    # (or from setup_system on first call). This is bonded + NB + K*slow.
    
    for istep in range(1, K + 1):
        is_outer = (istep == K)
        
        # First velocity kick using f_now
        factor = 0.5 * delt * rmass_vec[not_dummy_mask]
        state.atoms.v[:, not_dummy_mask] += factor * state.atoms.f[:, not_dummy_mask]
        
        # Drift (position update, PBC, dummy coords)
        velocity_verlet_step(state)
        
        # Compute fast forces (bonded + NB_splined). Zeros state.atoms.f first.
        from forces import compute_fast_forces, compute_slow_forces
        _t_force = _time.time()
        fast_energy = compute_fast_forces(state)
        f_ref = state.atoms.f.copy()  # save fast forces
        
        if is_outer:
            # Outer step: also compute slow forces and impulse-deliver
            f_short = f_ref.copy()
            energy = compute_slow_forces(state, fast_energy)
            # state.atoms.f now = bonded + NB + slow
            f_full = state.atoms.f.copy()
            
            # f_now = f_short + K * (f_full - f_short) = f_short + K * (slow)
            state.atoms.f = f_short + K * (f_full - f_short)
            state.energy = energy
        else:
            # Inner step: f_now = fast forces only
            state.atoms.f = f_ref
            state.energy = fast_energy
        
        state._t_forces_last = _time.time() - _t_force
        
        # Redistribute dummy atom forces to parent atoms
        do_dummy_forces(state)
        
        # SHAKE predictor-corrector
        if state.params.constrain and state.constraints.nbcon > 0:
            apply_shake_predictor_corrector(state)
        
        # Second velocity kick using current f (after SHAKE replacement if applicable)
        factor2 = 0.5 * delt * rmass_vec[not_dummy_mask]
        state.atoms.v[:, not_dummy_mask] += factor2 * state.atoms.f[:, not_dummy_mask]
    
    # Post-loop thermostat/barostat
    if state.params.nvt:
        nhc_integrate(state)
    elif state.params.npt:
        npt_integrate(state)
    else:  # NVE
        state.energy.kinen = compute_kinetic(
            state.atoms.v, state.atoms.mass, nat, ldummy
        )


def remove_com_velocity(state: SimulationState) -> None:
    """Remove center-of-mass velocity.
    
    Fortran formula: v_new = v_old - total_momentum / (natreal * mass[iat])
    This removes total momentum equally from all atoms (each loses P/N momentum).
    """
    # Vectorized COM removal
    nd_mask = ~state.atoms.ldummy
    natreal = state.atoms.natreal
    mass_real = state.atoms.mass[nd_mask]   # (nreal,)
    v_real = state.atoms.v[:, nd_mask]      # (3, nreal)

    # Total momentum = sum_i m_i * v_i
    vcum = (mass_real * v_real).sum(axis=1)   # (3,)

    # v_i -= vcum / (natreal * m_i) for each real atom
    inv_natmass = 1.0 / (natreal * mass_real)  # (nreal,)
    state.atoms.v[:, nd_mask] -= vcum[:, None] * inv_natmass[None, :]


def calculate_temperature(state: SimulationState) -> float:
    """Calculate instantaneous temperature.
    
    T = 2 * KE / (ndof * R)
    where KE is in kcal/mol and R = GASK_KCAL = 1.987e-3 kcal/(mol·K)
    """
    kinen = compute_kinetic(state.atoms.v, state.atoms.mass,
                           state.atoms.nat, state.atoms.ldummy)
    
    # KE already in kcal/mol, use simple formula
    temp = 2.0 * kinen / (state.ndof * GASK_KCAL)
    
    return temp



@jit(nopython=True, cache=True)
def _do_dummy_forces_inner(
    f, x, ndummy, jikdummy,
    adummy, bdummy, cdummy, LpGeomType,
    box, el
):
    """JIT'd inner loop of do_dummy_forces.
    
    Handles LpGeomType <= 2 (the common case, all type-2 dummies).
    Type-3 dummies are handled in a separate Python fallback in do_dummy_forces.
    
    Modifies f in place (parent atoms get force, dummy zeroed).
    Returns (n_skipped_type3, virdummy, tvirdummy).
    
    Performance: ~80 ms -> ~3 ms (JIT'd), ~25x speedup on the hot loop.
    """
    zero = 1.0e-12
    virdummy = 0.0
    tvirdummy = np.zeros((3, 3))
    n_skipped_type3 = 0
    
    for idummy in range(ndummy):
        jat = jikdummy[0, idummy]
        iat = jikdummy[1, idummy]
        kat = jikdummy[2, idummy]
        idat = jikdummy[3, idummy]
        itype = jikdummy[4, idummy]
        
        # Type-3 dummies handled outside JIT (rare)
        if LpGeomType[itype] > 2:
            n_skipped_type3 += 1
            continue
        
        # Get force on dummy atom (3 scalars, no array allocation)
        Fd0 = f[0, idat]; Fd1 = f[1, idat]; Fd2 = f[2, idat]
        
        aa = adummy[itype]
        bb = bdummy[itype]
        cc = cdummy[itype]
        
        # Vectors with PBC (manual unroll, no np.zeros)
        rij0 = x[0, jat] - x[0, iat]
        rij1 = x[1, jat] - x[1, iat]
        rij2 = x[2, jat] - x[2, iat]
        if abs(rij0) > el[0]:
            if rij0 > 0.0: rij0 -= box[0]
            else:          rij0 += box[0]
        if abs(rij1) > el[1]:
            if rij1 > 0.0: rij1 -= box[1]
            else:          rij1 += box[1]
        if abs(rij2) > el[2]:
            if rij2 > 0.0: rij2 -= box[2]
            else:          rij2 += box[2]
        rijabs2 = rij0*rij0 + rij1*rij1 + rij2*rij2
        
        rjk0 = x[0, kat] - x[0, jat]
        rjk1 = x[1, kat] - x[1, jat]
        rjk2 = x[2, kat] - x[2, jat]
        if abs(rjk0) > el[0]:
            if rjk0 > 0.0: rjk0 -= box[0]
            else:          rjk0 += box[0]
        if abs(rjk1) > el[1]:
            if rjk1 > 0.0: rjk1 -= box[1]
            else:          rjk1 += box[1]
        if abs(rjk2) > el[2]:
            if rjk2 > 0.0: rjk2 -= box[2]
            else:          rjk2 += box[2]
        rjkabs2 = rjk0*rjk0 + rjk1*rjk1 + rjk2*rjk2
        
        rik0 = x[0, kat] - x[0, iat]
        rik1 = x[1, kat] - x[1, iat]
        rik2 = x[2, kat] - x[2, iat]
        if abs(rik0) > el[0]:
            if rik0 > 0.0: rik0 -= box[0]
            else:          rik0 += box[0]
        if abs(rik1) > el[1]:
            if rik1 > 0.0: rik1 -= box[1]
            else:          rik1 += box[1]
        if abs(rik2) > el[2]:
            if rik2 > 0.0: rik2 -= box[2]
            else:          rik2 += box[2]
        rikabs2 = rik0*rik0 + rik1*rik1 + rik2*rik2
        rijdotrik = rij0*rik0 + rij1*rik1 + rij2*rik2
        
        rij_a_rjk0 = rij0 + aa * rjk0
        rij_a_rjk1 = rij1 + aa * rjk1
        rij_a_rjk2 = rij2 + aa * rjk2
        absrij_a_rjkinv2 = (rij_a_rjk0*rij_a_rjk0 +
                           rij_a_rjk1*rij_a_rjk1 +
                           rij_a_rjk2*rij_a_rjk2)
        
        rjkabs = np.sqrt(rjkabs2)
        if absrij_a_rjkinv2 < 1.0e-20:
            continue
        absrij_a_rjkinv = 1.0 / np.sqrt(absrij_a_rjkinv2)
        
        factor = -bb * absrij_a_rjkinv
        
        # rid = factor * rij_a_rjk
        rid0 = factor * rij_a_rjk0
        rid1 = factor * rij_a_rjk1
        rid2 = factor * rij_a_rjk2
        
        # Fd dot rid
        Fddotrid = Fd0*rid0 + Fd1*rid1 + Fd2*rid2
        
        # f1 term
        if abs(bb) > zero:
            inv_bb2 = 1.0 / (bb * bb)
            f1_0 = Fddotrid * rid0 * inv_bb2
            f1_1 = Fddotrid * rid1 * inv_bb2
            f1_2 = Fddotrid * rid2 * inv_bb2
        else:
            f1_0 = 0.0; f1_1 = 0.0; f1_2 = 0.0
        
        # Cross product [rij x rik]
        cross0 = rij1 * rik2 - rik1 * rij2
        cross1 = -rij0 * rik2 + rik0 * rij2
        cross2 = rij0 * rik1 - rik0 * rij1
        
        crossdotFd = cross0*Fd0 + cross1*Fd1 + cross2*Fd2
        
        crossabs2 = rijabs2 * rikabs2 - rijdotrik*rijdotrik
        if crossabs2 < 0.0: crossabs2 = 0.0
        crossabs = np.sqrt(crossabs2)
        if crossabs < 1.0e-10:
            continue
        
        tmp1_inv = 1.0 / crossabs
        
        fiout0 = (Fd1 * (rik2 - rij2) + Fd2 * (rij1 - rik1)) * tmp1_inv
        fiout1 = (Fd0 * (rij2 - rik2) + Fd2 * (rik0 - rij0)) * tmp1_inv
        fiout2 = (Fd0 * (rik1 - rij1) + Fd1 * (rij0 - rik0)) * tmp1_inv
        
        fjout0 = (Fd1 * (-rik2) + Fd2 * rik1) * tmp1_inv
        fjout1 = (Fd2 * (-rik0) + Fd0 * rik2) * tmp1_inv
        fjout2 = (Fd0 * (-rik1) + Fd1 * rik0) * tmp1_inv
        
        fkout0 = (Fd1 * rij2 - Fd2 * rij1) * tmp1_inv
        fkout1 = (Fd2 * rij0 - Fd0 * rij2) * tmp1_inv
        fkout2 = (Fd0 * rij1 - Fd1 * rij0) * tmp1_inv
        
        tmp3 = tmp1_inv * tmp1_inv * tmp1_inv
        # Add the second-derivative term to fiout/fjout/fkout
        cdf_t3 = crossdotFd * tmp3
        # Component 0
        fiout0 += cdf_t3 * (rikabs2 * rij0 + rijabs2 * rik0 - rijdotrik * (rik0 + rij0))
        fjout0 -= cdf_t3 * (rikabs2 * rij0 - rijdotrik * rik0)
        fkout0 -= cdf_t3 * (rijabs2 * rik0 - rijdotrik * rij0)
        # Component 1
        fiout1 += cdf_t3 * (rikabs2 * rij1 + rijabs2 * rik1 - rijdotrik * (rik1 + rij1))
        fjout1 -= cdf_t3 * (rikabs2 * rij1 - rijdotrik * rik1)
        fkout1 -= cdf_t3 * (rijabs2 * rik1 - rijdotrik * rij1)
        # Component 2
        fiout2 += cdf_t3 * (rikabs2 * rij2 + rijabs2 * rik2 - rijdotrik * (rik2 + rij2))
        fjout2 -= cdf_t3 * (rikabs2 * rij2 - rijdotrik * rik2)
        fkout2 -= cdf_t3 * (rijabs2 * rik2 - rijdotrik * rij2)
        
        # Update rid with cross product term
        rid0 += cc * cross0 * tmp1_inv
        rid1 += cc * cross1 * tmp1_inv
        rid2 += cc * cross2 * tmp1_inv
        
        # Vectors from dummy to other atoms (for virial)
        rkd0 = x[0, idat] - x[0, kat]
        rkd1 = x[1, idat] - x[1, kat]
        rkd2 = x[2, idat] - x[2, kat]
        if abs(rkd0) > el[0]:
            if rkd0 > 0.0: rkd0 -= box[0]
            else:          rkd0 += box[0]
        if abs(rkd1) > el[1]:
            if rkd1 > 0.0: rkd1 -= box[1]
            else:          rkd1 += box[1]
        if abs(rkd2) > el[2]:
            if rkd2 > 0.0: rkd2 -= box[2]
            else:          rkd2 += box[2]
        
        rjd0 = x[0, idat] - x[0, jat]
        rjd1 = x[1, idat] - x[1, jat]
        rjd2 = x[2, idat] - x[2, jat]
        if abs(rjd0) > el[0]:
            if rjd0 > 0.0: rjd0 -= box[0]
            else:          rjd0 += box[0]
        if abs(rjd1) > el[1]:
            if rjd1 > 0.0: rjd1 -= box[1]
            else:          rjd1 += box[1]
        if abs(rjd2) > el[2]:
            if rjd2 > 0.0: rjd2 -= box[2]
            else:          rjd2 += box[2]
        
        # Calculate forces on atoms jat, iat, kat due to lone pair (component-wise)
        # Component 0
        ftmp0 = factor * (Fd0 - f1_0)
        tmp1_0 = Fd0 - ftmp0 + fiout0 * cc
        tmp2_0 = (1.0 - aa) * ftmp0 + fjout0 * cc
        tmp3_0 = aa * ftmp0 + fkout0 * cc
        f[0, iat] += tmp1_0
        f[0, jat] += tmp2_0
        f[0, kat] += tmp3_0
        # Component 1
        ftmp1 = factor * (Fd1 - f1_1)
        tmp1_1 = Fd1 - ftmp1 + fiout1 * cc
        tmp2_1 = (1.0 - aa) * ftmp1 + fjout1 * cc
        tmp3_1 = aa * ftmp1 + fkout1 * cc
        f[1, iat] += tmp1_1
        f[1, jat] += tmp2_1
        f[1, kat] += tmp3_1
        # Component 2
        ftmp2 = factor * (Fd2 - f1_2)
        tmp1_2 = Fd2 - ftmp2 + fiout2 * cc
        tmp2_2 = (1.0 - aa) * ftmp2 + fjout2 * cc
        tmp3_2 = aa * ftmp2 + fkout2 * cc
        f[2, iat] += tmp1_2
        f[2, jat] += tmp2_2
        f[2, kat] += tmp3_2
        
        # Virial contribution (scalar)
        virdummy -= (rid0 * tmp1_0 + rjd0 * tmp2_0 + rkd0 * tmp3_0)
        virdummy -= (rid1 * tmp1_1 + rjd1 * tmp2_1 + rkd1 * tmp3_1)
        virdummy -= (rid2 * tmp1_2 + rjd2 * tmp2_2 + rkd2 * tmp3_2)
        
        # Virial tensor (3x3)
        # tvirdummy[kk, jj] -= rid[jj] * tmp1[kk] + rjd[jj] * tmp2[kk] + rkd[jj] * tmp3[kk]
        # So for each kk we accumulate over jj
        # Component (0, *)
        tvirdummy[0, 0] -= rid0 * tmp1_0 + rjd0 * tmp2_0 + rkd0 * tmp3_0
        tvirdummy[0, 1] -= rid1 * tmp1_0 + rjd1 * tmp2_0 + rkd1 * tmp3_0
        tvirdummy[0, 2] -= rid2 * tmp1_0 + rjd2 * tmp2_0 + rkd2 * tmp3_0
        # Component (1, *)
        tvirdummy[1, 0] -= rid0 * tmp1_1 + rjd0 * tmp2_1 + rkd0 * tmp3_1
        tvirdummy[1, 1] -= rid1 * tmp1_1 + rjd1 * tmp2_1 + rkd1 * tmp3_1
        tvirdummy[1, 2] -= rid2 * tmp1_1 + rjd2 * tmp2_1 + rkd2 * tmp3_1
        # Component (2, *)
        tvirdummy[2, 0] -= rid0 * tmp1_2 + rjd0 * tmp2_2 + rkd0 * tmp3_2
        tvirdummy[2, 1] -= rid1 * tmp1_2 + rjd1 * tmp2_2 + rkd1 * tmp3_2
        tvirdummy[2, 2] -= rid2 * tmp1_2 + rjd2 * tmp2_2 + rkd2 * tmp3_2
        
        # Zero the dummy atom force
        f[0, idat] = 0.0
        f[1, idat] = 0.0
        f[2, idat] = 0.0
    
    return n_skipped_type3, virdummy, tvirdummy


def do_dummy_forces(state: SimulationState) -> None:
    """
    Redistribute forces from dummy (lone pair) atoms to their parent atoms.
    
    Uses JIT'd inner loop (_do_dummy_forces_inner) for type-2 dummies
    (the common case, ~640 of them in this system). Type-3 dummies fall
    back to the Python path below.
    
    Performance: ~80 ms -> ~3 ms (JIT'd type-2 path).
    """
    ndummy = state.atoms.ndummy
    if ndummy == 0:
        return
    
    if state.atoms.jikdummy is None:
        return
    
    box = state.atoms.box
    el = state.atoms.el
    x = state.atoms.x
    f = state.atoms.f
    
    adummy = state.ff.adummy
    bdummy = state.ff.bdummy
    cdummy = state.ff.cdummy
    LpGeomType = state.ff.LpGeomType
    
    # Try JIT'd path for type-2 (common case)
    n_skipped_type3, _virdummy, _tvirdummy = _do_dummy_forces_inner(
        f, x, ndummy, state.atoms.jikdummy,
        adummy, bdummy, cdummy, LpGeomType, box, el
    )
    
    # If no type-3 dummies, we're done — the JIT'd path handled everything
    if n_skipped_type3 == 0:
        return
    
    # Fall back to Python for type-3 dummies (rare; system has 0 of them)
    # (This block is the original Python code, but now only runs for type-3.)
    zero = 1.0e-12
    virdummy = 0.0
    tvirdummy = np.zeros((3, 3))
    
    for idummy in range(ndummy):
        jat = state.atoms.jikdummy[0, idummy]
        iat = state.atoms.jikdummy[1, idummy]  # Central parent atom
        kat = state.atoms.jikdummy[2, idummy]
        idat = state.atoms.jikdummy[3, idummy]  # Dummy atom index
        itype = state.atoms.jikdummy[4, idummy]
        
        # Skip type-2 dummies (handled by JIT helper above)
        if LpGeomType[itype] <= 2:
            continue
        
        # Get force on dummy atom
        Fd = np.array([f[0, idat], f[1, idat], f[2, idat]])
        
        aa = adummy[itype]
        bb = bdummy[itype]
        cc = cdummy[itype]
        
        # Compute vectors with PBC
        rij = np.zeros(3)
        rjk = np.zeros(3)
        rik = np.zeros(3)
        
        rijabs2 = 0.0
        rjkabs2 = 0.0
        rikabs2 = 0.0
        absrij_a_rjkinv = 0.0
        rijdotrik = 0.0
        rij_a_rjk = np.zeros(3)
        
        for kk in range(3):
            rij[kk] = x[kk, jat] - x[kk, iat]
            if abs(rij[kk]) > el[kk]:
                rij[kk] -= np.sign(rij[kk]) * box[kk]
            rijabs2 += rij[kk] * rij[kk]
            
            rjk[kk] = x[kk, kat] - x[kk, jat]
            if abs(rjk[kk]) > el[kk]:
                rjk[kk] -= np.sign(rjk[kk]) * box[kk]
            rjkabs2 += rjk[kk] * rjk[kk]
            
            rij_a_rjk[kk] = rij[kk] + aa * rjk[kk]
            absrij_a_rjkinv += rij_a_rjk[kk] * rij_a_rjk[kk]
            
            rik[kk] = x[kk, kat] - x[kk, iat]
            if abs(rik[kk]) > el[kk]:
                rik[kk] -= np.sign(rik[kk]) * box[kk]
            rikabs2 += rik[kk] * rik[kk]
            rijdotrik += rij[kk] * rik[kk]
        
        rjkabs = np.sqrt(rjkabs2)
        if absrij_a_rjkinv < 1.0e-20:
            continue  # Skip degenerate dummy geometry
        absrij_a_rjkinv = 1.0 / np.sqrt(absrij_a_rjkinv)
        
        factor = -bb * absrij_a_rjkinv
        
        # Compute rid (displacement from iat to dummy)
        rid = factor * rij_a_rjk
        
        # Compute Fd dot rid
        Fddotrid = np.dot(Fd, rid)
        
        # f1 term
        if abs(bb) > zero:
            f1 = Fddotrid * rid / (bb * bb)
        else:
            f1 = np.zeros(3)
        
        if LpGeomType[itype] <= 2:
            # Cross product [rij x rik]
            cross = np.zeros(3)
            cross[0] = rij[1] * rik[2] - rik[1] * rij[2]
            cross[1] = -rij[0] * rik[2] + rik[0] * rij[2]
            cross[2] = rij[0] * rik[1] - rik[0] * rij[1]
            
            crossdotFd = np.dot(cross, Fd)
            
            # Gradient w.r.t iat, jat, kat
            crossabs = np.sqrt(max(0.0, rijabs2 * rikabs2 - rijdotrik**2))
            if crossabs < 1.0e-10:
                continue  # Skip degenerate geometry
            
            tmp1_inv = 1.0 / crossabs
            
            fiout = np.zeros(3)
            fjout = np.zeros(3)
            fkout = np.zeros(3)
            
            fiout[0] = (Fd[1] * (rik[2] - rij[2]) + Fd[2] * (rij[1] - rik[1])) * tmp1_inv
            fiout[1] = (Fd[0] * (rij[2] - rik[2]) + Fd[2] * (rik[0] - rij[0])) * tmp1_inv
            fiout[2] = (Fd[0] * (rik[1] - rij[1]) + Fd[1] * (rij[0] - rik[0])) * tmp1_inv
            
            fjout[0] = (Fd[1] * (-rik[2]) + Fd[2] * rik[1]) * tmp1_inv
            fjout[1] = (Fd[2] * (-rik[0]) + Fd[0] * rik[2]) * tmp1_inv
            fjout[2] = (Fd[0] * (-rik[1]) + Fd[1] * rik[0]) * tmp1_inv
            
            fkout[0] = (Fd[1] * rij[2] - Fd[2] * rij[1]) * tmp1_inv
            fkout[1] = (Fd[2] * rij[0] - Fd[0] * rij[2]) * tmp1_inv
            fkout[2] = (Fd[0] * rij[1] - Fd[1] * rij[0]) * tmp1_inv
            
            tmp3 = tmp1_inv ** 3
            for kk in range(3):
                fiout[kk] += crossdotFd * tmp3 * (rikabs2 * rij[kk] + 
                             rijabs2 * rik[kk] - rijdotrik * (rik[kk] + rij[kk]))
                fjout[kk] -= crossdotFd * tmp3 * (rikabs2 * rij[kk] - rijdotrik * rik[kk])
                fkout[kk] -= crossdotFd * tmp3 * (rijabs2 * rik[kk] - rijdotrik * rij[kk])
            
            # Update rid with cross product term
            for kk in range(3):
                rid[kk] += cc * cross[kk] * tmp1_inv
            
            # Compute vectors from dummy to other atoms (for virial)
            rkd = np.zeros(3)
            rjd = np.zeros(3)
            for kk in range(3):
                rkd[kk] = x[kk, idat] - x[kk, kat]
                if abs(rkd[kk]) > el[kk]:
                    rkd[kk] -= np.sign(rkd[kk]) * box[kk]
                rjd[kk] = x[kk, idat] - x[kk, jat]
                if abs(rjd[kk]) > el[kk]:
                    rjd[kk] -= np.sign(rjd[kk]) * box[kk]
            
            # Calculate forces on atoms jat, iat, kat due to lone pair
            for kk in range(3):
                ftmp = factor * (Fd[kk] - f1[kk])
                tmp1 = Fd[kk] - ftmp + fiout[kk] * cc
                tmp2 = (1.0 - aa) * ftmp + fjout[kk] * cc
                tmp3 = aa * ftmp + fkout[kk] * cc
                
                f[kk, iat] += tmp1
                f[kk, jat] += tmp2
                f[kk, kat] += tmp3
                
                # Virial contribution
                virdummy -= rid[kk] * tmp1 + rjd[kk] * tmp2 + rkd[kk] * tmp3
                for jj in range(3):
                    tvirdummy[kk, jj] -= (rid[jj] * tmp1 + rjd[jj] * tmp2 + rkd[jj] * tmp3)
        
        elif LpGeomType[itype] == 3:
            # Carbonyl-like lone pairs
            for kk in range(3):
                rid[kk] += cc * rjk[kk] / rjkabs
            
            # Compute vectors from dummy to other atoms
            rkd = np.zeros(3)
            rjd = np.zeros(3)
            for kk in range(3):
                rkd[kk] = x[kk, idat] - x[kk, kat]
                if abs(rkd[kk]) > el[kk]:
                    rkd[kk] -= np.sign(rkd[kk]) * box[kk]
                rjd[kk] = x[kk, idat] - x[kk, jat]
                if abs(rjd[kk]) > el[kk]:
                    rjd[kk] -= np.sign(rjd[kk]) * box[kk]
            
            Fddotrjk = np.dot(Fd, rjk)
            
            for kk in range(3):
                ftmp = factor * (Fd[kk] - f1[kk])
                tmp1 = Fd[kk] - ftmp
                tmp2 = (1.0 - aa) * ftmp - cc * (Fd[kk] - Fddotrjk * rjk[kk] / rjkabs2) / rjkabs
                tmp3 = aa * ftmp + cc * (Fd[kk] - Fddotrjk * rjk[kk] / rjkabs2) / rjkabs
                
                f[kk, iat] += tmp1
                f[kk, jat] += tmp2
                f[kk, kat] += tmp3
                
                # Virial contribution
                virdummy -= rid[kk] * tmp1 + rjd[kk] * tmp2 + rkd[kk] * tmp3
                for jj in range(3):
                    tvirdummy[kk, jj] -= (rid[jj] * tmp1 + rjd[jj] * tmp2 + rkd[jj] * tmp3)
        
        # Zero the force on dummy atom after distribution
        f[0, idat] = 0.0
        f[1, idat] = 0.0
        f[2, idat] = 0.0
    
    # DEBUG
    
    # Store virial contributions (could add to state.pressure.tvirpo if needed)
    # For now, just ensure forces are redistributed

