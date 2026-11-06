"""
Force Calculations for MD Simulation
Optimized with NumPy vectorization and Numba JIT compilation
"""

import numpy as np
from numba import jit, prange
from typing import Tuple
from datatypes import SimulationState, EnergyComponents
from constants import PI, COULOMB_CONST

from nonbonded_splined import compute_nonbonded_forces_with_splines
from ewald import compute_ewald

@jit(nopython=True, cache=True)
def minimum_image(dx: np.ndarray, box: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Apply minimum image convention."""
    for i in range(3):
        if abs(dx[i]) > el[i]:
            dx[i] = dx[i] - np.sign(dx[i]) * box[i]
    return dx


@jit(nopython=True, cache=True)
def minimum_image_batch(dx: np.ndarray, box: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Apply minimum image convention to batch of vectors."""
    for i in range(3):
        mask = np.abs(dx[i, :]) > el[i]
        dx[i, mask] = dx[i, mask] - np.sign(dx[i, mask]) * box[i]
    return dx


@jit(nopython=True, cache=True)
def compute_bond_forces(
    x: np.ndarray,
    bonds: np.ndarray,
    nbonds: int,
    stretch: np.ndarray,
    box: np.ndarray,
    el: np.ndarray,
    f: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute bond stretching forces.
    
    Bond potential: E = k * (r - r0)^2 + k3 * (r - r0)^3
    """
    ebond = 0.0
    
    for ibond in range(nbonds):
        iat = bonds[0, ibond]
        jat = bonds[1, ibond]
        itype = bonds[2, ibond]
        
        # Bond parameters
        k = stretch[itype, 0]
        r0 = stretch[itype, 1]
        # Note: stretch[itype, 2] is constrained r0 for SHAKE, NOT k3
        
        # Distance vector
        dx = np.zeros(3)
        dx[0] = x[0, jat] - x[0, iat]
        dx[1] = x[1, jat] - x[1, iat]
        dx[2] = x[2, jat] - x[2, iat]
        
        # Minimum image
        for i in range(3):
            if abs(dx[i]) > el[i]:
                dx[i] = dx[i] - np.sign(dx[i]) * box[i]
        
        r2 = dx[0]**2 + dx[1]**2 + dx[2]**2
        r = np.sqrt(r2)
        if r < 1.0e-10:
            continue  # Skip degenerate bond (atoms at same position)
        rinv = 1.0 / r
        
        dr = r - r0
        
        # Energy: E = 0.5 * k * (r - r0)^2  (Fortran line 923)
        ebond += 0.5 * k * dr * dr
        
        # Force magnitude: f1 = k * (r - r0) / r  (Fortran line 926)
        fmag = k * dr * rinv
        
        # Apply forces
        for i in range(3):
            fval = fmag * dx[i]
            f[i, iat] += fval
            f[i, jat] -= fval
    
    return ebond, f


@jit(nopython=True, cache=True)
def compute_bend_forces(
    x: np.ndarray,
    bends: np.ndarray,
    nbends: int,
    bend_params: np.ndarray,
    linear: np.ndarray,
    box: np.ndarray,
    el: np.ndarray,
    f: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute angle bending forces.
    
    Bend potential: E = k * (theta - theta0)^2
    """
    ebend = 0.0
    
    for ibend in range(nbends):
        iat = bends[0, ibend]
        jat = bends[1, ibend]
        kat = bends[2, ibend]
        itype = bends[3, ibend]
        
        # Bend parameters
        k = bend_params[itype, 0]
        theta0 = bend_params[itype, 1] * PI / 180.0  # Convert to radians
        
        # Vector j->i
        rji = np.zeros(3)
        rji[0] = x[0, iat] - x[0, jat]
        rji[1] = x[1, iat] - x[1, jat]
        rji[2] = x[2, iat] - x[2, jat]
        for i in range(3):
            if abs(rji[i]) > el[i]:
                rji[i] = rji[i] - np.sign(rji[i]) * box[i]
        
        # Vector j->k
        rjk = np.zeros(3)
        rjk[0] = x[0, kat] - x[0, jat]
        rjk[1] = x[1, kat] - x[1, jat]
        rjk[2] = x[2, kat] - x[2, jat]
        for i in range(3):
            if abs(rjk[i]) > el[i]:
                rjk[i] = rjk[i] - np.sign(rjk[i]) * box[i]
        
        rji2 = rji[0]**2 + rji[1]**2 + rji[2]**2
        rjk2 = rjk[0]**2 + rjk[1]**2 + rjk[2]**2
        if rji2 < 1.0e-20 or rjk2 < 1.0e-20:
            continue  # Skip degenerate angle
        rjiinv = 1.0 / np.sqrt(rji2)
        rjkinv = 1.0 / np.sqrt(rjk2)
        
        # Cosine of angle
        dot = rji[0]*rjk[0] + rji[1]*rjk[1] + rji[2]*rjk[2]
        costheta = dot * rjiinv * rjkinv
        
        # Clamp to valid range
        if costheta > 1.0:
            costheta = 1.0
        elif costheta < -1.0:
            costheta = -1.0
        
        theta = np.arccos(costheta)
        sintheta = np.sin(theta)
        
        if abs(sintheta) < 1.0e-10:
            continue
        
        dtheta = theta - theta0
        
        # Energy: E = 0.5 * k * (theta - theta0)^2  (Fortran en3cen: bendk2=k*0.5)
        ebend += 0.5 * k * dtheta * dtheta
        
        # Force: dE/dtheta = k * dtheta, chain rule through arccos gives 1/sin(theta)
        fmag = k * dtheta / sintheta
        
        # Derivatives of costheta with respect to atom positions
        # F_i = -dE/dx_i = +k*(theta-theta0)/sin(theta) * d(costheta)/dx_i
        # The chain rule through arccos gives the -1/sin factor, and
        # the overall negative from F=-grad(E) cancels it, yielding +fmag.
        for i in range(3):
            dcos_di = rjk[i] * rjiinv * rjkinv - costheta * rji[i] * rjiinv * rjiinv
            dcos_dk = rji[i] * rjiinv * rjkinv - costheta * rjk[i] * rjkinv * rjkinv
            
            f[i, iat] += fmag * dcos_di
            f[i, kat] += fmag * dcos_dk
            f[i, jat] -= fmag * (dcos_di + dcos_dk)
    
    return ebend, f


@jit(nopython=True, cache=True)
def compute_torsion_forces(
    x: np.ndarray,
    torts: np.ndarray,
    ntorts: int,
    twist: np.ndarray,
    nprms: np.ndarray,
    box: np.ndarray,
    el: np.ndarray,
    f: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute torsional forces.
    
    Torsion potential: E = sum_n V_n * [1 + cos(n*phi - gamma_n)]
    """
    etort = 0.0
    
    for itort in range(ntorts):
        iat = torts[0, itort]
        jat = torts[1, itort]
        kat = torts[2, itort]
        lat = torts[3, itort]
        itype = torts[4, itort]
        
        # Vectors
        rij = np.zeros(3)
        rjk = np.zeros(3)
        rkl = np.zeros(3)
        
        for i in range(3):
            rij[i] = x[i, iat] - x[i, jat]
            rjk[i] = x[i, jat] - x[i, kat]
            rkl[i] = x[i, kat] - x[i, lat]
            
            if abs(rij[i]) > el[i]:
                rij[i] = rij[i] - np.sign(rij[i]) * box[i]
            if abs(rjk[i]) > el[i]:
                rjk[i] = rjk[i] - np.sign(rjk[i]) * box[i]
            if abs(rkl[i]) > el[i]:
                rkl[i] = rkl[i] - np.sign(rkl[i]) * box[i]
        
        # Cross products for dihedral
        # m = rij x rjk
        m = np.zeros(3)
        m[0] = rij[1]*rjk[2] - rij[2]*rjk[1]
        m[1] = rij[2]*rjk[0] - rij[0]*rjk[2]
        m[2] = rij[0]*rjk[1] - rij[1]*rjk[0]
        
        # n = rjk x rkl
        n = np.zeros(3)
        n[0] = rjk[1]*rkl[2] - rjk[2]*rkl[1]
        n[1] = rjk[2]*rkl[0] - rjk[0]*rkl[2]
        n[2] = rjk[0]*rkl[1] - rjk[1]*rkl[0]
        
        mm = m[0]**2 + m[1]**2 + m[2]**2
        nn = n[0]**2 + n[1]**2 + n[2]**2
        rjk2 = rjk[0]**2 + rjk[1]**2 + rjk[2]**2
        
        if mm < 1.0e-10 or nn < 1.0e-10 or rjk2 < 1.0e-20:
            continue
        
        rjk_mag = np.sqrt(rjk2)
        mm_inv = 1.0 / mm
        nn_inv = 1.0 / nn
        
        # Cosine and sine of dihedral
        cosphi = (m[0]*n[0] + m[1]*n[1] + m[2]*n[2]) / np.sqrt(mm * nn)
        
        if cosphi > 1.0:
            cosphi = 1.0
        elif cosphi < -1.0:
            cosphi = -1.0
        
        # Sign from triple product
        sign = rij[0]*n[0] + rij[1]*n[1] + rij[2]*n[2]
        sinphi = np.sqrt(1.0 - cosphi*cosphi)
        if sign < 0:
            sinphi = -sinphi
        
        phi = np.arctan2(sinphi, cosphi)
        
        # Energy and force from torsion terms
        # Fortran: nfolds = nprms(itype) - 1; tortk(n) = twist(n,itype)*0.5
        # Fortran: entort = tk0 - sum(tortk(n)*cos(n*phi))
        # Fortran: f1 = sum(n*tortk(n)*sin(n*phi)/sin(phi))
        nterms = nprms[itype]
        dedphi = 0.0
        
        for nterm in range(nterms):
            vn = twist[nterm, itype]
            if nterm == 0:
                # Constant term (k0)
                etort += vn
            else:
                # E = -k_n/2 * cos(n*phi)  (Fortran line 1171)
                etort -= 0.5 * vn * np.cos(nterm * phi)
                # dE/dphi = k_n/2 * n * sin(n*phi)  (Fortran line 1173)
                dedphi += 0.5 * vn * nterm * np.sin(nterm * phi)
        
        if abs(dedphi) < 1.0e-12:
            continue
        
        # Force derivatives
        rijdotrjk = rij[0]*rjk[0] + rij[1]*rjk[1] + rij[2]*rjk[2]
        rjkdotrkl = rjk[0]*rkl[0] + rjk[1]*rkl[1] + rjk[2]*rkl[2]
        
        for i in range(3):
            dphi_di = -rjk_mag * mm_inv * m[i]
            dphi_dl = rjk_mag * nn_inv * n[i]
            dphi_dj = (rijdotrjk / rjk2 - 1.0) * dphi_di - (rjkdotrkl / rjk2) * dphi_dl
            dphi_dk = (rjkdotrkl / rjk2 - 1.0) * dphi_dl - (rijdotrjk / rjk2) * dphi_di
            
            f[i, iat] -= dedphi * dphi_di
            f[i, jat] -= dedphi * dphi_dj
            f[i, kat] -= dedphi * dphi_dk
            f[i, lat] -= dedphi * dphi_dl
    
    return etort, f


@jit(nopython=True, cache=True)
def compute_improper_forces(
    x: np.ndarray,
    deforms: np.ndarray,
    ndeforms: int,
    deform_k: np.ndarray,
    box: np.ndarray,
    el: np.ndarray,
    f: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute out-of-plane (improper torsion) forces.
    
    Improper potential: E = k * chi^2
    where chi is the out-of-plane angle.
    """
    eopbs = 0.0
    
    for idef in range(ndeforms):
        iat = deforms[0, idef]
        jat = deforms[1, idef]
        kat = deforms[2, idef]
        lat = deforms[3, idef]
        itype = deforms[4, idef]
        
        k = deform_k[itype]
        if abs(k) < 1.0e-10:
            continue
        
        # Similar to torsion calculation
        rij = np.zeros(3)
        rjk = np.zeros(3)
        rjl = np.zeros(3)
        
        for i in range(3):
            rij[i] = x[i, iat] - x[i, jat]
            rjk[i] = x[i, kat] - x[i, jat]
            rjl[i] = x[i, lat] - x[i, jat]
            
            if abs(rij[i]) > el[i]:
                rij[i] = rij[i] - np.sign(rij[i]) * box[i]
            if abs(rjk[i]) > el[i]:
                rjk[i] = rjk[i] - np.sign(rjk[i]) * box[i]
            if abs(rjl[i]) > el[i]:
                rjl[i] = rjl[i] - np.sign(rjl[i]) * box[i]
        
        # Normal to jkl plane
        n = np.zeros(3)
        n[0] = rjk[1]*rjl[2] - rjk[2]*rjl[1]
        n[1] = rjk[2]*rjl[0] - rjk[0]*rjl[2]
        n[2] = rjk[0]*rjl[1] - rjk[1]*rjl[0]
        
        nn = n[0]**2 + n[1]**2 + n[2]**2
        if nn < 1.0e-10:
            continue
        
        n_inv = 1.0 / np.sqrt(nn)
        
        # Distance of i from plane
        rij_mag = np.sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2)
        if rij_mag < 1.0e-10:
            continue
        
        dot = rij[0]*n[0] + rij[1]*n[1] + rij[2]*n[2]
        sinchi = dot * n_inv / rij_mag
        
        if sinchi > 1.0:
            sinchi = 1.0
        elif sinchi < -1.0:
            sinchi = -1.0
        
        chi = np.arcsin(sinchi)
        coschi = np.cos(chi)
        
        if abs(coschi) < 1.0e-10:
            continue
        
        # Energy
        eopbs += k * chi * chi
        
        # Force (simplified)
        dedchi = 2.0 * k * chi / (coschi * rij_mag)
        
        for i in range(3):
            fval = dedchi * n[i] * n_inv
            f[i, iat] -= fval
            f[i, jat] += fval / 3.0
            f[i, kat] += fval / 3.0
            f[i, lat] += fval / 3.0
    
    return eopbs, f


@jit(nopython=True, cache=True)
def compute_kinetic_energy(v: np.ndarray, mass: np.ndarray, nat: int) -> float:
    """Compute kinetic energy."""
    kinen = 0.0
    for iat in range(nat):
        kinen += mass[iat] * (v[0, iat]**2 + v[1, iat]**2 + v[2, iat]**2)
    return 0.5 * kinen / (4184.0 * 1000.0)  # Convert to kcal/mol

"""
ADD THIS TO forces.py - Safety Check Function and Updated compute_all_forces
============================================================================

Add this function BEFORE compute_all_forces():
"""

def check_for_force_anomalies(state: SimulationState) -> bool:
    """
    Check for anomalously large forces or energies that indicate a problem.

    Returns:
        True if anomalies detected, False otherwise.
    """
    anomaly_detected = False

    # Check potential energy magnitude
    if abs(state.energy.poten) > 1e9:  # More than 1 billion kcal/mol
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR: Potential energy is astronomically large!")
        print(f"{'='*80}")
        print(f"  PE = {state.energy.poten:.6e} kcal/mol")
        print(f"  Components:")
        print(f"    ebond = {state.energy.ebond:.6e}")
        print(f"    ebend = {state.energy.ebend:.6e}")
        print(f"    etort = {state.energy.etort:.6e}")
        print(f"    eopbs = {state.energy.eopbs:.6e}")
        print(f"    unbd  = {state.energy.unbd:.6e}")
        anomaly_detected = True

        # Find which component is exploding
        components = [
            ('bond', state.energy.ebond),
            ('bend', state.energy.ebend),
            ('torsion', state.energy.etort),
            ('improper', state.energy.eopbs),
            ('nonbonded', state.energy.unbd)
        ]
        max_component = max(components, key=lambda x: abs(x[1]))
        print(f"\n  Largest component: {max_component[0]} = {max_component[1]:.6e}")

    # Check force magnitudes
    f_max = np.max(np.abs(state.atoms.f))
    f_rms = np.sqrt(np.mean(state.atoms.f**2))

    if f_max > 1e6:  # Forces > 1 million kcal/mol/Ang
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR: Extremely large forces detected!")
        print(f"{'='*80}")
        print(f"  Max |F| = {f_max:.6e} kcal/mol/Ang")
        print(f"  RMS |F| = {f_rms:.6e} kcal/mol/Ang")
        anomaly_detected = True

        # Find atoms with huge forces
        print(f"\n  Atoms with |F| > 1e6:")
        count = 0
        for iat in range(state.atoms.nat):
            if state.atoms.ldummy[iat]:
                continue
            f_mag = np.sqrt(np.sum(state.atoms.f[:, iat]**2))
            if f_mag > 1e6:
                count += 1
                if count <= 10:  # Only show first 10
                    print(f"    Atom {iat}: |F| = {f_mag:.6e} kcal/mol/Ang")

                    # Check its close neighbors
                    if state.neighbors.list_neighbors is not None:
                        print(f"      Close neighbors:")
                        for k in range(min(5, state.neighbors.listm[iat])):
                            jat = state.neighbors.list_neighbors[k, iat]
                            dx = state.atoms.x[:, jat] - state.atoms.x[:, iat]
                            # Apply PBC
                            for kk in range(3):
                                if abs(dx[kk]) > state.atoms.el[kk]:
                                    dx[kk] -= np.sign(dx[kk]) * state.atoms.box[kk]
                            r = np.sqrt(np.sum(dx**2))
                            marker = " ⚠" if r < 1.0 else ""
                            print(f"        Atom {jat}: r = {r:.4f} Ang{marker}")

        if count > 10:
            print(f"    ... and {count - 10} more atoms with huge forces")

    # Check for NaN or Inf in forces
    if not np.all(np.isfinite(state.atoms.f)):
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR: NaN or Inf detected in forces!")
        print(f"{'='*80}")
        print(f"  NaN count: {np.sum(np.isnan(state.atoms.f))}")
        print(f"  Inf count: {np.sum(np.isinf(state.atoms.f))}")
        anomaly_detected = True

    return anomaly_detected


"""
REPLACE the compute_all_forces function with this version:
"""

def compute_fast_forces(state: SimulationState) -> EnergyComponents:
    """
    Compute FAST forces only: bonded + nonbonded splined real-space.
    
    These are the forces that run every inner step in r-RESPA.
    Cost: ~190 ms/step (NB_splined dominates).
    
    Modifies state.atoms.f IN-PLACE: zeros it, then writes fast forces.
    Returns: EnergyComponents with ebond, ebend, etort, eopbs, evdw set.
    
    NOTE: For RESPA, the caller is responsible for combining fast and slow
    forces appropriately. compute_all_forces() does this internally.
    """
    nat = state.atoms.nat
    state.atoms.f = np.zeros((3, nat))
    energy = EnergyComponents()
    
    import time as _time
    _t0 = _time.time()
    
    # Bonded
    energy.ebond, state.atoms.f = compute_bond_forces(
        state.atoms.x, state.connectivity.bonds, state.connectivity.nbonds,
        state.ff.stretch, state.atoms.box, state.atoms.el, state.atoms.f)
    energy.ebend, state.atoms.f = compute_bend_forces(
        state.atoms.x, state.connectivity.bends, state.connectivity.nbends,
        state.ff.bend, state.ff.linear, state.atoms.box, state.atoms.el, state.atoms.f)
    energy.etort, state.atoms.f = compute_torsion_forces(
        state.atoms.x, state.connectivity.torts, state.connectivity.ntorts,
        state.ff.twist, state.ff.nprms, state.atoms.box, state.atoms.el, state.atoms.f)
    energy.eopbs, state.atoms.f = compute_improper_forces(
        state.atoms.x, state.connectivity.deforms, state.connectivity.ndeforms,
        state.ff.deform, state.atoms.box, state.atoms.el, state.atoms.f)
    state._t_bonded = _time.time() - _t0
    
    # Nonbonded splined (includes erfc real-space term)
    _t0 = _time.time()
    if state.neighbors.list_neighbors is not None:
        unbd, unbde, virial_nb, state.atoms.f = compute_nonbonded_forces_with_splines(state)
        energy.evdw = unbd - unbde
        energy.eelec = unbde
        if hasattr(energy, 'unbd'):
            energy.unbd = unbd
        if hasattr(energy, 'unbde'):
            energy.unbde = unbde
        energy.vir = virial_nb
    else:
        energy.evdw = 0.0
        energy.eelec = 0.0
        if hasattr(energy, 'unbd'):
            energy.unbd = 0.0
        if hasattr(energy, 'unbde'):
            energy.unbde = 0.0
        energy.vir = 0.0
    state._t_nonbonded = _time.time() - _t0
    
    energy.poten = energy.ebond + energy.ebend + energy.etort + energy.eopbs + (
        energy.unbd if hasattr(energy, 'unbd') else energy.evdw + energy.eelec)
    
    return energy


def compute_slow_forces(state: SimulationState, fast_energy: EnergyComponents) -> EnergyComponents:
    """
    Compute SLOW forces only: Ewald reciprocal + polarization.
    
    These run every outer step in r-RESPA (every K inner steps).
    Cost: ~280 ms (recip_efield + ewald_recip + SCF + polF).
    
    The fast forces (already in state.atoms.f from compute_fast_forces) are
    PRESERVED — slow forces are added on top.
    
    Returns: updated EnergyComponents with erecip, ecorrect, epol added,
    plus updated unbd/poten totals.
    """
    import time as _time
    nat = state.atoms.nat
    energy = fast_energy  # mutate in place
    
    # Ewald reciprocal (forces accumulate ON TOP of fast forces in state.atoms.f)
    _t0 = _time.time()
    if state.neighbors.list_neighbors is not None and state.params.lewald:
        newpress = True
        erecip, ecorrect, ewself, f_ewald, vir_ewald = compute_ewald(state, newpress)
        state.atoms.f += f_ewald
        if hasattr(energy, 'unbd'):
            energy.unbd += erecip + ecorrect
        if hasattr(energy, 'unbde'):
            energy.unbde += erecip + ecorrect
        else:
            energy.eelec += erecip + ecorrect
        energy.vir += erecip
        for ii in range(3):
            for jj in range(3):
                state.pressure.tvirpo[ii, jj] += vir_ewald[ii, jj]
    state._t_ewald_recip = _time.time() - _t0
    
    # Polarization (also adds to state.atoms.f)
    if state.ff.lpolarizable:
        from polarization import compute_polarization, init_polarization
        if not hasattr(state, 'pol_state') or state.pol_state is None:
            init_polarization(state)
        epol, pol_virial = compute_polarization(state)
        energy.epol = epol
        if hasattr(energy, 'unbd'):
            energy.unbd += epol
        energy.poten += epol
        energy.vir += pol_virial
    else:
        energy.epol = 0.0
    
    # Recompute energy.evdw and energy.eelec from updated unbd/unbde
    # (compute_fast_forces set them based on real-space only; now correct)
    if hasattr(energy, 'unbd') and hasattr(energy, 'unbde'):
        energy.evdw = energy.unbd - energy.unbde
        energy.eelec = energy.unbde
    
    # Recompute poten as: bonded + nonbonded (incl. polarization in unbd) 
    if hasattr(energy, 'unbd'):
        energy.poten = energy.ebond + energy.ebend + energy.etort + energy.eopbs + energy.unbd
    else:
        energy.poten = energy.ebond + energy.ebend + energy.etort + energy.eopbs
        energy.poten += energy.evdw + energy.eelec + energy.epol
    
    return energy


def compute_all_forces(state: SimulationState) -> EnergyComponents:
    """
    Compute all forces and energies.

    INCLUDES SAFETY CHECKS for energy/force anomalies.
    """
    nat = state.atoms.nat

    # Zero forces
    state.atoms.f = np.zeros((3, nat))

    energy = EnergyComponents()
    virial_total = 0.0
    virial_tensor = np.zeros(3)  # Diagonal: xx, yy, zz

    # Bonded forces  (timing-instrumented)
    import time as _time
    _t0 = _time.time()
    energy.ebond, state.atoms.f = compute_bond_forces(
        state.atoms.x,
        state.connectivity.bonds,
        state.connectivity.nbonds,
        state.ff.stretch,
        state.atoms.box,
        state.atoms.el,
        state.atoms.f
    )

    energy.ebend, state.atoms.f = compute_bend_forces(
        state.atoms.x,
        state.connectivity.bends,
        state.connectivity.nbends,
        state.ff.bend,
        state.ff.linear,
        state.atoms.box,
        state.atoms.el,
        state.atoms.f
    )

    energy.etort, state.atoms.f = compute_torsion_forces(
        state.atoms.x,
        state.connectivity.torts,
        state.connectivity.ntorts,
        state.ff.twist,
        state.ff.nprms,
        state.atoms.box,
        state.atoms.el,
        state.atoms.f
    )

    energy.eopbs, state.atoms.f = compute_improper_forces(
        state.atoms.x,
        state.connectivity.deforms,
        state.connectivity.ndeforms,
        state.ff.deform,
        state.atoms.box,
        state.atoms.el,
        state.atoms.f
    )

    # End bonded timing, start nonbonded timing
    state._t_bonded = _time.time() - _t0
    _t0 = _time.time()

    # Non-bonded forces (if neighbor list exists)
    if state.neighbors.list_neighbors is not None:
        # Use spline-based calculation (includes real-space Ewald via erfc in splines)
        unbd, unbde, virial_nb, state.atoms.f = \
            compute_nonbonded_forces_with_splines(state)
        state._t_nonbonded = _time.time() - _t0
        _t0 = _time.time()

        # =====================================================================
        # EWALD RECIPROCAL SPACE + SELF-ENERGY + EXCLUSION CORRECTIONS
        # (Fortran: reciprocal() + correct() + ewself)
        # =====================================================================
        if state.params.lewald:
            newpress = True  # compute virial tensor
            erecip, ecorrect, ewself, f_ewald, vir_ewald = compute_ewald(state, newpress)
            
            # Add Ewald forces to atom forces
            state.atoms.f += f_ewald
            
            # Add Ewald energy contributions to nonbonded energy
            # unbd from splines already includes real-space erfc(αr)/r term
            # Now add reciprocal (minus self) and correction
            unbd += erecip + ecorrect
            unbde += erecip + ecorrect  # These are purely electrostatic
            
            # Add Ewald virial
            virial_nb += erecip  # Simplified scalar virial
            
            # Store Ewald virial tensor
            for ii in range(3):
                for jj in range(3):
                    state.pressure.tvirpo[ii, jj] += vir_ewald[ii, jj]
        state._t_ewald_recip = _time.time() - _t0

        # Store in energy components
        energy.evdw = unbd - unbde
        energy.eelec = unbde
        if hasattr(energy, 'unbd'):
            energy.unbd = unbd
        if hasattr(energy, 'unbde'):
            energy.unbde = unbde

        virial_total = virial_nb
        vir_tensor_nb = np.zeros(3)
    else:
        energy.evdw = 0.0
        energy.eelec = 0.0
        if hasattr(energy, 'unbd'):
            energy.unbd = 0.0
        if hasattr(energy, 'unbde'):
            energy.unbde = 0.0
        virial_total = 0.0
        vir_tensor_nb = np.zeros(3)

    # Total potential (before polarization)
    energy.poten = energy.ebond + energy.ebend + energy.etort + energy.eopbs + energy.unbd
    energy.vir = virial_total

    # =====================================================================
    # MANY-BODY POLARIZATION (APPLE&P force field)
    # Must be called AFTER nonbonded + Ewald, BEFORE return.
    # Adds polarization energy to energy.unbd and energy.poten.
    # Adds induced forces to state.atoms.f.
    # =====================================================================
    if state.ff.lpolarizable:
        from polarization import compute_polarization, init_polarization
        if not hasattr(state, 'pol_state') or state.pol_state is None:
            init_polarization(state)
        epol, pol_virial = compute_polarization(state)
        energy.epol = epol
        energy.unbd += epol
        energy.poten += epol
        energy.vir += pol_virial
    else:
        energy.epol = 0.0

    return energy
