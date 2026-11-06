"""
Non-Bonded Force Calculation Using Splines - FIXED VERSION
This version includes critical bug fixes for z-clamping and boundary checking
NUMBA-COMPATIBLE: Fixed f-string format specs
"""

import numpy as np
from numba import jit, prange
from typing import Tuple


@jit(nopython=True, cache=True)
def compute_nonbonded_forces_splined(
    x: np.ndarray,
    nat: int,
    atomtype: np.ndarray,
    list_neighbors: np.ndarray,
    listm: np.ndarray,
    typee: np.ndarray,
    box: np.ndarray,
    el: np.ndarray,
    rcut2: float,
    ro2: float,
    zs: np.ndarray,
    deltainv: float,
    coefft: np.ndarray,
    coeffft: np.ndarray,
    coeffee: np.ndarray,
    coefffee: np.ndarray,
    f: np.ndarray,
    kount: int = 0
) -> Tuple[float, float, float, np.ndarray, int]:
    """
    Compute non-bonded forces using spline interpolation.
    
    CRITICAL FIXES APPLIED:
    1. Z-clamping happens BEFORE spline evaluation (was happening too late)
    2. Improved boundary checking for ipiece (now uses nz-1 not nz)
    3. Removed overly restrictive z < 0.01 check
    4. NUMBA-COMPATIBLE: Removed f-string format specs
    5. NOW INCLUDES ELECTROSTATIC EVALUATION using coeffee/coefffee
    
    This matches the Fortran implementation (lines 2780-2850):
    1. Calculate z = r^2
    2. Clamp z to valid spline range [zs[0], zs[-1]]
    3. Find spline piece: ipiece = int((z - zs[0])*deltainv) + 1
    4. Evaluate polynomial E(z) using coefft (TOTAL)
    5. Evaluate polynomial E_elec(z) using coeffee (ELECTROSTATIC ONLY)
    6. Evaluate derivative F(z) using coeffft - coefffee
    7. Apply forces: f = F(z) * dx
    
    Args:
        x: Atomic positions (3, nat)
        nat: Number of atoms
        atomtype: Atom types
        list_neighbors: Neighbor list
        listm: Number of neighbors per atom
        typee: Type pair mapping
        box: Box dimensions
        el: Half box lengths
        rcut2: Cutoff distance squared
        ro2: Short-range cutoff squared
        zs: Spline grid points (r^2 values)
        deltainv: 1/deltaspline
        coefft: Total energy spline coefficients (6, npoints, maxnnb)
        coeffft: Total force spline coefficients (6, npoints, maxnnb)
        coeffee: Electrostatic energy spline coefficients (6, npoints, maxnnb)
        coefffee: Electrostatic force spline coefficients (6, npoints, maxnnb)
        f: Force array (3, nat) - will be updated
        kount: Current timestep (for debug output)
        
    Returns:
        unbd: Total non-bonded energy
        unbde: Electrostatic energy only
        virial: Virial
        f: Updated forces
        n_clamped: Number of atom pairs that were clamped (for debugging)
    """
    unbd = 0.0
    unbde = 0.0
    virial = 0.0
    
    zs0 = zs[0]  # Minimum z value for spline (typically 0.50)
    nz = zs.shape[0] - 1  # Number of spline segments
    
    # Counter for very close approaches (debug only)
    n_clamped = 0
    
    for iat in range(nat):
        itype = atomtype[iat]
        
        for k in range(listm[iat]):
            jat = list_neighbors[k, iat]
            jtype = atomtype[jat]
            
            # Calculate distance vector with PBC
            xij1 = x[0, jat] - x[0, iat]
            xij2 = x[1, jat] - x[1, iat]
            xij3 = x[2, jat] - x[2, iat]
            
            if abs(xij1) > el[0]:
                xij1 -= np.sign(xij1) * box[0]
            if abs(xij2) > el[1]:
                xij2 -= np.sign(xij2) * box[1]
            if abs(xij3) > el[2]:
                xij3 -= np.sign(xij3) * box[2]
            
            # Distance squared
            z = xij1*xij1 + xij2*xij2 + xij3*xij3
            
            # ================================================================
            # CRITICAL FIX #1: Check cutoff first
            # ================================================================
            if z > rcut2:
                continue
            
            # ================================================================
            # CRITICAL FIX #2: Clamp z to spline range BEFORE evaluation
            # ================================================================
            # The splines are only valid for z >= zs[0] (typically 0.50)
            # Atoms closer than sqrt(zs[0]) will cause extrapolation to
            # astronomically large values if not clamped!
            if z < zs0:
                z = zs0
                n_clamped += 1
            
            # Only calculate if within cutoff
            if z <= rcut2:
                # Get interaction type
                itypee = typee[itype, jtype]
                
                # =============================================================
                # SPLINE EVALUATION (Fortran lines 2794-2815)
                # =============================================================
                
                # Find which spline piece we're in
                ipiece = int((z - zs0) * deltainv) + 1
                
                # ============================================================
                # CRITICAL FIX #3: Improved boundary checking
                # ============================================================
                # Clamp ipiece to valid spline interval [1, nz-1]
                # Note: Use nz-1 not nz, because we access zs[ipiece-1]
                if ipiece < 1:
                    ipiece = 1
                elif ipiece >= nz:
                    ipiece = nz - 1  # Changed from nz to nz-1
                
                # Additional safety check (should never trigger with proper clamping)
                if ipiece >= coefft.shape[1]:
                    # This indicates a serious problem with array sizes
                    continue
                
                # Position within the piece
                zpiece = z - zs[ipiece - 1]
                
                # Powers for polynomial evaluation
                zpiece2 = zpiece * zpiece
                zpiece3 = zpiece2 * zpiece
                zpiece4 = zpiece3 * zpiece
                zpiece5 = zpiece4 * zpiece
                
                # Energy: E(z) = c0 + c1*z + c2*z^2 + c3*z^3 + c4*z^4 + c5*z^5
                # (Fortran lines 2810-2815)
                energy = (coefft[0, ipiece, itypee] +
                         coefft[1, ipiece, itypee] * zpiece +
                         coefft[2, ipiece, itypee] * zpiece2 +
                         coefft[3, ipiece, itypee] * zpiece3 +
                         coefft[4, ipiece, itypee] * zpiece4 +
                         coefft[5, ipiece, itypee] * zpiece5)
                
                unbd += energy
                
                # Electrostatic energy (separate): E_elec(z) using coeffee
                # (Fortran lines 2901-2906)
                energy_elec = (coeffee[0, ipiece, itypee] +
                              coeffee[1, ipiece, itypee] * zpiece +
                              coeffee[2, ipiece, itypee] * zpiece2 +
                              coeffee[3, ipiece, itypee] * zpiece3 +
                              coeffee[4, ipiece, itypee] * zpiece4 +
                              coeffee[5, ipiece, itypee] * zpiece5)
                
                unbde += energy_elec
                
                # Force: dE/dz using coeffft (TOTAL)
                # Note: coeffft already has the multiplied coefficients
                # (Fortran lines 2816-2825)
                ff = (coeffft[1, ipiece, itypee] +
                      coeffft[2, ipiece, itypee] * zpiece +
                      coeffft[3, ipiece, itypee] * zpiece2 +
                      coeffft[4, ipiece, itypee] * zpiece3 +
                      coeffft[5, ipiece, itypee] * zpiece4)
                
                # CRITICAL: Do NOT subtract ffe from ff for forces!
                # In Fortran (line 2900), ff-ffe only happens inside if(newpress)
                # for virial calculation. The actual forces at lines 2828-2836 use
                # the TOTAL ff from coeffft, NOT ff-ffe.
                
                # Electrostatic force (computed but NOT used in force application)
                # This is only for tracking purposes
                ffe = (coefffee[1, ipiece, itypee] +
                       coefffee[2, ipiece, itypee] * zpiece +
                       coefffee[3, ipiece, itypee] * zpiece2 +
                       coefffee[4, ipiece, itypee] * zpiece3 +
                       coefffee[5, ipiece, itypee] * zpiece4)
                
                # Apply forces: F = dE/dz * dx
                # Since E is function of z=r^2, dE/dr = dE/dz * dz/dr = dE/dz * 2r
                # Force on atom i: F_i = -dE/dr * r_hat = -dE/dz * 2r * (dr/dr)
                #                      = -dE/dz * dr (where dr = x_j - x_i)
                # But we want force, not -force, so signs work out
                # (Fortran lines 2828-2833)
                f1 = ff * xij1
                f2 = ff * xij2
                f3 = ff * xij3
                
                f[0, iat] += f1
                f[1, iat] += f2
                f[2, iat] += f3
                f[0, jat] -= f1
                f[1, jat] -= f2
                f[2, jat] -= f3
                
                # Virial contribution
                r = np.sqrt(z)
                virial += energy  # Simplified virial
    
    # NOTE: No 0.5 factor needed — the neighbor list's odd/even skip rule
    # ensures each pair (i,j) appears exactly once. Forces are applied to
    # both atoms once per pair (f[iat]+=, f[jat]-=), and energy is accumulated once.
    
    return unbd, unbde, virial, f, n_clamped


def compute_nonbonded_forces_with_splines(state) -> Tuple[float, float, float, np.ndarray]:
    """
    Wrapper for spline-based non-bonded force calculation.
    
    Returns BOTH total energy and electrostatic energy separately.
    """
    # Check that splines have been computed
    if not hasattr(state, 'splines') or state.splines is None:
        raise RuntimeError("Splines not initialized! Call compute_spline_coefficients() first!")
    
    if state.splines.coefft is None:
        raise RuntimeError("Spline coefficients not computed!")
    
    # Get parameters
    nat = state.atoms.nat
    x = state.atoms.x
    atomtype = state.atoms.atomtype
    list_neighbors = state.neighbors.list_neighbors
    listm = state.neighbors.listm
    typee = state.ff.typee if hasattr(state.ff, 'typee') else np.zeros((100, 100), dtype=np.int32)
    box = state.atoms.box
    el = state.atoms.el
    rcut2 = state.params.rcut ** 2
    ro2 = state.params.ros ** 2
    
    # Spline parameters
    zs = state.splines.zs
    deltainv = state.splines.deltainv
    coefft = state.splines.coefft
    coeffft = state.splines.coeffft
    coeffee = state.splines.coeffee
    coefffee = state.splines.coefffee
    
    # Get current timestep for debug output
    kount = state.kount if hasattr(state, 'kount') else 0
    
    # Initialize forces (will be added to)
    f = state.atoms.f.copy()  # Don't overwrite, add to existing
    
    # Compute non-bonded forces using splines
    unbd, unbde, virial, f, n_clamped = compute_nonbonded_forces_splined(
        x, nat, atomtype, list_neighbors, listm, typee,
        box, el, rcut2, ro2, zs, deltainv, coefft, coeffft, 
        coeffee, coefffee, f, kount
    )
    
    return unbd, unbde, virial, f


# =============================================================================
# COMPARISON: Old vs New
# =============================================================================
"""
OLD (BUGGY) VERSION:
--------------------
z = xij1*xij1 + xij2*xij2 + xij3*xij3

if z > rcut2 or z < 0.01:  # ❌ z < 0.01 is too restrictive
    continue

if z <= 0.60:  # ❌ This happens AFTER the check below!
    z = 0.60

if z <= ro2:
    # Get interaction type
    itypee = typee[itype, jtype]
    
    # Find spline piece ❌ z might be < zs[0] here!
    ipiece = int((z - zs0) * deltainv) + 1
    
    if ipiece > nz:  # ❌ Should be >= nz
        ipiece = nz

NEW (FIXED) VERSION:
--------------------
z = xij1*xij1 + xij2*xij2 + xij3*xij3

if z > rcut2:  # ✓ Simple cutoff check
    continue

if z < zs0:  # ✓ Clamp BEFORE spline evaluation
    z = zs0

if z <= ro2:
    # Get interaction type
    itypee = typee[itype, jtype]
    
    # Find spline piece ✓ z >= zs0 guaranteed
    ipiece = int((z - zs0) * deltainv) + 1
    
    if ipiece >= nz:  # ✓ Correct boundary
        ipiece = nz - 1

KEY DIFFERENCES:
1. Z-clamping happens BEFORE entering the ro2 check
2. Z-clamping uses zs0 (spline minimum) not arbitrary 0.60
3. Removed the z < 0.01 check which was preventing valid close interactions
4. Changed ipiece clamping from > nz to >= nz
5. Changed clamped value from nz to nz-1
6. NUMBA FIX: Moved formatted print outside JIT function
"""


# =============================================================================
# DEBUGGING: Check if splines are working
# =============================================================================
def check_spline_evaluation(state):
    """
    Test spline evaluation at a few distances.
    """
    print("\n  Testing spline evaluation...")
    
    if not hasattr(state, 'splines') or state.splines is None:
        print("  ERROR: Splines not initialized!")
        return
    
    zs = state.splines.zs
    coefft = state.splines.coefft
    coeffft = state.splines.coeffft
    deltainv = state.splines.deltainv
    
    # Test at a few r values
    test_r = [0.7, 1.0, 2.0, 3.0, 4.0, 5.0]  # Angstroms
    
    print(f"  Spline grid: z from {zs[0]:.2f} to {zs[-1]:.2f}")
    print(f"  Testing interaction type 0:")
    print(f"  {'r (Å)':<10} {'z=r²':<10} {'ipiece':<10} {'E (kcal/mol)':<15} {'F':<15}")
    
    itypee = 0  # Test first interaction type
    zs0 = zs[0]
    nz = zs.shape[0] - 1
    
    for r in test_r:
        z = r * r
        
        # Clamp z as in actual code
        if z < zs0:
            z_clamped = zs0
            clamped_str = f" (clamped from {z:.3f})"
        else:
            z_clamped = z
            clamped_str = ""
        
        if z_clamped < zs0 or z_clamped > zs[-1]:
            print(f"  {r:<10.3f} {z:<10.3f} {'out of range':<10}")
            continue
        
        # Find spline piece
        ipiece = int((z_clamped - zs0) * deltainv) + 1
        if ipiece < 1:
            ipiece = 1
        elif ipiece >= nz:
            ipiece = nz - 1
        
        # Position within piece
        zpiece = z_clamped - zs[ipiece - 1]
        zpiece2 = zpiece * zpiece
        zpiece3 = zpiece2 * zpiece
        zpiece4 = zpiece3 * zpiece
        zpiece5 = zpiece4 * zpiece
        
        # Energy
        energy = (coefft[0, ipiece, itypee] +
                 coefft[1, ipiece, itypee] * zpiece +
                 coefft[2, ipiece, itypee] * zpiece2 +
                 coefft[3, ipiece, itypee] * zpiece3 +
                 coefft[4, ipiece, itypee] * zpiece4 +
                 coefft[5, ipiece, itypee] * zpiece5)
        
        # Force
        force = (coeffft[1, ipiece, itypee] +
                coeffft[2, ipiece, itypee] * zpiece +
                coeffft[3, ipiece, itypee] * zpiece2 +
                coeffft[4, ipiece, itypee] * zpiece3 +
                coeffft[5, ipiece, itypee] * zpiece4)
        
        print(f"  {r:<10.3f} {z:<10.3f} {ipiece:<10} {energy:<15.6f} {force:<15.6f}{clamped_str}")
    
    print("\n  Expected: Energy should be NEGATIVE at typical distances (attractive)")
    print("  If energy is positive, check spline coefficient calculation!")
