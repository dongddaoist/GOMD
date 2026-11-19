"""
SHAKE Constraint Initialization - CORRECTED VERSION
Builds constraint data structures from bond information
"""

import numpy as np
from datatypes import SimulationState


def initialize_constraints(state: SimulationState) -> None:
    """
    Initialize SHAKE bond constraints from connectivity data.
    
    This follows the Fortran logic (lines 6195-6210):
    - Constrain all bonds EXCEPT dummy atom bonds
    - Set target distance to equilibrium bond length from stretch[itype, 1]
    - Calculate reduced masses: mu = 0.5 * m1*m2 / (m1 + m2)
    
    CRITICAL FIX: Correct indexing of stretch array
    stretch is (nbondtypes, 5) where:
        [itype, 0] = k (force constant)
        [itype, 1] = r0 (equilibrium distance) <- THIS IS WHAT WE NEED
        [itype, 2] = k3 (cubic term)
        [itype, 3] = r0^2 (pre-calculated)
        [itype, 4] = other parameters
    """
    if not state.params.constrain:
        print("  SHAKE constraints: DISABLED")
        return
    
    nbonds = state.connectivity.nbonds
    bonds = state.connectivity.bonds
    stretch = state.ff.stretch
    mass = state.atoms.mass
    ldummy = state.atoms.ldummy
    
    # Debug: Check stretch array shape
    if state.params.printout:
        print(f"  stretch array shape: {stretch.shape}")
        print(f"  Expected: (nbondtypes, 5)")
    
    # Count constraints (exclude bonds involving dummy atoms)
    nbcon = 0
    bondcon = []
    d2_list = []
    massred_list = []
    
    for ibond in range(nbonds):
        iat = bonds[0, ibond]
        jat = bonds[1, ibond]
        itype = bonds[2, ibond]
        
        # Skip bonds involving dummy atoms (Fortran: ldummybond check)
        if ldummy[iat] or ldummy[jat]:
            continue
        
        # Add constraint
        bondcon.append([iat, jat, itype])
        
        # FIXED: Correct indexing - stretch[itype, column]
        # Column 1 contains equilibrium distance r0
        r0 = stretch[itype, 1]
        
        # Target distance squared (for efficiency in SHAKE)
        d2_list.append(r0 * r0)
        
        # Reduced mass: mu = 0.5 * m1*m2 / (m1 + m2)
        # Factor of 0.5 matches Fortran convention
        massred = 0.5 * mass[iat] * mass[jat] / (mass[iat] + mass[jat])
        massred_list.append(massred)
        
        nbcon += 1
    
    if nbcon == 0:
        print("  SHAKE constraints: No bonds to constrain")
        state.params.constrain = False
        return
    
    # Store in state
    state.constraints.nbcon = nbcon
    state.constraints.bondcon[:, :nbcon] = np.array(bondcon).T
    state.constraints.d2[:nbcon] = np.array(d2_list)
    state.constraints.massred[:nbcon] = np.array(massred_list)
    
    print(f"\n  ========================================")
    print(f"  SHAKE CONSTRAINTS INITIALIZED")
    print(f"  ========================================")
    print(f"  Total bonds:       {nbonds}")
    print(f"  Constrained bonds: {nbcon}")
    print(f"  Tolerance:         {state.constraints.tol:.2e}")
    print(f"  Max iterations:    {state.constraints.max_iter}")
    
    # Show example constraint distances
    if state.params.printout:
        print(f"\n  Example constraint distances:")
        for icon in range(min(10, nbcon)):
            iat = bondcon[icon][0]
            jat = bondcon[icon][1]
            itype = bondcon[icon][2]
            r0 = np.sqrt(d2_list[icon])
            print(f"    Constraint {icon}: atoms {iat}-{jat}, type {itype}, r0 = {r0:.4f} Å")
    
    print(f"  ========================================\n")
