"""
Neighbor List Construction for MD Simulation
Uses cell lists for efficient O(N) neighbor finding
"""

import numpy as np
from numba import jit, prange
from typing import Tuple
from datatypes import SimulationState, NeighborList
from constants import DefaultParams


def build_cell_list(state: SimulationState) -> None:
    """
    Build cell list and neighbor lists.
    Equivalent to Fortran boxes() subroutine.
    """
    nat = state.atoms.nat
    box = state.atoms.box
    x = state.atoms.x
    rcut = state.params.rcut
    driftmax = state.params.driftmax
    chain = state.atoms.chain
    inter = state.params.inter
    
    # Half box lengths
    el = box * 0.5
    state.atoms.el = el
    
    # Short-range cutoffs
    ros = state.params.ros
    rshort = ros + driftmax
    rsbox = rshort + driftmax
    rshort2 = rshort * rshort
    rsbox2 = rsbox * rsbox
    rcut2 = rcut * rcut
    
    # Determine box subdivisions
    rbig = rcut
    rdimmax = rbig / 2.0
    
    idim = np.zeros(3, dtype=int)
    ncheck = np.zeros(3, dtype=int)
    size = np.zeros(3)
    
    for kk in range(3):
        rdimmin = box[kk] / nat**(1.0/3.0)
        
        if rdimmax < rdimmin:
            rdim = rdimmax
        else:
            icheck = 2
            while True:
                icheck += 1
                rcheck = rbig / icheck
                if rcheck <= rdimmin:
                    break
            rdim = rbig / (icheck - 1)
        
        idim[kk] = int(box[kk] / rdim)
        if idim[kk] > DefaultParams.maxdim:
            print(f"Warning: idim > maxdim, reassigning to maxdim")
            idim[kk] = DefaultParams.maxdim
        
        rdim = box[kk] / idim[kk]
        ncheck[kk] = int(rbig / rdim) + 1
        
        if idim[kk] < 2 * ncheck[kk] + 1:
            if idim[kk] % 2 == 0:
                idim[kk] += 1
                rdim = box[kk] / idim[kk]
            ncheck[kk] = (idim[kk] - 1) // 2
        
        size[kk] = box[kk] / idim[kk]
    
    if state.params.printout:
        print(f"Boxes: idim = {idim}")
    
    # Build cell list
    idim2 = idim[0] * idim[1]
    total_cells = idim[0] * idim[1] * idim[2]
    
    listsubbox = np.zeros(total_cells, dtype=int)
    subbox = [[] for _ in range(total_cells)]
    boxofatom = np.zeros((3, nat), dtype=int)
    
    # Assign atoms to cells
    for iat in range(nat):
        isub = np.zeros(3, dtype=int)
        for kk in range(3):
            # Handle coordinates outside box (apply PBC to get cell index)
            coord = x[kk, iat]
            # Wrap coordinate into box [0, box[kk])
            while coord < 0:
                coord += box[kk]
            while coord >= box[kk]:
                coord -= box[kk]
            
            isub[kk] = int(coord / size[kk])
            if isub[kk] >= idim[kk]:
                isub[kk] = idim[kk] - 1
            elif isub[kk] < 0:
                isub[kk] = 0
            boxofatom[kk, iat] = isub[kk]
        
        ibox = isub[2] * idim2 + isub[1] * idim[0] + isub[0]
        listsubbox[ibox] += 1
        subbox[ibox].append(iat)
    
    # Build neighbor lists
    list_neighbors = np.zeros((DefaultParams.maxnay, nat), dtype=int)
    listm = np.zeros(nat, dtype=int)
    lists = np.zeros((DefaultParams.maxnay // 2, nat), dtype=int)
    listms = np.zeros(nat, dtype=int)
    
    # Get exclusion list if it exists
    listex = state.neighbors.listex if state.neighbors.listex is not None else np.zeros((40, nat), dtype=int)
    listmex = state.neighbors.listmex if state.neighbors.listmex is not None else np.zeros(nat, dtype=int)
    
    for iat in range(nat):
        ich = chain[iat]
        nayo = 0
        nayos = 0
        
        ix = boxofatom[0, iat]
        iy = boxofatom[1, iat]
        iz = boxofatom[2, iat]
        
        # Loop over neighboring cells
        for isubz in range(iz - ncheck[2], iz + ncheck[2] + 1):
            izz = isubz
            if izz >= idim[2]:
                izz -= idim[2]
            if izz < 0:
                izz += idim[2]
            izbox = izz * idim2
            
            for isuby in range(iy - ncheck[1], iy + ncheck[1] + 1):
                iyy = isuby
                if iyy >= idim[1]:
                    iyy -= idim[1]
                if iyy < 0:
                    iyy += idim[1]
                iybox = iyy * idim[0]
                
                for isubx in range(ix - ncheck[0], ix + ncheck[0] + 1):
                    ixx = isubx
                    if ixx >= idim[0]:
                        ixx -= idim[0]
                    if ixx < 0:
                        ixx += idim[0]
                    ibox = izbox + iybox + ixx
                    
                    for jat in subbox[ibox]:
                        jch = chain[jat]
                        
                        # Skip self and already counted pairs
                        if iat >= jat:
                            if (jat + iat) % 2 == 0:
                                continue
                        else:
                            if (jat + iat) % 2 != 0:
                                continue
                        
                        # Check intermolecular flag
                        if not inter:
                            if ich != jch:
                                continue
                        
                        # Calculate distance
                        dx = x[:, jat] - x[:, iat]
                        
                        for kk in range(3):
                            if abs(dx[kk]) > el[kk]:
                                dx[kk] -= np.sign(dx[kk]) * box[kk]
                        
                        r2 = np.sum(dx * dx)
                        
                        if r2 <= rcut2:
                            # Check exclusion list
                            is_excluded = False
                            if ich == jch:
                                for iex in range(int(listmex[iat])):
                                    if listex[iex, iat] == jat:
                                        is_excluded = True
                                        break
                            
                            # DEBUG: Check for close approaches of excluded pairs
                            #if is_excluded and r2 < 4.0 and state.kount <= 10:  # < 2 Angstroms
                            #    print(f"  WARNING: Excluded pair {iat}-{jat} at distance {np.sqrt(r2):.3f} Å")
                            #    print(f"           This is correct (excluded from non-bonded forces)")
                            #    print(f"           If they're too close, check bend/bond forces!")
                            
                            if not is_excluded:
                                if r2 < rsbox2:
                                    if nayos < DefaultParams.maxnay // 2:
                                        lists[nayos, iat] = jat
                                        nayos += 1
                                
                                if nayo < DefaultParams.maxnay:
                                    list_neighbors[nayo, iat] = jat
                                    nayo += 1
        
        listm[iat] = nayo
        listms[iat] = nayos
    
    # Store in state
    state.neighbors.list_neighbors = list_neighbors
    state.neighbors.listm = listm
    state.neighbors.lists = lists
    state.neighbors.listms = listms
    state.neighbors.idim = idim
    state.neighbors.listsubbox = listsubbox


def build_exclusion_list(state: SimulationState) -> None:
    """
    Build exclusion list for 1-2, 1-3, and optionally 1-4 interactions.
    
    Follows Fortran exclude() subroutine logic exactly:
    - 1-2 pairs (bonded) are always excluded
    - 1-3 pairs (in angles) are always excluded
    - 1-4 pairs (in torsions) are excluded if ex14 = True
    """
    nat = state.atoms.nat
    nbonds = state.connectivity.nbonds
    nbends = state.connectivity.nbends
    bonds = state.connectivity.bonds
    bends = state.connectivity.bends
    ex14 = state.params.ex14  # Control 1-4 exclusions
    
    listex = np.zeros((40, nat), dtype=int)
    listmex = np.zeros(nat, dtype=int)
    
    # Build exclusion list for each atom separately
    # This matches the Fortran loop structure
    for iat in range(nat):
        nex = 0  # Number of exclusions for this atom
        
        # =================================================================
        # 1-2 EXCLUSIONS (bonded pairs)
        # =================================================================
        for ibond in range(nbonds):
            ix = -1
            
            if bonds[0, ibond] == iat:
                ix = bonds[1, ibond]
            elif bonds[1, ibond] == iat:
                ix = bonds[0, ibond]
            
            if ix >= 0:
                # Check if already in exclusion list
                found_duplicate = False
                for iex in range(nex):
                    if listex[iex, iat] == ix:
                        found_duplicate = True
                        break
                
                if not found_duplicate:
                    if nex < 40:
                        listex[nex, iat] = ix
                        nex += 1
                    else:
                        print(f"Warning: Exclusion list overflow for atom {iat}")
        
        # =================================================================
        # 1-3 EXCLUSIONS (angle pairs)
        # =================================================================
        for ibend in range(nbends):
            ix = -1
            ix_mid = -1  # Middle atom (for 1-4 search)
            
            # If iat is first atom in bend
            if bends[0, ibend] == iat:
                ix = bends[2, ibend]  # Third atom is 1-3 partner
                ix_mid = bends[1, ibend]  # Middle atom
            # If iat is third atom in bend
            elif bends[2, ibend] == iat:
                ix = bends[0, ibend]  # First atom is 1-3 partner
                ix_mid = bends[1, ibend]  # Middle atom
            
            if ix >= 0:
                # =========================================================
                # 1-4 EXCLUSIONS (if ex14 is True)
                # =========================================================
                # Find atoms bonded to the 1-3 partner (these are 1-4 to iat)
                if ex14:
                    for ibond in range(nbonds):
                        iy = -1
                        
                        if bonds[0, ibond] == ix:
                            iy = bonds[1, ibond]
                        elif bonds[1, ibond] == ix:
                            iy = bonds[0, ibond]
                        
                        if iy >= 0:
                            # Check if iy is already in exclusion list
                            found_neighbor = False
                            for ineigh in range(nex):
                                if listex[ineigh, iat] == iy:
                                    found_neighbor = True
                                    break
                            
                            if not found_neighbor:
                                if nex < 40:
                                    listex[nex, iat] = iy
                                    nex += 1
                                else:
                                    print(f"Warning: Exclusion list overflow for atom {iat} (1-4)")
                
                # =========================================================
                # Add the 1-3 pair itself
                # =========================================================
                found_duplicate = False
                for iex in range(nex):
                    if listex[iex, iat] == ix:
                        found_duplicate = True
                        break
                
                if not found_duplicate:
                    if nex < 40:
                        listex[nex, iat] = ix
                        nex += 1
                    else:
                        print(f"Warning: Exclusion list overflow for atom {iat} (1-3)")
        
        listmex[iat] = nex
    
    state.neighbors.listex = listex
    state.neighbors.listmex = listmex
    
    # Debug: Print exclusion statistics
    if state.params.printout:
        total_exclusions = np.sum(listmex)
        max_exclusions = np.max(listmex)
        print(f"  Exclusion list built:")
        print(f"    Total exclusions: {total_exclusions}")
        print(f"    Max per atom: {max_exclusions}")
        print(f"    1-4 exclusions: {'enabled' if ex14 else 'disabled'}")


def build_14_list(state: SimulationState) -> None:
    """
    Build 1-4 interaction list (for special scaling).
    """
    nat = state.atoms.nat
    ntorts = state.connectivity.ntorts
    torts = state.connectivity.torts
    
    list14 = np.zeros((25, nat), dtype=int)
    listm14 = np.zeros(nat, dtype=int)
    
    for itort in range(ntorts):
        iat = torts[0, itort]
        lat = torts[3, itort]  # 1-4 pair
        
        # Check if already in list
        already_in = False
        for i14 in range(int(listm14[iat])):
            if list14[i14, iat] == lat:
                already_in = True
                break
        
        if not already_in:
            idx = int(listm14[iat])
            if idx < 25:
                list14[idx, iat] = lat
                listm14[iat] += 1
            
            # Symmetric entry
            idx = int(listm14[lat])
            if idx < 25:
                list14[idx, lat] = iat
                listm14[lat] += 1
    
    state.neighbors.list14 = list14
    state.neighbors.listm14 = listm14


def check_update_needed(state: SimulationState, delxup: np.ndarray) -> bool:
    """
    Check if neighbor list needs updating based on atom displacements.
    """
    driftmax = state.params.driftmax
    max_drift = 0.0
    
    for iat in range(state.atoms.nat):
        drift = np.sqrt(np.sum(delxup[:, iat]**2))
        if drift > max_drift:
            max_drift = drift
    
    return max_drift > driftmax


def initialize_neighbor_lists(state: SimulationState) -> None:
    """
    Initialize all neighbor lists at simulation start.
    """
    build_exclusion_list(state)
    build_14_list(state)
    try:
        from neighbor_list_fast import build_cell_list_fast
        build_cell_list_fast(state)
    except ImportError:
        build_cell_list(state)
