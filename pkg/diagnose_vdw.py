"""
Targeted diagnostic for the -99 million vdW energy bug
"""
import numpy as np

def diagnose_vdw_explosion(state):
    """
    Check why van der Waals energy is -99 million
    """
    print("\n" + "="*80)
    print("VAN DER WAALS EXPLOSION DIAGNOSTIC")
    print("="*80)
    
    # Check force field parameters
    print("\n1. FORCE FIELD PARAMETERS")
    print("-" * 80)
    
    if hasattr(state.ff, 'nonbonded'):
        nonbonded = state.ff.nonbonded
        print(f"   nonbonded shape: {nonbonded.shape}")
        print(f"   Expected: (3, num_types)")
        
        # Check a few types
        print(f"\n   Sample nonbonded parameters (first 5 types):")
        print(f"   {'Type':<6} {'A':<15} {'B':<15} {'C':<15}")
        for i in range(min(5, nonbonded.shape[1])):
            a = nonbonded[0, i]
            b = nonbonded[1, i]
            c = nonbonded[2, i]
            print(f"   {i:<6} {a:<15.6e} {b:<15.6e} {c:<15.6e}")
        
        # Check for extremely large values
        a_max = np.max(np.abs(nonbonded[0, :]))
        b_max = np.max(np.abs(nonbonded[1, :]))
        c_max = np.max(np.abs(nonbonded[2, :]))
        
        print(f"\n   Max |A|: {a_max:.6e}")
        print(f"   Max |B|: {b_max:.6e}")
        print(f"   Max |C|: {c_max:.6e}")
        
        # Typical values for Buckingham: A ~ 1e5, B ~ 3-4, C ~ 1e3
        if a_max > 1e8:
            print(f"   ⚠ WARNING: A values are very large! (typical ~ 1e5)")
        if c_max > 1e6:
            print(f"   ⚠ WARNING: C values are very large! (typical ~ 1e3)")
    
    if hasattr(state.ff, 'electrostatic'):
        elec = state.ff.electrostatic
        print(f"\n   electrostatic shape: {elec.shape}")
        
        # Check if it's empty
        nonzero = np.count_nonzero(elec)
        if nonzero == 0:
            print(f"   ✗ ERROR: electrostatic array is ALL ZEROS!")
            print(f"      This is why electrostatic energy = 0")
        else:
            print(f"   ✓ electrostatic has {nonzero} non-zero values")
            
        # Sample values
        print(f"\n   Sample electrostatic parameters (first 5 types):")
        print(f"   {'Type':<6} {'Column 0':<15} {'Column 1':<15} {'Column 2':<15}")
        for i in range(min(5, elec.shape[0])):
            print(f"   {i:<6} {elec[i,0]:<15.6e} {elec[i,1]:<15.6e} {elec[i,2]:<15.6e}")
    
    # Check map array
    if hasattr(state.ff, 'map'):
        map_arr = state.ff.map
        print(f"\n   map shape: {map_arr.shape}")
        print(f"   map range: [{np.min(map_arr)}, {np.max(map_arr)}]")
        print(f"   Sample: {map_arr[:10]}")
    
    # Check spline coefficients
    print("\n2. SPLINE COEFFICIENTS")
    print("-" * 80)
    
    if hasattr(state, 'splines') and state.splines is not None:
        coefft = state.splines.coefft
        zs = state.splines.zs
        
        # Find first non-zero interaction type
        test_type = None
        for i in range(coefft.shape[2]):
            if np.any(coefft[:, 1:, i] != 0):
                test_type = i
                break
        
        if test_type is None:
            print("   ✗ ERROR: All spline coefficients are ZERO!")
            return
        
        print(f"   Testing interaction type {test_type}:")
        print(f"   Coefficient magnitudes at ipoint=1:")
        for ic in range(6):
            val = coefft[ic, 1, test_type]
            print(f"      coefft[{ic}] = {val:.6e}")
        
        # Evaluate at r = 2.0 Ang
        r_test = 2.0
        z_test = r_test * r_test
        
        deltainv = state.splines.deltainv
        zs0 = zs[0]
        nz = zs.shape[0] - 1
        
        if z_test < zs0:
            z_test = zs0
        
        ipiece = int((z_test - zs0) * deltainv) + 1
        if ipiece < 1:
            ipiece = 1
        elif ipiece >= nz:
            ipiece = nz - 1
        
        zpiece = z_test - zs[ipiece - 1]
        zpiece2 = zpiece * zpiece
        zpiece3 = zpiece2 * zpiece
        zpiece4 = zpiece3 * zpiece
        zpiece5 = zpiece4 * zpiece
        
        energy = (coefft[0, ipiece, test_type] +
                 coefft[1, ipiece, test_type] * zpiece +
                 coefft[2, ipiece, test_type] * zpiece2 +
                 coefft[3, ipiece, test_type] * zpiece3 +
                 coefft[4, ipiece, test_type] * zpiece4 +
                 coefft[5, ipiece, test_type] * zpiece5)
        
        print(f"\n   Energy at r={r_test} Ang for type {test_type}: {energy:.6e} kcal/mol")
        
        if abs(energy) > 1000:
            print(f"   ⚠ WARNING: Single pair energy is > 1000 kcal/mol!")
            print(f"      For ~10,000 atom pairs, total would be {energy*10000:.2e}")
    
    # Check actual atom pair distances
    print("\n3. CHECKING ACTUAL ATOM DISTANCES")
    print("-" * 80)
    
    nat = state.atoms.nat
    x = state.atoms.x
    box = state.atoms.box
    el = state.atoms.el
    
    # Check first 20 neighbor pairs
    if hasattr(state, 'neighbors') and state.neighbors.list_neighbors is not None:
        print(f"   Checking first 20 atom pairs in neighbor list:")
        print(f"   {'Pair':<12} {'r (Ang)':<10} {'Type':<8} {'Energy est.':<15}")
        
        count = 0
        for iat in range(min(10, nat)):
            for k in range(min(2, state.neighbors.listm[iat])):
                jat = state.neighbors.list_neighbors[k, iat]
                
                # Calculate distance
                dx = x[:, jat] - x[:, iat]
                for kk in range(3):
                    if abs(dx[kk]) > el[kk]:
                        dx[kk] -= np.sign(dx[kk]) * box[kk]
                
                r = np.sqrt(np.sum(dx**2))
                
                # Get types
                itype = state.atoms.atomtype[iat]
                jtype = state.atoms.atomtype[jat]
                itypee = state.ff.typee[itype, jtype]
                
                # Rough energy estimate (just C/r^6 term)
                if hasattr(state.ff, 'nonbonded') and hasattr(state.ff, 'map'):
                    imap = state.ff.map[itypee]
                    c = state.ff.nonbonded[2, imap]
                    energy_est = -c / (r**6)
                else:
                    energy_est = 0.0
                
                print(f"   {iat}-{jat:<8} {r:<10.4f} {itypee:<8} {energy_est:<15.6e}")
                
                count += 1
                if count >= 20:
                    break
            if count >= 20:
                break
    
    # Check total number of interactions
    print("\n4. COUNTING INTERACTIONS")
    print("-" * 80)
    
    total_pairs = 0
    for iat in range(nat):
        if hasattr(state, 'neighbors') and state.neighbors.list_neighbors is not None:
            total_pairs += state.neighbors.listm[iat]
    
    total_pairs = total_pairs // 2  # Divide by 2 for double counting
    
    print(f"   Total atom pairs in neighbor list: {total_pairs}")
    print(f"   Average energy per pair: {state.energy.evdw / total_pairs:.2f} kcal/mol")
    
    if abs(state.energy.evdw / total_pairs) > 100:
        print(f"   ⚠ WARNING: Average pair energy is {state.energy.evdw / total_pairs:.2f}")
        print(f"      Typical should be -1 to +10 kcal/mol per pair")
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if abs(state.energy.evdw / total_pairs) > 1000:
        print("\n✗ CRITICAL: Energy per pair is ~1000× too large!")
        print("\nMost likely causes:")
        print("  1. Force field parameters (A, B, C) have wrong units")
        print("     - Check if they need conversion (e.g., kcal/mol vs kJ/mol)")
        print("  2. Missing factor somewhere in potential function")
        print("  3. Distance units wrong (check if using Angstroms vs Bohr)")
        
    if hasattr(state.ff, 'electrostatic'):
        if np.count_nonzero(state.ff.electrostatic) == 0:
            print("\n✗ CRITICAL: Electrostatic array is all zeros!")
            print("  - Check force field loading")
            print("  - Verify charges are being read correctly")

# Usage: Add to main.py after compute_all_forces
# from diagnose_vdw import diagnose_vdw_explosion
# diagnose_vdw_explosion(state)

