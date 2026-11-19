#!/usr/bin/env python3
"""
GOMD Python Version
Molecular Dynamics Simulation Code

Converted from Fortran90 GOMD code
Supports NVE, NVT (Nose-Hoover), and NPT ensembles

Usage:
    python main.py [input_directory]
"""

import sys
import time
import numpy as np
from pathlib import Path

try:
    from splines_fast import compute_spline_coefficients_fast as compute_spline_coefficients
except ImportError:
    from splines import compute_spline_coefficients


from constants import GASK, GASK_KCAL, PI
from datatypes import SimulationState, EnergyComponents
from input_parser import load_simulation
from neighbor_list import initialize_neighbor_lists, check_update_needed
try:
    from neighbor_list_fast import build_cell_list_fast as build_cell_list
except ImportError:
    from neighbor_list import build_cell_list
from forces import compute_all_forces, compute_kinetic_energy
from integrator import (
    init_thermostat, init_integration, integrate_step,
    remove_com_velocity, calculate_temperature, compute_kinetic,
    do_dummy_forces,
    update_dummy_coords
)
from output import (
    MDLogWriter, TrajectoryWriter, VelocityWriter,
    write_restart, write_final_output, print_step_info
)

DEBUG_INPUT = False

def initialize_jikdummy(state: SimulationState) -> None:
    """
    Initialize jikdummy connectivity for dummy (lone pair) atoms.
    
    For type 2 or 3 lone pairs (paired), this function identifies twins
    (two lone pairs attached to the same central atom) and swaps jat/kat
    for the second one to place them on opposite sides of the plane.
    
    jikdummy[0, idummy] = jat (parent atom 1)
    jikdummy[1, idummy] = iat (central parent atom - bonded to dummy)
    jikdummy[2, idummy] = kat (parent atom 2)
    jikdummy[3, idummy] = idat (dummy atom index)
    jikdummy[4, idummy] = itype (dummy type index)
    """
    ndummy = state.atoms.ndummy
    if ndummy == 0:
        return
    
    # Allocate jikdummy array
    state.atoms.jikdummy = np.zeros((5, ndummy), dtype=int)
    
    # Get bond and bend data
    nbonds = state.connectivity.nbonds
    bonds = state.connectivity.bonds
    nbends = state.connectivity.nbends
    bends = state.connectivity.bends
    
    # Build a map of atom -> bonded atoms for quick lookup
    bonded_to = {i: [] for i in range(state.atoms.nat)}
    for ibond in range(nbonds):
        at1 = bonds[0, ibond]
        at2 = bonds[1, ibond]
        bonded_to[at1].append(at2)
        bonded_to[at2].append(at1)
    
    # Build a map of central atom -> list of (jat, kat) pairs from bends
    bends_by_central = {i: [] for i in range(state.atoms.nat)}
    for ibend in range(nbends):
        at1 = bends[0, ibend]
        at2 = bends[1, ibend]  # Central atom
        at3 = bends[2, ibend]
        bends_by_central[at2].append((at1, at3))
    
    # Find dummy atoms and their parent connectivity
    dummy_count = 0
    warnings_shown = 0
    max_warnings = 5
    
    for idat in range(state.atoms.nat):
        if state.atoms.ldummy[idat]:
            itype_atom = state.atoms.atomtype[idat]
            
            # Find dummy type index (index into adummy, bdummy, cdummy arrays)
            itype_dummy = state.ff.typedummy[itype_atom] - 1  # Convert to 0-indexed
            
            # Find parent atom - the real atom bonded to this dummy
            parent_atom = None
            for bonded_atom in bonded_to.get(idat, []):
                if not state.atoms.ldummy[bonded_atom]:
                    parent_atom = bonded_atom
                    break
            
            if parent_atom is None:
                if warnings_shown < max_warnings:
                    print(f"Warning: Could not find parent atom for dummy atom {idat}")
                    warnings_shown += 1
                # Use dummy's first bonded atom as fallback
                if bonded_to.get(idat):
                    parent_atom = bonded_to[idat][0]
                else:
                    parent_atom = max(0, idat - 1)  # Last resort fallback
            
            # Find a bend where parent_atom is central to get jat and kat
            bend_atoms = bends_by_central.get(parent_atom, [])
            
            if bend_atoms:
                # Use the first suitable bend (could be improved with LpBendAtomType matching)
                jat, kat = bend_atoms[0]
            else:
                # Fallback: use other atoms bonded to parent
                other_bonded = [a for a in bonded_to.get(parent_atom, []) if a != idat]
                if len(other_bonded) >= 2:
                    jat, kat = other_bonded[0], other_bonded[1]
                elif len(other_bonded) == 1:
                    jat = kat = other_bonded[0]
                else:
                    # Last resort
                    jat = kat = parent_atom
                    if warnings_shown < max_warnings:
                        print(f"Warning: Could not find bend atoms for dummy {idat}, parent {parent_atom}")
                        warnings_shown += 1
            
            state.atoms.jikdummy[0, dummy_count] = jat
            state.atoms.jikdummy[1, dummy_count] = parent_atom  # iat (central)
            state.atoms.jikdummy[2, dummy_count] = kat
            state.atoms.jikdummy[3, dummy_count] = idat
            state.atoms.jikdummy[4, dummy_count] = itype_dummy
            
            dummy_count += 1
    
    if warnings_shown >= max_warnings:
        print(f"  (suppressed {warnings_shown - max_warnings} additional warnings)")
    
    print(f"  Initialized jikdummy for {dummy_count} dummy atoms")
    
    # **CRITICAL FIX**: For paired lone pairs (type 2 or 3), swap jat and kat
    # for the second dummy in each pair to place them on opposite sides of the plane
    print(f"\n  Starting lone pair pairing...")
    
    # Debug: Check LpGeomType availability
    if not hasattr(state.ff, 'LpGeomType') or state.ff.LpGeomType is None:
        print(f"  ERROR: LpGeomType not found in force field!")
        print(f"  Cannot perform lone pair pairing - all type-2/3 LPs will overlap!")
        return
    
    has_been_changed = [False] * ndummy
    pairs_found = 0
    type2_count = 0
    type3_count = 0
    
    for idummy in range(ndummy):
        if not has_been_changed[idummy]:
            has_been_changed[idummy] = True
            
            itype = state.atoms.jikdummy[4, idummy]
            iat = state.atoms.jikdummy[1, idummy]  # Central atom
            idat = state.atoms.jikdummy[3, idummy]  # Dummy atom index
            
            # Get geometry type for this dummy type
            geom_type = state.ff.LpGeomType[itype]
            
            # Debug: Count geometry types
            if geom_type == 2:
                type2_count += 1
            elif geom_type == 3:
                type3_count += 1
            
            # For type 2 (ether-like) or type 3 (carbonyl-like) lone pairs
            if geom_type == 2 or geom_type == 3:
                # Find the twin dummy (another dummy attached to the same central atom)
                twin_found = False
                for jdummy in range(idummy + 1, ndummy):
                    if has_been_changed[jdummy]:
                        continue
                    
                    jat_central = state.atoms.jikdummy[1, jdummy]
                    
                    if iat == jat_central:  # Same central atom
                        if twin_found:
                            print(f"  Warning: Found more than two dummies on atom {iat}")
                        
                        twin_found = True
                        jdat = state.atoms.jikdummy[3, jdummy]
                        
                        # Debug: Print BEFORE swap
                        old_jat = state.atoms.jikdummy[0, jdummy]
                        old_kat = state.atoms.jikdummy[2, jdummy]
                        ## print(f"  Pairing dummies {idat} and {jdat} on atom {iat} (type {geom_type})")
                        ## print(f"    Before swap: jat={old_jat}, kat={old_kat}")
                        
                        # Swap jat and kat for the twin
                        # This flips the cross product direction: rik × rij = -(rij × rik)
                        state.atoms.jikdummy[0, jdummy] = old_kat  # jat = old kat
                        state.atoms.jikdummy[2, jdummy] = old_jat  # kat = old jat
                        
                        # Debug: Print AFTER swap
                        new_jat = state.atoms.jikdummy[0, jdummy]
                        new_kat = state.atoms.jikdummy[2, jdummy]
                        # print(f"    After swap:  jat={new_jat}, kat={new_kat}")
                        
                        has_been_changed[jdummy] = True
                        pairs_found += 1
                        break
                
                if not twin_found:
                    print(f"  Warning: Dummy {idat} (type {geom_type}) on atom {iat} has no twin!")
    
    print(f"  Found {type2_count} type-2 and {type3_count} type-3 dummies")
    print(f"  Successfully paired {pairs_found} sets of lone pairs")
    print(f"  Lone pair pairing completed\n")

def diagnose_energy_explosion(state):
    """
    Complete diagnostic for -99M energy bug
    """
    print("\n" + "="*80)
    print("ENERGY EXPLOSION DIAGNOSTIC")
    print("="*80)

    # Check 1: How many actual interaction types?
    print("\n1. ACTUAL INTERACTION TYPES")
    print("-" * 80)

    if hasattr(state.ff, 'typee'):
        typee = state.ff.typee
        # Count unique non-negative values
        unique_types = np.unique(typee[typee >= 0])
        n_actual = len(unique_types)
        max_typee = np.max(unique_types)

        print(f"  typee unique values: {n_actual}")
        print(f"  typee max value: {max_typee}")
        print(f"  typee range: [{np.min(unique_types)}, {max_typee}]")
    else:
        print("  ERROR: typee not found!")
        return False

    # Check 2: What's maxnnb?
    print("\n2. MAXNNB VALUE")
    print("-" * 80)

    if hasattr(state.ff, 'nonbonded'):
        maxnnb = state.ff.nonbonded.shape[1]
        print(f"  nonbonded shape: {state.ff.nonbonded.shape}")
        print(f"  maxnnb (column count): {maxnnb}")
    else:
        print("  ERROR: nonbonded not found!")
        return False

    if hasattr(state.ff, 'map'):
        print(f"  map length: {len(state.ff.map)}")
        map_max = np.max(state.ff.map)
        print(f"  map max value: {map_max}")

    # Check 3: Electrostatic array
    print("\n3. ELECTROSTATIC ARRAY")
    print("-" * 80)

    elec_ok = False
    if hasattr(state.ff, 'electrostatic'):
        elec = state.ff.electrostatic
        print(f"  electrostatic shape: {elec.shape}")
        nonzero = np.count_nonzero(elec)
        print(f"  non-zero elements: {nonzero}")

        if nonzero == 0:
            print(f"  ✗ CRITICAL: Electrostatic is ALL ZEROS!")
            print(f"    This means electrostatic energy = 0")
        else:
            elec_ok = True
            print(f"  ✓ Electrostatic has values")
            # Show first non-zero
            for i in range(elec.shape[0]):
                if elec[i, 0] != 0:
                    print(f"    Example: elec[{i}, 0] = {elec[i, 0]:.6e}")
                    break
    else:
        print("  ERROR: electrostatic not found!")

    # Check 4: Spline coefficient array
    print("\n4. SPLINE COEFFICIENTS")
    print("-" * 80)

    if hasattr(state, 'splines') and state.splines.coefft is not None:
        coefft = state.splines.coefft
        print(f"  coefft shape: {coefft.shape}")
        n_spline_types = coefft.shape[2]
        print(f"  Splines computed for: {n_spline_types} types")

        # Check which have non-zero values
        nonzero_types = []
        for i in range(n_spline_types):
            if np.any(coefft[:, 1:, i] != 0):
                nonzero_types.append(i)

        print(f"  Types with non-zero coefficients: {len(nonzero_types)}")

        if len(nonzero_types) < n_spline_types:
            n_zero = n_spline_types - len(nonzero_types)
            print(f"  ⚠ {n_zero} types have ALL ZERO coefficients!")

    # DIAGNOSIS
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    print(f"\nYour system:")
    print(f"  - Has {n_actual} actual interaction types (from typee)")
    print(f"  - But maxnnb = {maxnnb}")
    print(f"  - Splines computed for {n_spline_types} types")

    if n_spline_types > n_actual:
        print(f"\n✗ BUG CONFIRMED!")
        print(f"  You're computing splines for {n_spline_types - n_actual} EXTRA types!")
        print(f"  These have zero/garbage parameters and contribute wrong energies.")

    if not elec_ok:
        print(f"\n✗ ELECTROSTATIC BUG!")
        print(f"  Electrostatic array is all zeros")
        print(f"  This is why eelec = 0 in your output")

    # THE FIX
    print("\n" + "="*80)
    print("THE FIX")
    print("="*80)

    print(f"\nThere are TWO bugs to fix:")

    print(f"\n1. SPLINE LOOP RANGE:")
    print(f"   Current: for innb in range({n_spline_types})  # 0 to {n_spline_types-1}")
    print(f"   Should be: for innb in range({n_actual})  # 0 to {n_actual-1}")
    print(f"   OR match Fortran: for innb in range(1, {n_actual+1})  # 1 to {n_actual}")

    if not elec_ok:
        print(f"\n2. ELECTROSTATIC ARRAY:")
        print(f"   The electrostatic array is not being populated!")
        print(f"   Check force field loading code")
        print(f"   Should have: electrostatic[itype, 0] = 332.08 * q[i] * q[j]")

    print("\n" + "="*80)

    return True


def apply_quick_fix(state):
    """
    Quick fix: Zero out spline coefficients for invalid types
    """
    print("\n" + "="*80)
    print("APPLYING QUICK FIX")
    print("="*80)

    if not hasattr(state.ff, 'typee') or not hasattr(state, 'splines'):
        print("  ERROR: Cannot apply fix - missing data")
        return False

    # Find actual number of types
    typee = state.ff.typee
    unique_types = np.unique(typee[typee >= 0])
    n_actual = len(unique_types)
    max_type = np.max(unique_types)

    coefft = state.splines.coefft
    coeffft = state.splines.coeffft

    print(f"  Actual interaction types: {n_actual} (indices 0 to {max_type})")
    print(f"  Current spline types: {coefft.shape[2]}")

    # Zero out coefficients beyond max_type
    if max_type + 1 < coefft.shape[2]:
        n_zero = coefft.shape[2] - (max_type + 1)
        print(f"  Zeroing out {n_zero} invalid spline types...")

        coefft[:, :, max_type+1:] = 0.0
        coeffft[:, :, max_type+1:] = 0.0

        print(f"  ✓ Fix applied!")
        print(f"  Now only types 0 to {max_type} have non-zero splines")
    else:
        print(f"  ✓ Spline range already correct")

    print("="*80)
    return True


# Usage: Replace the initialize_jikdummy function in main.py with this version

def setup_system(state: SimulationState) -> None:
    """
    Complete system setup after reading input files.
    """
    nat = state.atoms.nat
    
    # Calculate total charge
    total_charge = 0.0
    for iat in range(nat):
        itype = state.atoms.atomtype[iat]
        total_charge += state.ff.q[itype]
    print(f"Total charge = {total_charge:.6f}")
    
    # Count real atoms
    state.atoms.natreal = nat - state.atoms.ndummy
    
    # Initialize dummy atom connectivity (jikdummy) from bend data
    if state.atoms.ndummy > 0:
        print(f"Initializing connectivity for {state.atoms.ndummy} dummy atoms...")
        initialize_jikdummy(state)
    # Initialize dummy atom coordinates BEFORE any force/neighbor calculations
    update_dummy_coords(state)

    
    # Degrees of freedom - PRELIMINARY (will be updated after constraints)
    # For now, just set it temporarily - will be corrected after constraint initialization
    state.ndof = 3 * state.atoms.natreal  # Temporary, will subtract nconst later
    
    # Initialize neighbor lists
    print("Building neighbor lists...")
    initialize_neighbor_lists(state)
   
    # Compute splines for potential 
    compute_spline_coefficients(state)
    # NOTE: typee is already 0-indexed from input_parser.py. Do NOT subtract 1.

    # Initialize constraints and degrees of freedom
    from initialize_constraints import initialize_constraints
    initialize_constraints(state)
    
    if state.params.constrain and hasattr(state.constraints, 'nbcon'):
        nconst = state.constraints.nbcon
    else:
        nconst = 0
    state.ndof = 3 * state.atoms.natreal - nconst
    
    # Initialize polarization (if APPLE&P force field)
    if state.ff.lpolarizable:
        from polarization import init_polarization
        init_polarization(state)

    # Install GPU acceleration for the recip E-field if a CUDA device is present.
    # If not, this silently sets state.gpu_ewald_recip = None and the polarization
    # module falls back to the JIT-compiled CPU version.
    try:
        from ewald_gpu import install_gpu_recip
        install_gpu_recip(state)
    except ImportError:
        state.gpu_ewald_recip = None

    # Install GPU pair builder (Stage 1 of polarization GPU port).
    # Falls back to CPU JIT if unavailable. Set GOMD_GPU_PAIR=0 to disable.
    try:
        from pair_build_gpu import install_gpu_pair_build
        install_gpu_pair_build(state)
        if state.gpu_pair_builder is not None:
            # Initial refresh now that init_polarization has set ps.c1
            state.gpu_pair_builder.refresh_static(state)
    except ImportError:
        state.gpu_pair_builder = None

    # Compute initial forces (with dummy redistribution)
    state.energy = compute_all_forces(state)
    do_dummy_forces(state)
    
    # Initial kinetic energy and temperature
    state.energy.kinen = compute_kinetic(
        state.atoms.v, state.atoms.mass, nat, state.atoms.ldummy
    )
    state.energy.toten = state.energy.kinen + state.energy.poten
    state.temp = calculate_temperature(state)
    
    # Print initial state summary
    print(f"\n  Degrees of freedom:  {state.ndof}")
    print(f"  Bond energy:         {state.energy.ebond:12.4f} kcal/mol")
    print(f"  Bend energy:         {state.energy.ebend:12.4f} kcal/mol")
    print(f"  Torsion energy:      {state.energy.etort:12.4f} kcal/mol")
    if hasattr(state.energy, 'evdw'):
        print(f"  van der Waals:       {state.energy.evdw:12.4f} kcal/mol")
    if hasattr(state.energy, 'eelec'):
        print(f"  Electrostatic:       {state.energy.eelec:12.4f} kcal/mol")
    if hasattr(state.energy, 'epol') and state.energy.epol != 0.0:
        print(f"  Polarization:        {state.energy.epol:12.4f} kcal/mol")
    print(f"  Potential energy:    {state.energy.poten:12.4f} kcal/mol")
    print(f"  Kinetic energy:      {state.energy.kinen:12.4f} kcal/mol")
    print(f"  Total energy:        {state.energy.toten:12.4f} kcal/mol")
    print(f"  Temperature:         {state.temp:12.2f} K")
    
    # Initialize thermostat and integration
    init_thermostat(state)
    init_integration(state)
    
    # Initialize pressure
    vol = np.prod(state.atoms.box)
    volinv = 1.0 / vol
    state.pressure.pke = 45.42 * (state.ndof - 3) * state.temp * volinv
    state.pressure.pres = state.pressure.pke
    
    for kk in range(3):
        akin2_kk = 0.0
        for iat in range(nat):
            if not state.atoms.ldummy[iat]:
                akin2_kk += state.atoms.mass[iat] * state.atoms.v[kk, iat]**2
        state.pressure.akin2[kk] = akin2_kk / (4184.0 * 1000.0)
        state.pressure.prt[kk] = 6.8571e4 * state.pressure.tvirpo[kk, kk] / vol


def run_simulation(state: SimulationState) -> None:
    """
    Main simulation loop.
    """
    nsteps = state.params.nsteps
    delt = state.params.delt
    multibig = state.params.multibig
    stepout = delt * multibig
    
    # Output frequencies (in steps)
    klog = 20                                                    # md.log every 20 steps
    kave = max(1, int(state.params.nave / stepout))
    koutput = max(1, int(state.params.noutput / stepout))

    # ============================================================
    # Trajectory output frequency  (FORCEFUL DEFAULT: 1000 steps)
    # ============================================================
    # The mdrun.params file historically has TWO fields that look like
    # trajectory frequency:
    #   * Section 5 "Coord Output(fs)"        -> params.noutput
    #   * Section 7 "Coordinate Time"         -> params.ncoords
    # Users naturally edit section 5, but section 7 is what gets used.
    # We avoid the confusion by:
    #   1) Using an explicit override if state.params.kcoords_steps is set
    #   2) Otherwise enforcing a hard floor of 1000 steps (~500 fs at dt=0.5)
    #      regardless of what the params file says.
    # If you genuinely want trajectory every <1000 steps, set
    # state.params.kcoords_steps explicitly.
    DEFAULT_KCOORDS_STEPS = 1000
    kcoords_override = getattr(state.params, 'kcoords_steps', None)
    if kcoords_override:
        kcoords = int(kcoords_override)
        kcoords_source = "kcoords_steps override"
    else:
        # Take the larger of noutput/ncoords (whichever the user actually edited),
        # convert to steps, then enforce floor.
        ncoords_fs = max(state.params.ncoords, state.params.noutput)
        kcoords_from_fs = int(ncoords_fs / stepout)
        kcoords = max(DEFAULT_KCOORDS_STEPS, kcoords_from_fs)
        kcoords_source = (f"max(noutput={state.params.noutput} fs, "
                          f"ncoords={state.params.ncoords} fs) -> "
                          f"{kcoords_from_fs} steps; floor={DEFAULT_KCOORDS_STEPS}")

    kvelocs = max(1, int(state.params.nvelocs / stepout))
    kinit = max(1, int(state.params.ninit / stepout))
    
    # Initialize output writers
    md_log = MDLogWriter("md.log", full_tensor=state.params.lstress)
    md_log.open()
    traj_writer = TrajectoryWriter("trajectory.xyz")
    
    if state.params.lvelocs:
        vel_writer = VelocityWriter("velocities.dat")
    else:
        vel_writer = None
    
    # Displacement tracking for neighbor list update
    delxup = np.zeros((3, state.atoms.nat))
    x_old = state.atoms.x.copy()
    
    print("\n" + "=" * 60)
    print("Starting MD simulation")
    print(f"  Steps: {nsteps}, dt = {delt} fs")
    print(f"  Output: md.log every {klog} steps,"
          f" trajectory every {kcoords} steps,"
          f" velocities every {kvelocs} steps")
    print(f"  trajectory source: {kcoords_source}")
    if state.params.multibig > 1:
        outer_dt = delt * state.params.multibig
        print(f"  [RESPA] multibig=K={state.params.multibig}: "
              f"slow forces (Ewald recip + polarization) every {state.params.multibig} steps "
              f"(outer dt = {outer_dt} fs)")
    else:
        print(f"  Integrator: vanilla velocity-Verlet (multibig=1)")
    print("=" * 60 + "\n")
    
    start_time = time.time()

    # Per-step timing instrumentation
    timer_phases = {
        'neighbor':   0.0,
        'pressure':   0.0,
        'integrate':  0.0,   # full integrate_step
        'forces':     0.0,   # subset of integrate: just compute_all_forces
        'kinetics':   0.0,
        'displ_track': 0.0,
        'output':     0.0,
    }
    timer_n = 0
    scf_iter_sum = 0
    scf_iter_n = 0
    pol_scf_time_sum = 0.0
    pol_force_time_sum = 0.0
    pair_build_time_sum = 0.0
    bonded_time_sum = 0.0
    nonbonded_time_sum = 0.0
    ewald_recip_time_sum = 0.0

    # Main dynamics loop
    while state.kount <= nsteps:
        t_step_start = time.time()

        # Rebuild neighbor list if needed
        if state.update:
            update_dummy_coords(state)
            build_cell_list(state)
            # Refresh GPU pair builder's static data (neighbor list changed)
            if hasattr(state, 'gpu_pair_builder') and state.gpu_pair_builder is not None:
                state.gpu_pair_builder.refresh_static(state)
            state.update = False
            delxup.fill(0.0)
            x_old = state.atoms.x.copy()
        timer_phases['neighbor'] += time.time() - t_step_start
        t_phase = time.time()
        
        # Directional pressure components for NPT (vectorized)
        vol = np.prod(state.atoms.box)
        not_dummy = ~state.atoms.ldummy  # bool mask, shape (nat,)
        v_real = state.atoms.v[:, not_dummy]              # (3, nreal)
        m_real = state.atoms.mass[not_dummy]              # (nreal,)
        # akin2[kk] = sum_i m_i * v[kk,i]^2 / (4184 * 1000)
        state.pressure.akin2[:] = (m_real * v_real**2).sum(axis=1) / 4184000.0
        state.pressure.prt[:] = 6.8571e4 * np.diag(state.pressure.tvirpo) / vol
        timer_phases['pressure'] += time.time() - t_phase
        t_phase = time.time()

        # Complete integration step (thermostat + VV + forces + thermostat)
        # This is the bulk of per-step cost: forces (incl. polarization SCF)
        integrate_step(state)
        timer_phases['integrate'] += time.time() - t_phase
        if hasattr(state, '_t_forces_last'):
            timer_phases['forces'] += state._t_forces_last
        if hasattr(state, '_last_scf_niter'):
            scf_iter_sum += state._last_scf_niter
            scf_iter_n += 1
        if hasattr(state, '_t_scf_iteration'):
            pol_scf_time_sum += state._t_scf_iteration
        if hasattr(state, '_t_pol_forces'):
            pol_force_time_sum += state._t_pol_forces
        if hasattr(state, '_t_pair_build'):
            pair_build_time_sum += state._t_pair_build
        if hasattr(state, '_t_bonded'):
            bonded_time_sum += state._t_bonded
        if hasattr(state, '_t_nonbonded'):
            nonbonded_time_sum += state._t_nonbonded
        if hasattr(state, '_t_ewald_recip'):
            ewald_recip_time_sum += state._t_ewald_recip
        t_phase = time.time()
        
        # Remove COM velocity and zero dummy velocities (vectorized)
        remove_com_velocity(state)
        state.atoms.v[:, state.atoms.ldummy] = 0.0
        
        # Calculate properties
        state.energy.kinen = compute_kinetic(
            state.atoms.v, state.atoms.mass,
            state.atoms.nat, state.atoms.ldummy
        )
        state.energy.toten = state.energy.kinen + state.energy.poten
        state.temp = calculate_temperature(state)

        # Scalar pressure
        vol = np.prod(state.atoms.box)
        volinv = 1.0 / vol
        state.pressure.pke = 45.42 * (state.ndof - 3) * state.temp * volinv
        state.pressure.pvir = 2.2857e4 * state.energy.vir * volinv
        state.pressure.pres = state.pressure.pke + state.pressure.pvir + state.pressure.ptrunc
        timer_phases['kinetics'] += time.time() - t_phase
        t_phase = time.time()

        # Track displacements for neighbor list update (vectorized minimum image)
        dx = state.atoms.x - x_old
        # Wrap each component to [-box/2, +box/2]
        for kk in range(3):
            mask = np.abs(dx[kk]) > state.atoms.el[kk]
            dx[kk, mask] -= np.sign(dx[kk, mask]) * state.atoms.box[kk]
        delxup += dx
        x_old = state.atoms.x.copy()
        if np.max(np.sqrt(np.sum(delxup**2, axis=0))) > state.params.driftmax:
            state.update = True
        timer_phases['displ_track'] += time.time() - t_phase
        t_phase = time.time()

        # Current time
        time_fs = state.kount * stepout

        # Write md.log every klog steps
        if state.kount % klog == 0:
            md_log.write(state, time_fs)
        
        # Write trajectory
        if state.params.lcoords and state.kount % kcoords == 0:
            traj_writer.write_frame(state, time_fs)
        
        # Write velocities
        if state.params.lvelocs and vel_writer and state.kount % kvelocs == 0:
            vel_writer.write_frame(state, time_fs)
        
        # Console progress (every koutput steps)
        if state.params.printout and state.kount % koutput == 0:
            print_step_info(state, time_fs)

        timer_phases['output'] += time.time() - t_phase
        timer_n += 1

        # Per-step timing breakdown printed every klog steps after step 0
        # (step 0 includes JIT warmup; skip it).
        if state.kount > 0 and state.kount % klog == 0:
            phases = timer_phases
            n = max(timer_n, 1)
            total_ms = sum(phases.values()) / n * 1000  # avg total per step (excludes 'forces' which is subset)
            total_ms = (phases['neighbor']+phases['pressure']+phases['integrate']
                        +phases['kinetics']+phases['displ_track']+phases['output']) / n * 1000
            scf_avg = (scf_iter_sum / scf_iter_n) if scf_iter_n > 0 else 0.0
            scf_t_ms = (pol_scf_time_sum / n) * 1000
            polf_t_ms = (pol_force_time_sum / n) * 1000
            pair_t_ms = (pair_build_time_sum / n) * 1000
            bond_t_ms = (bonded_time_sum / n) * 1000
            nb_t_ms = (nonbonded_time_sum / n) * 1000
            erecip_t_ms = (ewald_recip_time_sum / n) * 1000
            forces_ms = phases['forces']/n*1000
            unaccounted = forces_ms - scf_t_ms - polf_t_ms - pair_t_ms - bond_t_ms - nb_t_ms - erecip_t_ms
            print(f"  [t/step {n}]  total={total_ms:5.0f}  forces={forces_ms:5.0f}")
            print(f"             bonded={bond_t_ms:4.0f}  NB_splined={nb_t_ms:4.0f}  "
                  f"ewald_recip_F={erecip_t_ms:4.0f}")
            print(f"             POL: pair_build={pair_t_ms:4.0f}  "
                  f"SCF={scf_t_ms:4.0f}@{scf_avg:.1f}it  polF={polf_t_ms:4.0f}  "
                  f"unaccounted={unaccounted:.0f}")
            print(f"             neigh={phases['neighbor']/n*1000:4.0f}  "
                  f"misc={(phases['pressure']+phases['kinetics']+phases['displ_track']+phases['output'])/n*1000:4.0f}")
            for k in timer_phases:
                timer_phases[k] = 0.0
            timer_n = 0
            scf_iter_sum = 0
            scf_iter_n = 0
            pol_scf_time_sum = 0.0
            pol_force_time_sum = 0.0
            pair_build_time_sum = 0.0
            bonded_time_sum = 0.0
            nonbonded_time_sum = 0.0
            ewald_recip_time_sum = 0.0

    # Close output files
    md_log.close()
    traj_writer.close()
    if vel_writer:
        vel_writer.close()
    
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Simulation completed")
    print(f"Total steps: {state.kount}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if state.kount > 0:
        print(f"Performance: {state.kount / elapsed_time:.2f} steps/second")
        print(f"             {elapsed_time / state.kount * 1000:.1f} ms/step")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    # Clear stale Numba cache to prevent using outdated compiled code
    import shutil, os
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Cleared __pycache__ (Numba recompilation will occur)")

    print("=" * 60)
    print("GOMD Python Version")
    print("Molecular Dynamics Simulation")
    print("=" * 60)
    
    # Get input directory
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "."
    
    print(f"\nInput directory: {input_dir}")
    
    # Load simulation
   
    state = load_simulation(input_dir,DEBUG_INPUT)

    # Setup
    print("\nSetting up system...")
    setup_system(state)
    
    # Run
    run_simulation(state)
    
    # Write final output
    write_restart(state, "restart.inp")
    write_final_output(state, "final_output.dat")
    
    print("Simulation finished successfully!")


if __name__ == "__main__":
    main()
