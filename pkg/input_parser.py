"""
Input File Parsers for MD Simulation
Preserves the original Fortran input file formats
"""

import numpy as np
from pathlib import Path
from typing import TextIO, Optional, Tuple, List
from datatypes import (
    SystemParams, ForceFieldParams, Connectivity, AtomData,
    ThermostatState, BarostatState, SimulationState, TempRampData
)
from constants import DefaultParams, MAXNNB


def read_a_line(f: TextIO) -> str:
    """
    Read next non-comment line from file.
    Comments start with '*'.
    Equivalent to Fortran read_a_line subroutine.
    """
    while True:
        line = f.readline()
        if not line:  # EOF
            raise EOFError(f"Unexpected end of file")
        if not line.strip().startswith('*'):
            return line.rstrip()


def parse_line(line: str) -> List[str]:
    """Parse a line into whitespace-separated tokens."""
    return line.split()


def read_mdrun_params(filename: str = "mdrun.params") -> SystemParams:
    """
    Read simulation parameters from mdrun.params (Fortran unit 25).
    Preserves the exact input format of the original code.
    """
    params = SystemParams()
    
    with open(filename, 'r') as f:
        # Line 1: Ensemble selection
        line = read_a_line(f)
        tokens = parse_line(line)
        params.nve = tokens[0].upper() in ['T', 'TRUE', '.TRUE.']
        params.nvt = tokens[1].upper() in ['T', 'TRUE', '.TRUE.']
        params.npt = tokens[2].upper() in ['T', 'TRUE', '.TRUE.']
        params.lprofile = tokens[3].upper() in ['T', 'TRUE', '.TRUE.'] if len(tokens) > 3 else False
        
        # Validate ensemble
        ensemble_count = sum([params.nve, params.nvt, params.npt])
        if ensemble_count > 1:
            raise ValueError("Error: Multiple ensembles selected")
        if ensemble_count == 0:
            raise ValueError("Error: No ensemble specified")
        
        # Line 2: Cutoff, drift, timestep, steps
        line = read_a_line(f)
        tokens = parse_line(line)
        params.rcut = float(tokens[0])
        params.driftmax = float(tokens[1])
        params.delt = float(tokens[2])
        params.nsteps = int(tokens[3])
        
        # Line 3: Temperature control
        line = read_a_line(f)
        tokens = parse_line(line)
        params.tstart = float(tokens[0])
        params.nnos = int(tokens[1])
        params.qfreq = float(tokens[2])
        
        # Line 4: Pressure control
        line = read_a_line(f)
        tokens = parse_line(line)
        params.pfix = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        box_const = [tokens[3].upper() in ['T', 'TRUE', '.TRUE.'],
                     tokens[4].upper() in ['T', 'TRUE', '.TRUE.'],
                     tokens[5].upper() in ['T', 'TRUE', '.TRUE.']]
        press_avg = [tokens[6].upper() in ['T', 'TRUE', '.TRUE.'],
                     tokens[7].upper() in ['T', 'TRUE', '.TRUE.'],
                     tokens[8].upper() in ['T', 'TRUE', '.TRUE.']]
        params.wfreq = float(tokens[9])
        
        if params.npt:
            params.box_mult = np.array([1 if bc else 0 for bc in box_const])
            params.avg_mult = np.array([1 if pa else 0 for pa in press_avg])
            params.bUsePressure = all(press_avg)
        
        # Line 5: Polarization parameters
        line = read_a_line(f)
        tokens = parse_line(line)
        params.tbpol = tokens[0].upper() in ['T', 'TRUE', '.TRUE.']
        params.rop = float(tokens[1])
        params.rscalep = float(tokens[2])
        params.epsilonp = float(tokens[3])
        params.repsilonp = float(tokens[4])
        
        # Line 6: Output frequencies
        line = read_a_line(f)
        tokens = parse_line(line)
        params.ninit = int(float(tokens[0]))
        params.nave = float(tokens[1])
        params.nvir = float(tokens[2])
        params.nnben = float(tokens[3])
        params.noutput = float(tokens[4])
        
        # Line 7: Output flags
        line = read_a_line(f)
        tokens = parse_line(line)
        params.lboxdip = tokens[0].upper() in ['T', 'TRUE', '.TRUE.']
        params.lcoords = tokens[1].upper() in ['T', 'TRUE', '.TRUE.']
        params.lstress = tokens[2].upper() in ['T', 'TRUE', '.TRUE.']
        params.lvelocs = tokens[3].upper() in ['T', 'TRUE', '.TRUE.']
        
        # Line 8: Output frequencies for coordinates
        line = read_a_line(f)
        tokens = parse_line(line)
        params.nboxdip = float(tokens[0])
        params.ncoords = float(tokens[1])
        params.nstress = float(tokens[2])
        params.nvelocs = float(tokens[3])
        
        # Line 9: Box scaling
        line = read_a_line(f)
        tokens = parse_line(line)
        params.chvol = tokens[0].upper() in ['T', 'TRUE', '.TRUE.']
        params.boxnew = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        params.printout = tokens[4].upper() in ['T', 'TRUE', '.TRUE.']
        
        # Line 10: Interaction flags
        line = read_a_line(f)
        tokens = parse_line(line)
        params.inter = tokens[0].upper() in ['T', 'TRUE', '.TRUE.']
        params.ex14 = tokens[1].upper() in ['T', 'TRUE', '.TRUE.']
        params.fixtemp = tokens[2].upper() in ['T', 'TRUE', '.TRUE.']
        
        # Line 11: 1-4 reduction
        line = read_a_line(f)
        tokens = parse_line(line)
        params.lredonefour = tokens[0].upper() in ['T', 'TRUE', '.TRUE.']
        params.redfactor = float(tokens[1])
        params.lredQ_mu14 = tokens[2].upper() in ['T', 'TRUE', '.TRUE.']
        params.redQmufactor = float(tokens[3])
        
        # Line 12: Constraints
        line = read_a_line(f)
        tokens = parse_line(line)
        params.constrain = tokens[0].upper() in ['T', 'TRUE', '.TRUE.']
        params.tol = float(tokens[1])
        
        # Line 13: Electrostatics (with label)
        line = read_a_line(f)
        # First 12 chars are label
        rest = line[12:].split()
        params.alphai = float(rest[0])
        params.klimitx = int(rest[1])
        params.klimity = int(rest[2])
        params.klimitz = int(rest[3])
        params.lewald = rest[4].upper() in ['T', 'TRUE', '.TRUE.']
        
        # Line 14: Integrator (with label)
        line = read_a_line(f)
        rest = line[12:].split()
        params.multimed = int(rest[0])
        params.multibig = int(rest[1])
        params.ros = float(rest[2])
        
        # Line 15: Polarization convergence (with label)
        line = read_a_line(f)
        rest = line[22:].split()
        params.tolpol = float(rest[0])
        params.tapering = rest[1].upper() in ['T', 'TRUE', '.TRUE.']
        params.rtaper = float(rest[2])
        params.epsrf = float(rest[3])
    
    print(f"File: mdrun.params has been read")
    return params


def read_forcefield(filename: str = "ff.dat") -> ForceFieldParams:
    """
    Read force field parameters from ff.dat (Fortran unit 12).
    """
    ff = ForceFieldParams()
    
    # Initialize arrays
    ff.nonbonded = np.zeros((3, MAXNNB))  # FIXED: (3 parameters, MAXNNB types)
    ff.electrostatic = np.zeros((MAXNNB, 2))
    ff.atmasses = np.zeros(DefaultParams.maxtypes)
    ff.q = np.zeros(DefaultParams.maxcharges)
    ff.pol = np.zeros(DefaultParams.maxcharges)
    ff.non_of_charge = np.zeros(DefaultParams.maxcharges, dtype=int)
    ff.lonepairtype = np.zeros(DefaultParams.maxcharges, dtype=bool)
    ff.typen = np.zeros((DefaultParams.maxtypes, DefaultParams.maxtypes), dtype=int)
    ff.typee = np.zeros((DefaultParams.maxcharges, DefaultParams.maxcharges), dtype=int)
    ff.map_types = np.zeros(MAXNNB, dtype=int)
    ff.stretch = np.zeros((DefaultParams.maxbtypes, 5))
    ff.bend = np.zeros((DefaultParams.maxbtypes, 4))
    ff.linear = np.zeros(DefaultParams.maxbtypes, dtype=bool)
    ff.twist = np.zeros((DefaultParams.maxfolds + 1, DefaultParams.maxbtypes))
    ff.nprms = np.zeros(DefaultParams.maxbtypes, dtype=int)
    ff.deform = np.zeros(DefaultParams.maxbtypes)
    ff.typedummy = np.zeros(DefaultParams.maxcharges, dtype=int)
    ff.adummy = np.zeros(DefaultParams.maxcharges)
    ff.bdummy = np.zeros(DefaultParams.maxcharges)
    ff.cdummy = np.zeros(DefaultParams.maxcharges)
    ff.LpGeomType = np.zeros(DefaultParams.maxcharges, dtype=int)
    ff.LpBendAtomType = np.zeros((3, DefaultParams.maxcharges), dtype=int)
    
    with open(filename, 'r') as f:
        # Number of non-bonded types
        line = read_a_line(f)
        number = int(line.split()[0])
        ff.num_nbtypes = number
        
        # Assign type pairs
        intert = 0
        for itype in range(number):
            for jtype in range(itype, number):
                ff.typen[itype, jtype] = intert
                ff.typen[jtype, itype] = intert
                intert += 1
        ff.number_nnb = intert
        

        # Read non-bonded self terms
        ff.non_labels = []
        line = None  # Track if we've already read next line
        
        for itype in range(number):
            # Read self-term line (unless we already have it from previous iteration)
            if line is None:
                line = read_a_line(f)
            
            label = line[:3].strip()
            ff.non_labels.append(label)
            rest = line[3:].split()
            a, b, c, d, remass = float(rest[0]), float(rest[1]), float(rest[2]), float(rest[3]), float(rest[4])
            
            intert = ff.typen[itype, itype]
            ff.nonbonded[0, intert] = a
            ff.nonbonded[1, intert] = b
            ff.nonbonded[2, intert] = c
            ff.electrostatic[intert, 1] = -d * 2.0
            ff.atmasses[itype] = remass
            
            # Read charge types for this non-bonded type
            line = read_a_line(f)
            nq = 0
            
            # If line doesn't start with space, no charges given
            if not line or line[0] != ' ':
                # No charges, create one with zero charge
                nq = 1
                ff.ncharge += 1
                ff.q[ff.ncharge - 1] = 0.0
                ff.pol[ff.ncharge - 1] = 0.0
                ff.atom_labels.append(f"{label}{nq:2d}")
                ff.non_of_charge[ff.ncharge - 1] = itype
                # line already contains next self-term, keep it for next iteration
            else:
                # Read all charge lines (start with space)
                while line and line[0] == ' ':
                    nq += 1
                    ff.ncharge += 1
                    ff.lonepairtype[ff.ncharge - 1] = (label.lower() == 'lp')
                    ff.atom_labels.append(f"{label}{nq:2d}")
                    
                    parts = line[1:].split()
                    ff.q[ff.ncharge - 1] = float(parts[0])
                    ff.pol[ff.ncharge - 1] = float(parts[1])
                    
                    if ff.pol[ff.ncharge - 1] > 1.0e-4:
                        ff.lpolarizable = True
                    
                    ff.non_of_charge[ff.ncharge - 1] = itype
                    
                    # Read next line
                    line = read_a_line(f)
                # After while loop, 'line' contains next self-term or first cross-term
                # Keep it for next iteration
        
        # At this point, 'line' should contain first cross-term or '!'
        # If it's a self-term instead, we have a bug, but let's continue
        
        # Default off-diagonal terms (geometric/arithmetic mixing)
        intert = 0
        for itype in range(number):
            iinter = ff.typen[itype, itype]
            for jtype in range(itype, number):
                jinter = ff.typen[jtype, jtype]
                intert = ff.typen[itype, jtype]  # FIXED: Use typen mapping, not sequential
                a = np.sqrt(ff.nonbonded[0, iinter] * ff.nonbonded[0, jinter])
                b = 0.5 * (ff.nonbonded[1, iinter] + ff.nonbonded[1, jinter])
                c = np.sqrt(ff.nonbonded[2, iinter] * ff.nonbonded[2, jinter])
                ff.nonbonded[0, intert] = a
                ff.nonbonded[1, intert] = b
                ff.nonbonded[2, intert] = c
        
        # Read explicit cross-terms (overwrites mixing rules)
        # 'line' should already contain first cross-term or '!'
        while not line.startswith('!'):
            label1 = line[:3].strip()
            itype = None
            for i, lab in enumerate(ff.non_labels):
                if lab == label1:
                    itype = i
                    break
            
            if itype is not None:
                label2 = line[4:7].strip()
                jtype = None
                for j in range(itype + 1, number):
                    if ff.non_labels[j] == label2:
                        jtype = j
                        break
                
                if jtype is not None:
                    rest = line[7:].split()
                    a, b, c, d = float(rest[0]), float(rest[1]), float(rest[2]), float(rest[3])
                    intert = ff.typen[itype, jtype]
                    ff.nonbonded[0, intert] = a
                    ff.nonbonded[1, intert] = b
                    ff.nonbonded[2, intert] = c
                    ff.electrostatic[intert, 1] = -d * 2.0
            
            line = read_a_line(f)
 
        # Build charge-charge interaction types
        intert = 0
        for itype in range(ff.ncharge):
            inontype = ff.non_of_charge[itype]
            for jtype in range(itype, ff.ncharge):
                jnontype = ff.non_of_charge[jtype]
                ff.typee[itype, jtype] = intert
                ff.typee[jtype, itype] = intert
                inb = ff.typen[inontype, jnontype]
                ff.map_types[intert] = inb
                intert += 1
        
        # Set electrostatic interactions
        for irow in range(ff.ncharge):
            for icol in range(ff.ncharge):
                itype = ff.typee[irow, icol]
                ff.electrostatic[itype, 0] = 332.08 * ff.q[irow] * ff.q[icol]
        
        # Read bond types
        line = read_a_line(f)
        ff.nbondt = int(line.split()[0])
        for ibond in range(ff.nbondt):
            line = read_a_line(f)
            rest = line[11:].split()
            ff.stretch[ibond, 0] = float(rest[0])  # k
            ff.stretch[ibond, 1] = float(rest[1])  # r0
            ff.stretch[ibond, 2] = float(rest[2])  # r0^2 or other param
        
        # Read bend types
        line = read_a_line(f)
        ff.nbendt = int(line.split()[0])
        for ibend in range(ff.nbendt):
            line = read_a_line(f)
            rest = line[15:].split()
            ff.bend[ibend, 0] = float(rest[0])    # k
            ff.bend[ibend, 1] = float(rest[1])    # theta0
            ff.linear[ibend] = rest[2].upper() in ['T', 'TRUE', '.TRUE.']
        
        # Read torsion types
        line = read_a_line(f)
        ff.ntortt = int(line.split()[0])
        for itort in range(ff.ntortt):
            line = read_a_line(f)
            rest = line[20:].split()
            n2 = int(rest[0])
            ff.nprms[itort] = n2
            for ii in range(n2):
                ff.twist[ii, itort] = float(rest[ii + 1])
        
        # Read deformation (improper) types
        line = read_a_line(f)
        ff.ndeformt = int(line.split()[0])
        for ideform in range(ff.ndeformt):
            line = read_a_line(f)
            rest = line[20:].split()
            ff.deform[ideform] = float(rest[0])
        
        # Read dummy atom types
        line = read_a_line(f)
        ff.ndummytypes = int(line.split()[0])
        for itype in range(ff.ndummytypes):
            line = read_a_line(f)
            rest = line[20:].split()
            i = int(rest[0])
            ff.adummy[itype] = float(rest[1])
            ff.bdummy[itype] = float(rest[2])
            ff.cdummy[itype] = float(rest[3])
            ff.LpGeomType[itype] = int(rest[4])
            ff.LpBendAtomType[0, itype] = int(rest[5])
            ff.LpBendAtomType[1, itype] = int(rest[6])
            ff.LpBendAtomType[2, itype] = int(rest[7])
            ff.typedummy[i-1] = itype + 1  # 1-indexed in Fortran
    
    # CRITICAL: Create alias so code referencing ff.map finds ff.map_types
    ff.map = ff.map_types
    
    print(f"File: ff.dat has been read")
    return ff


def read_connectivity(filename: str = "connectivity.dat") -> Connectivity:
    """
    Read molecular connectivity from connectivity.dat (Fortran unit 11).
    """
    conn = Connectivity()
    
    with open(filename, 'r') as f:
        # Number of chains and ions
        line = read_a_line(f)
        tokens = parse_line(line)
        nchains = int(tokens[0])
        conn.nions = int(tokens[1])
        conn.nch = nchains + conn.nions
        
        # Read chain information
        natch = []
        nbondsch = []
        nbendsch = []
        ntortsch = []
        ndeformsch = []
        
        for ich in range(conn.nch):
            line = read_a_line(f)
            tokens = parse_line(line)
            natch.append(int(tokens[0]))
            nbondsch.append(int(tokens[1]))
            nbendsch.append(int(tokens[2]))
            ntortsch.append(int(tokens[3]))
            ndeformsch.append(int(tokens[4]))
        
        conn.nat = sum(natch)
        conn.nbonds = sum(nbondsch)
        conn.nbends = sum(nbendsch)
        conn.ntorts = sum(ntortsch)
        conn.ndeforms = sum(ndeformsch)
        
        # Allocate arrays
        conn.bonds = np.zeros((3, conn.nbonds), dtype=int)
        conn.bends = np.zeros((6, conn.nbends), dtype=int)
        conn.torts = np.zeros((10, conn.ntorts), dtype=int)
        conn.deforms = np.zeros((8, conn.ndeforms), dtype=int)
        conn.chain = np.zeros(conn.nat, dtype=int)
        
        # Read bonds
        itbond = 0
        for ich in range(conn.nch):
            if nbondsch[ich] == 0:
                line = read_a_line(f)
                iat = int(line.split()[0]) - 1  # Convert to 0-indexed
                conn.chain[iat] = ich
            
            for ibond in range(nbondsch[ich]):
                line = read_a_line(f)
                tokens = parse_line(line)
                conn.bonds[0, itbond] = int(tokens[0]) - 1  # atom1
                conn.bonds[1, itbond] = int(tokens[1]) - 1  # atom2
                conn.bonds[2, itbond] = int(tokens[2]) - 1  # type
                conn.chain[conn.bonds[0, itbond]] = ich
                conn.chain[conn.bonds[1, itbond]] = ich
                itbond += 1
        
        # Read bends
        itbend = 0
        for ich in range(conn.nch):
            for ibend in range(nbendsch[ich]):
                line = read_a_line(f)
                tokens = parse_line(line)
                conn.bends[0, itbend] = int(tokens[0]) - 1  # atom1
                conn.bends[1, itbend] = int(tokens[1]) - 1  # atom2
                conn.bends[2, itbend] = int(tokens[2]) - 1  # atom3
                conn.bends[3, itbend] = int(tokens[3]) - 1  # type
                itbend += 1
        
        # Build bond lookup dictionary: (min_atom, max_atom) -> bond_index
        bond_lookup = {}
        for ibond in range(conn.nbonds):
            a1, a2 = conn.bonds[0, ibond], conn.bonds[1, ibond]
            key = (min(a1, a2), max(a1, a2))
            bond_lookup[key] = ibond
        
        # Determine bonds for each bend using O(1) dict lookup
        for itbend in range(conn.nbends):
            ib1 = conn.bends[0, itbend]
            ib2 = conn.bends[1, itbend]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.bends[4, itbend] = imult * (ibond + 1)
            
            ib1 = conn.bends[1, itbend]
            ib2 = conn.bends[2, itbend]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.bends[5, itbend] = imult * (ibond + 1)
        
        # Read torsions
        ittort = 0
        for ich in range(conn.nch):
            for itort in range(ntortsch[ich]):
                line = read_a_line(f)
                tokens = parse_line(line)
                conn.torts[0, ittort] = int(tokens[0]) - 1  # atom1
                conn.torts[1, ittort] = int(tokens[1]) - 1  # atom2
                conn.torts[2, ittort] = int(tokens[2]) - 1  # atom3
                conn.torts[3, ittort] = int(tokens[3]) - 1  # atom4
                conn.torts[4, ittort] = int(tokens[4]) - 1  # type
                ittort += 1
        
        # Build bend lookup dictionary: (atom1, central, atom3) -> bend_index
        bend_lookup = {}
        for ibend in range(conn.nbends):
            a1 = conn.bends[0, ibend]
            a2 = conn.bends[1, ibend]  # central
            a3 = conn.bends[2, ibend]
            bend_lookup[(a1, a2, a3)] = ibend
            bend_lookup[(a3, a2, a1)] = ibend  # reversed key
        
        # Determine bonds and bends for each torsion using dict lookups
        for ittort in range(conn.ntorts):
            # First bond (1-2)
            ib1 = conn.torts[0, ittort]
            ib2 = conn.torts[1, ittort]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.torts[5, ittort] = imult * (ibond + 1)
            
            # Second bond (2-3)
            ib1 = conn.torts[1, ittort]
            ib2 = conn.torts[2, ittort]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.torts[6, ittort] = imult * (ibond + 1)
            
            # Third bond (3-4)
            ib1 = conn.torts[2, ittort]
            ib2 = conn.torts[3, ittort]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.torts[7, ittort] = imult * (ibond + 1)
            
            # First bend (1-2-3)
            a1 = conn.torts[0, ittort]
            a2 = conn.torts[1, ittort]
            a3 = conn.torts[2, ittort]
            ibend = bend_lookup.get((a1, a2, a3))
            if ibend is not None:
                # Check original ordering to determine sign
                if conn.bends[0, ibend] == a1 and conn.bends[2, ibend] == a3:
                    conn.torts[8, ittort] = ibend + 1
                else:
                    conn.torts[8, ittort] = -(ibend + 1)
            
            # Second bend (2-3-4)
            a1 = conn.torts[1, ittort]
            a2 = conn.torts[2, ittort]
            a3 = conn.torts[3, ittort]
            ibend = bend_lookup.get((a1, a2, a3))
            if ibend is not None:
                if conn.bends[0, ibend] == a1 and conn.bends[2, ibend] == a3:
                    conn.torts[9, ittort] = ibend + 1
                else:
                    conn.torts[9, ittort] = -(ibend + 1)
        
        # Read deformations
        itdefs = 0
        for ich in range(conn.nch):
            for ideform in range(ndeformsch[ich]):
                line = read_a_line(f)
                tokens = parse_line(line)
                conn.deforms[0, itdefs] = int(tokens[0]) - 1  # atom1
                conn.deforms[1, itdefs] = int(tokens[1]) - 1  # atom2
                conn.deforms[2, itdefs] = int(tokens[2]) - 1  # atom3
                conn.deforms[3, itdefs] = int(tokens[3]) - 1  # atom4
                conn.deforms[4, itdefs] = int(tokens[4]) - 1  # type
                itdefs += 1
        
        # Determine bonds for deformations using dict lookup
        for itdefs in range(conn.ndeforms):
            # Bond 1-2
            ib1, ib2 = conn.deforms[0, itdefs], conn.deforms[1, itdefs]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.deforms[5, itdefs] = imult * (ibond + 1)
            
            # Bond 2-3
            ib1, ib2 = conn.deforms[1, itdefs], conn.deforms[2, itdefs]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.deforms[6, itdefs] = imult * (ibond + 1)
            
            # Bond 2-4
            ib1, ib2 = conn.deforms[1, itdefs], conn.deforms[3, itdefs]
            imult = 1
            if ib2 < ib1:
                imult = -1
                ib1, ib2 = ib2, ib1
            ibond = bond_lookup.get((ib1, ib2))
            if ibond is not None:
                conn.deforms[7, itdefs] = imult * (ibond + 1)
    
    print(f"File: connectivity.dat has been read")
    print(f"  nat={conn.nat}, nbonds={conn.nbonds}, nbends={conn.nbends}, ntorts={conn.ntorts}, ndeforms={conn.ndeforms}")
    return conn


def read_coords(filename: str, nat: int, ff: ForceFieldParams, 
                nnos: int, lTempRamp: bool = False) -> Tuple[AtomData, np.ndarray, np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Read coordinates and velocities from coords.inp (Fortran unit 26).
    """
    atoms = AtomData()
    atoms.nat = nat
    atoms.x = np.zeros((3, nat))
    atoms.v = np.zeros((3, nat))
    atoms.accel = np.zeros((3, nat))
    atoms.f = np.zeros((3, nat))
    atoms.atomtype = np.zeros(nat, dtype=int)
    atoms.mass = np.zeros(nat)
    atoms.massinv = np.zeros(nat)
    atoms.ldummy = np.zeros(nat, dtype=bool)
    atoms.chain = np.zeros(nat, dtype=int)
    
    xlogs = np.zeros(10)
    vlogs = np.zeros(10)
    glogs = np.zeros(10)
    xlogv = np.zeros(3)
    vlogv = np.zeros(3)
    glogv = np.zeros(3)
    
    Tset = 0.0
    Tdelta = 0.0
    
    with open(filename, 'r') as f:
        # Check for temperature ramp continuation
        if lTempRamp:
            first_line = f.readline()
            if first_line.startswith('*!T'):
                parts = first_line.split()
                lContinue = parts[1].upper() in ['T', 'TRUE', '.TRUE.']
                if lContinue:
                    Tset = float(parts[2])
                    Tdelta = float(parts[3])
            else:
                f.seek(0)
        
        # Read atomic data
        shift = np.array([10000.0, 10000.0, 10000.0])
        
        for iat in range(nat):
            line = read_a_line(f)
            tokens = line.split()
            
            # Position
            atoms.x[0, iat] = float(tokens[0])
            atoms.x[1, iat] = float(tokens[1])
            atoms.x[2, iat] = float(tokens[2])
            
            for kk in range(3):
                if atoms.x[kk, iat] < shift[kk]:
                    shift[kk] = atoms.x[kk, iat]
            
            # Atom type label
            label = tokens[3][:3]
            icharge = int(tokens[4])
            check = f"{label}{icharge:2d}"
            
            # Find matching atom type
            found = False
            for itype, alabel in enumerate(ff.atom_labels):
                if check == alabel or check.replace(' ', '') == alabel.replace(' ', ''):
                    atoms.atomtype[iat] = itype
                    if ff.typedummy[itype] > 0:
                        atoms.ldummy[iat] = True
                        atoms.ndummy += 1
                    imap = ff.map_types[itype] if itype < len(ff.map_types) else ff.non_of_charge[itype]
                    atoms.mass[iat] = ff.atmasses[imap]
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Atom type not found: {check}")
            
            # Velocity
            line = read_a_line(f)
            tokens = line.split()
            atoms.v[0, iat] = float(tokens[0])
            atoms.v[1, iat] = float(tokens[1])
            atoms.v[2, iat] = float(tokens[2])
        
        # Read extended Lagrangian variables
        line = read_a_line(f)
        tokens = line.split()
        atoms.box = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        
        line = read_a_line(f)
        tokens = line.split()
        for inos in range(nnos):
            xlogs[inos] = float(tokens[inos])
        
        line = read_a_line(f)
        tokens = line.split()
        for inos in range(nnos):
            vlogs[inos] = float(tokens[inos])
        
        line = read_a_line(f)
        tokens = line.split()
        for inos in range(nnos):
            glogs[inos] = float(tokens[inos])
        
        line = read_a_line(f)
        tokens = line.split()
        xlogv = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        
        line = read_a_line(f)
        tokens = line.split()
        vlogv = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        
        line = read_a_line(f)
        tokens = line.split()
        glogv = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
    
    # Shift positions if necessary
    if np.any(shift < 0):
        print("Shifting positions")
        for iat in range(nat):
            atoms.x[:, iat] = atoms.x[:, iat] - shift + 1.0
    
    # Calculate inverse masses (avoid divide-by-zero for dummy atoms with mass=0)
    atoms.massinv = np.zeros_like(atoms.mass)
    mask = atoms.mass > 1.0e-5
    atoms.massinv[mask] = 1.0 / atoms.mass[mask]
    
    # Count real atoms
    atoms.natreal = nat - atoms.ndummy
    
    # Half box
    atoms.el = atoms.box * 0.5
    
    print(f"File: coords.inp has been read")
    print(f"  ndummy = {atoms.ndummy}, natreal = {atoms.natreal}")
    
    return atoms, xlogs, vlogs, glogs, xlogv, vlogv, glogv, Tset, Tdelta


def read_temp_ramp(filename: str = "temp.ramp") -> Tuple[TempRampData, bool]:
    """
    Read temperature ramping data from temp.ramp (Fortran unit 47).
    """
    ramp = TempRampData()
    lTempRamp = False
    
    if not Path(filename).exists():
        print("No temp.ramp file found - temperature ramping disabled")
        return ramp, False
    
    with open(filename, 'r') as f:
        line = read_a_line(f)
        tokens = line.split()
        ramp.tempSegments = int(tokens[0])
        
        if ramp.tempSegments > 0:
            lTempRamp = True
            ramp.rampSTemp = np.zeros(ramp.tempSegments)
            ramp.rampETemp = np.zeros(ramp.tempSegments)
            ramp.rampDelT = np.zeros(ramp.tempSegments)
            ramp.rampSteps = np.zeros(ramp.tempSegments, dtype=int)
            ramp.rampType = np.zeros(ramp.tempSegments, dtype=int)
            ramp.rampShape = []
            
            for i in range(ramp.tempSegments):
                line = read_a_line(f)
                tokens = line.split()
                ramp.rampSTemp[i] = float(tokens[0])
                ramp.rampETemp[i] = float(tokens[1])
                ramp.rampSteps[i] = int(tokens[2])
                ramp.rampType[i] = int(tokens[3])
                ramp.rampShape.append(tokens[4] if len(tokens) > 4 else "linear")
                
                # Calculate delta T per step
                if ramp.rampSteps[i] > 0:
                    ramp.rampDelT[i] = (ramp.rampETemp[i] - ramp.rampSTemp[i]) / ramp.rampSteps[i]
    
    if lTempRamp:
        print(f"File: temp.ramp has been read - {ramp.tempSegments} segments")
    
    return ramp, lTempRamp


def verify_input_data(state, debug=True):
    """Print all force field parameters if debug=True."""
    if not debug:
        return
    
    print("\n" + "="*80)
    print("FORCE FIELD PARAMETERS FROM ff.dat")
    print("="*80)
    
    # =========================================================================
    # NONBONDED PARAMETERS
    # =========================================================================
    print("\n### NONBONDED PARAMETERS (A, B, C) ###")
    print(f"Array shape: {state.ff.nonbonded.shape}")
    print(f"Total interaction types: {state.ff.nonbonded.shape[1]}")
    print(f"\n{'Type':<8} {'A':<18} {'B':<18} {'C':<18}")
    print("-" * 62)
    for i in range(state.ff.nonbonded.shape[1]):
        A = state.ff.nonbonded[0, i]
        B = state.ff.nonbonded[1, i]
        C = state.ff.nonbonded[2, i]
        print(f"{i:<8} {A:<18.10e} {B:<18.10e} {C:<18.10e}")
    
    # =========================================================================
    # CHARGES
    # =========================================================================
    if hasattr(state.ff, 'q') and len(state.ff.q) > 0:
        print(f"\n### CHARGES ###")
        print(f"Total charge types: {len(state.ff.q)}")
        print(f"\n{'Type':<8} {'Charge':<18}")
        print("-" * 26)
        for i, q in enumerate(state.ff.q):
            print(f"{i:<8} {q:<18.10f}")
    
    # =========================================================================
    # ELECTROSTATIC PARAMETERS
    # =========================================================================
    if hasattr(state.ff, 'electrostatic') and state.ff.electrostatic is not None:
        print(f"\n### ELECTROSTATIC PARAMETERS ###")
        print(f"Array shape: {state.ff.electrostatic.shape}")
        try:
            nonzero_count = np.count_nonzero(state.ff.electrostatic)
            print(f"Non-zero elements: {nonzero_count}")
            if nonzero_count > 0:
                print(f"\nFirst 20 non-zero electrostatic values:")
                print(f"{'Index':<8} {'Column':<8} {'Value':<18}")
                print("-" * 34)
                count = 0
                for i in range(min(state.ff.electrostatic.shape[0], 100)):
                    if state.ff.electrostatic.ndim == 1:
                        if state.ff.electrostatic[i] != 0 and count < 20:
                            print(f"{i:<8} {'0':<8} {state.ff.electrostatic[i]:<18.10e}")
                            count += 1
                    else:
                        for j in range(state.ff.electrostatic.shape[1]):
                            if state.ff.electrostatic[i, j] != 0 and count < 20:
                                print(f"{i:<8} {j:<8} {state.ff.electrostatic[i, j]:<18.10e}")
                                count += 1
                    if count >= 20:
                        break
        except Exception as e:
            print(f"Error reading electrostatic array: {e}")
    
    # =========================================================================
    # BOND STRETCH PARAMETERS
    # =========================================================================
    if hasattr(state.ff, 'stretch'):
        print(f"\n### BOND STRETCH PARAMETERS ###")
        print(f"Total bond types: {state.ff.stretch.shape[0]}")
        print(f"\n{'Type':<8} {'k':<18} {'r0':<18} {'k3':<18} {'k4':<18} {'k_ub':<18}")
        print("-" * 98)
        for i in range(state.ff.stretch.shape[0]):
            print(f"{i:<8} {state.ff.stretch[i,0]:<18.10f} {state.ff.stretch[i,1]:<18.10f} "
                  f"{state.ff.stretch[i,2]:<18.10f} {state.ff.stretch[i,3]:<18.10f} {state.ff.stretch[i,4]:<18.10f}")
    
    # =========================================================================
    # BEND PARAMETERS
    # =========================================================================
    if hasattr(state.ff, 'bend'):
        print(f"\n### ANGLE BEND PARAMETERS ###")
        print(f"Total bend types: {state.ff.bend.shape[0]}")
        print(f"\n{'Type':<8} {'k':<18} {'theta0':<18} {'k_ub':<18} {'r_ub':<18}")
        print("-" * 70)
        for i in range(state.ff.bend.shape[0]):
            print(f"{i:<8} {state.ff.bend[i,0]:<18.10f} {state.ff.bend[i,1]:<18.10f} "
                  f"{state.ff.bend[i,2]:<18.10f} {state.ff.bend[i,3]:<18.10f}")
    
    # =========================================================================
    # TORSION PARAMETERS
    # =========================================================================
    if hasattr(state.ff, 'twist'):
        print(f"\n### TORSION PARAMETERS ###")
        print(f"Total torsion types: {state.ff.twist.shape[0]}")
        print(f"\n{'Type':<8} {'V1':<18} {'V2':<18} {'V3':<18} {'V4':<18}")
        print("-" * 80)
        for i in range(min(20, state.ff.twist.shape[0])):  # Show first 20
            print(f"{i:<8} {state.ff.twist[i,0]:<18.10f} {state.ff.twist[i,1]:<18.10f} "
                  f"{state.ff.twist[i,2]:<18.10f} {state.ff.twist[i,3]:<18.10f}")
        if state.ff.twist.shape[0] > 20:
            print(f"... ({state.ff.twist.shape[0] - 20} more torsion types)")
    
    # =========================================================================
    # IMPROPER PARAMETERS
    # =========================================================================
    if hasattr(state.ff, 'deform') and state.ff.deform is not None:
        print(f"\n### IMPROPER PARAMETERS ###")
        if state.ff.deform.ndim == 1:
            print(f"Array shape: {state.ff.deform.shape} (1D array)")
            print(f"Values: {state.ff.deform}")
        else:
            print(f"Total improper types: {state.ff.deform.shape[0]}")
            if state.ff.deform.shape[0] > 0:
                print(f"\n{'Type':<8} {'k':<18} {'psi0':<18}")
                print("-" * 44)
                for i in range(state.ff.deform.shape[0]):
                    print(f"{i:<8} {state.ff.deform[i,0]:<18.10f} {state.ff.deform[i,1]:<18.10f}")
    
    print("\n" + "="*80)
    print("END FORCE FIELD PARAMETERS")
    print("="*80 + "\n")


def load_simulation(directory: str = ".", debug_input=False) -> SimulationState:
    """
    Load complete simulation state from all input files.
    """
    import os
    orig_dir = os.getcwd()
    os.chdir(directory)
    
    try:
        state = SimulationState()
        
        # Read parameters
        state.params = read_mdrun_params()
        
        # Read force field
        state.ff = read_forcefield()
        
        # Read temperature ramp
        ramp_data, state.lTempRamp = read_temp_ramp()
        state.Tset = state.params.tstart
        state.Tdelta = 0.0
        
        # Read connectivity
        state.connectivity = read_connectivity()
        
        # Read coordinates
        (state.atoms, xlogs, vlogs, glogs, xlogv, vlogv, glogv, 
         Tset_file, Tdelta_file) = read_coords(
            "coords.inp", 
            state.connectivity.nat, 
            state.ff,
            state.params.nnos,
            state.lTempRamp
        )
        
        # Update temperature settings if from file
        if Tset_file > 0:
            state.Tset = Tset_file
            state.Tdelta = Tdelta_file
        
        # Initialize thermostat
        state.thermostat = ThermostatState(
            nnos=state.params.nnos,
            xlogs=xlogs,
            vlogs=vlogs,
            glogs=glogs,
            qtmass=np.zeros(10),
            wdt1=np.zeros(10),
            wdt2=np.zeros(10),
            wdt4=np.zeros(10),
            wdt8=np.zeros(10)
        )
        
        # Initialize barostat
        state.barostat = BarostatState(
            xlogv=xlogv,
            vlogv=vlogv,
            glogv=glogv,
            wtmass=np.zeros(10)
        )
        
        # Copy chain info
        state.atoms.chain = state.connectivity.chain.copy()
        
        # VERIFY INPUT DATA (controlled by debug_input flag)
        verify_input_data(state, debug=debug_input)
        
        return state
        
    finally:
        os.chdir(orig_dir)
