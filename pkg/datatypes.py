"""
Data Structures for MD Simulation
Uses dataclasses for clean, efficient data organization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from constants import DefaultParams, GASK_KCAL


@dataclass
class SystemParams:
    """System-wide simulation parameters from mdrun.params"""
    # Ensemble selection
    nve: bool = False
    nvt: bool = False
    npt: bool = False
    lprofile: bool = False
    
    # Cutoffs and timestep
    rcut: float = 10.0          # Cutoff distance (Å)
    driftmax: float = 1.0       # Max drift for neighbor list
    delt: float = 0.5           # Timestep (fs)
    nsteps: int = 1000          # Total steps
    
    # Temperature control
    tstart: float = 300.0       # Initial temperature (K)
    nnos: int = 3               # Nose-Hoover chain length
    qfreq: float = 100.0        # Thermostat frequency
    
    # Pressure control
    pfix: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    box_mult: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))
    avg_mult: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    wfreq: float = 100.0        # Barostat frequency
    bUsePressure: bool = False
    
    # Polarization
    tbpol: bool = False
    rop: float = 10.0
    rscalep: float = 8.0
    epsilonp: float = 0.1
    repsilonp: float = 1.0
    
    # Output frequencies
    ninit: int = 0
    nave: float = 100.0
    nvir: float = 10.0
    nnben: float = 10.0
    noutput: float = 100.0
    
    # Output flags
    lboxdip: bool = False
    lcoords: bool = True
    lstress: bool = False
    lvelocs: bool = False
    nboxdip: float = 100.0
    ncoords: float = 100.0
    nstress: float = 100.0
    nvelocs: float = 100.0
    
    # Box scaling
    chvol: bool = False
    boxnew: np.ndarray = field(default_factory=lambda: np.zeros(3))
    printout: bool = True
    
    # Interactions
    inter: bool = True          # Intermolecular interactions
    ex14: bool = True           # Exclude 1-4 interactions
    fixtemp: bool = False
    
    # 1-4 reduction factors
    lredonefour: bool = False
    redfactor: float = 0.5
    lredQ_mu14: bool = False
    redQmufactor: float = 1.0
    
    # Constraints
    constrain: bool = False
    tol: float = 1.0e-6
    
    # Electrostatics
    alphai: float = 0.2
    klimitx: int = 8
    klimity: int = 8
    klimitz: int = 8
    lewald: bool = True
    
    # Multiple timestep
    multimed: int = 1
    multibig: int = 1
    ros: float = 6.0
    
    # Polarization convergence
    tolpol: float = 1.0e-6
    tapering: bool = False
    rtaper: float = 10.0
    epsrf: float = 1.0


@dataclass
class ForceFieldParams:
    """Force field parameters from ff.dat"""
    # Non-bonded types
    num_nbtypes: int = 0
    non_labels: List[str] = field(default_factory=list)
    nonbonded: np.ndarray = None      # (maxnnb, 6) - A, B, C parameters
    electrostatic: np.ndarray = None   # (maxnnb, 2) - charge interactions
    atmasses: np.ndarray = None        # Atomic masses by type
    
    # Charge types
    ncharge: int = 0
    atom_labels: List[str] = field(default_factory=list)
    q: np.ndarray = None              # Charges
    pol: np.ndarray = None            # Polarizabilities
    non_of_charge: np.ndarray = None  # Map charge type to non-bonded type
    lonepairtype: np.ndarray = None   # Lone pair flags
    
    # Type mapping
    typen: np.ndarray = None          # Non-bonded type pairs
    typee: np.ndarray = None          # Charge type pairs
    map_types: np.ndarray = None      # Charge to non-bonded mapping
    
    # Bond parameters
    nbondt: int = 0
    stretch: np.ndarray = None        # (maxbtypes, 5) - k, r0, r0^2, etc.
    
    # Bend parameters
    nbendt: int = 0
    bend: np.ndarray = None           # (maxbtypes, 4) - k, theta0, etc.
    linear: np.ndarray = None         # Linear bend flags
    
    # Torsion parameters
    ntortt: int = 0
    twist: np.ndarray = None          # (maxfolds+1, maxbtypes)
    nprms: np.ndarray = None          # Number of terms per torsion
    
    # Out-of-plane (improper) parameters
    ndeformt: int = 0
    deform: np.ndarray = None         # (maxbtypes,)
    
    # Dummy atom parameters
    ndummytypes: int = 0
    typedummy: np.ndarray = None
    adummy: np.ndarray = None
    bdummy: np.ndarray = None
    cdummy: np.ndarray = None
    LpGeomType: np.ndarray = None
    LpBendAtomType: np.ndarray = None
    
    # Polarization flag
    lpolarizable: bool = False


@dataclass
class Connectivity:
    """Molecular connectivity from connectivity.dat"""
    nch: int = 0                # Number of chains/molecules
    nions: int = 0              # Number of ions
    nat: int = 0                # Total atoms
    
    # Bond connectivity: (3, maxbonds) - atom1, atom2, type
    nbonds: int = 0
    bonds: np.ndarray = None
    
    # Bend connectivity: (6, maxbends) - atom1, atom2, atom3, type, bond1, bond2
    nbends: int = 0
    bends: np.ndarray = None
    
    # Torsion connectivity: (10, maxtorts)
    ntorts: int = 0
    torts: np.ndarray = None
    
    # Out-of-plane deformations: (8, maxdeforms)
    ndeforms: int = 0
    deforms: np.ndarray = None
    
    # Chain assignment for each atom
    chain: np.ndarray = None


@dataclass
class AtomData:
    """Atomic data from coords.inp"""
    nat: int = 0
    
    # Positions: (3, nat)
    x: np.ndarray = None
    
    # Velocities: (3, nat)
    v: np.ndarray = None
    
    # Accelerations: (3, nat)
    accel: np.ndarray = None
    
    # Forces: (3, nat)
    f: np.ndarray = None
    
    # Atom types
    atomtype: np.ndarray = None
    
    # Masses
    mass: np.ndarray = None
    massinv: np.ndarray = None
    
    # Dummy atom flags
    ldummy: np.ndarray = None
    ndummy: int = 0
    natreal: int = 0
    
    # Dummy atom connectivity: (5, ndummy)
    # [0,:] = jat, [1,:] = iat (central), [2,:] = kat, [3,:] = idat (dummy index), [4,:] = itype
    jikdummy: np.ndarray = None
    
    # Simulation box
    box: np.ndarray = field(default_factory=lambda: np.zeros(3))
    el: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Half box
    
    # Chain assignment
    chain: np.ndarray = None


@dataclass
class ThermostatState:
    """Nose-Hoover thermostat variables"""
    nnos: int = 3
    xlogs: np.ndarray = None    # Positions
    vlogs: np.ndarray = None    # Velocities
    glogs: np.ndarray = None    # Forces
    qtmass: np.ndarray = None   # Masses
    
    # Integration weights
    wdt1: np.ndarray = None
    wdt2: np.ndarray = None
    wdt4: np.ndarray = None
    wdt8: np.ndarray = None
    
    # Temperature factors
    gkt: float = 0.0            # k_B * T
    gnkt: float = 0.0           # N * k_B * T
    gn3kt: float = 0.0          # (N+3) * k_B * T


@dataclass
class BarostatState:
    """Barostat variables for NPT"""
    xlogv: np.ndarray = None    # Box position variables
    vlogv: np.ndarray = None    # Box velocity variables
    glogv: np.ndarray = None    # Box forces
    wtmass: np.ndarray = None   # Barostat masses


@dataclass
class NeighborList:
    """Neighbor list data structures"""
    # Full neighbor list
    list_neighbors: np.ndarray = None   # (maxnay, nat)
    listm: np.ndarray = None            # Number of neighbors per atom
    
    # Short-range neighbor list
    lists: np.ndarray = None            # (maxnay//2, nat)
    listms: np.ndarray = None           # Number of short-range neighbors
    
    # Exclusion list
    listex: np.ndarray = None           # (40, nat)
    listmex: np.ndarray = None          # Number of exclusions per atom
    
    # 1-4 interaction list
    list14: np.ndarray = None           # (25, nat)
    listm14: np.ndarray = None          # Number of 1-4 pairs per atom
    
    # Box subdivision
    idim: np.ndarray = None             # Box dimensions
    listsubbox: np.ndarray = None       # Atoms per subbox


@dataclass
class SplineCoeffs:
    """Spline coefficients for non-bonded interactions"""
    zs: np.ndarray = None               # Spline knots
    deltaspline: float = 0.0
    intzs0: int = 0
    
    # Energy spline coefficients
    coeffee: np.ndarray = None          # (6, maxpoints, maxnnb)
    coefffee: np.ndarray = None         # Force coefficients
    
    # Short-range coefficients
    coefft: np.ndarray = None
    coeffft: np.ndarray = None
    coeffft1: np.ndarray = None


@dataclass
class EnergyComponents:
    """Energy components for tracking"""
    kinen: float = 0.0          # Kinetic energy
    poten: float = 0.0          # Potential energy
    toten: float = 0.0          # Total energy
    
    ebond: float = 0.0          # Bond energy
    ebend: float = 0.0          # Bend energy
    etort: float = 0.0          # Torsion energy
    eopbs: float = 0.0          # Out-of-plane energy
    
    unbd: float = 0.0           # Non-bonded energy
    unbde: float = 0.0          # Electrostatic energy
    evdw: float = 0.0           # van der Waals energy
    eelec: float = 0.0          # Electrostatic energy (total)
    epol: float = 0.0           # Polarization energy
    
    ewself: float = 0.0         # Ewald self energy
    esumninj: float = 0.0       # Truncation correction
    
    vir: float = 0.0            # Virial


@dataclass
class PressureData:
    """Pressure and stress tensor data"""
    pres: float = 0.0           # Scalar pressure
    pke: float = 0.0            # Kinetic pressure
    pvir: float = 0.0           # Virial pressure
    ptrunc: float = 0.0         # Truncation pressure
    psumninj: float = 0.0       # Pressure correction
    
    # Stress tensor
    stress: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    tvirpo: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    prtrunc: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    prtkine: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Per-direction pressure
    prt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    akin2: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ConstraintData:
    """SHAKE constraint data for bond length constraints"""
    # Arrays sized using maxbonds from DefaultParams
    bondcon: np.ndarray = field(default_factory=lambda: np.zeros((3, DefaultParams.maxbonds), dtype=int))
    nbcon: int = 0                      # Number of constrained bonds (actual count)
    d2: np.ndarray = field(default_factory=lambda: np.zeros(DefaultParams.maxbonds))
    massred: np.ndarray = field(default_factory=lambda: np.zeros(DefaultParams.maxbonds))
    
    # Aromatic constraints (if any)
    iaromatic: np.ndarray = field(default_factory=lambda: np.zeros((4, 100), dtype=int))
    numaromatic: int = 0
    
    # SHAKE convergence parameters
    max_iter: int = 100
    tol: float = 1.0e-6


@dataclass
class SimulationState:
    """Complete simulation state"""
    # Main data containers
    params: SystemParams = field(default_factory=SystemParams)
    ff: ForceFieldParams = field(default_factory=ForceFieldParams)
    connectivity: Connectivity = field(default_factory=Connectivity)
    atoms: AtomData = field(default_factory=AtomData)
    
    # Constraints (SHAKE)
    constraints: ConstraintData = field(default_factory=ConstraintData)
    
    # Thermostat/Barostat
    thermostat: ThermostatState = field(default_factory=ThermostatState)
    barostat: BarostatState = field(default_factory=BarostatState)
    
    # Neighbor lists
    neighbors: NeighborList = field(default_factory=NeighborList)
    
    # Splines
    splines: SplineCoeffs = field(default_factory=SplineCoeffs)
    
    # Energies
    energy: EnergyComponents = field(default_factory=EnergyComponents)
    
    # Pressure
    pressure: PressureData = field(default_factory=PressureData)
    
    # Simulation counters
    kount: int = 0
    temp: float = 300.0
    ndof: int = 0
    
    # Temperature ramping
    lTempRamp: bool = False
    Tset: float = 300.0
    Tdelta: float = 0.0
    
    # Update flag
    update: bool = True
    
    # Polarization data
    px: np.ndarray = None       # Induced dipoles x
    py: np.ndarray = None       # Induced dipoles y
    pz: np.ndarray = None       # Induced dipoles z
    totdipole: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass 
class TempRampData:
    """Temperature ramping data from temp.ramp file"""
    tempSegments: int = 0
    rampSTemp: np.ndarray = None     # Start temperatures
    rampETemp: np.ndarray = None     # End temperatures
    rampDelT: np.ndarray = None      # Temperature deltas
    rampSteps: np.ndarray = None     # Steps per segment
    rampType: np.ndarray = None      # Ramp type
    rampShape: List[str] = field(default_factory=list)
