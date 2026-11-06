"""
Physical Constants and Default Parameters for MD Simulation
Converted from Fortran90 GOMD code
"""

import numpy as np

# Physical Constants
KB = 1.38066e-23          # Boltzmann constant (J/K)
AVOGADRO = 6.02205e+23    # Avogadro's number
GASK = 8.3144036          # Gas constant (J/(mol·K))
GASK_KCAL = 1.98709e-3    # Gas constant (kcal/(mol·K))
PI = np.pi
ROOTPI = np.sqrt(PI)
DELZERO = 1.0e-10         # Numerical zero threshold

# Conversion factors
KCAL_TO_KJ = 4.184
COULOMB_CONST = 332.08    # kcal·Å/(mol·e²)

# Default array dimensions (can be overridden)
class DefaultParams:
    """Default parameters matching params.h"""
    maxnch = 392          # Maximum molecules + atoms
    maxat = 8716          # Maximum force centers
    maxbonds = 8324       # Maximum bonds
    maxbends = 12628      # Maximum bends
    maxtorts = 12696      # Maximum torsions
    maxdeforms = 1        # Maximum deformations
    maxtypes = 10         # Max repulsion-dispersion types
    maxcharges = 16       # Max charge types
    maxdummy = 1316       # Max dummy (lone pair) atoms
    maxbtypes = 17        # Max bond/bend/torsion types
    maxfolds = 101        # Max torsion terms
    maxbox = 6000         # Max atoms per box
    maxnay = 2000         # Max neighbors per atom
    maxdim = 100          # Max box subdivisions per axis
    maxpoints = 2000      # Max spline points
    maxprop = 40          # Max properties
    buffersize = 2000     # Output buffer size
    maxRamp = 10          # Max temperature ramp segments
    
    # Ewald parameters
    kmax_x = 8
    kmax_y = 8
    kmax_z = 8
    
    # Thole damping
    a_thole = 0.20

# Derived parameters
MAXNNB = DefaultParams.maxcharges * (DefaultParams.maxcharges + 1) // 2
