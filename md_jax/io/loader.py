"""
loader.py — bridge between existing Python input parsers and the new SystemState.

We don't reinvent the parsers. The existing python-current/ parsers handle
mdrun.params, ff.dat, connectivity.dat, coords.inp correctly. We import them
and then convert the resulting (rich, multi-dataclass) state into our
simpler JAX-friendly SystemState.

Note on coordinate convention:
    Reference Python uses (3, nat) column-major layout (Fortran convention).
    We use (nat, 3) row-major because:
    - jax.numpy default
    - jax-md uses (nat, 3)
    - DMFF uses (nat, 3)
    - Cleaner for sparse pair indexing later

    Conversion: x_jax[i, k] = x_ref[k, i]   (transpose at load time)
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np

from ..state import SystemState


# We add the python-current/ directory to sys.path so we can import the
# existing parsers. This is a temporary M1 measure; in M2+ we'll either
# port them properly or make them a sibling package.
_PYTHON_CURRENT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent  # repo root
    / "python-current"
)


def _ensure_python_current_on_path():
    """Make python-current/ importable. Idempotent."""
    p = str(_PYTHON_CURRENT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_simulation_state(input_dir: str | Path) -> SystemState:
    """Parse mdrun.params, ff.dat, connectivity.dat, coords.inp from a directory.
    
    Returns a numpy-backed SystemState (call .to_jax() before passing to
    jit'd functions).
    
    Args:
        input_dir: directory containing the input files.
    
    Returns:
        SystemState with M1 fields populated. Other fields are None/0 and
        will be added in M2+.
    """
    _ensure_python_current_on_path()
    
    # Reuse existing parsers — they produce a rich SimulationState
    from input_parser import load_simulation as _load_legacy
    
    cwd_save = os.getcwd()
    try:
        os.chdir(input_dir)
        legacy = _load_legacy(str(input_dir))
    finally:
        os.chdir(cwd_save)
    
    # Convert legacy state to our SystemState. Transpose positions/velocities/forces
    # from (3, nat) to (nat, 3).
    nat = legacy.atoms.nat
    x = np.asarray(legacy.atoms.x).T.copy()                 # (nat, 3)
    v = np.asarray(legacy.atoms.v).T.copy() if legacy.atoms.v is not None else np.zeros((nat, 3))
    f = np.zeros((nat, 3))                                   # forces start at zero
    
    mass = np.asarray(legacy.atoms.mass).copy()              # (nat,)
    massinv = np.asarray(legacy.atoms.massinv).copy()
    
    # Charges may live in different places depending on the FF. For APPLE&P
    # the charges are stored per-atom-type and dispatched by atomtype.
    # For M1 (Coulomb test) we just need a per-atom charge. If there's a
    # charge array on the FF, use it; otherwise zeros.
    if hasattr(legacy.ff, 'q') and legacy.ff.q is not None:
        # ff.q[atomtype[i]] is the charge for atom i
        charge = np.asarray(legacy.ff.q)[legacy.atoms.atomtype].astype(np.float64)
    else:
        # M1 fallback: try to read directly from the SimulationState's FF
        charge = np.zeros(nat, dtype=np.float64)
    
    atomtype = np.asarray(legacy.atoms.atomtype, dtype=np.int32).copy()
    ldummy = np.asarray(legacy.atoms.ldummy, dtype=bool).copy()
    box = np.asarray(legacy.atoms.box, dtype=np.float64).copy()
    
    state = SystemState(
        nat=nat,
        x=x, v=v, f=f,
        mass=mass, massinv=massinv,
        charge=charge,
        atomtype=atomtype,
        ldummy=ldummy,
        box=box,
        dt=float(legacy.params.delt),
        rcut=float(legacy.params.rcut),
        step=0,
        pe=0.0,
        ke=0.0,
    )
    return state
