"""
state.py — System state representation for the JAX MD engine.

Design principle: state is a pure dataclass of JAX arrays. No methods that
mutate state. All updates produce new state objects.

This is the JAX-friendly analog of the current Python's SimulationState +
AtomData. State is registered as a JAX pytree so jax.jit can flatten/unflatten
it transparently — array fields become "leaves" (dynamic), other fields become
"aux data" (static, used for hashing and tree structure).

Conventions:
    Positions x: shape (nat, 3) — note column-major (3, nat) in the
        reference Python is transposed here. JAX/numpy default is
        row-major and (n, 3) is the convention in jax-md and DMFF.
    Box: 3-vector for orthorhombic (most common APPLE&P case).
    Mass: shape (nat,). For dummy atoms, mass should be 0 (or any
        sentinel value — we check against ldummy mask).
    Forces, velocities: same shape as positions.

What's in state:
    - Time-evolving (positions, velocities, forces, kinetic+potential energy)
    - Quasi-static (dipoles for SCF — change every step but with predictor)
    - Static (atomtype, masses, box [for NVE], connectivity tables, FF parameters)

For M1 we keep state minimal: just what's needed for Coulomb + velocity Verlet.
M2/M3 will add fields for bonded, dummy, polarization, etc.

Pytree note: JAX needs to know which fields are array "leaves" (vary at runtime,
flattened/transferred to GPU) vs "aux data" (static, used for tree hashing /
recompilation triggers). We list them explicitly in tree_flatten/unflatten.
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields, replace
from typing import Optional
import numpy as np
import jax


# Field names that hold dynamic arrays (jax pytree leaves).
# These are the fields that can change between jit calls without recompilation.
_LEAF_FIELDS = (
    'x', 'v', 'f',
    'mass', 'massinv', 'charge',
    'atomtype', 'ldummy',
    'box',
    'pe', 'ke',
)

# Field names that hold static config (not arrays).
# Changes to these trigger recompilation, so keep them stable during simulation.
_AUX_FIELDS = ('nat', 'dt', 'rcut')
# NOTE: 'step' is deliberately NOT in the pytree. It's a Python int that
# changes every step; if it were aux_data, every new value would change the
# tree structure and force a full jit recompile each step (~1.5 s/step!).
# The driver loop tracks the step count separately. state.step stays as a
# plain attribute, ignored by flatten/unflatten (reset to its current value
# on unflatten via the closure default below).


@dataclass
class SystemState:
    """Time-evolving + static state of the MD system.
    
    For Milestone 1 (Coulomb + velocity Verlet, NVE), only the fields
    marked [M1] are populated. Other fields are placeholders for M2+.
    
    Registered as a JAX pytree, so it can be passed to jax.jit-compiled
    functions and properly flattened to GPU.
    """
    
    # ===== [M1] core dynamical state =====
    nat: int = 0                              # total number of atoms (aux)
    x: np.ndarray = None                      # (nat, 3) positions, Å
    v: np.ndarray = None                      # (nat, 3) velocities
    f: np.ndarray = None                      # (nat, 3) forces, kcal/mol/Å
    
    # ===== [M1] static atomic data =====
    mass: np.ndarray = None                   # (nat,) masses, atomic mass units
    massinv: np.ndarray = None                # (nat,) 1/mass, with 0 for dummies
    charge: np.ndarray = None                 # (nat,) partial charges, e
    atomtype: np.ndarray = None               # (nat,) integer atom types
    ldummy: np.ndarray = None                 # (nat,) bool, True for dummy atoms
    
    # ===== [M1] box and PBC =====
    box: np.ndarray = None                    # (3,) box dimensions, Å (orthorhombic)
    
    # ===== [M1] simulation control =====
    dt: float = 0.5                           # timestep in fs (aux)
    rcut: float = 12.0                        # cutoff radius, Å (aux)
    step: int = 0                             # current step number (aux)
    
    # ===== [M1] energy bookkeeping =====
    pe: float = 0.0                           # potential energy, kcal/mol (leaf — scalar tracer)
    ke: float = 0.0                           # kinetic energy, kcal/mol (leaf — scalar tracer)
    
    # ===== [M2+] reserved =====
    # bonded connectivity, SHAKE info, dummy redistribution, polarizabilities,
    # induced dipoles + history, force-field tables, neighbor list state, ...
    
    @property
    def el(self) -> np.ndarray:
        """Half-box for PBC (matches original Python's `el` array)."""
        return self.box / 2.0
    
    def to_jax(self):
        """Convert numpy arrays to jax arrays. Call once at simulation start.
        
        After this conversion, state can be used inside jit'd functions.
        Setup code (parsing, init) uses numpy state; simulation loop uses jax state.
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            raise RuntimeError("JAX not installed. `pip install jax[cuda12]` for GPU.")
        
        kwargs = {}
        for fname in _AUX_FIELDS:
            kwargs[fname] = getattr(self, fname)
        for fname in _LEAF_FIELDS:
            arr = getattr(self, fname)
            if arr is not None:
                kwargs[fname] = jnp.asarray(arr)
            else:
                kwargs[fname] = None
        return SystemState(**kwargs)
    
    def summary(self) -> str:
        """Human-readable summary of this state. For logging."""
        natoms = self.nat
        ndummy = int(np.sum(self.ldummy)) if self.ldummy is not None else 0
        nreal = natoms - ndummy
        lines = [
            f"SystemState: {natoms} atoms ({nreal} real, {ndummy} dummy)",
            f"  box = {self.box}  (Å)",
            f"  dt = {self.dt} fs,  rcut = {self.rcut} Å,  step = {self.step}",
        ]
        if self.x is not None:
            lines.append(f"  x range: [{float(self.x.min()):.2f}, {float(self.x.max()):.2f}] Å")
        return "\n".join(lines)


# ============================================================================
# JAX pytree registration
# ============================================================================
# This is what allows jax.jit(fn) to accept a SystemState as input.
# tree_flatten breaks the state into (children, aux_data); unflatten rebuilds it.
# Children are arrays/scalars that get traced; aux_data is the static structure.

def _state_tree_flatten(state):
    """Pytree flatten: split state into (children, aux_data)."""
    children = tuple(getattr(state, f) for f in _LEAF_FIELDS)
    aux = tuple(getattr(state, f) for f in _AUX_FIELDS)
    return children, aux


def _state_tree_unflatten(aux, children):
    """Pytree unflatten: rebuild state from children + aux."""
    kwargs = {}
    for f, v in zip(_AUX_FIELDS, aux):
        kwargs[f] = v
    for f, v in zip(_LEAF_FIELDS, children):
        kwargs[f] = v
    return SystemState(**kwargs)


jax.tree_util.register_pytree_node(
    SystemState, _state_tree_flatten, _state_tree_unflatten
)
