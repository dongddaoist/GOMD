"""
forces — energy and force evaluators.

Architecture:
    Each force module exports `compute_<name>(state) -> (energy, forces)` where
    state is a SystemState (after .to_jax()) and energy/forces are jax arrays.
    
    All force functions are jit-compatible: pure functions of state, no
    side effects, no Python control flow that depends on data values.

Modules:
    coulomb.py  — [M1] direct Coulomb sum, no Ewald (test architecture)
    pair.py     — [M2] APPLE&P splined pair potential (12-6-1 + smooth cutoff)
    bonded.py   — [M2] bonds, angles, torsions, impropers
    dummy.py    — [M2] lone-pair virtual sites
    pme.py      — [M2] particle mesh Ewald
    shake.py    — [M2] SHAKE constraints
    polarization.py — [M3] APPLE&P induced dipoles + Thole damping + SCF
"""
