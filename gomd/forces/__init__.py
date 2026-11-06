"""
forces — energy and force evaluators (pure, jit-compatible JAX functions).

Modules:
    neighbor.py             host cell-list neighbor pairs (used once, to size capacity)
    neighbor_ondevice.py    on-device neighbor build, fixed-capacity padded pairs
    pair_splined.py         APPLE&P splined vdW + real-space electrostatics (fp64/fp32)
    pair_splined_segsum.py  fp32 pair kernel with segment_sum scatter (default)
    bonded.py               bonds, bends, torsions, impropers
    ewald.py                explicit-k Ewald reciprocal + self + exclusion correction
    dummy.py                lone-pair site placement + force redistribution
    all_forces.py           classical force field: sums the terms above
    polarization.py         induced-dipole SCF solver, energy, main induced force
    pol_field.py            charge electric field that drives the SCF
    pol_corrections.py      exclusion and 1-4 induced-force corrections
    polarization_full.py    full polarizable term: field -> SCF -> energy -> forces
"""
