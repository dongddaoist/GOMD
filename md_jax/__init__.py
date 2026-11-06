"""
wmi_md_jax — JAX-native molecular dynamics engine for APPLE&P force field.

Roadmap:
    1. (CURRENT) Build efficient MD engine reproducing APPLE&P with classical definition
    2. Constant-potential method for EDL simulation
    3. ML to bypass SCF
    4. ML/MM multi-scale or full ML force field

Design principles:
    - Own the engine end-to-end. Use external libraries only for primitives
      that won't be touched (FFT, basic GPU ops). Everything in the science
      stack — pair functional forms, polarization SCF, integrator step
      structure — we write ourselves so it can be replaced/modified later.
    - Pure functional JAX style: state passed in/out, no mutation.
    - Mixed precision (M4): fp32 forces+positions, fp64 dipoles+SCF+energy.
    - Validate at every milestone: PE matches reference Python code to 1e-3 kcal/mol.

Milestones:
    M1 (current): minimum viable engine — Coulomb + velocity Verlet, NVE only
    M2: full classical APPLE&P (splined pair, bonded, SHAKE, dummy atoms, PME)
    M3: polarization SCF (own implementation, prepared for ML replacement)
    M4: optimization to 80 ms/step (mixed precision, Triton kernels if needed)
    M5: ML/MM integration for steps 3-4 of roadmap
"""

__version__ = "0.0.1-M1"
