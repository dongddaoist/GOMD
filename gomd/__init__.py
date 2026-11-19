"""
GOMD — GPU-accelerated Open MD.

A JAX-native molecular dynamics engine for the APPLE&P polarizable force field.

Roadmap:
    1. (done) Efficient MD engine reproducing APPLE&P, classical and polarizable
    2. Constant-potential method for EDL simulation
    3. ML to bypass SCF
    4. ML/MM multi-scale or full ML force field

Design principles:
    - Own the engine end-to-end. Use external libraries only for primitives
      that won't be touched (FFT, basic GPU ops). Everything in the science
      stack — pair functional forms, polarization SCF, integrator step
      structure — we write ourselves so it can be replaced/modified later.
    - Pure functional JAX style: state passed in/out, no mutation.
    - Mixed precision: fp32 pair kernel, fp64 dipoles + SCF + energy.
    - Validate at every milestone against the reference Python package (pkg/).

Milestones:
    M1: minimum viable engine (scaffold, velocity Verlet)            — done
    M2: full classical APPLE&P (splined pair, bonded, Ewald, SHAKE,
        dummy atoms)                                                 — done
    M3: polarization SCF wired into the step loop                    — done
    M4: performance (on-device neighbor list, lax.scan loop,
        segment_sum pair kernel)                                     — done
    M5: NVT (Nose-Hoover chain), MLIP provider (UMA via fairchem in its
        own process), PFF/MLIP side-by-side comparison, Colab runner — done
    M6: hybrid PFF/MLIP regions, ML replacement of the SCF

Entry point: python -m gomd.main_scan  (see GOMD_DOCUMENTATION.md)
"""

__version__ = "0.6.0.dev0"   # 2026-11-17: profiler, PCG SCF, neighbor-list checks (not yet run on a GPU)
