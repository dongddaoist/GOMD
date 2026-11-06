# GOMD — project status

_GOMD (GPU-accelerated Open MD): a from-scratch JAX/GPU engine for the APPLE&P
polarizable force field (PEO + Li⁺ + FSI⁻, 8,716 sites = 7,400 atoms + 1,316
lone-pair sites). Platform: OSC Pitzer, Tesla V100-PCIE-16GB, conda env
`newenv2` (JAX 0.4.13, Python 3.8). The reference Python package in `pkg/` is
ground truth; every milestone is checked bit-for-bit against it._

This file records what is DONE and what is OPEN so the project can be resumed
cold. Technical detail lives in `GOMD_DOCUMENTATION.md`.

---

## Roadmap

1. Efficient MD engine (classical + polarizable)  — DONE
2. Constant-potential EDL simulations             — in development
3. ML replaces the polarization SCF               — not started (SCF seam is ready)
4. Full ML/MM                                     — not started (MLIP path in development)

---

## DONE

| Milestone | Result |
|-----------|--------|
| M1 scaffold | pytree state, velocity Verlet (scaffold code since retired) |
| M2 classical engine | splined pair, bonded, explicit-k Ewald, Jacobi SHAKE, lone-pair sites; PE 6.7e-10, F 1.4e-10 vs reference |
| M3 polarization in the loop | field 1.4e-15, dipoles 2.8e-15, E_pol 1.2e-11, induced forces 8.0e-9; initial PE −3611.5363 vs −3611.5355 |
| M4 performance | on-device neighbor list, `lax.scan` loop, fp32 `segment_sum` pair kernel |

V100, 200-step runs: **9.2 ms/step classical** (from ~800), **53–58 ms/step
polarizable** (56 average).

Key fixes along the way: the 800 ms/step was the ~4.5 s host neighbor build,
not the force kernels; SHAKE must constrain against the post-drift position;
the SCF field needed the 1-4 Q-μ fold-in (it was ~23% off); polarization forces
must mask skin pairs.

---

## IN DEVELOPMENT (outside this tree)

- MLIP energy/forces retrieved in parallel with the PFF, through the same
  `force_fn(x) → E, F` interface
- Constant-potential method
- Coupled electron–ion transport

---

## OPEN (priority order)

1. **Energy conservation and ensemble.** The 200-step V100 runs gain ~18
   kcal/mol per step in total energy, classical and polarizable alike
   (polarizable: TE 3990 → 7567, KE 7602 → 9398). The earlier "~3/step with
   polarization" came from a 6-step CPU run and does not hold. Also, GOMD is
   **NVE only**, while `md-input/mdrun.params` requests **NVT** (Nose–Hoover,
   296.15 K) and the reference supports NVE/NVT/NPT; GOMD ignores the ensemble
   flags without a warning. Next: run the reference for the same 200 steps in
   NVE to tell an integration problem from a GOMD bug, then port the
   thermostat.

2. **Contracting SCF solver.** ASPC warm start is unsafe with the Jacobi SCF (a
   1% dipole error → E_pol ~24 kcal/mol off; `min_iter` did not fix it), so the
   SCF cold-starts every step. Gauss-Seidel sweeps, a damped update, or a
   residual test would make ASPC safe and cut SCF iterations. Scaffolding
   (`use_aspc`, dipole history in the scan carry, `scf_min_iter`) is in place.

3. **Profile the polarizable step per phase on the V100.** The 56 ms breakdown
   is unknown. FFT-PME waits for this profile (the old "reciprocal = 32%" is a
   CPU-sandbox figure).

4. **Self-contained validation test.** The development-time component tests
   depended on one-off `/tmp` reference dumps and were removed; a script that
   builds its reference from `pkg/` on the fly would restore regression checks.

5. `compute_recipQmu` (k-space Q-μ) is a TODO in the reference itself and not in
   its forces; needed later for absolute accuracy, not for parity.

6. `gomd.__version__` still reads `0.0.1-M1`.

---

## How to run (see `run_scan_md.slurm`)

From the project root (folder containing `gomd/`, `pkg/`, `md-input/`):

Classical:
    python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
        --steps 200 --print-every 50 --rebuild-every 100 \
        --out md-output/md_scan_classical.out

Polarizable:
    python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
        --steps 200 --print-every 50 --rebuild-every 100 --polarizable \
        --out md-output/md_scan_polarizable.out

`--cpu` forces the CPU backend for debugging only; never use it on the V100.

---

## Change log

### 2026-11-06 — clean version, renamed to GOMD

Renamed (final name: GOMD, GPU-accelerated Open MD):
- project folder `gpu-jax-md/` → `GOMD/`; package `md_jax/` → `gomd/`
  (`python -m gomd.main_scan`); all imports and log headers
- `MD_JAX_DOCUMENTATION.md` → `GOMD_DOCUMENTATION.md` (rewritten to the current
  engine; absorbs the M3 and M4 READMEs)
- reference banners and comments "WMI-MD" → "GOMD"; env vars
  `WMI_GPU_PAIR` / `WMI_GPU_PAIR_VERIFY` → `GOMD_GPU_PAIR` /
  `GOMD_GPU_PAIR_VERIFY`
- SLURM job name `wmi_md_scan` → `gomd`
- Left as is: "jax-md" where it names the third-party JAX-MD library; inputs in
  `md-input/`.

Fixed: in the uploaded tree the package folder was `md_jax/` but every import
still read `wmi_md_jax`, so it would not run; the imports now match the folder.

Removed:
- generated files: all `__pycache__/` (cpython-38), `md_scan.out` (copy of
  `md-output/md_scan_polarizable.out`), empty `md_scan.err`,
  `md-output/md_jax.out`, `md-output/md_scan_polarizable.out`
- superseded engine code: `main.py`, `integrator.py`, `state.py`,
  `io/loader.py`, `forces/coulomb.py` (M1 Coulomb-only scaffold);
  `main_m2.py`, `integrator_m2.py`, `md-slurm` (host-loop driver, ~800 ms/step,
  replaced by `main_scan`)
- `tests/` (10 files): depended on one-off `/tmp/*.npz` dumps and a sandbox
  path, not runnable outside the sessions that made them
- reference scripts nothing imports: `pkg/check_ff_quick.py` (empty stub),
  `diagnose_vdw.py`, `verify_spline_fix.py` (one-off bug checks),
  `print_energy_breakdown.py`, `perf_snapshot.py`, `polarization_gpu.py`
  (CuPy SCF never wired in)
- `M3_POLARIZATION_README.md`, `M4_FIXES_README.md` (merged into
  `GOMD_DOCUMENTATION.md`)

Every remaining `.py` file is reachable from `gomd/main_scan.py` or
`pkg/main.py`.

### Earlier — M3 and M4

On-device neighbor list, `lax.scan` loop, segsum pair kernel (M4); full
polarizable force field wired into the loop (M3). See
`GOMD_DOCUMENTATION.md` §4 and §7.
