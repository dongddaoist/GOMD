# WMI-MD-JAX — project status

_Snapshot of a from-scratch JAX/GPU port of the APPLE&P polarizable MD engine
(PEO+Li+FSI, 8,716 atoms = 7,400 real + 1,316 dummy). Target platform: OSC
Pitzer, Tesla V100-PCIE-16GB, conda env `newenv2` (JAX 0.4.13). The 26-module
Python package in `pkg/` is the ground-truth validator; every milestone is
checked bit-for-bit against it._

This file records what is DONE and what is OPEN so the project can be resumed
cold. It reflects the reconstructed tree (original upload + the M4/M3 work).

---

## Roadmap (unchanged)

1. Efficient classical MD engine            — DONE
2. Constant-potential EDL simulations        — not started
3. ML replaces the polarization SCF          — not started (SCF seam is ready)
4. Full ML/MM                                — not started

---

## DONE

### M1–M2 — classical engine (from the original upload)
NVE velocity-Verlet, SystemState pytree, IO loader; on-device-ready neighbor
pairs, quintic splined pair forces, bonds/bends/torsions, explicit-k Ewald,
Jacobi SHAKE, dummy placement + force redistribution. Classical PE/forces
validated to ~1e-10 vs reference.

### M4 — performance (this collaboration)
The measured 800 ms/step was the **host neighbor build** (~4.5 s, mislabeled in
the old docs as "~150 ms, not the bottleneck"), amortized into every 10-step
window. Fixes:
- `forces/neighbor_ondevice.py` — all-pairs neighbor build on device; returns
  the **bit-identical** reference pair set. No host build, no host<->device sync.
- `integrator_scan.py` — whole trajectory in one `lax.scan`; neighbor rebuild
  via `lax.cond` inside the loop. One compile, no Python between steps.
- `forces/pair_splined_segsum.py` — fp32 + `segment_sum` scatter pair kernel
  (segment_sum vs `.at[].add` is bit-identical; fp32 vs fp64 is 1.6e-7 energy /
  3.3e-6 force).
- `all_forces.py` — `use_segsum_pair` branch (fp64/fp32 paths untouched).

Result on V100: **9.2 ms/step** (target was 10). Classical trajectory matches
reference `main_m2` to every printed digit.

A real bug this surfaced: the scan step must take the SHAKE reference position
from the **post-drift** position, not pre-drift (pre-drift caused KE->1e11).

### M3 — full polarizable force field wired into the loop (this collaboration)
The blocker was `charge_field_realspace` being ~23% off. Root cause: it omitted
the **1-4 Q-mu field fold-in** the reference folds into its main loop (also used
exact instead of tabulated erfc, and lacked the rcut mask). All fixed.
- `forces/pol_field.py` — complete SCF charge-field builder: real-space (+1-4
  fold-in) + reciprocal + exclusion correction + 1-4 correction. Total field
  matches reference to **1.4e-15**.
- `forces/pol_corrections.py` — induced-force phases 6 (exclusion) and 7 (1-4).
- `forces/polarization_full.py` — unified on-device compute_polarization(x,i,j):
  field -> SCF -> energy -> forces.
- `forces/polarization.py` — added a backward-compatible per-pair charge
  override (masks skin pairs in `pol_forces`) and a `min_iter` floor in
  `solve_scf`.
- `io/assembly_loader.py` — exposes `pol_data` (pol arrays, 1-4/exclusion
  lists, Ewald tables, factors).
- `integrator_scan.py` / `main_scan.py` — `--polarizable` flag; dipole history
  carried in the scan state for the (optional) ASPC predictor.

Validation (full system, vs reference): field 1.4e-15, dipoles 2.8e-15,
E_pol -7025.987542 (diff 1.2e-11), total pol force 8.0e-9. **Full-loop initial
PE -3611.5363 vs reference total -3611.5355.** Polarizable trajectory is stable;
TE drift dropped from ~18 kcal/mol/step (classical) to ~3/step.

---

## OPEN / NOT DONE

1. **ASPC warm-start is unsafe with the current solver — cold-start is the
   default.** Cold-start Jacobi converges to reference dipoles at 8e-16 every
   step. The ASPC predictor (wired, `use_aspc=False`) can converge to the WRONG
   fixed point from a biased warm start (a 1% dipole error -> E_pol ~24 kcal/mol
   off; a `min_iter` floor did not fix it — the Jacobi iteration stalls). To use
   ASPC safely the solver must be made properly contracting: Gauss-Seidel-style
   sweeps, a damped update, or a residual-based convergence test. **This is the
   main open engineering thread.** All scaffolding (history carry, `use_aspc`,
   `scf_min_iter`) is in place.

2. **Polarizable step is not yet profiled on the V100.** The 9.2 ms/step is
   CLASSICAL. The polarizable step adds the field build + ~4 SCF iterations
   (dipole-dipole contraction over ~3.3M pairs) + three correction passes. Likely
   hot spots: SCF contraction and the reciprocal field (same primitives — pair
   scatter, k-space cos/sin — as already-optimized pieces).

3. **FFT-PME for Ewald reciprocal — do NOT start until GPU-profiled.** The old
   "recip = 32% of cost / 609 ms" number is a CPU-sandbox figure; explicit-k is
   transcendental-heavy and cheap on V100 SFUs. Confirm with a real profile
   before paying the full-revalidation cost of FFT-PME.

4. **Roadmap step 2 (constant-potential EDL)** — not started.

5. **Roadmap step 3 (ML replaces SCF)** — not started, but the SCF is now a
   clean swappable seam (`solve_scf` / `make_polarization_fn`), which is exactly
   what an ML replacement plugs into.

6. `compute_recipQmu` (k-space Q-mu, ~5-10% of E_pol) is a TODO in the reference
   itself and is NOT applied to reference forces, so parity does not require it.
   Would be needed for absolute accuracy later.

---

## How to run (see run_scan_md.slurm for the full script)

From the project root (folder containing `wmi_md_jax/`, `pkg/`, `md-input/`):

Classical:
    python -m wmi_md_jax.main_scan --input-dir ./md-input --ref-dir ./pkg \
        --steps 200 --print-every 50 --rebuild-every 100 \
        --out md-output/md_scan_classical.out

Polarizable (M3):
    python -m wmi_md_jax.main_scan --input-dir ./md-input --ref-dir ./pkg \
        --steps 200 --print-every 50 --rebuild-every 100 --polarizable \
        --out md-output/md_scan_polarizable.out

`--cpu` forces the CPU backend for debugging only; never use it on the V100.
The original `main_m2` driver still exists and is unchanged (host-loop
classical reference).

---

## File map

New (M4):  forces/neighbor_ondevice.py, forces/pair_splined_segsum.py,
           integrator_scan.py, main_scan.py, run_scan_md.slurm
New (M3):  forces/pol_field.py, forces/pol_corrections.py,
           forces/polarization_full.py
Modified:  forces/all_forces.py (M4), forces/polarization.py (M3),
           io/assembly_loader.py (M3), integrator_scan.py & main_scan.py
           (created M4, extended M3 — the merge points)
Docs:      M4_FIXES_README.md, M3_POLARIZATION_README.md, PROJECT_STATUS.md
Unchanged: everything else (pkg/, md-input/, tests/, the original engine files)

## Caveat on this snapshot
This tree is the original upload + the files saved from the M4/M3 sessions. It
is faithful to where that work ended, but it does not capture anything run or
edited on the V100 afterward. If a newer version exists, it supersedes this.
