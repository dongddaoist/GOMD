# GOMD — project status

_GOMD (GPU-accelerated Open MD): a from-scratch JAX/GPU engine for the APPLE&P
polarizable force field (PEO + Li⁺ + FSI⁻, 8,716 sites = 7,400 atoms + 1,316
lone-pair sites). Platforms: Google Colab GPUs (`GOMD_colab.ipynb`) and OSC
Pitzer V100 (conda env `newenv2`, JAX 0.4.13, Python 3.8). The reference
Python package in `pkg/` is the ground truth; each milestone is checked against
it, and where GOMD departs from it on purpose the difference is a switch
(`--reference-parity`)._

This file records what is DONE and what is OPEN so the project can be resumed
cold. Technical detail lives in `GOMD_DOCUMENTATION.md`.

---

## Roadmap

1. Efficient MD engine (classical + polarizable)  — DONE
2. Constant-potential EDL simulations             — task 3: design agreed, P0 (fixed
                                                    electrodes) done 2026-11-18
3. ML replaces the polarization SCF               — not started (SCF seam is ready)
4. Full ML/MM                                     — MLIP provider DONE (UMA alongside the
                                                    PFF, or driving the dynamics);
                                                    hybrid regions not started

---

## DONE

| Milestone | Result |
|-----------|--------|
| M1 scaffold | pytree state, velocity Verlet (scaffold code since retired) |
| M2 classical engine | splined pair, bonded, explicit-k Ewald, Jacobi SHAKE, lone-pair sites; PE 6.7e-10, F 1.4e-10 vs reference |
| M3 polarization in the loop | field 1.4e-15, dipoles 2.8e-15, E_pol 1.2e-11, induced forces 8.0e-9; initial PE −3611.5363 vs −3611.5355 |
| M4 performance | on-device neighbor list, `lax.scan` loop, fp32 `segment_sum` pair kernel |
| M5 ensembles + MLIP | NVT (Nose–Hoover chain) matching the reference; MLIP provider (UMA via fairchem, own process); PFF/MLIP comparison; Colab runner; `gomd.selftest` |

V100, 200-step runs: **9.2 ms/step classical** (from ~800), **53–58 ms/step
polarizable** (56 average). Not yet re-timed with the 2026-11-16 defaults or
on Colab GPUs.

Key fixes along the way: the 800 ms/step was the ~4.5 s host neighbor build,
not the force kernels; SHAKE must constrain against the post-drift position;
the SCF field needed the 1-4 Q-μ fold-in (it was ~23% off); polarization forces
must mask skin pairs; the torsion force had two sign errors (fixed
2026-11-09); the SHAKE tolerance and the missing k-space charge–dipole force
(fixed 2026-11-16, see change log) were the remaining energy-conservation
errors.

---

## OPEN (priority order)

0. **Task 1 (2026-11-17): efficiency + ensemble validation.** The self-test
   passes locally, 12/12, including the two new SCF checks:
   - PCG = Jacobi: dipoles to 8.4e-13, in 17 vs 32 T·μ products;
   - warm start from a 1%-biased guess: PCG ends +0.0006 kcal/mol from
     converged after 3 products, Jacobi −0.19 after 4.

   Built in this task:
   - `gomd/profile_step.py` (Colab cell 10): per-phase GPU profile, stale
     neighbor-list check, Jacobi vs PCG SCF table.
   - PCG SCF solver (`--scf-solver pcg`, `--scf-guess predict`).
   - Neighbor-list drift trigger, overflow stop, missed-pair count.

   The first profile on the local RTX 5070 is in, and PCG + predict passed a
   2,000-step NVE check and is now the default (see the change log). Next,
   in this order:
   1. ~~A longer NVE run with PCG + predict, then make it the default.~~ Done.
   2. ~~A faster neighbor build.~~ Done: 186 → 16 ms; 60.5 → 46.2 ms/step.
   3. ~~A scatter-free or compacted pair layout.~~ Done: padding dropped +
      per-step rcut compaction; 46.2 → 27.8 ms/step.
   4. ~~One shared reciprocal-space pass.~~ Done and validated, but
      27.8 → 27.6 ms/step: XLA was already sharing the work.
      ~~Kernel-level trace of the fused step.~~ Done (`gomd/tests/trace_step.py`).
      ~~`pol_forces` erfc/exp~~ now from the reference tables: 27.6 → 21.6 ms/step.
      Remaining levers, from the trace:
      - merging the pair scatters, ~13 ms over ~10 passes;
      - fewer, larger k-space matmuls, ~4 ms.
   5. ~~Ensemble validation.~~ Done (see the change log):
      - NVE drift is negligible, and constrained-bond velocities are ~0.
      - The polarizable fluctuation has a dt-independent floor from the hard
        μ-μ cutoff in this input.
      - NVT gives T and canonical KE fluctuations exactly.
      - Induced forces = −∇E_pol.
      - Equipartition is pending an equilibrated configuration.
   6. ~~**Restart output.**~~ Done: `--restart-dir` / `--restart-every`
      write a ready-to-use input directory, and Colab cell 4 has
      `INPUT_DIR` / `RESTART_EVERY`. Next:
      - equilibrate in chained runs (ns scale), then repeat the
        equipartition test;
      - trajectory output is still missing.
   5. Ensemble validation: NVE timestep halving, NVT KE fluctuations and
      per-element temperatures, velocity along constrained bonds.

   Colab cell 10 on the L4 then gives the production numbers.
1. **Validate UMA on a Colab GPU** (task 2). Use `gomd.mlip_md` (Colab
   cell 11) on `md-input2/sphere_2000.xyz`; the 7,400-atom box does not fit
   a 40 GB A100. First run (2026-11-18, L4, `high-precision`; see the change
   log):
   - ~~runs, memory, ms per evaluation~~: fits the L4, 583 ms/step;
   - ~~energy resolution~~: ~1e-3 kcal/mol, not the fp32 step;
   - ~~forces vs the energy~~: consistent within that noise. The check is
     now a 4-point stencil at 0.01 Å, so the next run gives a clean number.

   - ~~first NVT~~: run (1 ps), and it exposed two driver issues:
     the thermostat ran in fp32 (H_ext drift) and chain 1 rang. Both are
     fixed (fp64; chain 3 by default), checked on the mock.

   - ~~NVT, fixed driver~~ (1 ps, chain 3, fp64): H_ext drift +2e-5
     kcal/mol/fs; T 295.0 K with sd 5.45 K, the canonical value being 5.40.
   - ~~restart continuity~~: across two hand-offs KE carries over exactly
     and PE to 4e-4 / 8e-4 kcal/mol. That is UMA's noise from one process to
     the next; positions are written with 12 decimals.

   - ~~NVE dt test~~: from the 296 K NVT restart, sd(TE) 0.0851 / 0.0214
     kcal/mol at 0.5 / 0.25 fs, a ratio of 3.98 (dt²); drift ≤ 2e-5
     kcal/mol/fs.

   **Task 2 validated on the cluster** (2026-11-18). The scorecard is in
   `GOMD_DOCUMENTATION.md` §5. Optional follow-ups:
   - `default` (TF32) vs `high-precision`: speed and energy noise;
   - omc vs omol;
   - the periodic 7,400-atom box. It needs more than 40 GB as run; the
     `lowmem` / `lowmem-fp64` settings, with activation checkpointing,
     are untested.
   - Notebook clean-up, with the Colab tab closed:
     - cell 5: its own `INSTALL_UMA` switch, VM cache by default, and a
       check that the checkpoint file is real;
     - cell 11: expose `--temp`, `--nhc-chain` and `--fd-h`.
1b. **Constant potential** (task 3, started 2026-11-18).
   The old Fortran constant-potential code is on a NAS that is currently
   disconnected. The design follows Siepmann & Sprik 1995 and Reed, Lanning
   & Madden 2007, with 2D Ewald per Hu 2014.

   **Requirements (user):**
   - fixed electrodes along z by default; the lower one negative, the upper
     one positive;
   - charge on the electrode sites;
   - a Poisson potential profile along z;
   - control over the rate of charge adjustment;
   - 2D Ewald instead of 3D;
   - polarization and constant potential must work together (in OpenMM the
     AMOEBA force could not be combined with a constant-potential setup).

   **Design:**
   - Gaussian charges on the electrode sites (η ≈ 1.979 Å⁻¹, to be tested).
   - Electrode charges q and induced dipoles μ minimize one energy.
     Forces are Hellmann–Feynman: −∂U/∂R at fixed q and μ.
   - q–μ couplings use one shared kernel in both directions, so the joint
     matrix stays symmetric.
   - Per step, q is eliminated with A⁻¹, inverted once (3,584 sites →
     103 MB), and the existing PCG runs on μ with (α⁻¹ − T) − C A⁻¹ Cᵀ.
   - Neutrality comes from a Lagrange multiplier.
   - 2D Ewald is computed as the trapezoid (EW3DC-equivalent) form in the
     existing explicit-k pass, plus the planar dipole term. It is validated
     against the exact pairwise 2D Ewald on small systems.
   - The Poisson profile includes the dipoles' bound charge. Its total drop
     must equal ΔV.

   **Phases:**
   - ~~P0 fixed electrodes~~: done, see the change log and doc §4.23.
   - P1: slab electrostatics (planar dipole term, Gaussian kernels, exact
     2D reference). This includes choosing α / kmax for a converged k-space;
     the input's α 0.23 with kmax 7/7/16 is loose.
   - P2: constant potential without polarization: capacitor, image-charge
     and test-charge checks, gauge invariance, NVE, Poisson drop = ΔV.
   - P3: coupling with polarization: forces vs finite differences of the
     minimized energy, and the α → 0 limit.
   - P4: production: ΔV ramp, per-electrode charge output, Poisson tool,
     performance.

   **Open questions for the user:**
   1. How did the old code update the charges: an exact solve, fictitious
      mass, or Poisson feedback? What did "rate" control?
   2. Keep the frozen geometry flat? Done by default; `--electrode-geometry
      snapshot` keeps the input.
   3. Is the slab dispersion tail correction needed? The old run used it
      (`surftrunc`, refreshed every 50 ps from the z-density profile).
   4. Is the trapezoid form of 2D Ewald acceptable?
2. **NPT.** Refused: the reference barostat is fed an invalid pressure (pair
   virial set equal to the pair energy; bonded, SHAKE-constraint and
   lone-pair virials missing; Ewald virial never reset). A correct virial,
   from autodiff of E with respect to a box strain plus the constraint virial,
   is the prerequisite.
3. **Contracting SCF solver.** Converged E_pol needs tolpol ≈ 1e-8 (about 12
   Jacobi iterations instead of 4). ASPC warm starts would cut that, but are
   unsafe with the Jacobi step-change test. The reference's own warm-start
   SCF drifts its logged PE by +0.04 kcal/mol per step from a fresh
   evaluation. *2026-11-17, done:* PCG with a residual test, the
   variational energy and a predicted start is the default. It matches
   Jacobi's dipoles to 4.9e-14 and its 2,000-step NVE drift, with 2.3 T·μ
   products per step instead of 10.
4. **Profile on the GPU** with the new defaults (SHAKE ~11 sweeps, k-space
   charge–dipole force by autodiff, tighter tolpol), per phase. FFT-PME waits
   for this profile. *2026-11-17:* the profiler is in (`gomd.profile_step`,
   Colab cell 10) and has not been run yet.
5. ~~**Polarization force check.**~~ Done 2026-11-17:
   `gomd/tests/check_polforce_fd.py` checks the total induced force against
   −dE_pol/dx by finite differences (8.5e-4 relative, table-limited), with
   the taper on and off. It runs on the GPU, separately from the CPU
   self-test.
6. **Hybrid PFF/MLIP** (subtractive ONIOM region first), multiple time
   stepping with an MLIP correction, ML replacement of `solve_scf`.
7. Smaller items:
   - ~~The on-device neighbor list drops pairs silently past its capacity.~~
     2026-11-17: the build returns its pair count; an overflow stops the run.
   - RESPA (Large Steps > 1) is not ported; the driver warns.
   - The reference ignores the SHAKE tolerance in `mdrun.params` (it uses a
     hard-coded 1e-6).

---

## How to run

Google Colab: put the project at `My Drive/GOMD/code/GOMD` (with `pkg/`),
open `GOMD_colab.ipynb`, choose a GPU runtime and run the cells in order.
For UMA, add the Colab secret `HF_TOKEN` with access to `facebook/UMA`.

Command line, from the project root (folder containing `gomd/`, `pkg/`,
`md-input/`):

    python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
        --steps 200 --print-every 50 --polarizable --tolpol 1e-8 \
        --out md-output/md_scan_polarizable.out
    # + --ensemble nve|nvt (default: mdrun.params)
    # + --mlip uma --compare-every 50            PFF dynamics, UMA alongside
    # + --mlip uma --propagate mlip               UMA dynamics, PFF alongside
    # + --reference-parity                        reproduce the reference pkg
    python -m gomd.selftest --input-dir ./md-input --ref-dir ./pkg
    python -m gomd.profile_step --input-dir ./md-input --ref-dir ./pkg \
        --polarizable --tolpol 1e-8 --out md-output/profile.txt
    # main_scan extras: --scf-solver pcg --scf-guess predict --skin 1.0
    #                   --no-drift-rebuild --nb-capacity 1.25

`--cpu` forces the CPU backend for debugging only.

Locally on the Windows laptop: WSL Ubuntu, micromamba env `gomd` (Python
3.12, jax 0.12 with CUDA 12, numpy, scipy, numba), RTX 5070 with 8 GB. G: is
not mounted inside WSL, so run from Git Bash in the project root:

    tools/wsl_run.sh -m gomd.selftest --input-dir ./md-input --ref-dir ./pkg
    tools/wsl_run.sh -m gomd.profile_step --input-dir ./md-input --ref-dir ./pkg \
        --polarizable --tolpol 1e-8 --out md-output/profile_local.txt

The script mirrors gomd/, pkg/ and md-input/ into `~/gomd_run` in WSL, runs
there, and copies md-output/ back.

---

## Change log

### 2026-11-18 — constant potential P0: fixed electrodes, graphite cell `md-input3`

**Input:** `md-input3/` is the user's interface run (`~/run4-resize` in WSL),
copied unchanged with a README:
- 17,544 sites: 3,584 graphite C in two electrodes of 4 sheets each,
  480 DME, 40 Zn(OTf)₂, 80 Pyr14TFSI;
- box 33.84 × 34.19 × 131.6 Å; the inner sheets are 107.7 Å apart;
- the old run was NVT at 300 K with no SHAKE, flexible sheets, and the two
  electrodes back to back across the z boundary.
- The Fortran `WMI-MD.f` in that folder has no constant-potential code. Its
  z slicing is the slab dispersion tail correction `surftrunc`.

**Loading fixes in `pkg`:**
- `params.h` raises the hard-coded array limits;
- Fortran `1.d-14` / `1.0d-9` literals in `mdrun.params` are accepted;
- `md-input` behaves exactly as before.

**`gomd/electrodes.py`** and `main_scan --electrode-lower/--electrode-upper/
--electrode-geometry/--slab-height` (doc §4.23):
- the electrodes are selected, flattened (graphene snapped to the ideal
  lattice) and frozen;
- an optional slab box adds vacuum outside the electrodes;
- terms among frozen atoms only are dropped, and so are frozen–frozen
  neighbor pairs;
- ndof becomes 46632 → 35880;
- the driver now refuses out-of-plane terms on mobile atoms, which GOMD
  never evaluated.

Restarts copy `params.h` and keep a scaled kmax_z. `main_scan --dt` was added.

**Results** (RTX 5070, polarizable, tolpol 1e-9):
- **Check script:** `gomd/tests/check_electrodes.py` all pass. The bulk
  self-test is still 14/14, and the bulk restart test still passes.
- **Frozen atoms:** moved 0 Å with zero velocity in every run.
- **Neighbor list:** 1.54 M frozen–frozen pairs left out (periodic box),
  leaving 7.75 M.
- **Speed:** 57 ms/step in the periodic box, 67 ms/step in the 205 Å slab
  (kmax_z 25). The old Fortran took about 225 ms/step on 40 MPI ranks.
- **NVE, 200 fs:** TE within ±1 kcal/mol (dt 0.5 fs) and ±0.4 (dt 0.25 fs),
  with no drift.
- **NVT from 313 K** (the electrolyte's temperature over its own 35,880
  dof):
  - H_ext fell 2.4 kcal/mol over 90 fs while about 1,000 kcal/mol of heat
    was removed; at dt 0.25 fs it fell 0.35.
  - So this is the thermostat's time-step error under strong heat flow, not
    the freezing. Bulk `md-input` cooling from 503 K shows it too: −3.0 per
    4,100 kcal/mol of heat removed. It is larger here because this input
    has no SHAKE.
- **Slab restart round trip:** the continuation tracks the uninterrupted run
  to 0.005 kcal/mol in PE, KE and H_ext.

### 2026-11-18 — UMA on Colab: what failed, and a 2,000-atom cluster

**UMA attempts on `md-input2/system.xyz` (7,400 atoms), in order:**
1. **Model load failed.** huggingface_hub's cache uses symbolic links, and
   Drive stored them as 79-byte text files. Fix: cache on the VM
   (`UMA_CACHE_ON_DRIVE` off).
2. **L4, `default` (= nitro):** out of memory at 21 of 22 GB on the first
   force evaluation.
3. **A100-40, `high-precision`:** first failed to import, because Colab's
   `MPLBACKEND` is unknown to the UMA venv. Fixed in the server (Agg).
4. **A100-40, `high-precision` again:** two evaluations completed:
   - PE −208,180,776.6 kcal/mol (−1219.94 eV/atom), |F| rms 18.7 kcal/mol/Å,
     ΣF 2.9e-4;
   - the same configuration twice gives identical E;
   - the third evaluation ran out of memory at 37.2 GB. The server now sets
     `expandable_segments`.
5. **A100-40, `default`:** out of memory at 38.7 GB on the first evaluation.
   Fairchem's presets turn activation checkpointing off. The server now also
   offers `lowmem` / `lowmem-fp64`, untested on fairchem.

**Energy resolution:** at |E| ≈ 2×10⁸ kcal/mol an fp32 sum can only resolve
~25 kcal/mol, so it decides whether finite-difference and NVE energy checks
are possible. Answered on the cluster below: the energy is resolved.

**Smaller system** (`gomd/io/make_subsystem.py`, user request: complete
molecules, real atoms only):
- `md-input2/sphere_2000.xyz`: a cluster in vacuum, R 18.56 Å, 2,002 atoms
  in 108 complete molecules (86 solvent, 11 Li⁺, 11 FSI⁻), neutral,
  pbc F F F. Closest contact between molecules 1.845 Å.
- The periodic `cube` variant overlapped at its faces (0.089 Å), so it was
  discarded.
- `pbc=False` is supported in `mlip_md`, the client and the mock.
- Mock NVE on the cluster: sd(TE) scales as dt²; forces = −∇E to 1.5e-4.

**First UMA run on the cluster** (`md-output/mlip_md_uma_nve_20261118-151039.out`;
Colab L4, uma-s-1p2p1 omol, `high-precision`, NVE, 200 steps of 0.5 fs):
- **Fits the 22 GB L4. 583 ms per evaluation**, ≈ 74 ps/day at 0.5 fs.
  Model load 9 s.
- **Initial state:** PE −57,078,476.47 kcal/mol (−1236.34 eV/atom),
  |F| rms 18.66 kcal/mol/Å, ΣF 2.1e-4. The same configuration twice gives
  identical E.
- **Energy is resolved to ~1e-3 kcal/mol**, not the fp32 step of 6.8 kcal/mol
  at that |E|. The residual roughness is the fp32 network itself.
  - Probe: ΔE = −|F|·s + ½k s², with k fitted from the 0.01 Å point. At
    1e-4 / 1e-3 Å the misfit is 1.7e-3 / 8.3e-4 kcal/mol at the start and
    5.1e-4 / 1.1e-5 at the end. It does not grow with s, so it is noise,
    not a wrong slope.
  - **Forces = −∇E within that noise.** With 2-point differences at
    h = 1e-3 Å the errors were 0.002–1.2 kcal/mol/Å (rms 0.4), which is
    ε/2h for ε ≈ 1e-3. That shows as "max rel diff 1.8e-1".
  - The check is now a 4-point stencil at h = 0.01 Å (`--fd-h`): truncation
    O(h⁴), noise ~0.1 kcal/mol/Å. On the mock: 7e-7, vs 1.5e-4 before.
- **NVE:** after the start-up, TE sd is 0.18 kcal/mol about its fit while KE
  swings over 230 kcal/mol. The drift, −0.0024 kcal/mol/fs, is not
  significant over 90 fs.
- **Start-up transient:** within the first 5 fs PE falls 366 kcal/mol and T
  rises 353 → 414 K, then settles at a 398 K mean; TE shifts −0.9 kcal/mol.
  That time is a quarter of an X–H vibration, so it is the bonds relaxing
  from the PFF's constrained lengths, with no bond velocity to start. The
  source state is also hot: 503.7 K over the PFF's constrained dof, at
  t = 1.5 ps of a 296.15 K NVT run. A UMA NVT equilibration should come
  before production or a dt-scaling test.

**First UMA NVT** (`md-output/mlip_md_uma_nvt_20261118-153123.out`; 2,000 steps
of 0.5 fs at 296.15 K, chain 1, OmegaQ 0.01, from the NVE restart; 602
ms/step):
- **The 4-point force check now resolves the forces:** max abs diff 0.044
  and 0.088 kcal/mol/Å at the start and end, vs up to 1.2 with 2 points.
  The relative diffs, 1e-3 to 7e-2, are larger only on the small
  components. Forces = −∇E to the energy's own noise.
- **T rang instead of settling:** 387 → 219 → 373 → 248 K, at the
  thermostat's own period (~600 fs for OmegaQ 0.01 1/fs). It averaged
  283 K with sd 46 K, and the thermostat removed 1,766 kcal/mol of heat.
- **H_ext fell 3.9 kcal/mol** (−0.0021 kcal/mol/fs), mostly while heat was
  being removed.

**Driver bug found from it, fixed in `gomd/mlip_md.py`:**
- **What was wrong:** `mlip_md` never enabled `jax_enable_x64`, so the
  thermostat (JAX, on the CPU) ran in fp32 and rounded v and the chain
  state every half step. `main_scan`, `profile_step` and `selftest` all
  enable it.
- **How it showed on the mock (cluster, 1 ps):** H_ext drift −0.0014 /
  −0.0022 kcal/mol/fs at dt 0.5 / 0.25 fs, not falling with dt. That is
  ~ −6e-4 kcal/mol per step. The 7,400-atom mock NVT of 2026-11-17
  (−0.004) had the same cause. NVE never used JAX and is unaffected.
- **After the fix:** +2e-5 / +1e-5 kcal/mol/fs, and sd(H_ext) 0.046 /
  0.011 kcal/mol, a ratio of 4.0, which is dt².
- **Chain length now defaults to 3.** With 1, the thermostat rings: sd(T)
  23.8 K vs 5.9 K with 3, the canonical value being 5.4 K (mock, 1 ps,
  same mean T). With 3 the H_ext drift is −0.00000.
- **Fresh chain on a length change:** a restart whose chain has another
  length starts a fresh chain, and the log says so.

**Second UMA NVT, fixed driver** (`md-output/mlip_md_uma_nvt_20261118-160311.out`;
1 ps at 296.15 K, chain 3, fp64, from the first NVT's restart):
- **H_ext drift +0.00002 kcal/mol/fs**, sd 0.089 kcal/mol about the fit,
  the same as the mock.
- **T rose 273 → ~296 K in ~350 fs, then held:** mean 295.02 K, sd 5.45 K,
  the canonical value being 296.15·√(2/6003) = 5.40 K.
- **Restart continuity:**
  - KE carries over exactly across both hand-offs (NVE → NVT → NVT).
  - PE carries over to 4e-4 and 8e-4 kcal/mol: UMA's noise between
    processes, since positions are written with 12 decimals.
- **The thermostat cost 57 ms/step on Colab** (644 vs 587 ms/step for the
  MLIP). The chain's unrolled scalar updates were dispatched one op at a
  time. Now `jax.jit`, compiled once: on the mock, chain 3 went from 19.4 to
  1.1 ms/step with identical results, and a resume reads the chain state.

**NVE time-step test** (`md-output/mlip_md_uma_nve_20261118-162854.out`,
`-163545.out`; both from the second NVT's restart at 296 K, 200 fs each):

| dt | steps | sd(TE) about the fit | drift | TE mean − initial |
|----|-------|----------------------|-------|-------------------|
| 0.5 fs | 400 | 0.0851 kcal/mol | +0.00002 kcal/mol/fs | −0.048 |
| 0.25 fs | 800 | 0.0214 kcal/mol | −0.00001 kcal/mol/fs | −0.021 |

- **The ratio is 3.98, which is dt².** UMA's energy noise, ~1e-3 kcal/mol,
  sets no floor at these step sizes.
- **No start-up shift from an equilibrated start.** The first NVE's
  −0.0024 kcal/mol/fs was its non-equilibrium relaxation.
- **Restart continuity:** both runs start from KE 1768.7403, exactly the
  NVT's last step. PE matches to 1.2e-3 and 1.0e-3 kcal/mol.
- **613 ms per evaluation**, NVE, L4.

### 2026-11-17 — task 2 started: lone-pair-free system + MLIP-only MD driver

MLIP runs are tested separately from the PFF, on systems without lone pairs.

- **`md-input2/`** (built once with `python -m gomd.io.make_lp_free`):
  - `system.xyz`: the `md-input` `coords.inp` state with the 1,316
    lone-pair sites removed, in extended XYZ. That is 7,400 atoms,
    H2880 Li36 C1920 N36 O784 F1672 S72, cubic 46.9566 Å, periodic.
  - It holds exactly what UMA takes (elements, positions, cell, pbc,
    charge 0, spin 1), plus velocities (m/s), masses and `site`, the index in
    the source system.
  - No connectivity and no force field. Read back exactly. Its velocities are
    a 344.7 K state over 3N − 3 = 22,197 dof, the same state that counts
    as 503.7 K over the PFF's 15,192 constrained dof.
  - A README records the provenance.
- **`gomd/io/xyz.py`**: extended-XYZ read/write (ASE-compatible layout,
  velocities in m/s).
- **`gomd/mlip_md.py`**: MD driven by the MLIP alone, from an XYZ:
  - velocity Verlet in the PFF integrator's units, with no constraints;
  - NVE, or NVT with the same Nose–Hoover chain module;
  - COM momentum removed;
  - `--fd-check N` checks the forces against finite differences of the
    energy, at the start and at the end;
  - `<out>_restart.xyz` carries the chain state, plus an optional
    trajectory.
  - The MLIP server and client are reused unchanged.
- **Colab cell 11** runs it on `md-input2` with UMA (after cell 5) or the
  mock.
- **Mock-MLIP validation (local):**
  - NVE drift ≤ 7e-5 kcal/mol/fs; sd(TE) 0.035 / 0.139 / 0.572 at
    dt 0.25 / 0.5 / 1.0 fs, ratios 4.0 and 16.5, which is dt².
  - Forces vs finite differences: 2.1e-4, the truncation error for
    h = 1e-3 Å.
  - NVT 2,000 steps: mean T 296.0 K.
  - The restart carries KE exactly and the chain state. The mock's own
    springs are rebuilt from whatever frame it first sees, so continuity
    across a restart has to be checked with UMA.
- **Not done yet: UMA itself.** It needs a fairchem environment, access to
  the gated `facebook/UMA` model, and enough GPU memory for 7,400 atoms.
  The doc warns a 15 GB T4 may not suffice; the local RTX 5070 has 8 GB.

### 2026-11-17 — restart output

- **Writer:** `gomd/io/restart.py` writes `coords.inp` from the input
  `coords.inp` as a template, so every site keeps its label and charge type:
  - positions are written with 12 decimals and velocities with 9;
  - the Nose–Hoover chain state is written for NVT runs;
  - all sites are wrapped into [0, L), and `Time` is advanced.
  It also copies the other inputs, so the next run is `--input-dir <restart dir>`.
- **`main_scan`:** `--restart-dir` (default `<out>_restart`, `none` to
  disable) and `--restart-every N`. Nothing is written after an overflow.
- **Colab cell 4:** `INPUT_DIR` (continue from a restart folder in
  `md-output/`) and `RESTART_EVERY` (default 10,000). Cell 6 prints the
  restart folder to use next.
- **Round trip** (`gomd/tests/check_restart.py`, polarizable NVT, 100 + 100
  steps):
  - read back through the reference parser: positions 1.4e-12 Å, velocities
    5e-10 m/s, chain state exact;
  - the continued run tracks the uninterrupted one: ΔPE 0.0002 at step 1 and
    0.14 at step 100; ΔH_ext 0.05 → 0.42 kcal/mol.
  - The step-1 offset is the SHAKE part of the last force, which
    `coords.inp` cannot hold.
- **End to end:** a 100-step run with `--restart-every 50`, then 50 steps
  from its restart. The second run starts with the same KE, PE within
  5e-4 kcal/mol, and H_ext continuing the first run's trend.

### 2026-11-17 — ensemble validation (NVE, NVT, constraints, induced forces)

Tools: `gomd/tests/ensemble_check.py` (`nve` / `nvt`; observables computed
on the GPU) and `gomd/tests/check_polforce_fd.py`. Reports are in
`md-output/ensemble_*.txt`. Everything starts from `coords.inp` on the RTX
5070.

**Induced forces = −∇E_pol** (finite differences on 8 atoms × 3 components,
tight SCF): max relative difference 8.5e-4, set by the piecewise-linear erfc
table. The taper on (rtaper 10 Å) agrees equally well. This closes OPEN item
5. The one outlier is a pair 7e-5 Å from rcut that the ±1e-4 Å displacement
carries across the cutoff: E_pol jumps there by ~1e-4 kcal/mol. The check
now detects such crossings and leaves them out.

**NVE, 1 ps per timestep, polarizable (tolpol 1e-8, PCG + predict):**

| dt | TE drift (kcal/mol/fs) | sd(TE) | sd(TE)/sd(KE) |
|---|---|---|---|
| 0.25 fs | −0.0009 | 0.384 | 3.1e-3 |
| 0.5 fs | −0.0004 | 0.420 | 3.4e-3 |
| 1.0 fs | +0.0007 | 0.770 | 6.2e-3 |

- **Drift:** negligible, with no trend in dt.
- **Fluctuation:** it does **not** scale as dt². A floor of ~0.38 kcal/mol
  is set by the hard dipole–dipole cutoff at 12 Å. The input has
  rtaper 14.5 > rcut 12, so no taper applies, in the reference too, and each
  pair crossing the cutoff makes E_pol jump by ~1e-4 kcal/mol.
- **Classical runs scale as dt²:** sd(TE) 0.061 / 0.154 / 0.583
  (ratios 2.5 and 9.6 against an ideal 4 and 16).
- **With a taper** (`--rtaper 10`): the floor is gone (0.056 / 0.162 /
  0.633), but a small dt-dependent drift appears (+0.0025 kcal/mol/fs at
  0.5 fs). Not adopted; the taper also changes the model.
- **Velocity along constrained bonds:** 0.000% of KE at every dt; rms
  0.2 / 1.0 / 4.0 m/s, scaling as dt². The SHAKE predictor–corrector keeps
  velocities consistent with the constraints, so no RATTLE step is needed and
  ndof is right.
- **ndof:** all 7,008 real bonds are constrained (21 per solvent molecule, 8
  per FSI⁻, no rings), so ndof = 320·45 + 36·19 + 36·3 = 15,192, as used.

**NVT, polarizable:** 15 ps equilibration, then 20 ps production.

| check | result |
|---|---|
| mean T | 296.14 ± 0.05 K (Tset 296.15) |
| sd(KE) | 51.31 ± 2.7 kcal/mol vs canonical √(ndof/2)·k·Tset = 51.29 (ratio 1.000) |
| H_ext drift | +0.00074 kcal/mol/fs (~0.4 kJ/mol/ns per atom) |

The classical NVT gives T 296.14 ± 0.11 K and an sd(KE) ratio of
1.12 ± 0.08, with H_ext drift −0.00004.

**Equipartition** (molecular center-of-mass temperature against T):
**not testable yet, because the system is still relaxing.**
- In NVT production, solvent T_com is 315 K (polarizable) and 316 K
  (classical), against T = 296 K.
- But PE is still falling by 6.6 and 4.5 kcal/mol/ps, and the gap closes over
  the run: polarizable solvent T_com goes 324 → 296 K over the ten 2-ps
  blocks, while the internal T rises 293.6 → 296.0 K.
- The gap persists in an NVE continuation (classical: solvent 305 against an
  internal 291.5 K), so it is not the thermostat.
- The 503 K `coords.inp` state is far from equilibrium. The relaxation
  energy goes into intermolecular motion, and the global thermostat cools
  every mode equally.
- Repeat the test from an equilibrated configuration. That needs restart
  output, which GOMD does not write yet.

**Also found:** the reference `dtau_func` = (r − ro)(r − rtaper)²/tauden is
not the derivative of its own `tau_func`. GOMD now implements the taper with
the true derivative (r − rtaper)(r − ro)/tauden, verified against
finite differences. It is inactive for this input.

### 2026-11-17 — induced-force pair term uses the reference erfc/exp tables

- The reference `compute_pol_forces` (`pkg/polarization.py:505`) evaluates
  erfc(αr) and exp(−α²r²) from the 16,384-knot tables in `ewald_tables.py`,
  as the SCF field does. GOMD's port had used exact fp64 `erfc`.
- `pol_forces(..., erfc_tab, exp_tab, dx_inv)` now uses those tables (via
  `pol_field._tab_lookup`), and `polarization_full` passes them.
- `gomd/tests/bench_polforces.py`, against exact fp64:
  - tables: max |ΔF| 2.7e-7 kcal/mol/Å (1e-8 of the largest force);
    7.91 → 5.81 ms;
  - fp32 transcendentals: 10× the error for the same speed, so not used.
- **Kernel trace:** the induced-force block went from 17.2 to 12.8 ms, and
  total kernel time from 38.9 to 34.6 ms.
- **2,000-step NVE: 27.6 → 21.6 ms/step.** Drift fit −0.00036 kcal/mol/fs
  and TE sd 0.42, both unchanged; PE within 0.003 kcal/mol of the previous
  run over 1,000 steps. Self-test 14/14.
- Polarizable step on the RTX 5070 today: 103 → **21.6 ms** (4.8×), at the
  same tolpol and the same energy conservation.

### 2026-11-17 — kernel-level trace of the fused step

`gomd/tests/trace_step.py`:
- builds the step as `main_scan` does;
- records 20 steps of the `lax.scan` with the JAX profiler;
- maps every GPU kernel to its HLO op, its name path (jit names plus the new
  `gomd_*` named scopes, which cost nothing at run time) and its source line.

XLA command buffers (CUDA graphs) hide the individual kernels, so the tool
turns them off (`--command-buffers` keeps them). That adds ~3 ms/step of
launch gaps: 4.3 ms off against 1.4 ms on.

Report: `md-output/trace_polarizable_RTX5070.txt`. Polarizable NVE, PCG +
predict, tolpol 1e-8. The laptop GPU was in a slower state here: 41.6 ms/step
wall, 38.9 ms of kernels, 328 launches per step. Shares of kernel time:

| block | ms | share | main kernels |
|---|---|---|---|
| induced forces | 17.2 | 44% | `pol_forces` pair fusion 8.4 (exact fp64 erfc/exp per pair); its scatters 4.7; k-space Q–μ matmuls 3.0 + 0.9 |
| SCF | 5.5 | 14% | T·μ scatters 4.7 (2.25 products/step) |
| classical | 5.3 | 14% | pair kernel 2.7, scatter 1.7 |
| k-space tables | 3.2 | 8% | cos/sin + S(k) 1.8, matmuls 1.2 |
| SCF field | 2.9 | 8% | scatters 2.0 |
| geometry | 1.6 | 4% | |
| neighbor rebuild | 1.1 | 3% | 2 rebuilds in 20 steps |
| compaction / SHAKE / lone pairs | 0.9 | 2% | |

Levers, largest first:
1. **The `pol_forces` pair fusion (8.4 ms)** evaluates erfc and exp exactly
   in fp64 for 2.9 M pairs, and fp64 is 7–9× slower than fp32 on this GPU.
   The SCF field already uses the 16,384-knot erfc/exp table, and the
   reference's force routine uses a fine erfc table too. Using the table, or
   fp32 transcendentals, here should save most of the 8.4 ms.
2. **Pair scatters (~13 ms)**: about ten passes per step. Precision does not
   matter (fp32 is as slow); the number of passes does. The classical and
   induced pair forces could share one scatter.
3. **k-space matmuls (~4 ms)** are memory-bound fp64 (nat × 709) passes.
   Merging them into fewer, larger passes should save 1–2 ms.

### 2026-11-17 — one shared k-space pass (correct, but no real speedup)

`forces/recip_shared.py` builds cos/sin(k·r) once per step and derives all
three reciprocal-space terms from it:
- the classical energy and force;
- the SCF field;
- the k-space Q–μ force, now in closed form instead of `jax.grad` of the
  field pass.

The table itself is factorized as exp(ik·r) = exp(ik_x t_x) exp(ik_y t_y)
exp(ik_z t_z). That needs 38 cos/sin evaluations per site instead of 709,
plus two complex multiplies per (site, k).

- **Agreement** with the separate passes (self-test check 8, 14/14):
  energy 9e-16, force 9e-15, field 7e-15, Q–μ force 2e-15.
- **Timed one evaluation at a time:** 16.0 → 6.0 ms (2.7×).
- **In the fused MD step:** 27.8 → 27.6 ms/step. XLA had already shared most
  of the common work inside the one jit'd step, so the separate-pass timings
  overstated the opportunity. The change stays: it is validated, and it
  drops the autodiff backward pass.
- Conservation is unchanged (drift −0.00035 kcal/mol/fs, TE sd 0.42), and PE
  stays within 0.001 kcal/mol of the previous run for 1,000 steps.
- Classical NVE, 2,000 steps: 6.6 ms/step on the RTX 5070; drift +0.00016
  kcal/mol/fs, TE sd 0.15.

**Lesson:** per-phase timings of separately jit'd pieces no longer predict
the fused step. The next profile has to time GPU kernels inside the fused
step, from an XLA/JAX profiler trace.

### 2026-11-17 — pair kernels: padding dropped, per-step rcut compaction

`gomd/tests/bench_pairs.py` timed one T·μ product on the RTX 5070 in several
layouts.
- The padded `(0,0)` slots were the problem: 838 K zeros added into atom 0 in
  every pair scatter.
- Sending them to a dropped index halved the product, 6.78 → 3.59 ms.
- A prefix-sum (scatter-free) layout reached about the same, 3.50 ms; a
  sorted `segment_sum` gave nothing (6.87); a dense full neighbor table was
  slower (9.06).

Two changes followed, each checked against a 2,000-step NVE run:

1. **Padding dropped** in every pair scatter: `town_product`,
   `_field_pairs_tab` (also drops pairs outside the cut), `pol_forces`, and
   the classical pair kernels. 46.2 → **32.9 ms/step**.
2. **Per-step compaction** (`compact_pairs`, `--cut-capacity 1.10`).
   - `force_fn` cuts the rlist list to the pairs inside rcut + 0.05 Å
     (2.67 M of 3.35 M; 2.94 M slots instead of 4.19 M).
   - Every pair kernel runs on that list. The margin covers both the fp32
     and the fp64 rcut tests.
   - An overflow stops the run, as for the neighbor list. The peak was 91%
     of capacity.
   - The compaction itself costs 1.3 ms. 32.9 → **27.8 ms/step**.

Conservation was unchanged at every stage (drift fit −0.00036 kcal/mol/fs,
TE sd 0.42). PE stayed within 0.001 kcal/mol of the previous run for 1,000
steps. The self-test passes 13/13.

Today's progress on this GPU, polarizable NVE, 2,000 steps, tolpol 1e-8,
same energy conservation throughout:

| configuration | ms/step |
|---|---|
| Jacobi, cold start (morning) | 103.0 |
| PCG + predicted start | 60.5 |
| + fast neighbor build | 46.2 |
| + padding dropped | 32.9 |
| + per-step rcut compaction | **27.8** (3.7×) |

Profile after both changes (`md-output/profile_polarizable_RTX5070_20261117b.txt`,
one evaluation each, compacted list):
- T·μ product 2.5 ms (was 9.2); field real-space 3.6 (12.8); induced-force
  pair term 7.8 (22.3); geometry 2.4 (4.8); classical pair 2.5 (8.0).
- The three reciprocal-space passes (Ewald 4.7, field 4.8, k-space Q–μ force
  5.7) now make up the largest group, about 15 ms.
- The small kernels each show a ~1.3 ms floor when timed alone, which is
  launch and sync latency; they are cheaper inside the fused step.
- The laptop clocks vary: the profiler's own MD section read 46 ms/step right
  after the 2,000-step run measured 27.8. Use the standalone runs for
  comparisons.

### 2026-11-17 — neighbor build 11.7× faster

- `build_neighbor_ondevice_count` was rewritten in two passes:
  1. candidates inside rlist are compacted, and rejected ones dropped through
     an out-of-range scatter slot;
  2. exclusions are removed on the ~3.4 M compacted keys, with int32 keys,
     and the list is compacted again.
- The old build sent every rejected candidate (about 73 M per build) to one
  dummy slot. That serialized the scatter.
- RTX 5070: **16 ms per build instead of 186 ms**. The output is element for
  element identical to the old build (`gomd/tests/bench_neighbor.py`), and
  overflow is still reported.
- 2,000-step polarizable NVE (PCG + predict, 1e-8): **46.2 ms/step instead of
  60.5**. It made the same 159 rebuilds, and the drift fit matches to five
  digits. PE stays within 0.0005 kcal/mol of the old-build run for 1,000
  steps. Rebuilds now cost ~1.3 ms/step.
- Versus the Jacobi cold-start 1e-8 run at the start of the day (103 ms/step),
  the step is now 2.2× faster on this GPU, with the same energy conservation.
- Self-test check 7 is new: the on-device build equals the host build. The
  one difference is a pair 9e-7 Å outside rlist, kept by fp32 rounding, which
  the old build had too. 13/13 pass.

### 2026-11-17 — PCG + predicted start is the default SCF

Three 2,000-step (1 ps) polarizable NVE runs from `coords.inp` on the RTX
5070. Logs are `md-output/nve_check_*.out`. Drift is a linear fit of TE from
step 10; "block" compares the means of the first and last tenth.

| run | drift kcal/mol/fs (fit / 2nd half / block) | TE sd | T·μ per step | ms/step |
|---|---|---|---|---|
| Jacobi, cold, 1e-8 | −0.0010 / +0.0008 / −0.00003 | 0.43 | 9.99 | 103.0 |
| PCG, predict, 1e-8 | −0.0004 / +0.0014 / +0.0005 | 0.42 | 2.34 | 60.5 |
| PCG, predict, 1e-6 | +0.0016 / +0.0034 / +0.0025 | 0.42 | 2.00 | 58.6 |

- PCG + predict at 1e-8 conserves energy as well as Jacobi and is 1.7×
  faster per step.
- The TE fluctuation is the same in all three runs, so it comes from the
  integrator (0.5 fs, 503 K start), not the SCF.
- PCG at 1e-6 drifts slightly upward and saves only 3%, so 1e-8 is the
  setting to use.
- All three runs rebuilt 159 times in 2,000 steps (139 on drift) and missed
  0 pairs. At ~186 ms per build, rebuilds are ~15 ms of the 60 ms step: the
  next target.

Defaults now:
- `main_scan`: `--scf-solver pcg --scf-guess predict`. `--reference-parity`
  falls back to Jacobi + cold start. `tolpol` still comes from
  `mdrun.params` unless `--tolpol` is given.
- Colab cell 4: `TOLPOL = 1e-8`, `SCF_SOLVER = pcg`, `SCF_GUESS = predict`.
- `profile_step`: `--scf-solver pcg`, with the predicted start in section 4.

### 2026-11-17 — first profile (local RTX 5070 Laptop, 8 GB)

Report: `md-output/profile_polarizable_RTX5070_20261117.txt`. Smoke test:
`md-output/smoke_pcg_predict_nve_100.out`. Laptop clocks vary, so read the
times as relative; the same MD step ran at 84–145 ms.

- **The old 100-step rebuild was not safe.** One list built at step 0:
  - after 25 steps (12.5 fs, from the 503 K start) a site has already moved
    0.73 Å, more than skin/2;
  - after 100 steps it has moved 1.47 Å, and 139 pairs inside rcut are
    missing from the list; 2,423 by step 200.
  - With the drift trigger, the smoke test rebuilt 8 times in 100 steps and
    missed 0 pairs.
- **The neighbor build is now a cost.** It takes 186 ms (about 1.9 steps).
  At one rebuild every ~12 steps that is ~15 ms/step, so a faster build is
  needed.
- **PCG halves the SCF cost** (cold start, T·μ products per solve):

  | tolpol | Jacobi | PCG |
  |---|---|---|
  | 1e-8 | 10 | 5 |
  | 1e-10 | 12 | 7 |
  | 1e-12 | 15 | 8 |

  - Energy error against the converged answer: Jacobi at 1e-4 (the
    `mdrun.params` value) is +19.07 kcal/mol; PCG is +0.006 at 1e-6
    (4 products) and +0.0002 at 1e-8.
  - With `--scf-guess predict` in MD: 2.25 products per step at 1e-8.
  - The two references agree to 4.9e-14 in μ.
- **Where the time goes** (one evaluation each, ms):

  | phase | ms |
  |---|---|
  | SCF, Jacobi 1e-8 (10 × ~9 ms products) | 89 |
  | induced force, main pair term | 22 |
  | field, real-space pairs | 13 |
  | three reciprocal-space passes | 28 |
  | classical pair kernel | 8 |
  | geometry | 5 |

- **Pair work is padded.** The kernels process 4.19M slots for 2.64M
  in-cutoff pairs (1.59×).
- **The scatter dominates a T·μ product.** A scatter of 2 × capacity rows
  alone takes 7.9 ms of the product's ~9 ms. `segment_sum` is no faster
  (7.8 ms), and fp32 hardly helps (6.8 ms). On this GPU the lever is fewer
  pairs or a scatter-free layout (full list, row reduction), not precision.
- fp64 is 7.4× slower than fp32 for elementwise math.
- Short NVE (100 steps, PCG + predict, tolpol 1e-8): TE 3971.30 → 3971.26,
  ±0.12. Too short to judge drift.

### 2026-11-17 — profiler, PCG SCF, neighbor-list checks (task 1)

**What the last L4 run showed.** This is
`md-output/gomd_polarizable_auto_20261117-170648.out`: polarizable NVT, 20,000
steps, tolpol 1e-4 from `mdrun.params`, 75.9 ms/step.
- H_ext steps up and down by about 34 kcal/mol, and the steps track ms/step:
  - about 67–71 ms/step goes with an H_ext near −251,480;
  - about 75–79 ms/step goes with an H_ext near −251,515.
- So the SCF flips between 4 and 5 Jacobi iterations, and each iteration moves
  E_pol by tens of kcal/mol. One iteration costs about 8 ms on the L4.
- Between the flips, H_ext drifts by −0.004 to −0.014 kcal/mol/fs.
- The input velocities start at T = 503 K, and the one-link chain with
  OmegaQ 0.01/fs takes several ps to settle near 296 K.

**Neighbor list.** The old rule was a fixed rebuild every 100 steps (50 fs)
with a 1 Å skin, with no check that the list was still valid.
- Now `scan_body` also rebuilds when any site has moved more than
  skin/2 − 0.01 Å since the last build. The test runs after the drift, on the
  exact force-evaluation positions, so it holds for any dt.
- The scheduled rebuild stays at the start of the step, as before.
- The first build now uses the placed lone-pair sites.
- Every rebuild counts the pairs inside rcut that the stale list was missing:
  count(fresh) − count(stale) at the same positions.
- The build returns its true pair count. An overflow stops the run, and the
  message gives the `--nb-capacity` to use.
- `--no-drift-rebuild` restores the fixed-interval rule. The missed-pair count
  then shows whether 100 steps was safe.
- The reference itself rebuilds on drift (`driftmax`, `pkg/main.py:698`).

**PCG SCF** (`solve_scf_pcg`, `--scf-solver pcg`):
- Solves A·μ = E with A = diag(1/α − arf) − T, using a diagonal preconditioner.
- Same stopping metric as the reference, max|α·r|² ≤ tolpol, but applied to the
  true residual. That makes a warm start safe (`--scf-guess predict`: quadratic
  extrapolation, history seeded with μ0).
- A cold start skips the product A·0.
- The logged energy is the variational −½E·μ − ½μ·r. It equals −½E·μ at
  convergence and has a second-order error before it.
- Jacobi stays the default until the profiler's section 5 and an NVE run
  validate PCG.

**Profiler** (`gomd/profile_step.py`, Colab cell 10), in five sections:
1. fp64/fp32 and scatter throughput.
2. List vs in-cutoff pair counts. The polarization kernels run over the whole
   padded capacity.
3. Every force phase timed on its own.
4. The scan step without rebuilds, plus drift and missed pairs at each
   checkpoint.
5. Jacobi vs PCG at tolpol 1e-4 … 1e-12 against tight references.

**Also:**
- `polarization_full.make_polarization_parts` exposes each stage.
- The polarization function returns `(E_pol, f_pol, μ, {'niter'})`.
- The log has an `scf_it` column (T·μ products per step) and warns when the
  SCF hits `--scf-max-iter`.
- `--skin` and `--nb-capacity` flags.

The Jacobi arithmetic is unchanged: `town_product` is the old inner function,
moved to module level.

### 2026-11-16 — NVT, MLIP (UMA), Colab, two energy-conservation fixes

None of the three uploaded tarballs had MLIP, Colab or ensemble code (the last
was byte-identical to the Nov 9 release), so these were built on the Nov 9
engine.

**NVT.** `gomd/thermostat.py` ports the reference Nose–Hoover chain (MTK,
5 × 5 Yoshida–Suzuki):
- Half steps around velocity Verlet, chain state restarted from `coords.inp`,
  SHAKE after the drift, COM-velocity removal each step as the reference.
- `--ensemble auto` follows `mdrun.params` (NVT for this input); NPT is
  refused with the reason. T and the conserved H_ext are logged.
- Half step vs the reference: 1.4e-14 relative (chains of 1 and 3).
- 10-step polarizable NVT trajectory vs the reference (tight tolerances, same
  forces): KE agrees to 2.4e-3 kcal/mol, and the fresh PE at the same
  positions agrees to 7e-3.
- The reference's *logged* PE drifts from its own fresh evaluation by
  +0.04 kcal/mol per step (0.42 after 10 steps), most likely from its
  warm-started SCF.
- The reference's clamp / NaN-reset / [0.1, 10] scale-clip guards are left
  out on purpose.

**MLIP.** UMA support lives in `gomd/mlip/`:
- `server.py` runs the MLIP in its own process, optionally from a separate
  virtual environment. fairchem-core 2.25 pins torch 2.13 and numpy < 2.5,
  and PyTorch and JAX must not share a process or GPU memory pool.
- `client.py` exposes `MLIPProvider(x) → (E kcal/mol, F kcal/mol/Å)`; only
  real atoms are sent, and elements come from the masses.
- `compare.py` logs PFF-vs-MLIP agreement: energy-difference drift, force
  RMSE / MAE / max / relative error / cosine, and per-element RMSE.
- Driver modes:
  - `--mlip uma|mock --compare-every N`: the PFF drives the dynamics in the
    on-device scan; the MLIP is evaluated at the same configuration every N
    steps.
  - `--propagate mlip`: the MLIP drives the dynamics (one host round trip per
    step), the PFF is evaluated alongside, and NVE/NVT work the same way.
- `mock`, a numpy elastic network, tests the path without fairchem.

**Energy-conservation fixes** (CPU, 0.5 fs, from step 2):

| Run | Conserved-energy behaviour |
|-----|---------------------------|
| classical NVE, Nov 9 (SHAKE 1e-6) | TE alternates ±4 kcal/mol every step; −0.05 kcal/mol/fs on even steps |
| classical NVE, SHAKE 1e-12 | ±0.1 fluctuation, −0.002 kcal/mol/fs over 20 fs |
| polarizable NVE, tolpol 1e-10, SHAKE 1e-12, reference forces | +0.13 kcal/mol/fs |
| same + k-space charge–dipole force | +0.006 kcal/mol/fs |
| polarizable NVT, same defaults | H_ext +0.005 kcal/mol/fs (TE −28 kcal/mol from the thermostat) |
| classical NVT, defaults | H_ext −0.013 kcal/mol/fs over 20 fs |

1. *SHAKE tolerance.* The reference ignores the `mdrun.params` value (1e-4)
   and uses 1e-6, at which the Jacobi loop stops after ~2 sweeps, leaving
   bond errors up to 3e-4 Å. The SHAKE-combined force then rings with period
   two steps, about 20 kcal/mol/Å on the heavy atoms, and TE swings ±4 kcal/mol
   (the reference too). Printing every 10 steps had hidden this. The default
   is now 1e-12 (~11 sweeps).
2. *k-space charge–dipole force.* E_pol includes the reciprocal-space charge
   field, but the reference forces omit its gradient (`compute_recipQmu` is a
   TODO in `pkg/`). GOMD now adds it exactly, with `jax.grad` at fixed μ
   (rms 0.19 kcal/mol/Å). Its power, −0.095 kcal/mol/fs, matches the drift it
   removes.
3. *SCF tolerance.* Not changed (it is an input), but the driver now warns:
   `tolpol` 1e-4 from `mdrun.params` leaves E_pol 19 kcal/mol unconverged
   (initial PE −3611.54 vs −3630.60 converged; the reference gives −3630.60
   at the same tight tolerance). `--tolpol 1e-8` is converged.
4. SHAKE after the drift now also runs in NVE, as in the reference. At the
   reference tolerance it is a no-op, so `--reference-parity` NVE is
   bit-identical to the Nov 9 engine.

**Colab.** `GOMD_colab.ipynb` was rewritten for Drive `My Drive/GOMD/code/GOMD`:
- Finds the project under `MyDrive` or `My Drive`, accepting `GOMD\code\GOMD`.
- Checks the layout and the torsion fix.
- Checks GPU and JAX in a subprocess; the notebook kernel never imports JAX.
- Builds an optional UMA environment (`uv`, fairchem, Hugging Face token
  from Colab secrets, model cached on Drive).
- Writes logs straight to `md-output/` on Drive and keeps a run history in
  `md-output/colab_runs.csv`.
- Plots the run and offers the self-test.

**Also:** `gomd.selftest` (OPEN item 4 of Nov 9); new flags `--tolpol`,
`--scf-max-iter`, `--shake-tol`, `--shake-max-iter`, `--reference-pol-forces`,
`--reference-parity`; `gomd.__version__` = 0.5.0.

### 2026-11-09 — torsion force fixed (energy drift root cause)

The NVE drift (~36 kcal/mol per fs) was a bug in the analytic torsion force,
present in both `gomd/forces/bonded.py` and the reference `pkg/forces.py`:
- Two sign-convention errors in the dφ/dx formulas: the dihedral sign
  (`−rij·n`) and the j/k coefficients (written for r_kj = x_k − x_j while the
  code uses rjk = x_j − x_k). The energy is even in φ, so energies were always
  right; the Fortran uses a different route, so this was a Python porting
  error.
- Fixed in both files: torsion force vs autodiff 5.6e-12, fixed reference vs
  GOMD 3.9e-14.
- Every earlier run had non-conservative torsion forces, so re-check any
  production data from `pkg/`.

### 2026-11-06 — clean version, renamed to GOMD

Package `md_jax/` → `gomd/`, folder → `GOMD/`, documentation rewritten.
Generated files, superseded drivers, one-off tests and unused reference scripts
were removed; every remaining `.py` is reachable from `gomd/main_scan.py`,
`gomd/selftest.py` or `pkg/main.py`.

### Earlier — M3 and M4

On-device neighbor list, `lax.scan` loop, segsum pair kernel (M4); full
polarizable force field wired into the loop (M3). See
`GOMD_DOCUMENTATION.md` §4 and §7.
