# GOMD — Complete Technical Documentation

**GOMD (GPU-accelerated Open MD)** is a from-scratch JAX/GPU molecular dynamics
engine for the polarizable force field (like APPLE&P, AMOEBA, MU-PFF or equivalance). It reproduces a validated
Python/Fortran reference (kept in `pkg/`) bit-for-bit and owns every layer of
the science stack, so the polarization SCF can later be replaced by ML
(roadmap step 3).

Target system: PEO polymer electrolyte, Li⁺ + FSI⁻, **8,716 sites** (7,400 atoms + 1,316 lone-pair/dummy sites). Hardware: Tesla V100-PCIE-16GB,
conda env `newenv2`, JAX 0.4.13 / jaxlib 0.4.12, CUDA 11.8, Python 3.8.

---

## 1. Status at a glance

| Milestone | What | State |
|-----------|------|-------|
| M1 | JAX scaffold (pytree state, velocity Verlet) | done (scaffold code retired) |
| M2 | Full **classical** polarizable MD | done, validated |
| M3 | **Polarization** — field, SCF, energy, forces — wired into the step loop | done, validated |
| M4 | Performance: on-device neighbor list, `lax.scan` loop, `segment_sum` pair kernel | done |

Measured on the V100 (200-step runs):

| Force field | ms/step |
|-------------|---------|
| Classical | **9.2** (down from ~800 with the host-loop driver) |
| Full polarizable | **53–58**, 56 average |

In development (not in this tree): MLIP energy/force retrieval in parallel
with the polarizable force field, the constant-potential method, and coupled
electron–ion transport.

---

## 2. Repository layout

```
GOMD/
├── gomd/                          # the engine (import root: python -m gomd.main_scan)
│   ├── main_scan.py               # driver — THE one you run
│   ├── integrator_scan.py         # whole trajectory in one lax.scan (VV + SHAKE + dummies)
│   ├── forces/
│   │   ├── neighbor.py            # host cell-list neighbor pairs (setup only: sizes capacity)
│   │   ├── neighbor_ondevice.py   # on-device neighbor build, fixed-capacity padded pairs
│   │   ├── pair_splined.py        # splined vdW + real-space electrostatics (fp64 / fp32)
│   │   ├── pair_splined_segsum.py # fp32 pair kernel with segment_sum scatter (default)
│   │   ├── bonded.py              # bonds / bends / torsions / impropers
│   │   ├── ewald.py               # explicit-k reciprocal + self + exclusion correction
│   │   ├── dummy.py               # lone-pair placement + force redistribution
│   │   ├── all_forces.py          # classical force field: sums the terms above
│   │   ├── polarization.py        # SCF solver, polarization energy, main induced force
│   │   ├── pol_field.py           # charge electric field that drives the SCF
│   │   ├── pol_corrections.py     # exclusion + 1-4 induced-force corrections
│   │   └── polarization_full.py   # full polarizable term: field → SCF → energy → forces
│   ├── constraints/
│   │   └── shake.py               # Jacobi SHAKE (lax.while_loop)
│   └── io/
│       └── assembly_loader.py     # numpy → JAX boundary: one-time reference setup
├── pkg/                           # reference Python package (20 modules) — ground truth + setup
├── md-input/                      # ff.dat, coords.inp, connectivity.dat, mdrun.params
├── md-output/                     # logs are written here
├── run_scan_md.slurm              # SLURM script (classical or polarizable)
├── GOMD_DOCUMENTATION.md          # this file
└── PROJECT_STATUS.md              # done / open tracker and change log
```

`gomd/` depends on `pkg/` at setup time only: `assembly_loader` runs the
reference parsers and `setup_system` once, then everything per-step is pure JAX.

---

## 3. Design principles (why it's built this way)

1. **Own every layer.** External libraries only for primitives that the
   science never touches (FFT, low-level GPU ops). DMFF / jax-md are rejected
   as functional dependencies because roadmap step 3 requires surgically
   replacing the SCF with ML — you cannot swap code you do not own.
2. **Validate bit-for-bit against the reference.** Every component is checked
   against the Python/Fortran reference on the real 8,716-site system. The bar
   is machine-precision agreement (or, where an algorithm legitimately differs
   like Jacobi-vs-Gauss-Seidel SHAKE, the same physical invariant).
3. **Clean numpy→JAX boundary at the loader.** All one-time setup (parsing,
   spline-coefficient construction, connectivity, constraints, polarization
   arrays) runs once in the reference (numpy) and is extracted as JAX-ready
   arrays. The per-step evaluation is pure JAX, jit'd, on GPU.
4. **The SCF is a swappable pure function.** `solve_scf(field, pair_factors,
   alpha) -> dipoles`, wrapped by `make_polarization_fn`, is the exact seam
   where ML will later replace physics.
5. **Profile before optimizing.** Twice the documented bottleneck was not the
   real one (§7, §10).

---

## 4. Implementation specifics — per module

Conventions throughout: positions are `(nat, 3)` row-major (the reference uses
`(3, nat)`; converted once at load). The classical kernels use the minimum
image `d - box*round(d/box)` (orthorhombic box); the polarization kernels use
the reference's single conditional shift (`d -= sign(d)*box` where
`|d| > box/2`), which agrees inside the cutoff but matches the reference form
exactly. Forces are scatter-summed, so pair ordering is irrelevant.

### 4.1 `io/assembly_loader.py` — the numpy→JAX boundary
- `load_m2_system(input_dir, ref_pkg_dir)`: adds `ref_pkg_dir` to `sys.path`,
  `chdir`s into `input_dir`, runs the reference `load_simulation` +
  `setup_system` ONCE, then extracts JAX-ready arrays: positions/velocities/
  masses, per-atom charges, spline tables (+ moved-axis `coefft_m`),
  connectivity, exclusions (set + directed qq list), Ewald params, constraints,
  dummy connectivity.
- **`pol_data`** (M3): per-atom polarizability, Thole `c1`, `arf`, `rtaper`,
  `tauden`, `tolpol`, half-box `el`, directed 1-4 and exclusion pair lists,
  `lredQ_mu14` / `redQmufactor`, `lredonefour` / `redfactor`. `None` for a
  non-polarizable force field.
- Both directories are resolved to absolute paths up front (the reference
  `load_simulation` also `chdir`s).

### 4.2 `forces/neighbor.py` — host neighbor list (setup only)
- Cell list: `ncell = floor(box/(rcut+skin))` cells per dim, 14-entry
  half-stencil, brute-force fallback below 3 cells/dim. OrderedSparse
  `(idx_i, idx_j)` with `i < j`, exclusions removed.
- **Used once** in `main_scan` to size the fixed pair capacity (1.25× the first
  build). All rebuilds during MD happen on device (§4.3).
- Validation: bit-identical to the reference — 2,635,583 pairs at rcut 12 Å,
  20,952 exclusions.
- Cost: **~4.5 s** on the host for this box. It only yields 3×3×3 cells at
  rlist 13 Å, so it is near all-pairs work in Python. (Older docs said ~150 ms;
  that was wrong and hid the real bottleneck — §7.)

### 4.3 `forces/neighbor_ondevice.py` — on-device neighbor build (M4)
- `build_neighbor_ondevice(x, box, rlist2, excl_flat, nat, capacity, tile=512)`:
  all-pairs r² on device in fp32, tiled over rows (`lax.fori_loop`), exclusions
  removed by `searchsorted` membership against `exclusions_to_flat(...)`, valid
  pairs compacted per tile into a fixed `capacity` buffer padded with `(0,0)`.
- For this box all-pairs is the right primitive (~76 M distances, a few ms on
  the V100) and needs no host↔device traffic.
- Validation: returns the **bit-identical** reference pair set.
- `tile`: 512 by default; raise on the V100 (1024–2048) for more parallelism.
- With rcut 12 Å + skin 1 Å the list holds 3,354,520 pairs; capacity
  4,193,150.

### 4.4 `forces/pair_splined.py` — splined non-bonded
- **Physics:** APPLE&P short-range = exp-6 / 12-6 vdW + erfc-damped real-space
  electrostatics, pre-tabulated by the reference as **quintic splines in
  z = r²**.
- **Per pair:** z = r² (min image), clamp to the spline domain, piece index
  `ipiece = floor((z - zs0)*deltainv) + 1`, gather the 6 coefficients via
  `coefft_m[ipiece, itypee]` (`itypee = typee[type_i, type_j]`), Horner
  evaluation of energy and force, scatter `f_i += ff*dx`, `f_j -= ff*dx`.
- **Table layout:** reference `(6, npiece, ninter)` moved to
  `(npiece, ninter, 6)` so one gather yields all six coefficients.
- **Mask:** `in_cut = (z <= rcut2) & (idx_i != idx_j)` removes skin and padded
  pairs.
- Holds the fp64 kernel and an fp32 `.at[].add` variant; the driver uses the
  segsum kernel (§4.5).
- Validation: energy 1.5e-10, forces 1.3e-13 (fp64).

### 4.5 `forces/pair_splined_segsum.py` — default pair kernel (M4)
- fp32 end to end (positions, box, tables, distances, spline evaluation,
  reduction); returns fp64 energy and forces.
- **`jax.ops.segment_sum`** over the concatenated `(+fvec on i, −fvec on j)`
  replaces two contended atomic `.at[].add` scatters.
- Validation: segment_sum vs `.at[].add` force difference **0.0**; fp32 vs fp64
  energy rel 1.6e-7, force rel 3.3e-6.

### 4.6 `forces/bonded.py` — bonded terms
- **Bonds:** `E = ½k(r−r0)²`; vmap over `bonds (3, nbond)`.
- **Bends:** `E = ½k(θ−θ0)²`; analytic dcosθ/dr; θ0 degrees → radians.
- **Torsions:** dihedral φ from m = r_ij×r_jk, n = r_jk×r_kl, sign from the
  triple product; Fourier series `V0 − Σ ½Vₙcos(nφ)` bounded by a static
  `max_terms`; forces via dφ/dr on all four atoms.
- **Impropers:** ported; `ndeforms = 0` on this system.
- Validation: bond E 9e-16 / F 4e-12; bend exact / F 2e-13; torsion E 3e-13 /
  F 1e-10.

### 4.7 `forces/ewald.py` — reciprocal electrostatics
- **Explicit-k Ewald** (not FFT-PME), to match the reference exactly:
  half-space k-vectors within the spherical cutoff (klimit 7 → 709 vectors).
- Structure factor `S(k)=Σ qᵢ exp(ik·rᵢ)` as `(nat × nk)` cos/sin; energy
  `Σ |S(k)|²·akvec`; forces `Σ akvec·(cossum·sin − sinsum·cos)·h`.
- Self-energy `COULOMB·α·Σqᵢ²/√π`; exclusion correction removes the erf field
  for bonded pairs.
- **COULOMB_CONST = 332.08** (the reference value, not 332.0716 — the wrong one
  gave a ~0.4 kcal/mol self-energy error).
- Validation: reciprocal E 3e-14 / F 4e-14; self 5e-10; correction 4e-12.

### 4.8 `constraints/shake.py` — SHAKE
- The reference is sequential (Gauss-Seidel). GOMD is **Jacobi**: all 7,008
  constraints compute corrections from the same start-of-iteration positions,
  then scatter-add at once, inside `lax.while_loop`. Converges when
  `mean((d²−|r|²)²) ≤ tol`.
- Both reach the same constraint surface; trajectories track the reference but
  are not bit-identical step to step.

### 4.9 `forces/dummy.py` — lone-pair virtual sites
- **Placement:** `rid = −bb·(r_ij+aa·r_jk)/|r_ij+aa·r_jk| + cc·(r_ij×r_ik)/|r_ij×r_ik|`,
  `x_dummy = x_iat + rid` (LpGeomType ≤ 2; all 1,316 sites here).
- **Force redistribution:** the force on each site is pushed onto its three
  parents through the chain rule of the placement formula, then zeroed.
- Validation: placement 7e-15, redistribution 9e-16.

### 4.10 `forces/all_forces.py` — classical force field
- `compute_all_forces(x, D, idx_i, idx_j, kx, ky, kz, ewself)` sums splined pair
  + bonded + reciprocal − self + exclusion correction:
  `PE = E_bond + E_bend + E_tort + E_vdw + E_realelec + E_recip − E_self + E_corr`.
- Kernel choice: `D['use_segsum_pair']` (set by the driver) → §4.5;
  `D['use_fp32_pair']` → fp32 `.at[].add`; otherwise fp64.

### 4.11 `forces/polarization.py` — SCF, energy, main induced force
- **`solve_scf(Efield, alpha, pairs, factors, mu_init, arf, tolpol, nat,
  max_iter, min_iter=0)`** — `lax.while_loop`; each iteration contracts the
  dipole-dipole tensor over pairs with **Thole damping**
  (λ3 = 1−exp(−c1r³), λ5 = 1−exp(−c1r³)(1+c1r³)) and the tau taper, then
  `μ_i = α_i·(E + T·μ + arf·μ_i)`; converged when `max|Δμ|² ≤ tolpol` (and at
  least `min_iter` iterations).
- **`pol_energy`** = `−½·COULOMB·Σ E·μ`.
- **`pol_forces`** — dipole-dipole + charge-dipole gradients over pairs (exact
  erfc). Optional per-pair charge overrides let the caller zero skin and
  padded pairs (§4.14).
- `pack_pairs` precomputes per-pair factors for a fixed pair list.
- `charge_field_realspace` is the **legacy, incomplete** real-space field (it
  lacks the 1-4 fold-in and is ~23% off). Superseded by `pol_field.py`; not
  used by the engine.
- System specifics: arf = 0, tau = 1 everywhere (rtaper 14.5 > ro 12).

### 4.12 `forces/pol_field.py` — SCF charge field (M3)
The reference builds the field that drives the SCF in four parts; all are here:

| Part | Reference | GOMD |
|------|-----------|------|
| real-space field + 1-4 Q-μ fold-in (tabulated erfc, rcut mask) | `compute_charge_efield_and_pairs` | `_field_pairs_tab` |
| reciprocal-space field (no COULOMB factor) | `compute_recip_efield` | `_recip_field` |
| exclusion field correction (erf, ½) | `_apply_elfown_correction` | `_field_correction` |
| 1-4 Q-μ field correction | `_apply_elfown_14_correction` | `_field_correction` |

- Erfc/exp tables match `ewald_tables.py` (16,384 knots on [0, 6], linear
  interpolation); the corrections use the A&S erfc polynomial, as the
  reference does.
- Validation: total field **1.4e-15**.

### 4.13 `forces/pol_corrections.py` — induced-force corrections (M3)
- Phase 6 `qmu_exclusion_correction` — erf charge-dipole correction over
  excluded pairs (½).
- Phase 7 `qmu_14_correction` — the `lredQ_mu14` branch (active here,
  factor 0.8). The `lredonefour` branch is off on this system and not ported.
- Phase 8 (`recipQmu`) is a TODO in the reference itself and is not applied
  to its forces, so parity does not need it.
- Validation: phase 6 2.6e-14, phase 7 5.8e-16.

### 4.14 `forces/polarization_full.py` — full polarizable term (M3)
- `make_polarization_fn(D, kx, ky, kz, scf_max_iter=50, scf_min_iter=0)`
  returns jit'd `compute_polarization(x, idx_i, idx_j, mu_init=None)
  → (E_pol, f_pol, μ)`: pair geometry → field (§4.12) → SCF → energy →
  forces (main + phases 6, 7) × COULOMB.
- Takes the **same rcut+skin, padded** pairs as the classical kernels. Skin and
  padded pairs are masked out of the field, the SCF factors and — through
  zeroed charges and displacements — the main induced force. (Without that
  mask the skin pairs gave a force error of ~4.6e4 relative.)
- Validation (positions in, everything self-contained): dipoles 2.8e-15,
  E_pol −7025.987542 (diff 1.2e-11), total induced force 8.0e-9.

### 4.15 `integrator_scan.py` — the step loop (M4)
`make_scan_step(...)` builds one jit-able scan body; `make_runner` wraps it in
`lax.scan` for a fixed chunk length. Carry =
`(x, v, f, idx_i, idx_j, step, μ1, μ2, μ3)`. Per step:
1. neighbor rebuild on device when `step % rebuild_every == 0` (`lax.cond`)
2. first half-kick `v += ½dt·(41.84·massinv)·f` (real atoms)
3. drift `x += v·dt·1e-5`, PBC wrap
4. place dummies → classical force (+ polarization when enabled) →
   redistribute dummy forces
5. SHAKE predictor-corrector: `x2 = x + 1e-5·dt·v + 1e-5·dt²·f/m`,
   `shake_constrain(x2)` against the **post-drift** position, then the
   combined force `f = (x2c − x − 1e-5·dt·v)·m / (41.84e-5·dt²)`
6. second half-kick; KE.

- `use_aspc=False` (default): the SCF cold-starts from μ = 0 each step. The
  ASPC predictor `μ_pred = 3(μ1 − μ2) + μ3` is wired but off — see §6.
- Ensemble: **NVE only** (see §6).

### 4.16 `main_scan.py` — the driver
Loads via the assembly loader, builds k-vectors and self-energy, sizes the pair
capacity from one host build, builds the initial pairs on device, builds the
polarization function if `--polarizable`, then runs the trajectory in chunks of
`--print-every` steps (one compiled scan per chunk; the first chunk is a
discarded warm-up compile). Flags: `--input-dir`, `--ref-dir` (required),
`--steps`, `--print-every`, `--rebuild-every`, `--polarizable`, `--cpu`,
`--out`.

---

## 5. Validation scorecard (vs the reference, 8,716 sites)

| Component | Energy diff | Force diff |
|-----------|------------|-----------|
| Neighbor list (host and on-device) | exact pair set | — |
| Splined pair (vdW + real elec), fp64 | 1.5e-10 | 1.3e-13 |
| Splined pair, fp32 segment_sum (default) | 1.6e-7 rel | 3.3e-6 rel |
| Bonded — bond | 9e-16 | 4e-12 |
| Bonded — bend | exact | 2e-13 |
| Bonded — torsion | 3e-13 | 1e-10 |
| Ewald reciprocal | 3e-14 | 4e-14 |
| Ewald self-energy | 5e-10 | — |
| Ewald exclusion correction | 4e-12 | 6e-14 |
| SHAKE | constraints satisfied to tol | — |
| Dummy placement | 7e-15 | — |
| Dummy force redistribution | — | 9e-16 |
| **Assembled classical** | **6.7e-10** | **1.4e-10** |
| SCF charge field (all four parts) | 1.4e-15 | — |
| Induced dipoles | 2.8e-15 | — |
| Polarization energy | 1.2e-11 | — |
| Induced forces (phases 5 + 6 + 7) | — | 8.0e-9 rel |

Trajectory checks:
- Classical: the on-device scan loop reproduces the earlier host-loop driver
  (itself validated against the reference) to every printed digit (step 10 PE
  3414.6104 / KE 7661.5807; step 20 PE 3499.6967 / TE 11222.7591).
- Polarizable: initial PE −3611.5363 vs the reference total −3611.5355.

Component validations were run in development sessions against one-off
reference dumps; those test scripts are not part of this tree (§6).

---

## 6. Known gaps / caveats (honest status)

- **NVE only.** `mdrun.params` in `md-input/` requests **NVT** (Nose–Hoover,
  296.15 K); the reference supports NVE/NVT/NPT. GOMD reads none of the
  ensemble flags and integrates NVE regardless. Thermostat and barostat are
  not ported.
- **Energy drift.** The 200-step V100 runs gain ~18 kcal/mol per step in total
  energy, classical and polarizable alike (polarizable: TE 3990 → 7567,
  KE 7602 → 9398). An earlier note that polarization cut the drift to ~3/step
  came from a 6-step CPU run and does not hold. Cause not yet known: needs a
  same-length reference comparison under the same ensemble.
- **ASPC warm start is unsafe with the Jacobi SCF.** From a biased start the
  step-change test `max|Δμ|² ≤ tolpol` stops at the wrong fixed point (a 1%
  dipole error → E_pol ~24 kcal/mol off; a `min_iter` floor did not fix it).
  Cold start converges to the reference dipoles at 8e-16 and is the default.
  ASPC needs a contracting solver (Gauss-Seidel sweeps, damped update, or a
  residual test).
- **SHAKE Jacobi vs reference Gauss-Seidel** — same constraint surface,
  trajectories not bit-identical; together with ASPC this is why polarizable
  trajectories track but drift apart from the reference step to step.
- **Explicit-k Ewald, not FFT-PME** — bit-matches the reference; FFT-PME would
  need full re-validation.
- **No self-contained test suite.** The component tests used during
  development loaded one-off reference dumps (`/tmp/*.npz`) and a sandbox path,
  so they could not run anywhere else and were removed.

---

## 7. Performance — measurements and findings

| Stage | V100 ms/step |
|-------|--------------|
| Host-loop driver, recompile every neighbor rebuild | 2253 |
| Fixed-capacity padded pairs (no recompile) | 843 → ~810 |
| On-device neighbor build + `lax.scan` + segsum, classical | **9.2** |
| Same, full polarizable | **53–58** (56 avg) |

Findings:
1. **The 800 ms/step was the host neighbor build**, not the force kernels:
   ~4.5 s per build, every 10 steps. Moving the build on device (and the loop
   into `lax.scan`) gave the ~90× win.
2. **fp32 alone did nothing** (810 → 807) while the scatter was atomic and
   fp64: the pair kernel is gather/scatter-bound. fp32 pays off together with
   the `segment_sum` scatter.
3. The old CPU-sandbox breakdown (pair 67%, reciprocal 32%) does not carry to
   the GPU. **Profile the polarizable step per phase on the V100** before any
   further work; FFT-PME waits for that profile.

Likely next levers (to confirm by profile): a contracting SCF solver (fewer
iterations, and makes ASPC safe), sharing pair geometry between the field,
SCF and force passes, and the k-space cos/sin used by both the reciprocal force
and the reciprocal field.

---

## 8. How to run

Layout (run from the folder that holds all four):
```
GOMD/
├── gomd/        (the engine)
├── pkg/         (20 reference .py modules)
├── md-input/    (4 input files)
└── md-output/   (logs)
```

```bash
cd GOMD
conda activate newenv2
module load cuda/11.8.0

# classical
python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
    --steps 200 --print-every 50 --rebuild-every 100 \
    --out md-output/md_scan_classical.out

# full polarizable
python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
    --steps 200 --print-every 50 --rebuild-every 100 --polarizable \
    --out md-output/md_scan_polarizable.out
```
`--cpu` forces the CPU backend (debugging only). SLURM: `sbatch
run_scan_md.slurm` from `GOMD/` (pick run A or B inside).

Pre-flight check:
```bash
python -c "import gomd; print(gomd.__file__)"   # ends in gomd/__init__.py
ls md-input/ff.dat pkg/input_parser.py
ls pkg/*.py | wc -l                             # 20
```

Expected: ~40–110 s setup, then `Initial classical PE: 3414.451` or
`Initial polarizable PE: -3611.536` (the last digit varies by ~5e-4 with the
fp32 summation order: V100 gives 3414.4512 / −3611.5363, CPU 3414.4507 /
−3611.5368), then one log line per chunk.

Environment variable for the reference setup: `GOMD_GPU_PAIR=0` disables the
GPU pair builder used during setup (default on); `GOMD_GPU_PAIR_VERIFY=1`
cross-checks it against the CPU path.

---

## 9. Roadmap position

- Done: classical and fully polarizable APPLE&P MD on the GPU, validated
  against the reference, 9.2 / ~56 ms/step.
- In development: MLIP energy/forces in parallel with the PFF (same
  `force_fn(x) → E, F` interface), constant-potential method, coupled
  electron–ion transport.
- Next engine work: per-phase GPU profile of the polarizable step, contracting
  SCF solver, thermostat (NVT), energy-drift diagnosis.
- Roadmap: step 2 constant-potential EDL, step 3 ML replaces `solve_scf`,
  step 4 ML/MM.

---

## 10. Lessons recorded

- Profile before optimizing. The documented bottleneck ("the force kernels")
  was wrong; the host neighbor build was ~4.5 s, not the ~150 ms on record.
- `step` in a pytree's aux_data → recompile every step. Keep changing counters
  out of the pytree.
- Pair-array shape changes → recompile every rebuild. Pad to fixed capacity;
  pass pairs as arguments, not closure constants.
- fp32 helps compute and gathers but not contended atomic scatters; use
  `segment_sum`.
- The SHAKE reference position is the **post-drift** position; using the
  pre-drift one blew KE up to 1e11.
- The real-space SCF field must include the 1-4 Q-μ fold-in, tabulated erfc
  and the rcut mask; missing the fold-in made it ~23% off.
- Polarization forces must mask skin pairs explicitly: the charge-dipole terms
  do not vanish just because the Thole factors are zeroed.
- A Jacobi SCF with a step-change convergence test is only safe from a cold
  start.
- A few steps of MD say little about energy drift; judge conservation over
  hundreds of steps.
- COULOMB_CONST must be 332.08 (reference value). Fortran `derfc()` is
  double-precision erfc, not a derivative.
- Validate every component bit-for-bit, or errors hide until they compound.
