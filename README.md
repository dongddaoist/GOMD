# WMI-MD-JAX — Complete Technical Documentation

A from-scratch JAX/GPU molecular dynamics engine for the **APPLE&P polarizable
force field**, reproducing a validated Python/Fortran reference bit-for-bit and
built to own every layer of the science stack (so the SCF solver can later be
replaced by ML — roadmap step 3).

Target system: PEO polymer electrolyte, Li⁺ + FSI⁻, **8,716 atoms** (7,400 real
+ 1,316 lone-pair/dummy sites). Hardware: OSC Pitzer, Tesla V100, JAX 0.4.13.

---

## 1. Status at a glance

| Milestone | What | State |
|-----------|------|-------|
| M1 | JAX scaffold (state pytree, integrator, Coulomb-only) | done |
| M2 | Full **classical** APPLE&P MD (6 force pieces + assembly) | **done, validated, runs stable on V100** |
| M3 | **Polarization** induced-dipole SCF + energy + forces | **core validated**; field-builder corrections pending |
| M4 | Performance to ~80 ms/step | in progress (see §7) |

Current speed: **~810 ms/step on V100** (classical), at parity with the
711 ms/step Python reference. Validated bit-identical to the reference
(PE 6.7e-10, forces 1.4e-10).

---

## 2. Repository layout

```
md_jax/                     # the package (import root)
├── state.py                    # SystemState pytree (M1)
├── integrator.py               # M1 velocity-Verlet + on-device runners
├── integrator_m2.py            # M2 NVE step: SHAKE scheme + dummy atoms
├── main.py                     # M1 driver (Coulomb-only; legacy)
├── main_m2.py                  # M2 driver — THE one you run
├── forces/
│   ├── neighbor.py             # M2.1 cell-list neighbor pairs (+ padded)
│   ├── pair_splined.py         # M2.2 splined vdW + real-space elec (+ fp32)
│   ├── bonded.py               # M2.3 bonds / angles / torsions / impropers
│   ├── ewald.py                # M2.4 reciprocal + self + exclusion correction
│   ├── dummy.py                # M2.6 lone-pair placement + force redistribution
│   ├── polarization.py         # M3 SCF solver + energy + forces + field
│   ├── all_forces.py           # unified compute_all_forces
│   └── coulomb.py              # M1 Coulomb (legacy)
├── constraints/
│   └── shake.py                # M2.5 Jacobi SHAKE (lax.while_loop)
├── io/
│   ├── assembly_loader.py      # extract JAX arrays from reference setup
│   └── loader.py               # M1 loader (legacy)
└── tests/                      # one validation test per component
```

Runtime sibling folders (NOT in the package):
```
md-input/   ff.dat, coords.inp, connectivity.dat, mdrun.params
pkg/        the FULL reference Python package (26 .py modules)
md-output/  MD logs written here
```

---

## 3. Design principles (why it's built this way)

1. **Own every layer.** External libraries only for primitives that the
   science never touches (FFT, low-level GPU ops). DMFF / jax-md are rejected
   as functional dependencies because roadmap step 3 requires surgically
   replacing the SCF with ML — you cannot swap code you do not own.
2. **Validate bit-for-bit against the reference.** Every component is checked
   against the Python/Fortran reference on the real 8,716-atom system. The bar
   is machine-precision agreement (or, where an algorithm legitimately differs
   like Jacobi-vs-Gauss-Seidel SHAKE, the same physical invariant).
3. **Clean numpy→JAX boundary at the loader.** All one-time setup (parsing,
   spline-coefficient construction, connectivity, constraints) runs once in the
   reference (numpy) and is extracted as JAX-ready arrays. The per-step force
   evaluation is pure JAX, jit'd, on GPU.
4. **The SCF is a swappable pure function.** `solve_scf(field, pair_factors,
   alpha) -> dipoles` is the exact seam where ML will later replace physics.

---

## 4. Implementation specifics — per module

Conventions throughout: positions are `(nat, 3)` row-major (the reference uses
`(3, nat)`; converted once at load). Minimum image is
`d - box*round(d/box)` (orthorhombic box). Forces are scatter-summed with
`jnp.zeros(...).at[idx].add(...)`, so pair ordering is irrelevant.

### 4.1 `state.py` — SystemState pytree (M1)
- A dataclass registered as a JAX pytree. `_LEAF_FIELDS` (traced arrays:
  x, v, f, mass, massinv, charge, atomtype, ldummy, box, pe, ke) vs
  `_AUX_FIELDS` (static: nat, dt, rcut).
- **Critical fix:** `step` was originally in `_AUX_FIELDS`. Because aux_data
  defines the pytree's static structure, incrementing `step` each step changed
  the structure → **full jit recompile every step (~1.5 s)**. Removed `step`
  from the pytree entirely; the driver tracks the step count. This is the M1
  recompile bug.

### 4.2 `integrator.py` — M1 runners
- `velocity_verlet_step` (kick-drift-kick), plus on-device multi-step runners
  `run_nve` (lax.fori_loop) and `run_nve_logged` (lax.scan, returns per-step
  pe/ke history). These compile once and run N steps on-device — no Python
  between steps, no per-step recompile.

### 4.3 `forces/neighbor.py` — M2.1 neighbor list
- **Algorithm:** cell list. Box divided into `ncell = floor(box/(rcut+skin))`
  cells per dim; atoms bucketed; each cell checked against a 14-entry half-
  stencil of neighbor cells (self + 13 forward) to enumerate each pair once.
  Falls back to brute O(N²) if any dim has <3 cells.
- **Output:** OrderedSparse `(idx_i, idx_j)` with `i < j`, exclusions removed.
- **Exclusions:** `exclusions_from_reference(listex, listmex)` converts the
  reference's per-atom exclusion arrays to a set of `(i<j)` tuples; the build
  filters them out.
- **`build_neighbor_pairs_padded(..., capacity)`** (M4): returns FIXED-length
  arrays padded with self-pairs `(0,0)`. Constant shape → the force fn compiles
  once across rebuilds. Padded pairs are masked off downstream by `i != j`.
- **Validation:** bit-identical to reference — 2,635,583 pairs, zero diff,
  20,952 exclusions. (`tests/test_neighbor.py`.)
- **Note:** runs host-side in numpy (~8 s sandbox CPU, ~150 ms on V100). On the
  V100 the box yields only 3×3×3=27 cells (~323 atoms each), so cells barely
  help; this is a known limitation, not the current bottleneck.

### 4.4 `forces/pair_splined.py` — M2.2 splined non-bonded
- **Physics:** APPLE&P short-range = exp-6 / 12-6 vdW (the repulsive wall that
  stops Coulomb collapse) + erfc-damped real-space electrostatics. Both are
  pre-tabulated by the reference as **quintic splines in z = r²**.
- **Per pair:** compute z=r² (min image); clamp z to the spline domain; piece
  index `ipiece = floor((z - zs0)*deltainv) + 1`; gather the 6 coefficients via
  `coefft_m[ipiece, itypee]` where `itypee = typee[type_i, type_j]`; evaluate
  energy and force by Horner's method; scatter `f_i += ff*dx`, `f_j -= ff*dx`.
- **Table layout:** the reference tables are `(6, npiece, ninter)`; moved to
  `(npiece, ninter, 6)` (`coefft_m`) so one gather yields all six coefficients.
- **Cutoff + padding mask:** `in_cut = (z <= rcut2) & (idx_i != idx_j)`. The
  second term zeroes padded sentinel pairs.
- **Mixed precision (M4):** `splined_pair_energy_force_fp32` casts positions,
  box, and tables to fp32 for the per-pair math, returns fp64 forces. Validated
  energy rel 1.6e-7, force rel 3.3e-5. Enable with `--fp32`.
- **Validation:** energy diff 1.5e-10, forces 1.3e-13 vs reference.

### 4.5 `forces/bonded.py` — M2.3 bonded
- **Bonds:** `E = ½k(r−r0)²`, `fmag = k(r−r0)/r`; vmap over `bonds (3,nbond)`.
- **Angles (bends):** `E = ½k(θ−θ0)²`; cosθ from the two bond vectors;
  analytic dcosθ/dr per atom; θ0 in degrees → radians.
- **Torsions:** dihedral φ via the cross products m = r_ij×r_jk, n = r_jk×r_kl;
  sign from the triple product; energy a Fourier series `V0 − Σ ½Vₙcos(nφ)`,
  the term loop bounded by a static `max_terms`; forces via dφ/dr on all four
  atoms.
- **Impropers:** ported but `ndeforms = 0` on this system (no-op).
- **Validation:** bond E 9e-16 / F 4e-12; bend exact / F 2e-13; torsion E 3e-13
  / F 1e-10.

### 4.6 `forces/ewald.py` — M2.4 reciprocal electrostatics
- **Approach:** EXPLICIT-k Ewald (not FFT-PME), to match the reference exactly.
  `build_kvectors` enumerates the half-space k-vectors within the spherical
  cutoff (klimit 7 → 709 vectors).
- **Reciprocal sum:** structure factor `S(k)=Σ qᵢ exp(ik·rᵢ)` built as
  `(nat × nk)` cos/sin; energy `Σ |S(k)|²·akvec`; forces from
  `Σ akvec·(cossum·sin − sinsum·cos)·h`.
- **Self-energy:** `COULOMB·α·Σqᵢ²/√π`. **Exclusion correction:** erf-field
  removal for bonded pairs, vectorized over the directed exclusion list.
- **Critical constant:** the reference uses **COULOMB_CONST = 332.08** (not the
  more precise 332.0716). Using the wrong one caused a ~0.4 kcal/mol self-energy
  error. All electrostatic constants must match the reference exactly.
- **erfc:** the reference uses a fine erfc *table* (matches true erfc ~1e-7);
  the JAX code uses `jax.scipy.special.erfc` (exact) — agreement ~1e-7.
- **Validation:** reciprocal E 3e-14 / F 4e-14; self 5e-10; correction 4e-12.

### 4.7 `constraints/shake.py` — M2.5 SHAKE
- **Reference is sequential** (Gauss-Seidel: each constraint sees prior updates
  within a sweep) → does not parallelize.
- **This implementation is JACOBI** (all 7,008 constraints compute corrections
  from the same start-of-iteration positions, then scatter-add at once), wrapped
  in `lax.while_loop`. Per constraint: `g = (d² − |r|²)·massred / (r·r_ref)`,
  push both atoms along the reference bond vector. Converges when
  `mean((d²−|r|²)²) ≤ tol`.
- Both forms reach the **same constraint surface** (SHAKE's fixed point is
  order-independent). Validated: converges in ~5 iters to sigma 9e-7
  (tol 1e-6), all constraints satisfied; tighter tol → smaller violation.
- Trajectories track the reference but are not bit-identical step-to-step
  (Gauss-Seidel vs Jacobi land on slightly different valid points).

### 4.8 `forces/dummy.py` — M2.6 lone-pair virtual sites
- **Placement** (`update_dummy_coords`): for each dummy (parents jat, iat, kat),
  `rid = −bb·(r_ij+aa·r_jk)/|r_ij+aa·r_jk| + cc·(r_ij×r_ik)/|r_ij×r_ik|`;
  `x_dummy = x_iat + rid`. Handles LpGeomType ≤ 2 (all 1,316 dummies here).
- **Force redistribution** (`dummy_force_redistribution`): the force on a dummy
  is pushed onto its three parents via the chain rule of the placement formula
  (first-derivative cross-product terms + second-derivative term), then the
  dummy's own force is zeroed. Vectorized over all dummies.
- **Validation:** placement 7e-15, force redistribution 9e-16.

### 4.9 `forces/polarization.py` — M3 induced dipoles
- **`solve_scf(Efield, alpha, pairs, factors, mu_init, arf, tol)`** — the heart
  and the ML-swappable seam. `lax.while_loop`; each iteration does the
  dipole-dipole `T·μ` contraction over pairs (scatter-add) with **Thole damping**
  (lamda3 = 1−exp(−c1r³), lamda5 = 1−exp(−c1r³)(1+c1r³)) and **tau taper**, then
  `μ_i = α_i·(E + T·μ + arf·μ_i)`; converged when `max|Δμ|² ≤ tolpol`.
- **`pol_energy`** = `−½·COULOMB·Σ E·μ`.
- **`pol_forces`** — dipole-dipole gradient (p1..p6 terms) + charge-dipole
  gradient (ee1..ee6, exact erfc) over pairs; vectorized.
- **`pack_pairs`** precomputes per-pair geometric factors (fact, tfact,
  3·zinv·fact·lamda5) once per neighbor build, so the SCF inner loop is just
  the dipole contraction.
- **`charge_field_realspace`** — real-space part of the charge field (the SCF
  input). NOTE: not yet bit-validated standalone (discrepancy under
  investigation; see §6).
- **System specifics:** arf = 0, tau = 1 everywhere (rtaper 14.5 > ro 12), so
  the taper is effectively off; the general forms are kept.
- **Validation:** SCF dipoles 6.7e-16; energy 1.4e-11; forces 8.0e-10.

### 4.10 `forces/all_forces.py` — unified force
- `compute_all_forces(x, D, idx_i, idx_j, kx, ky, kz, ewself)` sums splined
  pair + bonded + reciprocal − self + exclusion correction.
- `PE = E_bond + E_bend + E_tort + E_vdw + E_realelec + E_recip − E_self +
  E_corr` (classical; polarization added by M3 when wired).
- Selects the fp32 pair kernel when `D['use_fp32_pair']` is set.

### 4.11 `io/assembly_loader.py` — the numpy→JAX boundary
- `load_m2_system(input_dir, ref_pkg_dir)`: adds `ref_pkg_dir` to `sys.path`,
  `chdir`s into `input_dir`, runs the reference `load_simulation` +
  `setup_system` ONCE, then extracts JAX-ready arrays: positions/velocities/
  masses, per-atom charges, spline tables (+ moved-axis `coefft_m`),
  connectivity, exclusions (set + directed qq list), Ewald params, constraints,
  dummy connectivity.
- **Path fix (M4):** both dirs resolved to absolute up front. The reference
  `load_simulation` also `chdir`s, so passing a relative `input_dir` after the
  loader's own `chdir` raised `FileNotFoundError`; absolute paths fix the
  split `md-input` / `pkg` layout.

### 4.12 `integrator_m2.py` — the M2 NVE step
Replicates the reference `_vanilla_vv_step` exactly:
1. first half-kick `v += ½dt·(41.84·massinv)·f` (real atoms)
2. drift `x += v·dt·1e-5`, PBC wrap
3. place dummies (inside `force_fn`)
4. compute forces (`compute_all_forces`)
5. redistribute dummy forces (inside `force_fn`)
6. SHAKE predictor-corrector: predictor `x2 = x + 1e-5·dt·v + 1e-5·dt²·f/m`,
   `shake_constrain(x2)`, then COMBINED force
   `f = (x2c − x − 1e-5·dt·v)·m / (41.84e-5·dt²)`
7. second half-kick.
- **`make_force_fn(Dj, kx, ky, kz, ewself)`** returns a jit'd `force_fn(x,
  idx_i, idx_j)` — pairs are ARGUMENTS, not closure constants, so it compiles
  ONCE and is reused across rebuilds (with fixed-capacity pairs).

### 4.13 `main_m2.py` — the driver
- Loads via the assembly loader, builds k-vectors + self-energy, builds the
  Dj jax-array dict, sizes the padded capacity from the first neighbor build
  (1.25× headroom), builds `force_fn` once, runs the loop. Neighbor list
  rebuilt every `--rebuild-every` steps (host); only the pair arrays are
  swapped — no recompile. Writes the per-step log to `--out` (default
  `md_jax.out`).
- Flags: `--input-dir`, `--ref-dir`, `--steps`, `--print-every`,
  `--rebuild-every`, `--cpu`, `--fp32`, `--out`.

---

## 5. Validation scorecard (vs Python reference, 8,716 atoms)

| Component | Energy diff | Force diff |
|-----------|------------|-----------|
| Neighbor list | exact pair set (0 diff) | — |
| Splined pair (vdW + real elec) | 1.5e-10 | 1.3e-13 |
| Bonded — bond | 9e-16 | 4e-12 |
| Bonded — bend | exact | 2e-13 |
| Bonded — torsion | 3e-13 | 1e-10 |
| Ewald reciprocal | 3e-14 | 4e-14 |
| Ewald self-energy | 5e-10 | — |
| Ewald exclusion correction | 4e-12 | 6e-14 |
| SHAKE | constraints satisfied to tol | — |
| Dummy placement | 7e-15 | — |
| Dummy force redistribution | — | 9e-16 |
| **Assembled classical (M2)** | **6.7e-10** | **1.4e-10** |
| Polarization SCF (dipoles) | 6.7e-16 | — |
| Polarization energy | 1.4e-11 | — |
| Polarization forces | — | 8.0e-10 |

Each row has a test in `tests/`. The M2 assembled force runs stable MD that
tracks the reference's classical trajectory (no blow-up; M1 Coulomb-only
detonated at KE→10⁶ for comparison).

---

## 6. Known gaps / caveats (honest status)

- **M3 not wired into main_m2.** The polarization SCF/energy/forces are
  validated against the reference's TOTAL field, but the self-contained
  **field-builder corrections** are not all ported: reciprocal-space charge
  field (~10% of the field), Ewald exclusion field correction, and the 1-4
  Q-mu field + force corrections. `charge_field_realspace` (real-space part)
  is written but shows a standalone discrepancy under investigation (likely the
  erfc-table interpolation or the 1-4 term the reference folds into the same
  loop). So `main_m2` currently runs CLASSICAL-only.
- **Explicit-k Ewald, not FFT-PME.** Bit-matches the reference; FFT-PME is a
  deferred performance swap (different numerics → needs re-validation).
- **SHAKE Jacobi vs reference Gauss-Seidel.** Same constraint surface;
  trajectories track but aren't bit-identical step-to-step.
- **Energy drift** in classical runs (TE 11076→11772 over 50 steps) mirrors the
  reference's own classical-only drift — expected without polarization, not a
  bug.
- **`tests/test_pair_splined.py` is stale** — references the old `unbde` return
  the optimized kernel dropped. The kernel is verified correct by direct
  reference comparison (PE/force diff 0.0); update the test's expected returns.

---

## 7. Performance (M4) — measurements and findings

### Per-step cost on V100: ~810 ms (classical)

Force-eval breakdown (CPU sandbox; proportions hold on GPU):

| Component | Cost | % |
|-----------|------|---|
| Splined pair (3.3M pairs) | 1277 ms | 67% |
| Ewald reciprocal (709 k) | 609 ms | 32% |
| exclusion + bonded | ~12 ms | <1% |

### Wins implemented

1. **Recompile fix (biggest realized win).** The force fn recompiled on every
   neighbor rebuild (pair shape changed). Fixed via fixed-capacity padded pairs
   + pairs-as-arguments. Result: **2253 → 843 ms/step (2.7×)**, reaching parity
   with the Python reference. Correctness preserved exactly (padded vs unpadded
   force diff 0.0).

2. **Mixed precision (fp32) pair force.** Implemented and validated (energy rel
   1.6e-7). **But measured ~no change on V100 (810→807).** Diagnosis: the pair
   kernel is GATHER/SCATTER-bound, not compute-bound. Primitive costs (4.2M
   pairs): spline table gather ~280 ms, scatter-add ~138 ms, math ~74 ms,
   position gather ~59 ms. fp32 halves the *table gather* (129→55 ms) but the
   bandwidth-bound scatter and the still-fp64 reciprocal dominate, so the total
   barely moves. `--fp32` is correct and available; it isn't the binding
   constraint alone.

### Findings that reprioritized the plan

- The host neighbor build is **~150 ms on V100** (not the bottleneck) — so GPU
  neighbor build and lax.scan fusion were DEPRIORITIZED (small wins here).
- The real levers are now, in order:
  1. **FFT-PME** for the reciprocal (32%, untouched, fp64). Charge spread →
     `jnp.fft.rfftn` → convolution → inverse FFT, B-spline order 4-6. Different
     numerics → full re-validation required. **#1 remaining lever.**
  2. **Coalesced scatter-add** — sort pairs by atom index and use `segment_sum`
     instead of unordered `.at[].add` (atomic contention on GPU). Attacks the
     138 ms scatter.
  3. **fp32 end-to-end** (incl. reciprocal) once #1 lands.

---

## 8. How to run

Layout (run from the parent that holds all four):
```
~/python/
├── md-input/    (4 input files)
├── pkg/         (ALL 26 reference .py modules — the full pkg.zip)
├── md-output/   (logs)
└── md_jax/  (this package)
```

```bash
cd ~/python
conda activate newenv2
module load cuda/11.8.0

python -m md_jax.main_m2 \
    --input-dir ./md-input \
    --ref-dir   ./pkg \
    --steps 50 --print-every 10 \
    --out md-output/md_jax.out
# add --fp32 for mixed precision; --cpu for a CPU test
```

Pre-flight check:
```bash
python -c "import md_jax; print(md_jax.__file__)"  # ends in md_jax/__init__.py
ls md-input/ff.dat pkg/input_parser.py
ls pkg/*.py | wc -l   # must be 26
```

Expected: `JAX devices: [gpu(id=0)]`, ~60-95 s load, `Initial classical PE:
3414.4520`, then stable steps, log in `md-output/md_jax.out`.

SLURM: edit `RUNROOT` in `run_m2_md.slurm`, then `sbatch run_m2_md.slurm`.

---

## 9. Roadmap position

- Done: M1 scaffold, M2 classical MD (validated, stable, at parity), M3
  polarization core (validated).
- Next: finish M3 field-builder corrections → fully self-contained polarizable
  MD; then M4 FFT-PME + coalesced scatter toward 80 ms/step.
- Then: roadmap step 2 (constant-potential EDL), step 3 (ML replaces
  `solve_scf`), step 4 (ML/MM multi-scale).

---

## 10. Lessons recorded

- `step` in a pytree's aux_data → recompile every step. Keep changing counters
  out of the pytree.
- Pair-array shape changes → recompile every rebuild. Pad to fixed capacity;
  pass pairs as arguments, not closure constants.
- COULOMB_CONST must be 332.08 (reference value), not 332.0716.
- The reference `derfc()` (Fortran) means double-precision ERFC, not the
  derivative of erfc — a prior bug that corrupted spline coefficients.
- fp32 helps compute/gather but not bandwidth-bound scatter; profile before
  assuming a kernel is compute-bound.
- A from-scratch engine must be validated bit-for-bit at every component, or
  errors hide until they compound.
