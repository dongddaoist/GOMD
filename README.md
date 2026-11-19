# GOMD — Complete Technical Documentation

**GOMD (GPU-accelerated Open MD)** is a from-scratch JAX/GPU molecular dynamics
engine for the **APPLE&P polarizable force field**. It reproduces a validated
Python/Fortran reference (kept in `pkg/`) bit-for-bit and owns every layer of
the science stack, so the polarization SCF can later be replaced by ML
(roadmap step 3).

Target systems:
- `md-input`: PEO polymer electrolyte, Li⁺ + FSI⁻, **8,716 sites** (7,400
  atoms + 1,316 lone-pair/dummy sites), periodic;
- `md-input3`: Zn²⁺ electrolyte between two graphite electrodes held at
  constant potential, **17,544 sites**, slab (§4.23–4.28).

Runs locally on an RTX 5070 Laptop GPU (8 GB) under WSL (`tools/wsl_run.sh`,
conda env `gomd`), where the timings below were measured, and on Google Colab
GPUs (`GOMD_colab.ipynb`, current JAX / Python 3.12). The code still keeps to
the Python 3.8 grammar of the older OSC Pitzer environment (V100, JAX 0.4.13).

---

## 1. Status at a glance

| Milestone | What | State |
|-----------|------|-------|
| M1 | JAX scaffold (pytree state, velocity Verlet) | done (scaffold code retired) |
| M2 | Full **classical** APPLE&P MD | done, validated |
| M3 | **Polarization** — field, SCF, energy, forces — wired into the step loop | done, validated |
| M4 | Performance: on-device neighbor list, `lax.scan` loop, `segment_sum` pair kernel, PCG with a predicted SCF start | done |
| M5 | NVT (Nose–Hoover chain); MLIP provider (UMA via fairchem) run alongside the PFF or driving the dynamics; Colab runner; self-test | done (UMA validated on a Colab L4, §6) |
| M6 | Constant-potential electrodes: fixed electrodes, slab (2D) Ewald, PME, constant potential coupled to the induced dipoles | P0–P4 done and validated (§4.23–4.28) |

Measured on the RTX 5070 Laptop GPU, 2026-11-17 to 11-19 (NVE runs of
200–2,000 steps; §7 has the history):

| System | Force field | ms/step |
|--------|-------------|---------|
| `md-input`, 8,716 sites, periodic | classical | **6.6** |
| `md-input` | polarizable (PCG + predicted start, tolpol 1e-8) | **21.6** |
| `md-input3`, 17,544 sites, 205 Å slab, PME | classical, constant potential at 2 V | **15.7** (16.4 with fixed charges) |
| `md-input3` | polarizable, fixed electrode charges | **44** |
| `md-input3` | polarizable, constant potential at 2 V (tolpol 1e-9) | **64** (69 before P4) |

The old Fortran code ran `md-input3` at about 225 ms/step on 40 MPI ranks. The
laptop GPU switches between two clock states (up to ±50 %), so compare only
runs made back to back. The V100 numbers quoted before 2026-11-17 (9.2 ms/step
classical, 56 polarizable) came from the earlier engine; §7 keeps them as
history.

The 2026-11-16 release also changes two defaults for energy conservation
(§4.8, §4.14): SHAKE tolerance 1e-12 (the reference's 1e-6 makes the total
energy ring by ±4 kcal/mol every other step) and the k-space charge–dipole
polarization force, which the reference omits. `--reference-parity` restores
the reference behaviour.

In development (not in this tree): hybrid PFF/MLIP regions, coupled
electron–ion transport.

---

## 2. Repository layout

```
GOMD/
├── gomd/                          # the engine (import root: python -m gomd.main_scan)
│   ├── main_scan.py               # driver — THE one you run
│   ├── integrator_scan.py         # MD step (pre_force / force / post_force) + lax.scan runner
│   ├── thermostat.py              # Nose–Hoover chain (NVT), port of the reference
│   ├── selftest.py                # python -m gomd.selftest: forces vs autodiff, NHC + torsion vs reference, MLIP plumbing
│   ├── profile_step.py            # python -m gomd.profile_step: per-phase GPU timing, stale-list check, SCF solvers
│   ├── mlip/
│   │   ├── client.py              # MLIPProvider: x -> (E kcal/mol, F kcal/mol/Å), starts the server
│   │   ├── server.py              # standalone MLIP process (UMA via fairchem, or mock); numpy only
│   │   ├── elements.py            # atomic numbers from force-field masses
│   │   └── compare.py             # PFF vs MLIP agreement metrics -> CSV
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
├── md-input2/                     # system.xyz: md-input without lone pairs, for MLIP runs (§4.22)
├── md-output/                     # logs are written here
├── run_scan_md.slurm              # SLURM script (classical or polarizable)
├── GOMD_colab.ipynb               # Google Colab runner (Drive: My Drive/GOMD/code/GOMD)
├── GOMD_DOCUMENTATION.md          # this file
└── PROJECT_STATUS.md              # done / open tracker and change log
```

`gomd/` depends on `pkg/` at setup time only: `assembly_loader` runs the
reference parsers and `setup_system` once, then everything per-step is pure JAX
(or, for the MLIP, a separate process). `gomd/mlip/server.py` imports nothing
from `gomd` and runs from any Python that has numpy (plus fairchem for UMA).

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
- **`ensemble`** (M5): NVE/NVT/NPT flags, `Tset`, chain length `nnos`,
  `qfreq` (OmegaQ), `ndof` (3·natreal − nconst = 15,192 here), `natreal`,
  `multibig`, and the thermostat restart state (`xlogs`, `vlogs`, `glogs`)
  read from `coords.inp`. `atom_labels` per force-field type.
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
- `build_neighbor_ondevice_count(x, box, rlist2, excl_flat, nat, capacity,
  tile=512) → (idx_i, idx_j, count)` works in two passes (2026-11-17):
  1. All-pairs r² in fp32, tiled over rows (`lax.fori_loop`). Every candidate
     inside rlist is compacted into the list in (i, j) order. Rejected
     candidates (about 95% of the 76 M) go to an out-of-range slot with
     `mode='drop'`.
  2. Exclusions are removed once, by `searchsorted` over the ~3.4 M compacted
     keys rather than all 76 M, and the list is compacted again. Real pairs
     come first; the padding is `(0,0)`.

  Keys `i*nat + j` are int32 when nat² < 2³¹.
- **Speed:** 16 ms on the RTX 5070, down from 186 ms. The old one-pass build
  sent every rejected candidate to one dummy slot, which serialized the
  scatter, and it searched exclusions with int64 keys over all candidates.
  The output is element-for-element identical to the old build
  (`python -m gomd.tests.bench_neighbor`).
- For this box all-pairs is the right primitive (~76 M distances) and needs no
  host↔device traffic.
- **Validation:** the same pair set as the host cell-list build, except pairs
  within fp32 rounding of rlist. On `coords.inp` that is one pair, 9e-7 Å
  outside 13 Å. Such pairs lie in the skin and every kernel masks them by
  rcut. The earlier claim of a "bit-identical" pair set was off by exactly
  this one pair. Self-test check 7 runs this comparison.
- `tile`: 512 by default; a larger GPU (V100, A100) may gain from 1024–2048.
- With rcut 12 Å + skin 1 Å the list holds 3,354,520 pairs; capacity
  4,193,150.
- **Validity checks (2026-11-17)** — written, not yet run on a GPU:
  - `build_neighbor_ondevice_count` also returns the true pair count;
    count > capacity is an overflow.
  - `count_pairs_within(x, i, j, box, rcut²)` counts the listed pairs inside
    rcut. At a rebuild, count(fresh) − count(stale) at the same positions is
    exactly the number of interacting pairs the stale list was missing.
  - The MD loop rebuilds on the `--rebuild-every` schedule and whenever a site
    has drifted more than skin/2 − 0.01 Å since the last build (§4.15).

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
  triple product `−r_ij·n` (IUPAC convention, matching the derivative
  formulas); Fourier series `V0 − Σ ½Vₙcos(nφ)` bounded by a static
  `max_terms`; forces via dφ/dr on all four atoms, with the j/k coefficients
  written for r_kj = x_k − x_j (hence `a = −r_ij·r_jk/|r_jk|²`,
  `b = −r_jk·r_kl/|r_jk|²`).
- **Impropers:** ported; `ndeforms = 0` on this system.
- Validation: bond E 9e-16 / F 4e-12; bend exact / F 2e-13; torsion E 3e-13.
  Torsion F vs −∇E (autodiff) 5.6e-12 and vs the fixed reference 3.9e-14.
  Before 2026-11-09 both GOMD and the reference had two sign errors in the
  torsion force (it matched the buggy reference to 1e-10); that bug drove the
  NVE energy drift (§6).

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

### 4.7b `forces/recip_shared.py` — one shared k-space pass (2026-11-17)
- `make_recip(box, q, kx, ky, kz, alpha)` returns `tables(x)`, `energy(rt)`,
  `force(rt)` and `qmu_grad(rt, μ)`.
- `tables` builds cos/sin(k·r) (nat × nk), the structure factors C_k and S_k,
  and the reciprocal field, once per step. The integrator passes them (`rt`)
  to the classical term and to the polarization function.
- The table is factorized per axis: exp(ik·r) = exp(ik_x t_x) exp(ik_y t_y)
  exp(ik_z t_z), with t = 2πr/L. That is 8 + 15 + 15 cos/sin per site instead
  of 709, plus two complex multiplies per (site, k).
- **Classical:** the force is COULOMB · q_i times the field (the same
  prefactors), and the energy comes from C_k² + S_k².
- **k-space Q–μ force**, in closed form at fixed μ, with m_lk = μ_l·h_k,
  A_k = Σ m_ik sin_ik and B_k = Σ m_ik cos_ik:
  - dW/dr_l = q_l Σ_k P_k h_k (−A_k sin_lk − B_k cos_lk)
    + Σ_k P_k h_k m_lk (C_k cos_lk + S_k sin_lk).
  - This replaced `jax.grad` through the field pass.
- **Validation** (self-test check 8, `gomd/tests/bench_recip.py`): against
  `ewald.reciprocal_energy_force`, `pol_field._recip_field` and autodiff,
  within 1e-14 relative.
- **Speed:** timed alone, 6.0 instead of 16.0 ms for the three separate
  passes. The fused MD step barely changed (27.8 → 27.6 ms), because XLA
  already shared that work inside one jit'd program.

### 4.8 `constraints/shake.py` — SHAKE
- The reference is sequential (Gauss-Seidel). GOMD is **Jacobi**: all 7,008
  constraints compute corrections from the same start-of-iteration positions,
  then scatter-add at once, inside `lax.while_loop`. Converges when
  `mean((d²−|r|²)²) ≤ tol`.
- Both reach the same constraint surface; trajectories track the reference but
  are not bit-identical step to step.
- **Tolerance.** `mdrun.params` says 1e-4, but the reference never copies that
  value into `constraints.tol` and uses its dataclass default 1e-6. At 1e-6
  the Jacobi loop stops after ~2 sweeps with bond errors up to 3e-4 Å (rms
  4e-5 Å). The next predictor-corrector removes them, and the SHAKE-combined
  force rings with period two steps: its alternating part is ~20 kcal/mol/Å
  on the heavy atoms (the physical force's is ~0.1). PE, and so TE, alternate
  by ±4 kcal/mol, in GOMD and in the reference alike. Printing every 10 steps
  hid it. At 1e-12 (~11 sweeps, bond errors 2e-6 Å) the ringing is gone. GOMD
  now defaults to **1e-12**; `--shake-tol` sets it, `--reference-parity` uses
  1e-6.

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
- **`town_product`** is one T·μ product: a gather over pairs plus two
  `.at[].add` scatters. The Jacobi SCF calls it once per iteration; it is the
  old inner function, moved out.
- **Padding (2026-11-17):** padded `(0,0)` slots now scatter to a dropped
  index (`mode='drop'`) instead of adding zeros into atom 0. That contention
  serialized every pair scatter, here and in `pol_forces`,
  `_field_pairs_tab` and the classical pair kernels. The product went from
  6.8 to 3.6 ms on the RTX 5070; the arithmetic is unchanged.
- **`solve_scf_pcg`** (2026-11-17; the default since its 2,000-step NVE
  check, §5) is a preconditioned conjugate gradient.
  - It solves A·μ = E with A = diag(1/α − arf) − T, restricted to sites with
    α > 0. T is symmetric, so A is SPD when there is no polarization
    catastrophe.
  - Preconditioner: 1/A_ii.
  - Stopping test: max|α_i·r_i|² ≤ tolpol on the current residual r = E − A·μ.
    That is the size of the next Jacobi step, so the number means the same as
    the Jacobi `tolpol`. Because it measures the equation itself, a warm start
    is safe.
  - μ_init=None is a cold start and skips the product A·0.
  - Returns (μ, niter, r). niter is the number of T·μ products, as for Jacobi.
  - If p·A·p ≤ 0 (no longer positive definite: a polarization catastrophe or
    a CG breakdown), it stops at once and reports niter = max_iter. The
    driver then warns that the SCF is unconverged.
  - Energy: CG minimizes U = ½μ·A·μ − E·μ, and U = −½E·μ − ½μ·r. This equals
    the reference −½E·μ at convergence, with a second-order error before it.
    The PCG path logs U.
- **`pol_energy`** = `−½·COULOMB·Σ E·μ`.
- **`pol_forces`** — dipole-dipole + charge-dipole gradients over pairs.
  Optional per-pair charge overrides let the caller zero skin and padded
  pairs (§4.14).
  - erfc(αr) and exp(−α²r²) come from the reference's 16,384-knot tables,
    the same ones the SCF field and the reference `compute_pol_forces` use
    (`erfc_tab`/`exp_tab`/`dx_inv`, passed by `polarization_full`).
  - Before 2026-11-17 GOMD used exact fp64 erfc. The two differ by
    ≤ 2.7e-7 kcal/mol/Å, but exact fp64 erfc made this the most expensive
    kernel of the step.
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
- Phase 8 (`recipQmu`, the k-space charge–dipole force) is a TODO in the
  reference and missing from its forces. GOMD now adds it in
  `polarization_full.py` (§4.14).
- Validation: phase 6 2.6e-14, phase 7 5.8e-16.

### 4.14 `forces/polarization_full.py` — full polarizable term (M3)
- `make_polarization_fn(D, kx, ky, kz, scf_max_iter=50, scf_min_iter=0,
  recip_qmu_force=True, solver='jacobi'|'pcg')` returns jit'd
  `compute_polarization(x, idx_i, idx_j, mu_init=None) → (E_pol, f_pol, μ,
  {'niter'})`. The stages are: pair geometry → field (§4.12) → SCF → energy →
  forces (main + phases 6, 7 + k-space Q–μ) × COULOMB.
- `make_polarization_parts(...)` returns each stage as its own function
  (`geometry`, `field`, `field_parts`, `town`, `scf`, `energy`, `forces`,
  `force_parts`, `compute`), so the profiler can time them one by one. The
  arithmetic and its order are the same as before the split.
- Takes the **same rcut+skin, padded** pairs as the classical kernels. Skin and
  padded pairs are masked out of the field, the SCF factors and — through
  zeroed charges and displacements — the main induced force. (Without that
  mask the skin pairs gave a force error of ~4.6e4 relative.)
- Validation (positions in, everything self-contained): dipoles 2.8e-15,
  E_pol −7025.987542 (diff 1.2e-11), total induced force 8.0e-9 (with
  `recip_qmu_force=False`, the reference's forces).
- **k-space charge–dipole force** (`recip_qmu_force=True`, default). E_pol
  contains the reciprocal-space charge field, so −dE_pol/dx contains
  `COULOMB·∂/∂x[Σᵢ μᵢ·E0_recip,i(x)]` at fixed μ (E_pol is stationary in μ at
  the SCF solution). The reference leaves this term out. It is computed
  exactly with `jax.grad` of `_recip_field` with μ held fixed. On this system
  it has rms 0.19 and max 3.2 kcal/mol/Å, and its power at t = 0 is
  −0.095 kcal/mol/fs. That accounts for the +0.12 kcal/mol/fs NVE drift of the
  polarizable runs (§5). `--reference-pol-forces` drops it again.
- **SCF tolerance.** `tolpol` 1e-4 (`mdrun.params`) stops the cold-start
  Jacobi SCF after 4 iterations. E_pol is then 19 kcal/mol from converged:
  initial PE −3611.537 at 1e-4, −3631.319 at 1e-6, −3630.603 at 1e-10 (12
  iterations, matching the reference's −3630.602 at the same tolerance). The
  driver warns when tolpol > 1e-6. `--tolpol` overrides the file.

### 4.15 `integrator_scan.py` — the MD step (M4, M5)
`make_md_step(Dj, ..., polarization_fn=None, nhc_params=None, natreal=None,
shake_max_iter=100)` returns the step as pieces that share one arithmetic:

- `pre_force(S)`: [NHC half step] → first half kick
  `v += ½dt·(41.84·massinv)·f` → drift `x += v·dt·1e-5` + PBC wrap → SHAKE of
  the drifted positions with bond vectors from the pre-drift positions
  (reference `velocity_verlet_step`).
- `force_fn(x, idx_i, idx_j, mu_init)`: place lone-pair sites → classical PFF
  (+ polarization) → redistribute site forces.
- `post_force(S, x, pe, f)`: SHAKE predictor-corrector
  `x2 = x + 1e-5·dt·v + 1e-5·dt²·f/m`, constrained against the **post-drift**
  x, combined force `f = (x2c − x − 1e-5·dt·v)·m/(41.84e-5·dt²)` → second half
  kick → [NHC half step → COM-velocity removal] → KE, H_ext.
- `scan_body`, all on device:
  1. a scheduled neighbor rebuild (`lax.cond`) at the start of the step when
     `step % rebuild_every == 0`, as before;
  2. `pre_force`;
  3. a drift rebuild when any site of the force-evaluation configuration
     (after the drift, lone pairs placed) is more than skin/2 − 0.01 Å from
     its position at the last build. That test is exact for any dt; the
     0.01 Å only absorbs fp32 rounding in the build;
  4. the force and `post_force`.

  `make_runner` runs it in `lax.scan` for a fixed chunk length.

`S = {x, v, f, nhc, step}`. The scan carry, built with `init_carry`, adds:
- the pairs and the positions they were built at (`x_nb`);
- neighbor statistics `nb`: rebuilds, rebuilds on drift alone, overflows,
  maximum missed pairs, maximum pair count, peak drift;
- the dipole history (μ1, μ2, μ3);
- the last physical PE and force (`pe`, `fphys`), so an MLIP evaluated
  alongside is compared with exactly the force the integrator used.

The per-step output adds `scf_it` (T·μ products), `rebuilt` and `ncut`.

**Per-step rcut compaction** (`cut_capacity`, driver `--cut-capacity 1.10`,
2026-11-17):
- `force_fn` first runs `neighbor_ondevice.compact_pairs`. That cuts the
  rlist pairs to those inside rcut + 0.05 Å (fp32 test), in list order, into
  `cut_capacity` slots padded with `(0,0)`.
- The classical pair kernel and every polarization pair kernel run on this
  list: 2.94 M slots instead of 4.19 M.
- The margin keeps every pair either kernel's own rcut test (fp32
  round-based, fp64 single-shift) could accept.
- `ncut > cut_capacity` stops the run.
- It costs 1.3 ms per step and took the 2,000-step NVE step from 32.9 to
  27.8 ms (§7).

For MLIP propagation the driver calls `jit(pre_force)`, the MLIP, then
`jit(post_force)` per step.

[...] = NVT only. This is the reference `_vanilla_vv_step` plus the COM
removal of its `run_simulation`. SHAKE after the drift is applied in NVE as
well, as in the reference. Once the trajectory is on the constraints the
predictor-corrector already lands the drift there and the loop exits
immediately. With the reference's 1e-6 tolerance it is a no-op even on step 1,
so NVE at `--reference-parity` is bit-identical to the Nov 9 engine.

- `use_aspc` (`--scf-guess predict`, the default with PCG): the SCF is
  warm-started from 3(μ1 − μ2) + μ3, with all three slots seeded with μ0.
  At tolpol 1e-8 that averages 2.3 T·μ products per step.
  `--scf-guess cold` starts from μ = 0 each step, which the Jacobi solver
  needs (§6).
- NPT is refused (§6).

### 4.16 `thermostat.py` — Nose–Hoover chain (M5)
- Port of the reference `init_thermostat` / `init_integration` /
  `nhc_integrate`: MTK chains, 5 × 5 Yoshida–Suzuki factorization, a half step
  before and after velocity Verlet, chain masses Q₁ = gNkT/ω², Qᵢ = kT/ω²
  (ω = OmegaQ), gNkT = ndof·k·Tset. The chain length is static, so its loops
  unroll at trace time.
- The initial chain state comes from `coords.inp` (restart).
- Logged: `T = 2KE/(ndof·k)` and the conserved energy
  `H_ext = KE + PE + Σ ½Qᵢvᵢ² + gNkT·ξ₁ + kT·Σᵢ₌₂ ξᵢ`, which the reference
  never reports.
- Left out on purpose: the reference's clamping of exp() arguments to ±700,
  chain-velocity reset on NaN, and velocity-scale clipping to [0.1, 10].
  These guards never fire in a healthy run and silently corrupt the dynamics
  when they do; a blow-up shows in H_ext instead.
- Validation: half step vs the reference, chain lengths 1 and 3, four
  successive half steps: v 1e-15, ξ 6e-16, v_ξ 6e-15, G_ξ 1.4e-14 relative.

### 4.17 `mlip/` — ML interatomic potentials (M5)
- **Process split.** `MLIPProvider.start(...)` launches `mlip/server.py` as
  its own process (`python -I`), optionally from another Python
  (`--mlip-python`, e.g. a venv with fairchem). PyTorch and JAX never share a
  process, CUDA context or memory pool, and fairchem can pin its own
  torch / numpy (fairchem-core 2.25 needs torch ~= 2.13 and numpy < 2.5).
- **Protocol.** Length-prefixed JSON header + raw float64 blob over
  stdin/stdout: `init` (atomic numbers, cell, pbc, charge, spin, positions),
  `eval` (positions → energy + forces), `quit`. Library prints to fd 1 are
  redirected to stderr so they cannot corrupt the channel. Errors come back
  with their traceback and are raised as `MLIPServerError`.
- **UMA backend.** `fairchem.core.pretrained_mlip.get_predict_unit(model,
  inference_settings, device)` + `FAIRChemCalculator(predictor,
  task_name)`. One persistent `ase.Atoms` (pbc, cell = diag(box),
  `info = {charge, spin}`), positions updated per call. Models: `uma-s-1p2p1`
  (default), `uma-s-1p2`, `uma-s-1p1`, `uma-m-1p1`, or a checkpoint path.
  Tasks: `omol` (default), `omc`, `omat`, `odac`, `oc20`.
- **Settings.** Any fairchem preset name works: in 2.26 `default` (= nitro,
  TF32), `high-precision` (FP32, compiled), `batch`, ….
- **GOMD's own settings:** `lowmem` is FP32 with activation checkpointing on,
  merged MOLE experts and no compilation; `lowmem-fp64` is the same in
  float64.
  - Both presets above turn checkpointing off. On 7,400 atoms they needed
    more than 21 GB (L4) and more than 37 GB (A100-40) for one force
    evaluation.
  - float64 removes the fp32 resolution limit on omol's absolute energy,
    about 25 kcal/mol at 2×10⁸ kcal/mol.
- The server sets `MPLBACKEND=Agg` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Cache the model on local disk, not on Drive. huggingface_hub's cache uses
  symbolic links, which Drive stores as small text files (2026-11-18).
- **Mock backend.** Harmonic springs (10 eV/Å²) over all pairs closer than
  1.9 Å in the first frame. It is conservative and numpy-only, for testing the
  path without fairchem.
- **Units and sites.** Only real atoms are sent (elements from the
  force-field masses, `elements.py`), so lone-pair sites get zero force. eV →
  kcal/mol × 23.0605 (96.48533212/4.184). The default charge is the rounded
  sum of PFF charges (0 here), with spin multiplicity 1.
- **Comparison** (`compare.py`), real atoms only, one CSV row per sample:
  - E_pff, E_mlip, and `dE_shift` = (E_mlip − E_pff) minus its value at the
    first sample (the two zeros are unrelated), also per atom.
  - Force RMSE, MAE, max per-atom |ΔF|, relative error ‖ΔF‖/‖F_pff‖, global
    cosine, rms of each force, and RMSE per element.
  - F_pff is the physical force after lone-pair redistribution, not the
    SHAKE-combined one.

### 4.18 `selftest.py`
`python -m gomd.selftest --input-dir ./md-input --ref-dir ./pkg` (CPU by
default; `--gpu` uses the JAX default backend) builds its reference from
`pkg/` on the fly and checks:
1. every classical force term against −∇E (autodiff);
2. lone-pair redistribution against the chain rule;
3. torsion against the reference;
4. the NHC half step against the reference;
5. the mock MLIP server round trip against finite differences;
6. (2026-11-17, polarizable systems) two SCF checks:
   - PCG and Jacobi give the same dipoles at a tight tolerance;
   - PCG, warm-started from a 1%-biased guess, comes back to the converged
     energy at tolpol 1e-8. The Jacobi result from the same guess is printed
     for contrast. `--skip-pol` skips both;
7. the on-device neighbor build against the host cell-list build: same pairs
   (except those within fp32 rounding of rlist), (i, j) order, `(0,0)`
   padding.

Exit code 0 only if every check passes.

### 4.19 `main_scan.py` — the driver
Loads via the assembly loader, resolves the ensemble, builds k-vectors and
self-energy, sizes the pair capacity from one host build, builds the
polarization function if `--polarizable`, starts the MLIP server if `--mlip`,
then runs the trajectory. PFF propagation runs in `lax.scan` chunks of
gcd(`--print-every`, `--compare-every`) steps; the first chunk is a discarded
warm-up compile. MLIP propagation runs one host round trip per step.

| Flag | Default | Meaning |
|------|---------|---------|
| `--input-dir`, `--ref-dir` (required) | | inputs; reference package |
| `--steps`, `--print-every`, `--rebuild-every` | 200, 50, 100 | rebuild-every = scheduled interval, 0 = only on drift |
| `--skin`, `--nb-capacity` | 1.0 Å, 1.25 | rlist − rcut; pair capacity as a multiple of the first build |
| `--cut-capacity` | 1.10 | per-step list of pairs inside rcut + 0.05 Å, as a multiple of the first count; 0 = off |
| `--restart-dir`, `--restart-every` | `<out>_restart`, 0 | restart input directory, written at the end (and every N steps); `none` disables (§4.21) |
| `--no-drift-rebuild` | off | no drift trigger; the log reports pairs a stale list missed |
| `--polarizable` | off | full induced-dipole force field |
| `--ensemble auto\|nve\|nvt` | auto | auto = `mdrun.params`; NPT is refused |
| `--tolpol`, `--scf-max-iter` | file, 50 | SCF tolerance on max\|Δμ\|² |
| `--scf-solver pcg\|jacobi` | pcg (jacobi with `--reference-parity`) | PCG: residual test, variational energy (§4.11) |
| `--scf-guess predict\|cold` | predict with pcg, cold with jacobi | predict = extrapolate the last 3 steps |
| `--shake-tol`, `--shake-max-iter` | 1e-12, 100 | SHAKE tolerance on mean(Δd²)² |
| `--reference-pol-forces` | off | drop the k-space charge–dipole force |
| `--reference-parity` | off | SHAKE 1e-6 + `--reference-pol-forces` |
| `--mlip none\|uma\|mock` | none | ML potential |
| `--propagate pff\|mlip` | pff | which engine drives the dynamics |
| `--compare-every N` | `--print-every` | evaluate the other engine every N steps |
| `--compare-out` | `<out>.compare.csv` | comparison table |
| `--mlip-model`, `--mlip-task`, `--mlip-settings`, `--mlip-device` | uma-s-1p2p1, omol, default, auto | fairchem options |
| `--mlip-python` | this Python | interpreter for the MLIP server |
| `--charge`, `--spin` | Σq rounded, 1 | UMA omol |
| `--electrode-lower`, `--electrode-upper`, `--electrode-geometry`, `--slab-height` | none, flat | fixed electrodes and the slab box (§4.23, §4.24) |
| `--kspace auto\|explicit\|pme`, `--pme-spacing`, `--pme-order` | auto, 1.0 Å, 6 | auto = PME with electrodes (§4.25) |
| `--cpm-dv`, `--cpm-psi-lower`, `--electrode-eta` | off, −ΔV/2, 1.979 Å⁻¹ | constant potential (§4.26, §4.27) |
| `--cpm-ramp` | 0 ps | ramp Ψ from 0 over this time; logs W_psi and TE + W_psi (§4.28) |
| `--charge-every` | 10 | electrode totals and layer charges to `<out>_charges.csv`; 0 = off |
| `--poisson-every`, `--poisson-start`, `--poisson-bin` | 10, end of ramp, 0.1 Å | time-averaged Poisson profile to `<out>_poisson.csv`; 0 = off |
| `--cpu`, `--out` | | |

Log: NVE `step PE KE TE [scf_it] ms/step`; NVT
`step PE KE TE T H_ext [scf_it] ms/step`.
- `scf_it` (polarizable PFF runs) is the mean number of T·μ products per step
  over the print interval. A change in it explains steps in PE when tolpol is
  loose.
- With `--propagate mlip` PE is the MLIP energy.
- ms/step is MD time only; comparisons are timed separately.
- Comparison lines start with `#`.
- Neighbor-list events are `#` lines:
  - extra rebuilds on drift;
  - missed pairs;
  - an overflow, which stops the run with exit code 2.
- A summary line closes the run. With an
MLIP the driver sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX starts,
so JAX does not reserve 75% of the GPU the MLIP server also uses.

### 4.28 Constant potential in production runs (P4, 2026-11-19)
- **Voltage ramp** (`--cpm-ramp PS`). The electrode potentials are s(n)·Ψ,
  with s(n) = min(1, n·dt / t_ramp) at MD step n. s = 0 at the step-0 force
  call, so a run starts from the uncharged capacitor (charges induced by the
  electrolyte only) and the double layer builds over the ramp instead of the
  electrode charges jumping at step 1. `cpm.solve`, `charge_terms` and
  `coupled_solve` take the scale; the MD step derives it from `S['step']`.
  - The ramp counts from step 0 of this run. Leave it off when continuing
    from a restart that already reached the full potential.
  - While s changes, the Hamiltonian depends on time and TE is not conserved:
    the source does work. At fixed positions, dW/ds = −Ψ_full·q (W =
    min over q, μ of U − sΨ·q; envelope theorem). The driver integrates
    W_psi = ∫ Ψ_full·q ds by the trapezoid rule over every step and logs
    `dV`, `W_psi` and `TE+W_psi` (NVE) or `H_ext+W` (NVT). That sum is the
    conserved quantity through the ramp and after it.
- **Electrode charge file** (`--charge-every N`, default 10):
  `<out>_charges.csv` with step, time, the applied ΔV, Q_lower, Q_upper and
  the charge of every electrode layer, numbered bottom to top in each
  electrode (the header line gives each layer's z). The layer sums come from
  the step (`make_md_step(cpm_layer_id=...)`), so they cost nothing extra.
- **Time-averaged Poisson profile** (`--poisson-every N`, default 10;
  `--poisson-start`, default the end of the ramp; `--poisson-bin`, 0.1 Å).
  - Inside the scan (`make_md_step(poisson=...)`, carry `pois`), every N-th
    step adds the point charges and the induced μ_z per z bin, and the
    electrode charges per site.
  - Point charges and dipoles are assigned to the two nearest bin centres
    linearly (cloud-in-cell). Nearest-bin assignment moves each charge by up
    to dz/2, which shifts the potential drop: on `md-input3` it read 1.984 V
    at 2 V (−0.8 %).
  - `poisson.profile_binned` turns the sums into ρ(z) (mobile, electrode
    Gaussian marginals, bound −dP_z/dz), E(z) and Ψ(z). The driver writes
    `<out>_poisson.csv` at the end and at every restart, and logs the drop
    between the electrode interiors (all layers but the one facing the
    electrolyte) against ΔV.
- **Speed of the coupled solve.** Measured in one process, alternating
  100-step NVE chunks (`md-input3`, polarizable, 2 V, tolpol 1e-9), and per
  operator inside a 20-iteration `fori_loop`, which removes the ~1.3 ms
  launch floor a standalone call shows:

  | | before P4 | P4 |
  |---|---|---|
  | one Schur application C Ã⁻¹ Cᵀ μ | 6.2 ms | 3.1 ms |
  | real-space part of one C q | 1.43 ms | 0.18 ms |
  | MD, coupled constant potential | 69.3 ms/step | **63.8 ms/step** |
  | MD, polarizable with fixed electrode charges (reference) | 44.2 | 44.1 |

  - The gain is the padding. The electrode–electrolyte pair list has 30 %
    spare slots, and they all pointed at atom 0 and electrode site 0, so
    every scatter over the pairs (C q, its transpose, b, the charge forces)
    piled ~200 k zeros into one row. Unused slots now hold out-of-range
    indices, and the gathers whose transposes are scatters use
    `mode='fill'`, so the scatters drop them. This is the same fix as for the
    bulk pair kernels (2026-11-17, 46.2 → 32.9 ms/step).
  - The k-space part of C now spreads only the electrode charges, whose
    B-spline stencils are built once (the electrodes are frozen), and
    interpolates the field only at the 11,960 polarizable sites, whose
    stencils are built once per step (`cpm.prep_coupling`, `pme.stencil /
    spread / potential_grid / field_from_grid`). It equals the full pass to
    3e-17 relative, and Cᵀ to 2e-16. It makes one Schur application 20 %
    cheaper in a loop, but the MD step does not get measurably faster (64.0
    vs 63.8 ms/step). `cpm.cfield_full` keeps the full-pass form for the
    checks.
  - What remains (~20 ms/step over fixed charges): about 3.6 Schur
    applications per step (2.6 PCG products + the initial residual), C q₀
    for the right-hand side, the final Cᵀμ, the stencils and the real-space
    electrode–dipole force. Every one of them is needed for the solve to be
    the exact minimum that the Hellmann–Feynman forces assume.
- **Validation** (`gomd/tests/check_cpm_p4.py`; `md-input3`, polarizable,
  2 V, tolpol 1e-12, coupled solve):
  1. the restricted coupling equals the full PME pass: C q to 5e-15 and Cᵀμ
     to 3e-15 relative; the step energy is identical, forces agree to 3e-13
     kcal/mol/Å and charges to 3e-16 e;
  2. dW/ds = −Ψ_full·q(s) at fixed positions, by central differences in s
     (charges and dipoles re-solved), to 4.6e-7 relative at s = 0.25 and
     0.75: the identity behind W_psi;
  3. the layer charges add up to the electrode totals (2e-16 e);
  4. the scan's Poisson accumulator after one step gives the same Ψ(z) as
     `poisson.profile` on that configuration, to 1.5e-13 V over 2,050 bins;
  5. NVE through a 0 → 2 V ramp over 40 steps (80 steps in all): TE falls by
     10.26 kcal/mol (20-step means at the start and end), W_psi = 11.46, and
     TE + W_psi moves by +0.02. Step to step it spreads 1.94 kcal/mol, the
     same as TE alone after the ramp.
  - Linear assignment also tightened the P2/P3 Poisson checks (one
    configuration, 2 V): 1.9999 V (was 2.0033) and, with dipoles, 1.9997 V
    (was 2.0031).
- **MD** (`md-input3`, classical, 2 V, NVE, 0.1 ps ramp in a 0.2 ps run,
  nearest-bin Poisson assignment):
  - TE falls by 6.7 kcal/mol over the ramp; the source does
    W_psi = 6.06 kcal/mol; TE + W_psi stays within ±0.8 kcal/mol, the band of
    the run without a ramp;
  - the electrode charges grow from ±0.045 e (step 0, Ψ = 0) to ±0.20 e at
    the end of the ramp;
  - almost all the charge sits on the layer facing the electrolyte (−0.233 e
    of −0.221 e on the lower electrode, with +0.012 e on the next layer);
  - 16.0 ms/step, the same as without the P4 output.
- **MD** (`md-input3`, polarizable, coupled, 2 V, NVE, 0.05 ps ramp in a
  0.2 ps run, `main_scan`, linear Poisson assignment):
  - TE + W_psi stays within 3631.2–3632.6 kcal/mol through and after the
    ramp, while TE falls by about 10 and W_psi reaches 11.98;
  - the electrode charges grow from ±0.042 e to ±0.42–0.49 e;
  - the time-averaged Poisson drop over 61 samples after the ramp is
    1.9998 V at 2 V (−0.01 %);
  - 2.63 PCG products per step; 63.4–64.8 ms/step once the GPU left its slow
    clock state (113 ms/step for the first 160 steps).

### 4.27 Constant potential with induced dipoles (P3, 2026-11-19)
- **One energy:**
  W(q, μ) = W_P2(q) + ½ μ·(α⁻¹ − arf − T)·μ − μ·(E_ff + C q)
  is minimised jointly in q and μ, at fixed total electrode charge.
  - C q is the field of the electrode charges at every site:
    - the PME pass with q on the electrode sites (k-space + planar);
    - the real-space Gaussian part −q v′(r) r̂ over the electrode–electrolyte
      pairs, with v = [erfc(αr) − erfc(ηr)]/r and no Thole damping.
  - Cᵀμ, the potential at the electrode sites from the dipoles, is `jax.vjp`
    of the same map. So the two directions are exact transposes and the
    coupled matrix stays symmetric. This is the consistency that two
    separately written forces, as in OpenMM's AMOEBA with constant
    potential, do not have.
- **Solve** (`cpm.coupled_solve`). q is eliminated exactly:
  - q(μ) = q₀ + Ã⁻¹Cᵀμ, with q₀ the P2 charges (μ = 0) and Ã⁻¹ the inverse
    on the fixed-charge subspace;
  - PCG solves (α⁻¹ − arf − T − C Ã⁻¹ Cᵀ) μ = E_ff + C q₀, through
    `solve_scf_pcg(extra_op=...)`. Every product applies the Schur term
    with one C and one Cᵀ;
  - the energy is −½ COULOMB (E_tot·μ + μ·r), with E_tot·μ =
    E_rhs·μ + dq·Cᵀμ;
  - the ASPC predictor warm-starts μ as before.
- **Forces** (Hellmann–Feynman at fixed q, μ):
  - the P2 charge forces with the final q;
  - the usual dipole forces on the k-space tables built with all charges
    (`qmu_grad` with the electrode charges included);
  - −COULOMB ∇ of the real-space electrode–dipole energy
    (`edip_energy_real`, by `jax.grad`).
- **Driver:** `main_scan --polarizable --cpm-dv V` (PCG required).
  `make_md_step(pol_parts=...)` routes the polarizable term through the
  coupled solve.
- **Validation** (`gomd/tests/check_cpm_pol.py`; `md-input3`, 2 V,
  tolpol 1e-12; all pass):
  1. coupled solve = an alternating solve (dipoles at fixed q, then q at
     fixed dipoles): charges to 5e-9 e, dipoles to 1.3e-6 e Å, the tolpol
     level. That takes 11 PCG products, against 57 full alternations.
  2. the dipole equation holds against a field rebuilt without the coupling
     operator (fresh PME with the final charges + numpy Gaussian field):
     max |α(E_tot − Aμ)| 5.8e-7.
  3. forces = −dW/dx by central differences with q and μ re-solved at every
     displacement, to 1.1e-5 relative. Checked on atoms 2 Å from the
     electrode and 66 Å away.
  4. Poisson profile with the dipoles' bound charge: a 1.9997 V drop at 2 V
     (−0.01 %; 2.0031 V with the nearest-bin assignment used before P4).
     Without the dipoles in ρ it would be 6.76 V, so Cᵀ carries
     the physics.
  - The default pair kernel sums its energy over 7.7 M pairs in fp32: the
    energy moves in about 5e-4 kcal/mol steps, too coarse for central
    differences at 1e-4 Å, though its forces equal the fp64 ones. The check
    therefore uses the fp64 kernel. Per term, all fp64 classical pieces pass
    finite differences on `md-input3` to ≤1.2e-6.
- **MD** (`md-input3`, polarizable, 2 V, NVE, 200 fs, tolpol 1e-9):
  - TE stays within ±0.9 kcal/mol with no drift, the same band as without
    electrodes;
  - 2.60 PCG products per step, against 2.38 without electrodes;
  - electrode charges ±0.43–0.51 e, against ±0.21 e for the non-polarizable
    electrolyte and ±0.45 e at step 0;
  - about 69 ms/step at P3, against 44 without constant potential; 64
    ms/step since P4 (§4.28).

### 4.26 `cpm.py` — constant-potential electrodes (P2, 2026-11-18)
- **Model** (Siepmann & Sprik 1995; Reed, Lanning & Madden 2007): every
  electrode site carries a Gaussian charge q (η = 1.979 Å⁻¹ by default,
  `--electrode-eta`). Each step the charges minimise U − Ψ·q at a fixed total
  (the cell is neutral), so the potential dU/dq at every site of an electrode
  equals its Ψ. The charges are solved exactly each step; there is no lag.
- **Setup:** `make_cpm`.
  - A, the electrode–electrode matrix, is built once from three parts:
    - real-space Gaussian–Gaussian pairs, [erf(ηr/√2) − erf(αr)]/r within
      r_c;
    - the self terms, 2(η/√(2π) − α/√π);
    - the k-space (+ planar) response of the very PME pass the dynamics
      uses, from unit charges vmapped in batches. That keeps the solve
      exactly consistent with the forces.
  - A is factored by Cholesky; A⁻¹ and A⁻¹1 are kept (3,584 sites:
    103 MB, about 6 s).
- **Per step** (`make_md_step(cpm=...)`; `--kspace pme` is required):
  1. b = the PME potential of the force-field charges at the electrode sites
     (the electrodes are at zero there), plus the real-space
     electrode–electrolyte part [erfc(αr) − erfc(ηr)]/r over the
     electrode–electrolyte pairs, compacted from the neighbor list each
     step.
  2. q = A⁻¹(Ψ − b) − A⁻¹1·(1ᵀA⁻¹(Ψ − b) − Q)/(1ᵀA⁻¹1).
  3. A second PME pass with all charges, plus the real-space
     electrode–electrolyte term, ½ qᵀA_real q and −Ψ·q.
  - Forces are Hellmann–Feynman (Coulomb at fixed q). The reported PE is
    U − Ψ·q, the conserved quantity.
- **Driver:**
  - `main_scan --cpm-dv V` sets Ψ_upper − Ψ_lower. The lower electrode is at
    −dV/2 and the upper at +dV/2, or `--cpm-psi-lower` sets the lower one.
  - Needs the electrodes and `--slab-height`. Refused with `--polarizable`
    until P3.
  - The log adds Q_lower / Q_upper columns. Overflowing electrode–electrolyte
    pairs stop the run.
- **Units:** Ψ is converted with 23.0605 kcal/mol/e per V. APPLE&P's Coulomb
  constant is 332.08, against CODATA's 332.0637, so the model's volt is
  5.1e-5 off the physical one. Compare potentials in model units.
- **`gomd/poisson.py`:** the laterally averaged ρ(z) is built from three
  parts: binned point charges, the 1D marginals of the electrode Gaussians,
  and −dP_z/dz from induced dipoles. Integrating with zero field below the
  stack gives Ψ(z) in V.
- **Validation** (`gomd/tests/check_cpm.py`, all pass):
  1. charges solved on the GPU, potentials evaluated by the exact 2D Ewald
     reference: φ − Ψ is the same on every electrode site to 5.6e-9 V
     (small cell, 144 Gaussian sites and 12 points);
  2. gauge: (−½, +½) V and (0, 1) V give the same charges to 4e-16 e;
  3. forces on the electrolyte at the solved charges = reference to
     7.5e-6 kcal/mol/Å;
  4. finite differences of W = U − Ψ·q, with the charges re-solved at every
     displacement = the Hellmann–Feynman force to 6e-8 kcal/mol/Å;
  5. `md-input3` empty capacitor at 1 V: Q = 0.059559 e against A·ΔV/(4πd) =
     0.059371 e (+0.32 %; Reed et al. +0.18 %). The inner layer carries
     105.2 % of the electrode charge; the layers behind it alternate
     −5.5 %, +0.29 %, −0.01 %, the metallic screening pattern of Siepmann &
     Sprik's Table II;
  6. a +1 e charge at z = d/4 between grounded electrodes induces
     −0.7508 / −0.2492 e (images: −0.75 / −0.25); at d/2, −0.5000 / −0.5000;
  7. Poisson profile of one `md-input3` configuration at 2 V: a 1.9999 V
     drop between the electrode interiors (−0.00 %; 2.0033 V with the
     nearest-bin assignment used before P4), flat to 0.2 mV inside
     each electrode.
- **MD** (`md-input3`, classical APPLE&P, slab 205 Å, PME, ΔV 2 V, NVE,
  200 fs):
  - U − Ψ·q + KE stays within ±0.8 kcal/mol with no drift.
  - The electrode charges grow from ±0.164 e to about ±0.23 e as the double
    layer forms; the empty capacitor would hold ±0.119 e.
  - 15.7 ms/step on the RTX 5070, against 16.4 ms/step for the same run
    without constant potential. The second PME pass, the
    electrode–electrolyte pairs and the 3,584² matvec cost nothing
    measurable, within the GPU's clock noise.

### 4.25 `forces/pme.py` — smooth particle-mesh Ewald (2026-11-18)
- **What:** `make_recip_pme(box, q, alpha, spacing=1.0, order=6, slab)` is
  SPME (Essmann et al. 1995) with the same interface as
  `recip_shared.make_recip`, so the classical force, the SCF field and the
  charge–dipole force use it unchanged. `rt` also carries `phi`, the
  reciprocal (+ planar) potential at each site, and the charges can be
  passed per call; both are for the constant-potential electrodes.
- **How:**
  - Cardinal B-spline spreading, FFT convolution with B(m)·C(m), and
    interpolation of φ and E.
  - force = q E, which is exactly −∂E/∂x of the SPME energy.
  - `qmu_grad` is the autodiff of Σ μ·E through the same discrete pipeline,
    so it is exactly consistent with the field the SCF uses. There is no
    hand-written dipole spreading.
  - The slab planar term is as in `recip_shared`.
- **Driver:**
  - `main_scan --kspace auto|explicit|pme` (`--pme-spacing`, default 1.0 Å;
    `--pme-order`, default 6).
  - `auto` picks PME when electrodes are set and the explicit-k sum
    otherwise, so bulk runs keep the reference code's k-space.
  - `--reference-parity` forces explicit.
- **Gaussian electrode charges enter k-space as points.** Their width is an
  exact short-ranged real-space correction, −erfc(ηr)/r: energy and forces
  agree to 4e-15 with the Gaussian-factor form; site potentials agree up to
  a uniform constant.
- **Validation:**
  - `gomd/tests/check_pme.py`, 300 random charges, slab and bulk:
    - at 0.5 Å / order 8, PME = converged explicit-k to 4e-9 (energy) and
      5e-10 (field), with errors falling steadily with spacing and order;
    - force = −∇E to 8e-14;
    - `qmu_grad` = analytic explicit-k to 3e-9.
  - Self-test check 8b (`md-input`): PME 1.0 Å / order 8 = explicit-k
    14/14/14 to 3e-7. The self-test is now 15/15.
- **Accuracy and cost on the `md-input3` slab** (α 0.23, vs explicit
  11/11/66; table in `md-output/kspace_convergence_pme_md-input3.txt`).
  The GPU time is one pass with energy, force and the Q–μ gradient on the
  RTX 5070.

  | k-space | F rms / max (kcal/mol/Å) | electrode φ rms / max | GPU |
  |---|---|---|---|
  | explicit 7/7/25 | 0.035 / 0.40 | 3.4 / 10.6 mV | 20–40 ms |
  | explicit 8/8/35 | 0.0022 / 0.036 | 0.18 / 0.60 mV | 70–84 ms |
  | PME 2.0 Å p6 (18×18×108) | 0.0009 / 0.021 | 0.014 / 0.077 mV | 2.7 ms |
  | PME 1.5 Å p6 (24×24×144) | 0.0001 / 0.0026 | 0.002 / 0.016 mV | 2.9 ms |
  | PME 1.0 Å p6 (36×36×216), default | <1e-4 / 0.0002 | 0.0001 / 0.0008 mV | 3.5 ms |
  | PME 1.0 Å p8 | <1e-4 / <1e-4 | ~0 | 5.6 ms |
- **MD:**
  - `md-input3` slab, polarizable NVE over 200 fs with PME: TE within
    ±1 kcal/mol; 44 ms/step, vs 57–67 with explicit 7/7/25.
  - Bulk `md-input`: about even, 22.5–24.4 vs 21.7 ms/step, with a
    standalone pass of 2.6 vs 3.6 ms.
  - The laptop GPU switches between two clock states (bulk: ~22 vs
    ~34 ms/step), so compare timings only within a run, or alternate runs.
- **Harness note:** the WSL VM has 7 GB of RAM. Scoring several large k-space
  settings in one process gets the process killed, so
  `kspace_convergence.py --cache` stores the prepared cell and the reference
  for one-setting-per-process runs.

### 4.24 Slab electrostatics: 2D Ewald (constant-potential task, P1, 2026-11-18)
- **`forces/ewald2d_ref.py`**: fp64 numpy references for point and Gaussian
  charges, for small N. Each returns the energy, the potential at each site
  (dU/dq, which a constant-potential electrode needs) and the forces.
  - **`ewald2d_exact`:** the 2D Ewald sum (Hu, JCTC 2014 eqs 37–39) with
    Gaussian charges as in Reed et al. 2007 (eq A31).
    - Pair smearings: 1/σ² = 1/α² + 1/η_i² + 1/η_j².
    - Unlike Reed's A31, the real-space kernel keeps the full
      erf(s r)/r − erf(σ r)/r. For graphite (C–C 1.41 Å, η 1.979 Å⁻¹) the
      term Reed drops is 0.6 % at nearest neighbours.
  - **`ewald3dc` / `recip3d`:** the 3D sum in a box taller than the slab plus
    the planar dipole term 2π M_z²/V. This is the trapezoid rule for the
    2D sum's k_z integral.
- **GPU path:** `recip_shared.make_recip(..., slab=True)` adds the planar term.
  - Energy: + 2π M_z²/V, with M_z = Σ q z.
  - The field gets a uniform −4π M_z/V in z, so the classical force, the
    SCF field and the charge–dipole energy all carry it.
  - `qmu_grad` gets −4π q_l Σ μ_z / V.
  - Induced dipoles do not enter M_z: APPLE&P's μ–μ interaction is real-space
    only (RF/taper), so it has no k = 0 term.
- **Driver:**
  - `main_scan --slab-height` turns the planar term on; `--no-planar-term`
    turns it off, for comparisons.
  - `--kmax kx,ky,kz` overrides the k-limits; restarts keep them.
  - The log names the k-space mode.
- **Validation** (`gomd/tests/check_ewald2d.py`; 184 sites: 4 Gaussian layers
  plus 40 points, in a 12 × 13 Å cell):
  - exact 2D sum independent of α (0.3 / 0.4 / 0.5): energy, site potentials
    and forces to 2e-15;
  - isolated Gaussian–Gaussian and point–Gaussian pairs in a 300 Å cell:
    analytic energy to the predicted lateral-image term (1.5e-6);
  - 3D sum + planar term vs exact, as the vacuum gap grows (Lₓ 12 Å):

    | gap | 18 Å | 24 Å | 36 Å | 48 Å |
    |---|---|---|---|---|
    | max dF | 5e-7 | 3e-8 | 7e-11 | 2e-13 |

    This tracks exp(−2π·gap/Lₓ). Without the planar term the error stays at
    6e-2.
  - GPU pass = numpy reference to 2e-13 kcal/mol/Å; force = −∇E; `qmu_grad`
    = autodiff.
  - For `md-input3` (Lₓ 33.8 Å) the 76.8 Å gap of the 205 Å slab puts the
    image error near exp(−14).
- **k-space convergence on the `md-input3` slab** (α 0.23, reference
  11/11/66; `gomd/tests/kspace_convergence.py`, table in
  `md-output/kspace_convergence_md-input3.txt`):

  | kmax | k-vectors | dE (kcal/mol) | F rms / max (kcal/mol/Å) | field rms (V/Å) | electrode φ rms / max | GPU pass |
  |---|---|---|---|---|---|---|
  | 7/7/25 (input, scaled) | 2,535 | −14.0 | 0.035 / 0.40 | 5e-3 | 3.4 / 10.6 mV | 36 ms |
  | 8/8/35 | 4,645 | −0.80 | 0.0022 / 0.036 | 3e-4 | 0.18 / 0.60 mV | 70 ms |
  | 9/9/45 | 7,639 | −0.028 | 0.0001 / 0.0015 | 9e-6 | 0.004 / 0.015 mV | does not fit 8 GB |

  - The real-space cutoff error at α 0.23 and 12 Å is roughly 0.005
    kcal/mol/Å (Kolafa–Perram estimate; erfc(α r_c) = 9.5e-5). That makes
    8/8/35 the balanced choice.
  - In a full polarizable step on the 8 GB RTX 5070, 8/8/35 does not fit:
    each (sites × k) fp64 table is 652 MB and the pass keeps several. It
    needs Colab (L4 / A100) until the pass is made memory-lean (k-chunked or
    PME).
- **NVE in the slab with the planar term** (7/7/25, 200 fs): TE within
  3640.3–3642.2 kcal/mol, the same band as without it.
  - With neutral electrodes the term is small (0.02 kcal/mol at step 0).
  - It will dominate once the electrodes carry charge: ±5 e, 108 Å apart,
    gives about 2,500 kcal/mol.

### 4.23 `electrodes.py` — fixed electrodes (constant-potential task, P0, 2026-11-18)
Sets up an electrode | electrolyte | electrode cell with both electrodes held
fixed. Their charges are still the force field's (0 for graphite Cg); P2 makes
them variable (design in `PROJECT_STATUS.md`, task 3).
- **Driver options** (`main_scan`):
  - `--electrode-lower SPEC --electrode-upper SPEC`. A SPEC is
    `first:N`, `last:N`, `a-b` (1-based, inclusive) or single indices,
    comma-separated. The electrodes are normal to z, lower below the
    electrolyte; other axes are not supported yet.
  - `--electrode-geometry flat|snapshot` (default flat).
  - `--slab-height L` (Å).
  - `--dt` (fs) overrides `mdrun.params`; it is a general option, added for
    the time-step tests.
- **Layers.** Each electrode is unwrapped in z about its first atom and split
  into layers at gaps > 1 Å.
- **Flat geometry.** Each layer goes to its mean z. A layer that is a graphene
  sheet commensurate with the box (4·nx·ny atoms; a rectangular cell of
  3b × √3b in either orientation) is snapped onto the ideal lattice:
  - every atom goes to its nearest ideal site, one-to-one, with the lattice
    offset fitted per layer;
  - other layers keep x, y and are only flattened.
  - On `md-input3` each sheet is 8 × 14 cells, b = 1.4100 Å, armchair along
    x. In-plane moves were rms 0.04–0.05 Å, at most 0.14 Å.
- **Slab layout.** The box becomes L tall. The stack is centred, with vacuum
  above and below; the vacuum must be at least rlist.
  - Every mobile site keeps its height above the lower electrode's bottom
    layer, measured going up the old box. A stack that the input wrapped
    across z comes out whole.
  - kmax_z scales with the height (16 → 25 for 131.6 → 205 Å), which keeps
    the z k-space cutoff.
  - The k-space sum is still 3D Ewald with tinfoil boundaries until P1 adds
    the planar dipole term.
- **Freezing.**
  - Electrode velocities are zeroed and `D['frozen']` is set.
    `make_md_step(frozen=...)` removes frozen atoms from the moved, kicked
    and thermostatted set.
  - NVT COM-velocity removal is off with fixed atoms present: they take
    momentum, so the mobile COM momentum is not conserved, and ndof does
    not count it.
  - ndof loses 3 per frozen atom and regains any SHAKE constraint dropped
    between two frozen atoms.
  - Dropped terms: bonds, bends and torsions among frozen atoms only; also
    out-of-plane terms, which GOMD never evaluates. The driver now refuses
    out-of-plane terms on mobile atoms; it used to ignore them silently.
  - The device neighbor build (`skip=` mask) and the one-time host build
    both leave out frozen–frozen pairs.
  - The PE therefore excludes the constant frozen–frozen terms.
- **Loader:**
  - `conn['deforms']` carries the out-of-plane list.
  - `pkg` reads `params.h` limits (`constants.apply_params_h`; only raised,
    and none is used inside numba code).
  - `pkg` reads Fortran `d` exponents in `mdrun.params` (`_ffloat`).
- **Tests:**
  - `gomd/tests/check_electrodes.py` (CPU): flat ideal graphene to 5e-10 Å,
    layer spacings equal in both layouts, the electrolyte inside the gap
    and moved rigidly (3e-14 Å over 621k pairs), and the dropped-term and
    ndof counts (46632 → 35880).
  - MD on `md-input3`, polarizable, RTX 5070: in every run the frozen atoms
    moved 0 Å with zero velocity.
    - NVE over 200 fs: TE within ±1 kcal/mol at dt 0.5 fs and ±0.4 at
      0.25 fs, with no drift.
    - Slab restart round trip: the continuation tracks the uninterrupted
      run to 0.005 kcal/mol.
    - Bulk self-test still 14/14.

### 4.22 MLIP-only systems and driver (2026-11-17, task 2)
MLIP runs use systems without lone-pair sites: the MLIP knows only real atoms.
They are validated separately from the PFF.

- **`io/make_lp_free.py`**: `python -m gomd.io.make_lp_free --input-dir
  md-input --out-dir md-input2` loads a GOMD system the usual way and drops
  every lone-pair site. It writes `system.xyz` (elements from the
  force-field masses; positions wrapped into the box; velocities and masses
  unchanged; `site` = the index in the source system; charge = the source's
  total, 0 here; spin 1) and a README.
- **`io/xyz.py`**: one-frame extended XYZ. The comment line holds
  `Lattice=`, `Properties=species:S:1:pos:R:3:vel:R:3:mass:R:1:site:I:1`,
  `pbc`, `charge`, `spin`, `time_ps` and, in restarts, `nhc_xlogs/vlogs/glogs`.
  Velocities are in m/s, not ASE's units. Orthorhombic cells only.
- **`mlip_md.py`**: `python -m gomd.mlip_md --input md-input2/system.xyz
  --mlip uma|mock [--ensemble nvt] [--fd-check N]`.
  - The MLIP runs in its own process through `mlip/server.py`, as in §4.17.
  - Velocity Verlet with v += dt/2 · 41.84 F/m and x += 1e-5 dt v. There are
    no constraints, so dt ≤ 0.5 fs and ndof = 3N − 3.
  - COM momentum is removed at the start and after each NVT step, as a
    uniform shift.
  - NVT uses `thermostat.py`'s chain on the CPU (`JAX_PLATFORMS=cpu`; the GPU
    is the MLIP's), in fp64 (`jax_enable_x64`).
    - **Until 2026-11-18 this ran in fp32**, which rounded v and the chain
      state every half step. H_ext then drifted ~ −6e-4 kcal/mol per step,
      independent of dt (2,002 atoms).
    - The chain length defaults to 3 (`--nhc-chain`), not mdrun.params' 1. A
      single thermostat rings at its own period on a flexible cluster:
      sd(T) 24 K with 1 vs 5.9 K with 3, the canonical value being 5.4 K
      (mock, 1 ps).
    - A restart whose chain has another length starts a fresh chain, and
      the log says so.
    - `nhc_half` and `nhc_energy` are `jax.jit`-compiled once. Run eagerly,
      the chain's unrolled scalar ops cost ~57 ms/step on Colab's CPU; they
      now cost ~1 ms/step.
  - Log columns: `step PE KE TE [T H_ext] ms/step mlip_ms`. A closing
    summary gives the drift and sd of TE / H_ext and the mean T.
  - `--fd-check N` checks forces against a 4-point central difference of
    the energy on N atoms, at the start and at the end. The step is `--fd-h`,
    default 0.01 Å. The error is O(h⁴) plus ~ε/h for an energy rough at ε;
    UMA in fp32 has ε ≈ 1e-3 kcal/mol, which a 2-point stencil at 1e-3 Å
    turned into ~0.5 kcal/mol/Å of noise. The log gives both the relative
    and the absolute difference.
  - Before each check, an **energy probe** looks at how finely the model
    resolves E. It evaluates E twice at the same configuration, then moves
    the atom with the largest force by 1e-4, 1e-3 and 1e-2 Å along that
    force and compares ΔE with −|F|·s.
    - This matters because UMA's omol head predicts absolute DFT energies
      (~ −9×10⁶ eV here). Summed in fp32, those move in ~1 eV steps.
    - If the probe shows such steps, finite-difference and NVE-energy checks
      are meaningless at kcal/mol level. Try `--mlip-settings high-precision`.
    - Measured (uma-s-1p2p1, `high-precision`, 2026-11-18): no such steps.
      ΔE follows −|F|·s + ½k s² to ~1e-3 kcal/mol at |E| = 5.7×10⁷ kcal/mol.
  - Writes `<out>_restart.xyz` (`--restart-every`) and, optionally, a
    trajectory (`--traj-every`). Colab cell 11 runs it.
- **Mock validation:** NVE sd(TE) scales as dt² (ratios 4.0 and 16.5);
  drift ≤ 7e-5 kcal/mol/fs; forces = −∇E to 2e-4 (h = 1e-3 Å); NVT mean T
  296.0 K.
  - That NVT run's H_ext drift, −0.004 kcal/mol/fs, was the fp32
    thermostat.
  - In fp64, on the cluster: H_ext drift ≤ 2e-5 kcal/mol/fs, and its sd
    scales as dt² (0.046 / 0.011 kcal/mol at 0.5 / 0.25 fs).
- **Smaller systems** (`io/make_subsystem.py`, 2026-11-18). UMA on all 7,400
  atoms did not fit a 40 GB A100, so the builder cuts a test system out of
  the GOMD system:
  1. Molecules are the connected components of the bond graph over all
     sites, lone pairs included, so each has an integer charge.
  2. Each molecule is made whole across the PBC and placed at the image of
     its COM nearest the center.
  3. Whole molecules are selected by COM: `sphere` takes all within R, grown
     to `--natoms`; `cube` takes those inside a cube of edge a.
  4. The nearest counter-ions are added until the system is neutral.
  5. Only real atoms are written, with velocities, masses and `site`. The
     closest contact between molecules is reported.
- **Sphere vs cube:**
  - `sphere` writes a cluster in vacuum (`pbc="F F F"`), the setting UMA's
    `omol` head was trained for. `mlip_md`, the client and the mock all
    honor `pbc=False`.
  - `cube` writes a periodic box. The molecules that cross its faces wrap
    onto others: 0.089 Å closest contact at 2,000 atoms, so it is not usable
    as built.
- **`md-input2/sphere_2000.xyz`:**
  - R 18.56 Å, 2,002 atoms (H774 Li11 C516 N11 O216 F452 S22);
  - 108 complete molecules: 86 solvent, 11 Li⁺, 11 FSI⁻ (2 counter-ions
    added);
  - charge 0; closest contact between molecules 1.845 Å; 353 K over
    6,003 dof.
  - Mock NVE on it: sd(TE) 0.020 / 0.080 kcal/mol at dt 0.25 / 0.5 fs
    (ratio 4.0); forces = −∇E to 1.5e-4 (2-point, 1e-3 Å), 7e-7 (4-point,
    0.01 Å).
- **UMA on `sphere_2000.xyz`** (Colab L4, uma-s-1p2p1 omol,
  `high-precision`, 2026-11-18):
  - it fits the 22 GB L4, at 583 ms per evaluation (≈ 74 ps/day at 0.5 fs);
  - PE −1236.34 eV/atom; the evaluation is deterministic;
  - energy resolved to ~1e-3 kcal/mol, and forces = −∇E within that;
  - NVE at 0.5 fs: TE sd 0.18 kcal/mol about its fit over 90 fs, with no
    significant drift.
  - **Start-up transient:** the cluster's bonds sit at the PFF's
    constrained lengths with no bond velocity. Under UMA they relax within
    5 fs: PE falls ~370 kcal/mol, T rises 353 → ~400 K, and TE shifts by
    −0.9 kcal/mol. The source velocities are themselves hot: 503.7 K over
    the PFF's dof. Equilibrate under UMA (NVT) before any measurement.
  - **First NVT** (1 ps at 296.15 K, chain 1, still the fp32 thermostat,
    from the NVE restart): heat was removed (TE −1,766 kcal/mol), but T
    rang between 219 and 387 K at the thermostat period (~600 fs). H_ext
    lost 3.9 kcal/mol, mostly while heat was being removed. Both were
    driver issues, fixed as described above; the run's final state is
    still a valid start.
  - **NVT with the fixed driver** (1 ps, chain 3, fp64):
    - H_ext drift +2e-5 kcal/mol/fs, sd 0.089 kcal/mol;
    - T 295.0 K, sd 5.45 K (canonical 5.40 K).
  - **Restarts:** KE carries over exactly. PE carries over to ≤ 8e-4
    kcal/mol, which is UMA's noise from one process to the next.
  - **NVE time-step test** (from the NVT restart at 296 K, 200 fs each):
    - sd(TE) 0.0851 kcal/mol at 0.5 fs and 0.0214 at 0.25 fs, a ratio of
      3.98, which is dt²;
    - drift +2e-5 / −1e-5 kcal/mol/fs.
    - The first NVE's −0.0024 kcal/mol/fs came from its non-equilibrium
      start.
    - §5 has the full UMA scorecard.

### 4.21 `io/restart.py` — restart input directories (2026-11-17)
- `write_restart_dir(input_dir, out_dir, x, v, box, nhc, time_add_ps, klimit)`
  writes `out_dir/coords.inp` and copies `mdrun.params`, `ff.dat`,
  `connectivity.dat` (and `temp.ramp`, `params.h` when present), so the next
  run is just `--input-dir out_dir`.
- `klimit`: the Ewald k-limits the run used. When they differ from the
  copied `mdrun.params` (a slab run scales kmax_z), `set_kmax` writes them
  into it. Read back with the input's kmax_z, the 205 Å slab started
  9.2 kcal/mol off.
- **The new `coords.inp` uses the run's input `coords.inp` as a template:**
  - every site keeps its label and charge type, so the reference parser
    reads it unchanged;
  - positions are written with 12 decimals and velocities (m/s) with 9;
  - Xlogs / Vlogs / Glogs are replaced by the chain state for NVT runs and
    kept otherwise, and the barostat lines are kept;
  - `Time = … picoseconds` is advanced.
- All sites, lone pairs included, are wrapped into [0, L). The parser shifts
  every atom when it meets a negative coordinate.
- **`main_scan`** writes it at the end of a valid run (default
  `<out>_restart`) and every `--restart-every` steps. Writes are atomic.
  After an overflow it is not written.
- **Round trip** (`gomd/tests/check_restart.py`):
  - read back: positions 1.4e-12 Å, velocities 5e-10 m/s, chain state exact;
  - over 100 steps the restarted run tracks the uninterrupted one to
    ΔPE ≤ 0.3 and ΔH_ext ≤ 0.42 kcal/mol, the chaotic divergence plus the
    cutoff noise floor.
- **Limitation:** `coords.inp` cannot hold the SHAKE constraint part of the
  last force. A restarted run's first half kick therefore uses the physical
  force, as any run from `coords.inp` does: ΔKE 0.05 kcal/mol at step 1. The
  neighbor list and the SCF predictor history also start fresh.

### 4.20 `profile_step.py` — per-phase profiler (2026-11-17, not yet run)
`python -m gomd.profile_step --input-dir ./md-input --ref-dir ./pkg
--polarizable --tolpol 1e-8 --out md-output/profile.txt`, or Colab cell 10.
Each phase is jit'd and timed on its own: the median of `--reps` calls, with
`block_until_ready` and the compile excluded. The report has five sections:
1. fp64/fp32 elementwise throughput, and scatter costs (`.at[].add` vs
   `segment_sum`, fp64 vs fp32).
2. Neighbor build time; pairs in rlist vs inside rcut vs capacity.
3. Classical terms, lone pairs, every polarization field part, one T·μ
   product, the SCF (and its products), every induced-force part, SHAKE
   (sweeps), `pre_force` / `post_force`, and the whole `force_fn`.
4. The MD step in `lax.scan` with no neighbor rebuild. At each `--chunk`
   checkpoint it prints ms/step, the largest site drift since the build, and
   the pairs inside rcut the stale list misses. That shows how long a list
   with this skin stays valid.
5. Polarizable only: Jacobi vs PCG at tolpol 1e-4 … 1e-12, from a cold start.
   For each: T·μ products, ms, max and rms dipole error, and E_pol and
   variational-energy error, all against a tightly converged PCG reference.
   It also prints the PCG and Jacobi references against each other, which is
   the PCG validation.

Phase times do not add up exactly to the step time, because XLA fuses across
phase boundaries in the full step.

---

## 5. Validation scorecard (vs the reference, 8,716 sites)

| Component | Energy diff | Force diff |
|-----------|------------|-----------|
| Neighbor list (host and on-device) | exact pair set | — |
| Splined pair (vdW + real elec), fp64 | 1.5e-10 | 1.3e-13 |
| Splined pair, fp32 segment_sum (default) | 1.6e-7 rel | 3.3e-6 rel |
| Bonded — bond | 9e-16 | 4e-12 |
| Bonded — bend | exact | 2e-13 |
| Bonded — torsion | 3e-13 | 3.9e-14 (fixed reference); 5.6e-12 vs −∇E |
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
reference dumps. `python -m gomd.selftest` (§4.18) now rebuilds the key
checks from `pkg/` on the fly.

Ensembles and MLIP (2026-11-16, CPU, fp64):

| Check | Result |
|-------|--------|
| NHC half step vs reference `nhc_integrate` (chain 1 and 3, 4 half steps) | 1.4e-14 rel |
| NVT polarizable, 10 steps vs reference (tolpol 1e-10, SHAKE 1e-12, reference forces) | KE 2.4e-3 kcal/mol; fresh PE at the same positions 7e-3 |
| Converged initial PE (tolpol 1e-10) | GOMD −3630.6030, reference −3630.6017 |
| MLIP server: forces vs finite differences of its energy (mock) | 3e-9 rel |
| MLIP units (stand-in calculator, −1000 eV) | −23060.5478 kcal/mol, exact |
| MLIP propagation, mock elastic network, NVE 60 steps | TE 7601.6–7603.8, no drift |
| MLIP propagation, mock, NVT 30 steps | H_ext within ±1 kcal/mol, no drift |
| PFF scan with MLIP alongside (chunked at 5 steps) | trajectory identical to the unchunked run |
| `--reference-parity` NVE classical, 20 steps vs Nov 9 engine | x, v, f bitwise equal |
| PCG vs Jacobi SCF, tight tolerance (2026-11-17, CPU, `gomd.selftest`) | dipoles 8.4e-13 rel, E_pol 2.4e-10 kcal/mol; 17 vs 32 T·μ products; converged E_pol −7045.0605 |
| PCG warm start from a 1%-biased guess, tolpol 1e-8 | energy +0.0006 kcal/mol after 3 products (Jacobi from the same guess: −0.19 after 4) |

Energy conservation, 0.5 fs, from step 2 (step 1 carries a start-up transient
from input velocities that have components along the constrained bonds):

| Run | TE / H_ext |
|-----|-----------|
| classical NVE, SHAKE 1e-6 (Nov 9) | ±4 kcal/mol alternation every step; −0.05 kcal/mol/fs on even steps |
| classical NVE, SHAKE 1e-12 (default) | ±0.1 fluctuation, −0.002 kcal/mol/fs over 20 fs |
| polarizable NVE, tolpol 1e-10, SHAKE 1e-12, reference forces | +0.13 kcal/mol/fs |
| polarizable NVE, same, with the k-space charge–dipole force (default) | +0.006 kcal/mol/fs |
| polarizable NVT, tolpol 1e-10, default forces | H_ext ±0.03, +0.005 kcal/mol/fs (TE falls 28 kcal/mol: thermostat) |
| classical NVT, default | H_ext ±0.15, −0.013 kcal/mol/fs over 20 fs |
| polarizable NVE, 2,000 steps, Jacobi cold 1e-8 (RTX 5070, 2026-11-17) | TE sd 0.43; −0.0010 kcal/mol/fs (block −0.00003) |
| same, PCG + predict 1e-8 (default since 2026-11-17) | TE sd 0.42; −0.0004 kcal/mol/fs (block +0.0005); 2.34 T·μ/step |
| same, PCG + predict 1e-6 | TE sd 0.42; +0.0016 kcal/mol/fs (block +0.0025): use 1e-8 |


Ensembles (2026-11-17, RTX 5070, from `coords.inp`; `gomd/tests/ensemble_check.py`):

| Check | Result |
|-------|--------|
| NVE polarizable, 1 ps at dt 0.25 / 0.5 / 1.0 fs | drift −0.0009 / −0.0004 / +0.0007 kcal/mol/fs; sd(TE) 0.38 / 0.42 / 0.77, a floor from the hard μ-μ cutoff (§6) |
| NVE classical, same | sd(TE) 0.061 / 0.154 / 0.583, ≈ dt² |
| Velocity along constrained bonds | 0.000% of KE; rms 0.2 / 1.0 / 4.0 m/s (∝ dt²) |
| NVT polarizable, 20 ps after 15 ps equilibration | T 296.14 ± 0.05 K (Tset 296.15); sd(KE) 51.31 ± 2.7 vs canonical 51.29; H_ext +0.00074 kcal/mol/fs |
| NVT classical, same | T 296.14 ± 0.11 K; sd(KE) ratio 1.12 ± 0.08; H_ext −0.00004 kcal/mol/fs |
| Equipartition (COM T of Li⁺, FSI⁻, solvent vs T) | not yet testable: PE still falls 4.5–6.6 kcal/mol/ps and the gap closes over the run (solvent 324 → 296 K) |
| Induced force vs −∇E_pol (finite differences, taper on and off) | 8.5e-4 relative (erfc-table limited); `gomd/tests/check_polforce_fd.py` |

UMA, MLIP-only MD (2026-11-18; `gomd.mlip_md` on `md-input2/sphere_2000.xyz`,
2,002 atoms in vacuum; uma-s-1p2p1 omol, `high-precision`; Colab L4):

| Check | Result |
|-------|--------|
| Memory, speed | fits the 22 GB L4; 587–613 ms per evaluation (≈ 70 ps/day at 0.5 fs) |
| Energy resolution (probe) | ~1e-3 kcal/mol at \|E\| = 5.7×10⁷ kcal/mol; deterministic within a process |
| Forces vs −∇E (4-point, h = 0.01 Å) | max abs diff 0.044 / 0.088 kcal/mol/Å: the energy's noise (2-point at 1e-3 Å: up to 1.2) |
| NVE from 296 K, 200 fs at dt 0.5 / 0.25 fs | sd(TE) 0.0851 / 0.0214 kcal/mol, ratio 3.98 (dt²); drift +2e-5 / −1e-5 kcal/mol/fs |
| NVT 296.15 K, 1 ps, chain 3 | T 295.02 K, sd 5.45 K (canonical 5.40); H_ext +2e-5 kcal/mol/fs |
| Restart continuity (4 hand-offs) | KE exact; PE ≤ 1.2e-3 kcal/mol (UMA's noise between processes) |
| From the PFF state directly | 5 fs transient: bonds relax from the PFF's constrained lengths, T 353 → ~400 K; equilibrate first |

Conservativeness (analytic F vs −∇E from JAX autodiff, fp64, 2026-11-09):
pair 2e-15, bond exact, bend 2e-15, torsion 5.6e-12 (after the fix), Ewald
reciprocal 1e-15, Ewald exclusion correction 9e-6 (A&S erfc in the energy),
lone-pair redistribution vs the chain rule 2e-16.

---

## 6. Known gaps / caveats (honest status)

- **UMA is untested on a GPU.** Everything around fairchem is tested: the
  protocol, units, elements, lone-pair handling, error reporting, and the
  calls against a stand-in with fairchem 2.25's signatures. Unknown:
  - memory and speed for 7,400 atoms (a T4's 15 GB may not be enough);
  - whether fairchem's torch 2.13 (CUDA 13 wheels) runs on the Colab driver
    (the notebook tries CUDA 12 builds first when the driver is older);
  - the science: UMA's `omol` head was trained on non-periodic molecules and
    is used here with periodic boundaries on a condensed-phase electrolyte.
    Compare with `omc`, and treat UMA-vs-PFF differences as model
    differences, not errors.
- **NPT is refused.** The reference NPT is driven by a wrong pressure (pair
  virial set equal to the pair energy; no bonded, SHAKE or lone-pair virial;
  the Ewald virial tensor is never reset), so there is no valid NPT to port.
  GOMD needs its own virial first.
- **Deliberate departures from the reference** (restore with
  `--reference-parity`):
  - SHAKE tolerance 1e-12 instead of 1e-6 (§4.8);
  - the k-space charge–dipole polarization force (§4.14).
- **SCF tolerance is an input.** `mdrun.params` sets 1e-4, which leaves E_pol
  19 kcal/mol unconverged; the driver warns. 1e-8 converges in about 12
  iterations instead of 4.
- **ASPC warm start is unsafe with the Jacobi SCF.** From a biased start the
  step-change test `max|Δμ|² ≤ tolpol` stops at the wrong fixed point (a 1%
  dipole error → E_pol ~24 kcal/mol off; a `min_iter` floor did not fix it).
  Cold start is the default. The reference warm-starts, and its logged PE
  drifts away from a fresh evaluation at its own positions by
  ~0.04 kcal/mol per step even at tolpol 1e-10.
- **SHAKE Jacobi vs reference Gauss-Seidel** — same constraint surface,
  trajectories not bit-identical step to step (2e-3 kcal/mol in KE after 10
  steps at tight tolerances).
- **RESPA** (Large Steps > 1) is not ported; the driver warns and uses dt.
- **Explicit-k Ewald, not FFT-PME** — bit-matches the reference; FFT-PME would
  need full re-validation.
- **Hard μ-μ cutoff in this input.** rtaper 14.5 > rcut 12, so the
  dipole–dipole terms stop abruptly at 12 Å, in the reference too.
  - Each pair crossing the cutoff makes E_pol jump by ~1e-4 kcal/mol.
  - In NVE that gives a dt-independent total-energy noise floor of
    ~0.38 kcal/mol over 1 ps, against ~0.05 classical. There is no
    systematic drift.
  - A taper (rtaper < rcut) removes the floor, but showed a small
    dt-dependent drift. GOMD implements the taper with the true derivative.
    The reference `dtau_func`, (r − ro)(r − rtaper)²/tauden, is not the
    derivative of its `tau_func`.
- **Restart output exists since 2026-11-17** (§4.21), but there is no
  trajectory output yet. The equipartition test is waiting on an
  equilibrated configuration, which can now be built in chained runs.
- **Neighbor capacity.** The on-device list holds 1.25× the first build's
  pairs (`--nb-capacity`). Since 2026-11-17 an overflow stops the run instead
  of dropping pairs silently.
- **Neighbor validity before 2026-11-17.** Lists were rebuilt every 100
  steps (50 fs) with a 1 Å skin and no drift check. Whether pairs were missed
  in those runs is not known yet; profiler section 4 or a
  `--no-drift-rebuild` run answers it.
- **Padded pair work** — fixed 2026-11-17. The pair kernels now run on a
  per-step list compacted to rcut + 0.05 Å (1.11× the interacting pairs,
  down from 1.59×), and padded slots are dropped from the scatters.
- **Use the kernel trace to see where the fused step's time goes.** The
  per-phase profiler times separately jit'd pieces, but XLA fuses and shares
  work across them inside the step. `python -m gomd.tests.trace_step
  --polarizable` traces the real step instead and maps each GPU kernel to
  its `gomd_*` named scope, jit function and source line. It turns off XLA
  command buffers, which otherwise hide the kernels.

  On the RTX 5070 (2026-11-17), shares of kernel time:
  - induced forces 44%, including the `pol_forces` pair fusion (exact fp64
    erfc/exp) at 8.4 ms;
  - SCF 14%; classical 14%; k-space tables 8%; field 8%; geometry 4%.
  - Pair scatter-adds total ~13 ms over ~10 passes per step.

  See `PROJECT_STATUS.md` for the table and the levers.
- **Self-test coverage.** `gomd.selftest` has no term-by-term check of the
  induced forces against the polarization functional; the polarizable energy
  conservation is the indirect check.

---

## 7. Performance — measurements and findings

All current numbers are from the RTX 5070 Laptop GPU (8 GB, local WSL). Its
clocks switch between two states (bulk polarizable: ~22 vs ~34 ms/step), so
compare only runs made back to back, or alternate them.

**Bulk `md-input`** (8,716 sites), polarizable NVE, 2,000 steps, tolpol 1e-8,
neighbor rebuilds on drift (159 in 2,000 steps). Energy conservation is the
same in every row (2026-11-17):

| Stage | ms/step |
|-------|---------|
| Jacobi, cold start | 103.0 |
| PCG + predicted start (2.35 T·μ products/step instead of 10) | 60.5 |
| + two-pass neighbor build (186 → 16 ms) | 46.2 |
| + padded slots dropped in every pair scatter | 32.9 |
| + per-step compaction to pairs inside rcut + 0.05 Å | 27.8 |
| + one shared k-space pass | 27.6 |
| + induced-force erfc/exp from the reference tables | **21.6** |
| classical force field, same run settings | **6.6** |

With PME instead of explicit k-space the bulk step is about even (22.5–24.4
vs 21.7 ms/step); explicit k stays the bulk default.

**Electrode slab `md-input3`** (17,544 sites, 205 Å slab, PME 1 Å / order 6,
NVE, 200 fs, 2026-11-18/19):

| Run | ms/step |
|-----|---------|
| classical, fixed electrode charges | 16.4 |
| classical, constant potential at 2 V | **15.7** |
| polarizable, fixed electrode charges | **44** (57–67 with explicit k 7/7/25) |
| polarizable, constant potential at 2 V, tolpol 1e-9 | **64** (69 before P4) |
| old Fortran code, 40 MPI ranks | ≈225 |

The constant-potential solve itself is free in the classical run (one extra
PME pass, a 3,584² matvec). Coupled to the dipoles it costs about 20 ms/step
(25 before P4 dropped the padded slots of the electrode–electrolyte pair
list): two PME-sized operations per Schur application, about 3.6
applications per step (§4.28).

**History: the earlier engine on the OSC Pitzer V100** (200-step runs, before
the 2026-11-16 defaults; not re-measured since):

| Stage | V100 ms/step |
|-------|--------------|
| Host-loop driver, recompile every neighbor rebuild | 2253 |
| Fixed-capacity padded pairs (no recompile) | 843 → ~810 |
| On-device neighbor build + `lax.scan` + segsum, classical | 9.2 |
| Same, full polarizable (Jacobi SCF) | 53–58 (56 avg) |

Findings:
1. **The 800 ms/step was the host neighbor build**, not the force kernels:
   ~4.5 s per build, every 10 steps. Moving the build on device (and the loop
   into `lax.scan`) gave the ~90× win.
2. **fp32 alone did nothing** (810 → 807) while the scatter was atomic and
   fp64: the pair kernel is gather/scatter-bound. fp32 pays off together with
   the `segment_sum` scatter.
3. **Padding and scatters dominated the GPU step**, not arithmetic. The
   per-phase profile on the RTX 5070
   (`md-output/profile_polarizable_RTX5070_20261117b.txt`) showed the padded
   `(0,0)` slots and the serialized neighbor-build scatter as the largest
   costs; dropping them gave 46.2 → 27.8 ms/step.
4. After those fixes the reciprocal-space passes were the largest group
   (~15 ms per evaluation); XLA already shared most of their work, so one
   shared pass saved only 0.2 ms.

Next levers: nothing large is left without a change of method: fewer SCF
products per step, or a learned SCF (roadmap step 3). For the coupled
constant-potential step the Schur applications are the floor of an exact
solve (§4.28).

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

**Google Colab** (primary): put the folder at `My Drive/GOMD/code/GOMD`, open
`GOMD_colab.ipynb`, select a GPU runtime and run the cells in order:

1. mount Drive and find the project;
2. check the layout and the torsion fix;
3. check the GPU and JAX (in a subprocess);
4. settings;
5. UMA environment (only for `MLIP = uma`);
6. run;
7. summary and run history (`md-output/colab_runs.csv`);
8. plots;
9. self-test.

Logs go straight to `md-output/` on Drive. For UMA, add the Colab secret
`HF_TOKEN` with access to `facebook/UMA`; the model is cached in
`My Drive/GOMD/fairchem_cache`. GPU choice: A100/H100 have fast fp64; T4/L4
do not, and the SCF, Ewald and bonded terms run in fp64.

**Command line:**
```bash
# classical / polarizable, ensemble from mdrun.params (NVT here)
python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
    --steps 200 --print-every 50 --rebuild-every 100 [--polarizable --tolpol 1e-8] \
    --out md-output/md_scan.out

# NVE instead of mdrun.params' ensemble
...  --ensemble nve

# PFF dynamics, UMA evaluated alongside every 50 steps (-> md_scan.compare.csv)
...  --mlip uma --compare-every 50 [--mlip-python /path/to/uma_env/bin/python]

# UMA dynamics, PFF evaluated alongside
...  --mlip uma --propagate mlip --compare-every 50

# reproduce the reference pkg (SHAKE 1e-6, no k-space charge-dipole force)
...  --reference-parity

# self-test (CPU, a few minutes)
python -m gomd.selftest --input-dir ./md-input --ref-dir ./pkg
```
`--cpu` forces the CPU backend (debugging only). SLURM on OSC: `sbatch
run_scan_md.slurm` from `GOMD/` (pick run A or B inside).

Pre-flight check:
```bash
python -c "import gomd; print(gomd.__version__, gomd.__file__)"
ls md-input/ff.dat pkg/input_parser.py
ls pkg/*.py | wc -l                             # 20
```

Expected: ~30–110 s setup, then `Initial classical PE: 3414.451` or
`Initial polarizable PE: -3611.536` (tolpol 1e-4; −3630.603 at 1e-8 or
tighter). The last digit varies by ~5e-4 with the fp32 summation order. Then
one log line per `--print-every` steps.

Environment variable for the reference setup: `GOMD_GPU_PAIR=0` disables the
GPU pair builder used during setup (default on); `GOMD_GPU_PAIR_VERIFY=1`
cross-checks it against the CPU path.

---

## 9. Roadmap position

- Done:
  - classical and fully polarizable APPLE&P MD on the GPU, validated against
    the reference (6.6 / 21.6 ms/step on the RTX 5070 laptop for
    `md-input`);
  - PCG SCF with a predicted start, per-phase GPU profile, PME;
  - NVT;
  - MLIP provider with PFF/MLIP comparison and MLIP-driven dynamics, UMA
    validated on a Colab L4;
  - constant-potential electrodes P0–P4 (fixed electrodes, slab Ewald, PME,
    constant potential coupled to the induced dipoles, voltage ramp, charge
    and Poisson output; §4.23–4.28).
- Next engine work:
  - constant potential: production EDL runs on Colab; the open questions in
    PROJECT_STATUS 1b (the old code's charge-update "rate", the slab
    dispersion correction);
  - a virial, then NPT;
  - UMA `omol` vs `omc`.
- Roadmap:
  - step 2: constant-potential EDL (engine side in progress, above);
  - step 3: ML replaces `solve_scf`;
  - step 4: ML/MM — the hybrid PFF/MLIP region (subtractive ONIOM first) on
    top of the provider interface.

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
- Matching the reference is not proof of correctness: the torsion force matched
  the reference to 1e-10 and both were wrong. Check each force term against
  −∇E from JAX autodiff, and run a timestep-halving test: drift that does not
  shrink with dt is a force/energy mismatch, not integration error.
- COULOMB_CONST must be 332.08 (reference value). Fortran `derfc()` is
  double-precision erfc, not a derivative.
- Validate every component bit-for-bit, or errors hide until they compound.
- Log every step when judging conservation at least once: printing every 10
  steps hid a ±4 kcal/mol period-two ringing caused by a loose SHAKE
  tolerance.
- When the energy includes a term (here the k-space charge field acting on
  the dipoles), the force must include its gradient. The power Σ F·v of the
  missing force predicts the drift (−0.095 vs the observed +0.12 kcal/mol/fs).
- Compare energies at the same positions, from a fresh evaluation, before
  blaming the dynamics: the reference's logged PE was biased by its warm-start
  SCF while its trajectory was fine.
- Keep PyTorch (MLIP) and JAX in separate processes. They pin different
  library versions and each wants the GPU memory.
- **Fixed-capacity padding needs a dropped target.** Padding slots that all
  scatter into one index (atom 0, or a dummy slot) serialize the scatter.
  That cost 11.7× in the neighbor build and 1.9× in every pair kernel.
  Point padding at an out-of-range index with `mode='drop'`.
- **Time the fused step, not only its pieces.** The three k-space passes
  cost 16 ms when timed separately and almost nothing inside the jit'd step,
  where XLA already shares common work. Confirm a gain in a real MD run
  before counting it.
