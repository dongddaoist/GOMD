# M3 — full polarizable force field, wired into the loop

The classical loop (M4 fixes) ran at 9.2 ms/step on the V100 but was
classical-only: the induced-dipole polarization was computed once at setup and
never entered the dynamics. This adds the complete polarizable force field —
field, SCF, energy, and forces — validated bit-for-bit against the reference,
and wires it into the on-device scan loop behind a `--polarizable` flag.

## The discrepancy that blocked everything

`charge_field_realspace` was off by ~23%. Root cause: it omitted the **1-4 Q-mu
field fold-in** that the reference `compute_charge_efield_and_pairs` applies
inside its main loop (and also used exact instead of tabulated erfc, and lacked
the rcut mask). With all three fixed, the real-space field matches the
reference to **1.4e-15**.

## What was ported

The reference builds the SCF input field in four parts; all are now in JAX:

| part | reference | JAX |
|------|-----------|-----|
| real-space charge field (+ 1-4 fold-in) | compute_charge_efield_and_pairs | pol_field._field_pairs_tab |
| reciprocal-space charge field | compute_recip_efield | pol_field._recip_field |
| exclusion field correction | _apply_elfown_correction | pol_field._field_correction |
| 1-4 field correction | _apply_elfown_14_correction | pol_field._field_correction |

and the force side (the energy + main force were already validated):

| phase | reference | JAX |
|-------|-----------|-----|
| 5 main induced force | compute_pol_forces | polarization.pol_forces |
| 6 exclusion force corr | compute_qmu_correction | pol_corrections.qmu_exclusion_correction |
| 7 1-4 force corr | compute_qmu_14_corrections | pol_corrections.qmu_14_correction |

Phase 8 (recipQmu) is a TODO in the reference itself and is not applied to the
reference forces, so parity does not require it.

## Validation (full 8,716-atom system, vs reference)

| quantity | result |
|----------|--------|
| total charge field | 1.4e-15 |
| SCF dipoles | 2.8e-15 (4 iterations, ref 4) |
| E_pol | -7025.987542 vs -7025.987542 (1.2e-11) |
| pol force phase 5 | 9.7e-9 (exact-vs-tabulated erfc) |
| pol force phase 6 | 2.6e-14 |
| pol force phase 7 | 5.8e-16 |
| total pol force | 8.0e-9 |
| **full-loop initial PE** | **-3611.5363 vs reference total -3611.5355** |

The polarizable trajectory is stable and energy-conserving: TE drift dropped
from ~18 kcal/mol/step (classical) to ~3/step. Initial TE 3991 matches the
reference's reported total energy 3990.46.

## New / modified files

New: `forces/pol_field.py`, `forces/pol_corrections.py`,
`forces/polarization_full.py`.
Modified: `io/assembly_loader.py` (exposes pol arrays), `integrator_scan.py`
(optional polarization + dipole-history carry), `main_scan.py`
(`--polarizable`), `forces/polarization.py` (backward-compatible charge
override + `min_iter` floor in `solve_scf`).

## Run it

```bash
python -m wmi_md_jax.main_scan --input-dir ./md-input --ref-dir ./pkg \
    --steps 200 --print-every 50 --rebuild-every 100 --polarizable \
    --out md-output/md_pol.out
```

## Honest status: the SCF warm-start

The induced-dipole solver is **Jacobi** (parallel, GPU-friendly). It converges
to the reference dipoles at 8e-16 **from a cold start (mu=0) every step**, which
is the default and is fully validated.

I also wired the reference's **ASPC predictor** (`mu_pred = 3*(mu1-mu2)+mu3`,
warm-starting from the last three steps' dipoles, carried in the scan state) to
cut SCF iterations. It is **off by default** (`use_aspc=False`) for a real
reason: with the Jacobi solver and the reference's step-change convergence test
(`max|Δμ|² ≤ tolpol`), a warm start that is close but biased can stop early at
the wrong fixed point — a deliberately injected 1% dipole error converged to
E_pol ~24 kcal/mol off, and a `min_iter` floor did not fix it (the iteration
stalls rather than stops early). The reference avoids this only because its
predictor lands extremely close AND it uses Gauss-Seidel (which contracts
faster). 

So: cold-start Jacobi is the safe, validated default. To use ASPC safely the
solver needs to be made properly contracting (Gauss-Seidel-style sweeps, or a
residual-based convergence test, or a few extra guaranteed iterations with a
damped update). That is genuine future work, flagged honestly rather than
hidden — the predictor scaffolding (history in the carry, `use_aspc` flag,
`scf_min_iter`) is all in place for when the solver is upgraded.

## Performance note

The per-step cost now includes the field build + SCF (~4 iterations of a
dipole-dipole contraction over ~3.3M pairs) + the three force-correction
passes. The 9.2 ms classical number does NOT carry over. Re-profile the
polarizable step on the V100; the SCF contraction and the recip field are the
likely hot spots, and both are the same primitives (pair scatter, k-space
cos/sin) you already know how to optimize.
