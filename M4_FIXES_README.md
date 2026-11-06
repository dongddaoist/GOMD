# M4 performance fixes — what changed and why

Your measured 800 ms/step was **not** the force kernels. It was the host
neighbor-list build, which the documentation listed as "~150 ms, not the
bottleneck." Timed directly on your real `coords.inp` (8,716 atoms, 46.96 Å
box), `build_neighbor_pairs` takes **~4.5 s**, runs on the CPU, and with
`--rebuild-every 10` lands inside every 10-step timing window — which *is* the
~800 ms/step you saw. (The box only yields 3 cells/dim at rlist=13 Å, so the
cell list barely prunes; it's near-all-pairs work in a Python loop.)

Three fixes, each validated against the reference on the full system.

## New files

| File | Fix | What it does |
|------|-----|--------------|
| `wmi_md_jax/forces/neighbor_ondevice.py` | **1** | All-pairs neighbor build on device (fp32 distances, per-tile compaction, fixed-capacity padded output). No host build, no device↔host sync. Returns the **bit-identical reference pair set**. |
| `wmi_md_jax/integrator_scan.py` | **2** | Whole trajectory in one `lax.scan`; periodic neighbor rebuild via `lax.cond` *inside* the loop. No Python between steps, one compile. |
| `wmi_md_jax/forces/pair_splined_segsum.py` | **3** | fp32 end-to-end pair kernel with `segment_sum` scatter (replaces the atomic-contention-bound `.at[].add`). |
| `wmi_md_jax/main_scan.py` | — | Driver wiring all three. Runs in chunks of `--print-every` so it can log a (PE,KE) sample between on-device scans. |
| `run_scan_md.slurm` | — | SLURM script for the new driver. |

## Modified files

- `wmi_md_jax/forces/all_forces.py` — added a `use_segsum_pair` branch that
  selects the Fix-3 kernel (mirrors the existing `use_fp32_pair` path). The
  default fp64 and fp32 paths are untouched.

## Validation (full 8,716-atom system)

Bit-for-bit match between the original host-loop driver (`main_m2`) and the new
on-device driver (`main_scan`):

| | reference `main_m2` | new `main_scan` |
|---|---|---|
| step 10 PE | 3414.6104 | 3414.6104 |
| step 10 KE | 7661.5807 | 7661.5807 |
| step 20 PE | 3499.6967 | 3499.6967 |
| step 20 TE | 11222.7591 | 11222.7591 |

Component checks:
- on-device neighbor build pair set ≡ reference pair set (exact, incl. exclusions)
- fp32 segsum pair kernel vs fp64 reference: energy rel 1.6e-7, force rel 3.3e-6
- segment_sum scatter vs `.at[].add` scatter: force diff **0.0** (bit-identical)
- one full VV step, scan vs original vv_step (both fp64): KE/PE identical

## A bug this surfaced

The scan step initially exploded (KE→1e11). Cause was the SHAKE reference
position: the reference `vv_step` constrains against the **post-drift**
position, and the first scan draft used the pre-drift position. Fixed; the
trajectory now matches to all printed digits. (Worth noting in case the same
slip exists anywhere else that reimplements the step.)

## How to run

```bash
cd ~/python           # folder with md-input/ pkg/ md-output/ wmi_md_jax/
conda activate newenv2
module load cuda/11.8.0
python -m wmi_md_jax.main_scan --input-dir ./md-input --ref-dir ./pkg \
    --steps 200 --print-every 50 --rebuild-every 100 --out md-output/md_scan.out
```

Capacity is sized from one host build at setup (the only remaining host build,
run once). If a rebuild ever overflows it, raise the 1.25× factor in
`main_scan.py`.

## Important: re-profile on the V100

The per-step numbers in this sandbox are CPU-only (1 core, 3 GB) and are
**not** indicative — the all-pairs distance work that's a few ms on a V100 is
seconds on one CPU core. On the GPU:

- Expect the neighbor build to drop from ~4.5 s (host) to single-digit ms.
- After that you're force-bound. **Re-profile before touching the reciprocal.**
  The 609 ms reciprocal figure is a CPU-sandbox number; the explicit-k sum is
  transcendental-heavy and V100 SFUs eat that cheaply, so FFT-PME (which forces
  full re-validation) may not be worth it at this N. Confirm with a GPU profile
  first.
- If the pair kernel is still hot, the fp32+segment_sum kernel here is the
  lever; the earlier fp32 attempt showed no gain only because the scatter and
  reciprocal stayed fp64 and masked it.

## Tuning knobs

- `--rebuild-every`: 100 is safe for skin=1.0 Å / max-drift 1.0 Å here. At
  0.5 fs and 296 K atoms move ~0.03 Å per 10 steps, so 10 was ~10× too eager.
  Widen skin to 2.0 Å if you want to rebuild even less often.
- `tile` in `build_neighbor_ondevice` (default 512): raise on the V100 (more
  parallelism, more memory) — e.g. 1024 or 2048. The 3 GB sandbox needs it
  small; the V100 does not.
