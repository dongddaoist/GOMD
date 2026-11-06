#!/bin/bash
#SBATCH --job-name=gomd
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gpus-per-node=1
#SBATCH --account=TBA
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=TBA
#SBATCH -o gomd.out
#SBATCH -e gomd.err
# =============================================================================
# GOMD (GPU-accelerated Open MD): on-device neighbor build + lax.scan loop +
# segment_sum pair kernel; optional full polarizable force field.
#
# Submit from the project root — the folder that contains:
#   gomd/   pkg/   md-input/   md-output/
#
# gomd.main_scan flags:
#   --input-dir DIR    (default '.')       folder with ff.dat/coords.inp/etc.  -> ./md-input
#   --ref-dir DIR      (REQUIRED)          the reference Python package        -> ./pkg
#   --steps N          (default 200)       number of MD steps
#   --print-every N    (default 50)        log (PE,KE) every N steps; also the scan chunk length
#   --rebuild-every N  (default 100)       neighbor-list rebuild interval (safe at 100 for skin=1.0 A)
#   --polarizable      (flag, default OFF) enable the full induced-dipole force field
#   --cpu              (flag)              force CPU backend (debug only; do NOT use on the V100)
#   --out FILE         (default md_scan.out) log file path
#
# Optional env var (reference setup): GOMD_GPU_PAIR=0 disables the GPU pair
# builder used during the one-time setup (default on).
# =============================================================================

module load miniconda3/24.1.2-py310
eval "$(conda shell.bash hook)"
conda activate newenv2
module load cuda/11.8.0

cd "$SLURM_SUBMIT_DIR"        # project root (has gomd/, pkg/, md-input/)

# ----------------------------------------------------------------------------
# Pick ONE of the two runs below (comment out the other).
# ----------------------------------------------------------------------------

# (A) CLASSICAL force field — matches the reference bit-for-bit, ~9 ms/step on V100.
# python -m gomd.main_scan \
#     --input-dir ./md-input \
#     --ref-dir   ./pkg \
#     --steps 200 \
#     --print-every 50 \
#     --rebuild-every 100 \
#     --out md-output/md_scan_classical.out

# (B) FULL POLARIZABLE force field — induced-dipole SCF each step, ~53-56 ms/step on V100.
python -m gomd.main_scan \
    --input-dir ./md-input \
    --ref-dir   ./pkg \
    --steps 200 \
    --print-every 50 \
    --rebuild-every 100 \
    --polarizable \
    --out md-output/md_scan_polarizable.out
