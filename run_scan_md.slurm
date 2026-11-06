#!/bin/bash
#SBATCH --job-name=wmi_md_scan
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gpus-per-node=1
#SBATCH --account=PYS1137
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=cherrydons@gmail.com
#SBATCH -o md_scan.out
#SBATCH -e md_scan.err
# =============================================================================
# WMI-MD-JAX driver: on-device neighbor build + lax.scan loop + segsum pair.
#
# Run this from the project root — the folder that contains:
#   wmi_md_jax/   pkg/   md-input/   md-output/
#
# main_scan.py flags (exact interface):
#   --input-dir DIR    (default '.')       folder with ff.dat/coords.inp/etc.  -> use ./md-input
#   --ref-dir DIR      (REQUIRED)          the 26-module Python reference      -> ./pkg
#   --steps N          (default 200)       number of MD steps
#   --print-every N    (default 50)        log (PE,KE) every N steps; also the scan chunk length
#   --rebuild-every N  (default 100)       neighbor-list rebuild interval (safe at 100 for skin=1.0 A)
#   --polarizable      (flag, default OFF) enable the full induced-dipole force field (M3)
#   --cpu              (flag)              force CPU backend (debug only; do NOT use on the V100)
#   --out FILE         (default md_scan.out) log file path
# =============================================================================

module load miniconda3/24.1.2-py310
eval "$(conda shell.bash hook)"
conda activate newenv2
module load cuda/11.8.0

cd "$SLURM_SUBMIT_DIR"        # project root (has wmi_md_jax/, pkg/, md-input/)

# ----------------------------------------------------------------------------
# Pick ONE of the two runs below (comment out the other).
# ----------------------------------------------------------------------------

# (A) CLASSICAL force field — matches reference main_m2 bit-for-bit, ~9 ms/step on V100.
#python -m wmi_md_jax.main_scan \
#    --input-dir ./md-input \
#    --ref-dir   ./pkg \
#    --steps 200 \
#    --print-every 50 \
#    --rebuild-every 100 \
#    --out md-output/md_scan_classical.out

# (B) FULL POLARIZABLE force field (M3) — induced-dipole SCF each step.
#     Cold-start Jacobi SCF (validated to 8e-16); slower per step than classical
#     (SCF + field build + correction passes). Re-profile on the V100.
 python -m wmi_md_jax.main_scan \
     --input-dir ./md-input \
     --ref-dir   ./pkg \
     --steps 200 \
     --print-every 50 \
     --rebuild-every 100 \
     --polarizable \
     --out md-output/md_scan_polarizable.out
