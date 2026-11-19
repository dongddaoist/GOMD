#!/bin/bash
# tools/wsl_run.sh — run GOMD locally in WSL (env "gomd": Python 3.12 + jax[cuda12]).
#
# Run from Git Bash on Windows, in the project root:
#   tools/wsl_run.sh -m gomd.selftest --input-dir ./md-input --ref-dir ./pkg
#   tools/wsl_run.sh -m gomd.profile_step --input-dir ./md-input --ref-dir ./pkg \
#       --polarizable --tolpol 1e-8 --out md-output/profile_rtx5070.txt
#   tools/wsl_run.sh -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg --steps 200 ...
#
# Google Drive (G:) is not mounted inside WSL, so this script:
#   1. mirrors gomd/, pkg/, md-input/ (and md-input2/, md-input3/) into ~/gomd_run in WSL. numba's cache
#      in pkg/__pycache__ there is kept, so later runs set up faster, and
#      nothing is written into the Drive copy;
#   2. runs the env's python with your arguments, from ~/gomd_run;
#   3. copies new files in ~/gomd_run/md-output back to md-output/ here.
# Arguments must not contain spaces (wsl.exe re-joins them).
# GOMD_WSL_DISTRO overrides the distro (default Ubuntu); --cpu runs on the CPU.
set -u
PROJ=$(cygpath -w "$(cd "$(dirname "$0")/.." && pwd)")
DISTRO=${GOMD_WSL_DISTRO:-Ubuntu}
WUSER=$(MSYS_NO_PATHCONV=1 wsl.exe -d "$DISTRO" --cd "~" whoami | tr -d '\r\000')
RUN=/home/$WUSER/gomd_run
UNC="\\\\wsl.localhost\\$DISTRO\\home\\$WUSER\\gomd_run"
PY=/home/$WUSER/micromamba/envs/gomd/bin/python

for d in gomd pkg md-input md-input2 md-input3; do
  [ -d "$d" ] || continue
  MSYS_NO_PATHCONV=1 robocopy "$PROJ\\$d" "$UNC\\$d" /MIR /XD __pycache__ .ipynb_checkpoints \
      /NFL /NDL /NJH /NJS /NP > /dev/null
  if [ $? -ge 8 ]; then echo "wsl_run: copying $d into WSL failed" >&2; exit 1; fi
done
MSYS_NO_PATHCONV=1 wsl.exe -d "$DISTRO" --cd "$RUN" mkdir -p md-output

# The laptop GPU is shared with Windows (~1 GB in use) and WSL cannot get the
# 75% block XLA preallocates by default, so allocate on demand.
MSYS_NO_PATHCONV=1 wsl.exe -d "$DISTRO" --cd "$RUN" env PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" "$@"
rc=$?

MSYS_NO_PATHCONV=1 robocopy "$UNC\\md-output" "$PROJ\\md-output" /E /XO \
    /NFL /NDL /NJH /NJS /NP > /dev/null
exit $rc
