#!/bin/bash
# Resolve this bundle itself, independent of the calling shell's current directory.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PLATFORM=${AF_PLATFORM:-aces}
case "$AF_PLATFORM" in
  aces)
    AF_DEFAULT_ROOT=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}/phase-c-paired-fit-v1
    ;;
  delta)
    AF_DEFAULT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-paired-fit-v1
    ;;
  *) echo 'AF_PLATFORM must be aces or delta.' >&2; exit 2 ;;
esac
export AF_PAIRED_FIT_ROOT=${AF_PAIRED_FIT_ROOT:-$AF_DEFAULT_ROOT}
AF_WORKER="$AF_REPO_ROOT/scripts/hpc/run_construction_paired_fit.sh"
case "${1:?Use run, status, report, or inspect}" in
  run)
    bash "$AF_WORKER" prepare \
      --inputs "${AF_PAIRED_FIT_INPUTS:-$AF_REPO_ROOT/inputs/paired-inputs.json}" \
      --config "${AF_PAIRED_FIT_CONFIG:-$AF_REPO_ROOT/configs/construction_paired_fit_v1.json}"
    exec bash "$AF_WORKER" submit
    ;;
  status)
    if command -v squeue >/dev/null 2>&1; then
      squeue -u "$USER" -o '%.18i %.40j %.10T %.10M %R' || true
    fi
    exec bash "$AF_WORKER" status
    ;;
  report|inspect) exec bash "$AF_WORKER" "$1" ;;
  *) echo 'Use run, status, report, or inspect.' >&2; exit 2 ;;
esac
