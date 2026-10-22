#!/bin/bash
# Portable reliability follow-up; inherited AF_OUTPUT_ROOT is deliberately unused.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PLATFORM=${AF_PLATFORM:-delta}
case "$AF_PLATFORM" in
  delta)
    AF_DEFAULT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-reliability-v1
    ;;
  aces)
    AF_DEFAULT_ROOT=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}/phase-c-construction-reliability-v1
    ;;
  *) echo 'AF_PLATFORM must be delta or aces.' >&2; exit 2 ;;
esac
export AF_RELIABILITY_ROOT=${AF_RELIABILITY_ROOT:-$AF_DEFAULT_ROOT}
AF_WORKER="$AF_REPO_ROOT/scripts/hpc/run_construction_reliability.sh"
case "${1:?Use run, status, report, or inspect}" in
  run)
    bash "$AF_WORKER" prepare \
      --inputs "${AF_RELIABILITY_INPUTS:-$AF_REPO_ROOT/inputs/reliability-inputs.json}" \
      --config "${AF_RELIABILITY_CONFIG:-$AF_REPO_ROOT/configs/construction_fit_reliability_v1.json}"
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
