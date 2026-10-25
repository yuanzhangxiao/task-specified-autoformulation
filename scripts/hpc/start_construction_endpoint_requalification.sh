#!/bin/bash
# Saved-endpoint requalification; inherited AF_OUTPUT_ROOT is deliberately unused.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PLATFORM=${AF_PLATFORM:-delta}
case "$AF_PLATFORM" in
  delta)
    AF_DEFAULT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-endpoint-requalification-v1
    ;;
  aces)
    AF_DEFAULT_ROOT=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}/phase-c-construction-endpoint-requalification-v1
    ;;
  *) echo 'AF_PLATFORM must be delta or aces.' >&2; exit 2 ;;
esac
export AF_ENDPOINT_REQUALIFICATION_ROOT=${AF_ENDPOINT_REQUALIFICATION_ROOT:-$AF_DEFAULT_ROOT}
AF_WORKER="$AF_REPO_ROOT/scripts/hpc/run_construction_endpoint_requalification.sh"
case "${1:?Use run, status, report, or inspect}" in
  run)
    bash "$AF_WORKER" prepare \
      --inputs "${AF_ENDPOINT_REQUALIFICATION_INPUTS:-$AF_REPO_ROOT/inputs/endpoint-requalification-inputs.json}" \
      --config "${AF_ENDPOINT_REQUALIFICATION_CONFIG:-$AF_REPO_ROOT/configs/construction_endpoint_requalification_v1.json}"
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
