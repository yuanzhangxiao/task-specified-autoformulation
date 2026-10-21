#!/bin/bash
# Use writable group scratch; no GPU, serving endpoint, or new model generation.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" && -w "$AF_GROUP" ]] || {
  echo 'Run on ACES with AF_GROUP set to writable group scratch.' >&2; exit 2;
}
export AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-$AF_GROUP/phase-c-construction-fitting-v1}
AF_WORKER="$AF_REPO_ROOT/scripts/hpc/run_construction_fit_aces.sh"
case "${1:?Use run, status, report, or inspect}" in
  run)
    bash "$AF_WORKER" prepare \
      --source "${AF_CONSTRUCTION_FIT_SOURCE:-$AF_REPO_ROOT/inputs/source-inspection.tar.gz}" \
      --public-cell "${AF_CONSTRUCTION_FIT_PUBLIC_CELL:-$AF_REPO_ROOT/inputs/public-cell}" \
      --config "${AF_CONSTRUCTION_FIT_CONFIG:-$AF_REPO_ROOT/configs/construction_fit_qualification_v1.json}"
    exec bash "$AF_WORKER" submit
    ;;
  status)
    squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %R'
    if [[ -f "$AF_OUTPUT_ROOT/report/summary.json" ]]; then
      cat "$AF_OUTPUT_ROOT/report/summary.json"
    else
      echo 'No report yet; the dependent report runs after all array tasks finish.'
    fi
    printf 'Logs: %s/logs\n' "$AF_OUTPUT_ROOT"
    ;;
  report) exec bash "$AF_WORKER" report ;;
  inspect)
    bash "$AF_WORKER" inspect
    printf 'Download: %s/inspection.tar.gz\n' "$AF_OUTPUT_ROOT"
    ;;
  *) echo 'Use run, status, report, or inspect.' >&2; exit 2 ;;
esac
