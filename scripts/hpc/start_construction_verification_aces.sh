#!/bin/bash
# Separate CPU-only verification of frozen saved endpoints.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" && -w "$AF_GROUP" ]] || {
  echo 'Run on ACES with AF_GROUP set to writable group scratch.' >&2; exit 2;
}
export AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-$AF_GROUP/phase-c-construction-verification-v1}
AF_WORKER="$AF_REPO_ROOT/scripts/hpc/run_construction_verification_aces.sh"
case "${1:?Use run, status, report, or inspect}" in
  run)
    bash "$AF_WORKER" prepare --source "${AF_VERIFICATION_SOURCE:-$AF_REPO_ROOT/inputs/source}"
    exec bash "$AF_WORKER" submit
    ;;
  status)
    squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %R'
    if [[ -f "$AF_OUTPUT_ROOT/report/summary.json" ]]; then
      cat "$AF_OUTPUT_ROOT/report/summary.json"
    else
      echo 'No report yet; the report runs after all verification tasks finish.'
    fi
    printf 'Logs: %s/logs\n' "$AF_OUTPUT_ROOT"
    ;;
  report|inspect) exec bash "$AF_WORKER" "$1" ;;
  *) echo 'Use run, status, report, or inspect.' >&2; exit 2 ;;
esac
