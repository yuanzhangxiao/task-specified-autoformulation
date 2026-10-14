#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_MEASUREMENT_ROOT=${1:-${AF_MEASUREMENT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/measurement-fitting-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_MEASUREMENT_ROOT/plan.json" ]] || { echo "No plan at $AF_MEASUREMENT_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-measurement-fitting-1"' "$AF_MEASUREMENT_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_MEASUREMENT_ROOT"
if [[ -f "$AF_MEASUREMENT_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | select(length > 0) | join(",")' "$AF_MEASUREMENT_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_measurement_fitting.py" report --root "$AF_MEASUREMENT_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,evaluation_status_counts,groups,paired_objectives,cost_complete,cost_scope,fit_wall_seconds_observed,evaluation_wall_seconds_observed,evaluation_cpu_seconds_observed,limitation}' "$AF_MEASUREMENT_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_MEASUREMENT_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_MEASUREMENT_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_MEASUREMENT_ROOT" "${AF_MEMBERS[@]}"
sha256sum "$AF_REVIEW"
printf '\nDownload: %s\n' "$AF_REVIEW"
