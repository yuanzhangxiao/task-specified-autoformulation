#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_SENSOR_DRIFT_ROOT=${1:-${AF_SENSOR_DRIFT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/sensor-drift-fitting-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_SENSOR_DRIFT_ROOT/plan.json" ]] || { echo "No plan at $AF_SENSOR_DRIFT_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-sensor-drift-fitting-1"' "$AF_SENSOR_DRIFT_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_SENSOR_DRIFT_ROOT"
if [[ -f "$AF_SENSOR_DRIFT_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | select(length > 0) | join(",")' "$AF_SENSOR_DRIFT_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_sensor_drift.py" report --root "$AF_SENSOR_DRIFT_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,evaluation_status_counts,diagnostic_status_counts,groups,paired_objectives,information_by_hypothesis,observation_hypothesis_comparisons,cost_by_hypothesis_loss_and_strategy,cost_complete,cost_scope,fit_wall_seconds_observed,evaluation_wall_seconds_observed,evaluation_cpu_seconds_observed,diagnostic_wall_seconds_observed,diagnostic_cpu_seconds_observed,diagnostic_budget_charge_seconds,diagnostic_calls_observed,diagnostic_completed_calls_observed,diagnostic_calls_total,diagnostic_completed_calls_total,diagnostic_calls_unknown_or_missing,limitation}' "$AF_SENSOR_DRIFT_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_SENSOR_DRIFT_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_SENSOR_DRIFT_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_SENSOR_DRIFT_ROOT" "${AF_MEMBERS[@]}"
sha256sum "$AF_REVIEW"
printf '\nDownload: %s\n' "$AF_REVIEW"
