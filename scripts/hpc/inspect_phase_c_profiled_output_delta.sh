#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_PROFILED_ROOT=${1:-${AF_PROFILED_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-profiled-output-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_PROFILED_ROOT/plan.json" ]] || { echo "No plan at $AF_PROFILED_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-profiled-output-1"' "$AF_PROFILED_ROOT/plan.json" >/dev/null || { echo "Wrong campaign: $AF_PROFILED_ROOT" >&2; exit 1; }
printf 'Campaign: %s\n' "$AF_PROFILED_ROOT"
if [[ -f "$AF_PROFILED_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_PROFILED_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_profiled_output.py" report --root "$AF_PROFILED_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,groups,rows:[.rows[] | {task_id,status,stop_reason,training_prediction_certified,fit_seconds,selected_origin:.selected.origin,training:.evaluation.training_nmse,validation:.evaluation.validation_nmse,max_coefficient_error:.evaluation.coefficient_recovery.maximum_coefficient_relative_error,initial_errors:.evaluation.coefficient_recovery.initial_parameter_absolute_errors,optimizer_stop:.optimizer.stop_reason,outer_calls:.optimizer.actual_residual_calls,profiled_block:.optimizer.profiled_output,inner:.optimizer.selected_inner}]}' "$AF_PROFILED_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_PROFILED_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_PROFILED_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_PROFILED_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
