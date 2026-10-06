#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_PORTFOLIO_ROOT=${1:-${AF_PORTFOLIO_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-start-portfolio-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_PORTFOLIO_ROOT/plan.json" ]] || { echo "No plan at $AF_PORTFOLIO_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-start-portfolio-1"' "$AF_PORTFOLIO_ROOT/plan.json" >/dev/null || { echo "Wrong campaign: $AF_PORTFOLIO_ROOT" >&2; exit 1; }
printf 'Campaign: %s\n' "$AF_PORTFOLIO_ROOT"
if [[ -f "$AF_PORTFOLIO_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_PORTFOLIO_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_start_portfolio.py" report --root "$AF_PORTFOLIO_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,groups,rows:[.rows[] | {task_id,status,stop_reason,training_prediction_certified,fit_seconds,selected_origin:.selected.origin,training:.evaluation.training_nmse,validation:.evaluation.validation_nmse,max_coefficient_error:.evaluation.coefficient_recovery.maximum_coefficient_relative_error,initial_errors:.evaluation.coefficient_recovery.initial_parameter_absolute_errors,meshes:[.levels[]?.mesh.actual_variables],portfolio:[.portfolio[]? | {source,retired,trials:(.trials|length),training:.best.training_nmse}]}]}' "$AF_PORTFOLIO_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_PORTFOLIO_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_PORTFOLIO_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_PORTFOLIO_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
