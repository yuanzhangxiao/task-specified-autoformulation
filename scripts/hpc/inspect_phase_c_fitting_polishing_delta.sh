#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_POLISHING_ROOT=${1:-${AF_POLISHING_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-polishing-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_POLISHING_ROOT/plan.json" ]] || { echo "No plan at $AF_POLISHING_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-fitting-polishing-1"' "$AF_POLISHING_ROOT/plan.json" >/dev/null || { echo "Wrong campaign: $AF_POLISHING_ROOT" >&2; exit 1; }
printf 'Campaign: %s\n' "$AF_POLISHING_ROOT"
if [[ -f "$AF_POLISHING_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_POLISHING_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_fitting_polishing.py" report --root "$AF_POLISHING_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,groups,polishing_groups,rows:[.rows[] | {task_id,status,stop_reason,fit_seconds,polishing:.polishing | {polishing_attempted,polishing_seconds,strict_prediction_certified,retained_stage,actual_residual_calls,accounting_complete},first:.comparison.first_evaluation | {training_nmse,validation_nmse,coefficient_recovery,coefficients_recovered,initials_recovered},final:.evaluation | {training_nmse,validation_nmse,coefficient_recovery,coefficients_recovered,initials_recovered}}]}' "$AF_POLISHING_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_POLISHING_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_POLISHING_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_POLISHING_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
