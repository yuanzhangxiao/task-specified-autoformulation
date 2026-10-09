#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_NONLINEAR_COMPARISON_ROOT=${1:-${AF_NONLINEAR_COMPARISON_ROOT:-/work/hdd/bibo/yxiao2/phase_c/nonlinear-conditional-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_NONLINEAR_COMPARISON_ROOT/plan.json" ]] || { echo "No plan at $AF_NONLINEAR_COMPARISON_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-nonlinear-conditional-comparison-1"' "$AF_NONLINEAR_COMPARISON_ROOT/plan.json" >/dev/null || { echo "Wrong campaign: $AF_NONLINEAR_COMPARISON_ROOT" >&2; exit 1; }
printf 'Campaign: %s\n' "$AF_NONLINEAR_COMPARISON_ROOT"
if [[ -f "$AF_NONLINEAR_COMPARISON_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_NONLINEAR_COMPARISON_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_nonlinear_comparison.py" report --root "$AF_NONLINEAR_COMPARISON_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,groups,cost_complete,
rows:[.rows[] | {task_id,status,assessment,routing,stage_costs,portfolio_triggered,trial_count,
additional_wall_seconds,additional_cpu_seconds,cost_complete,evaluation_seconds,
after_warm:(.evaluations.after_warm | {training_nmse,validation_nmse,coefficients_recovered,initials_recovered}),
after:(.evaluations.selected | {training_nmse,validation_nmse,coefficients_recovered,initials_recovered})}]}' "$AF_NONLINEAR_COMPARISON_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json assessment.md results submission_manifest.json submission logs; do
  [[ ! -e "$AF_NONLINEAR_COMPARISON_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_NONLINEAR_COMPARISON_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_NONLINEAR_COMPARISON_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
