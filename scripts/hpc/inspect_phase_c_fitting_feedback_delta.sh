#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_FEEDBACK_ROOT=${1:-${AF_FEEDBACK_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-feedback-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_FEEDBACK_ROOT/plan.json" ]] || { echo "No plan at $AF_FEEDBACK_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-fitting-feedback-qualification-1"' "$AF_FEEDBACK_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_FEEDBACK_ROOT"
if [[ -f "$AF_FEEDBACK_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_FEEDBACK_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_fitting_feedback.py" report --root "$AF_FEEDBACK_ROOT" >/dev/null
jq '{status,qualification_status,qualification_gates,expected,recorded,
status_counts,diagnosis_counts,fixture_check_status,fixture_checks,
packet_boundary_check_status,packet_boundary_checks,diagnostic_check_status,
diagnostic_checks,diagnostic_cost_totals,
fixture_fitting_cost_totals,rows:[.rows[] | {
task_id,kind,status,source_status,fitting_performed,
diagnosis:.feedback.diagnosis,action:.feedback.recommended_action,
diagnostic_cost:.feedback.cost,original_fit_cost,evaluator_labels
}]}' "$AF_FEEDBACK_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_FEEDBACK_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_FEEDBACK_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_FEEDBACK_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
