#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_STAGED_FEEDBACK_ROOT=${1:-${AF_STAGED_FEEDBACK_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-staged-feedback-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_STAGED_FEEDBACK_ROOT/plan.json" ]] || { echo "No plan at $AF_STAGED_FEEDBACK_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-staged-feedback-qualification-1"' "$AF_STAGED_FEEDBACK_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_STAGED_FEEDBACK_ROOT"
if [[ -f "$AF_STAGED_FEEDBACK_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_STAGED_FEEDBACK_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_staged_feedback.py" report --root "$AF_STAGED_FEEDBACK_ROOT" >/dev/null
jq '{status,qualification_status,qualification_gates,expected,recorded,
status_counts,diagnosis_counts,control_checks,packet_boundary_checks,output_checks,
accuracy_preservation_checks,
diagnostic_cost_totals,new_fixture_fitting_cost_totals,
historical_fitting_cost_totals,historical_diagnostic_cost_totals,comparisons}' "$AF_STAGED_FEEDBACK_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_STAGED_FEEDBACK_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_STAGED_FEEDBACK_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_STAGED_FEEDBACK_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
