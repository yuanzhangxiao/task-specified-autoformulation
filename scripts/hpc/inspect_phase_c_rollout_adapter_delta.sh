#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_ADAPTER_ROOT=${1:-${AF_ADAPTER_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-rollout-adapter-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_ADAPTER_ROOT/plan.json" ]] || { echo "No plan at $AF_ADAPTER_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-rollout-adapter-1"' "$AF_ADAPTER_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_ADAPTER_ROOT"
if [[ -f "$AF_ADAPTER_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_ADAPTER_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_rollout_adapter.py" report --root "$AF_ADAPTER_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,cost_complete,groups,
rows:[.rows[] | {task_id,status,
training:.fit.verification,
assessment:.fit.assessment,
cost:.fit.cost,
evaluation_status:.evaluation.status,
metrics:.evaluation.metrics,
solver_agreement_passed:.evaluation.solver_agreement_passed,
retry_used:.evaluation.retry_used,
coefficient_error:.evaluation.coefficient_recovery.maximum_coefficient_relative_error,
initial_errors:.evaluation.coefficient_recovery.initial_parameter_absolute_errors}]}' "$AF_ADAPTER_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_ADAPTER_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_ADAPTER_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_ADAPTER_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
