#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_INCUMBENT_ROOT=${1:-${AF_INCUMBENT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-incumbent-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_INCUMBENT_ROOT/plan.json" ]] || { echo "No plan at $AF_INCUMBENT_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-incumbent-continuation-1"' "$AF_INCUMBENT_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_INCUMBENT_ROOT"
if [[ -f "$AF_INCUMBENT_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_INCUMBENT_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_incumbent_continuation.py" report --root "$AF_INCUMBENT_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,cost_complete,shared_warm_cost_once_per_seed,
rows:[.rows[] | {task_id,status,assessment,trial_count,incumbent_continuation_run,continuation_run,
search_calls_started,search_calls_completed,search_call_accounting_complete,stage_costs,
additional_wall_seconds,additional_cpu_seconds,evaluation_seconds,
before:.evaluations.after_warm,after:.evaluations.selected}]}' "$AF_INCUMBENT_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_INCUMBENT_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_INCUMBENT_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_INCUMBENT_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
