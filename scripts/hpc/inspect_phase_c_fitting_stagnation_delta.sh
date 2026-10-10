#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_STAGNATION_ROOT=${1:-${AF_STAGNATION_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-stagnation-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_STAGNATION_ROOT/plan.json" ]] || { echo "No plan at $AF_STAGNATION_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-fitting-stagnation-1"' "$AF_STAGNATION_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_STAGNATION_ROOT"
if [[ -f "$AF_STAGNATION_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_STAGNATION_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_fitting_stagnation.py" report --root "$AF_STAGNATION_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,cost_complete,
rows:[.rows[] | {task_id,status,audit_status:.derivative_audit.value.status,
audit_cost_seconds:.derivative_audit.wall_seconds,evaluation_seconds,
arms:(.arms // {} | with_entries(.value |= {status,assessment,search_calls_started,search_calls_completed,
additional_wall_seconds,stage_costs,evaluation}))}]}' "$AF_STAGNATION_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_STAGNATION_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_STAGNATION_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_STAGNATION_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
