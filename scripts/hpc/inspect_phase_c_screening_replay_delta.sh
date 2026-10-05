#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_CAMPAIGN=${AF_CAMPAIGN:-/work/hdd/bibo/yxiao2/phase_c/fitting-screening-replay-v1}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_CAMPAIGN/plan.json" ]] || { echo "No plan at $AF_CAMPAIGN" >&2; exit 1; }
if [[ -f "$AF_CAMPAIGN/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_CAMPAIGN/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_screening_replay.py" report --root "$AF_CAMPAIGN" > /dev/null
jq '{status,expected,recorded,status_counts,calibration,rows: [.rows[] | {task_id,status,training_nmse,validation_nmse,selected_source,released_final_training_nmse,stop_reason}]}' "$AF_CAMPAIGN/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json calibration results submission_manifest.json submission logs; do
  [[ ! -e "$AF_CAMPAIGN/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_CAMPAIGN/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_CAMPAIGN" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
