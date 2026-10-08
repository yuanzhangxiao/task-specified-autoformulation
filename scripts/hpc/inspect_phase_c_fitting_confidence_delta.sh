#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_FITTING_CONFIDENCE_ROOT=${1:-${AF_FITTING_CONFIDENCE_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-confidence-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_FITTING_CONFIDENCE_ROOT/plan.json" ]] || { echo "No plan at $AF_FITTING_CONFIDENCE_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-fitting-confidence-1"' "$AF_FITTING_CONFIDENCE_ROOT/plan.json" >/dev/null || { echo "Wrong campaign: $AF_FITTING_CONFIDENCE_ROOT" >&2; exit 1; }
printf 'Campaign: %s\n' "$AF_FITTING_CONFIDENCE_ROOT"
if [[ -f "$AF_FITTING_CONFIDENCE_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_FITTING_CONFIDENCE_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_fitting_confidence.py" report --root "$AF_FITTING_CONFIDENCE_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,groups,confidence_counts,
rows:[.rows[] | {task_id,status,confidence:.confidence.level,
additional_wall_seconds,additional_cpu_seconds,cost_complete,evaluation_seconds,
before:(.before | {training_nmse,validation_nmse,coefficients_recovered,initials_recovered}),
after:(.evaluations.selected | {training_nmse,validation_nmse,coefficients_recovered,initials_recovered})}]}' "$AF_FITTING_CONFIDENCE_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_FITTING_CONFIDENCE_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_FITTING_CONFIDENCE_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_FITTING_CONFIDENCE_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
