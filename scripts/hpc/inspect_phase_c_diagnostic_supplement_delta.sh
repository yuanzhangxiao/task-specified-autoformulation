#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_SUPPLEMENT_ROOT=${1:-${AF_SUPPLEMENT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-diagnostic-supplement-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -f "$AF_SUPPLEMENT_ROOT/plan.json" ]] || { echo "No plan at $AF_SUPPLEMENT_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-diagnostic-supplement-1"' "$AF_SUPPLEMENT_ROOT/plan.json" >/dev/null
if [[ -f "$AF_SUPPLEMENT_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_SUPPLEMENT_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_diagnostic_supplement.py" report --root "$AF_SUPPLEMENT_ROOT" >/dev/null
jq '{status,qualification_status,qualification_gates,portable_audit:{status:.portable_audit.status,expected:.portable_audit.expected,verified:.portable_audit.verified},target,original_diagnosis,diagnosis,piecewise_status,required_calls,new_calls_started,new_calls_completed,remaining_keys,original_cost,new_cost,combined_cost}' "$AF_SUPPLEMENT_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json portable_audit.json summary.json source_snapshot results submission_manifest.json submission logs; do
  [[ ! -e "$AF_SUPPLEMENT_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_SUPPLEMENT_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_SUPPLEMENT_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
