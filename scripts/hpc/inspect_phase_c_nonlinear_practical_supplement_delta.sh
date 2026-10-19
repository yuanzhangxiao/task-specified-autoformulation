#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_NONLINEAR_SUPPLEMENT_ROOT=${1:-${AF_NONLINEAR_SUPPLEMENT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-supplement-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_NONLINEAR_SUPPLEMENT_ROOT/plan.json" ]] || { echo "No plan at $AF_NONLINEAR_SUPPLEMENT_ROOT" >&2; exit 1; }
printf 'Supplement: %s\n' "$AF_NONLINEAR_SUPPLEMENT_ROOT"
if [[ -f "$AF_NONLINEAR_SUPPLEMENT_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | select(length > 0) | join(",")' "$AF_NONLINEAR_SUPPLEMENT_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%40,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_nonlinear_practical_supplement.py" report --root "$AF_NONLINEAR_SUPPLEMENT_ROOT" >/dev/null
jq 'del(.rows)' "$AF_NONLINEAR_SUPPLEMENT_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json portable_audit.json source_snapshot results summary.json submission_manifest.json submission logs; do
  [[ ! -e "$AF_NONLINEAR_SUPPLEMENT_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_NONLINEAR_SUPPLEMENT_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_NONLINEAR_SUPPLEMENT_ROOT" "${AF_MEMBERS[@]}"
sha256sum "$AF_REVIEW"
printf '\nDownload: %s\n' "$AF_REVIEW"
