#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_INTEGRATION_ROOT=${1:-${AF_INTEGRATION_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-public-integration-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_INTEGRATION_ROOT/plan.json" ]] || { echo "No plan at $AF_INTEGRATION_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-public-fitting-integration-1"' "$AF_INTEGRATION_ROOT/plan.json" >/dev/null
printf 'Campaign: %s\n' "$AF_INTEGRATION_ROOT"
if [[ -f "$AF_INTEGRATION_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_INTEGRATION_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_public_integration.py" report --root "$AF_INTEGRATION_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,costs,
rows:[.rows[] | {task_id,kind,status,source_status,
setup_status:.fit.preparation.status,
setup_error:(.fit.preparation.value.error // .fit.preparation.error),
routing:.fit.fit.routing.selected,training:.fit.fit.verification,
assessment:.fit.fit.assessment,cost:.fit.fit.cost,
evaluation_status:.evaluation.status,metrics:.evaluation.metrics,
per_target_nmse:.evaluation.per_target_nmse,
solver_agreement_passed:.evaluation.solver_agreement_passed,
coefficient_error:.evaluation.coefficient_recovery.maximum_coefficient_relative_error}]}' "$AF_INTEGRATION_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json results submission_manifest.json submission logs; do
  [[ ! -e "$AF_INTEGRATION_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_INTEGRATION_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_INTEGRATION_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
