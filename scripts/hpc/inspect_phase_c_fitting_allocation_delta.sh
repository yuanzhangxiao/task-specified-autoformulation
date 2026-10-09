#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_FITTING_ALLOCATION_ROOT=${1:-${AF_FITTING_ALLOCATION_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-allocation-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_FITTING_ALLOCATION_ROOT/plan.json" ]] || { echo "No plan at $AF_FITTING_ALLOCATION_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-nonlinear-allocation-1"' "$AF_FITTING_ALLOCATION_ROOT/plan.json" >/dev/null || { echo "Wrong campaign: $AF_FITTING_ALLOCATION_ROOT" >&2; exit 1; }
printf 'Campaign: %s\n' "$AF_FITTING_ALLOCATION_ROOT"
if [[ -f "$AF_FITTING_ALLOCATION_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_FITTING_ALLOCATION_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_nonlinear_comparison.py" report --root "$AF_FITTING_ALLOCATION_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,groups,cost_complete,
diagnostics:(.diagnostics | {status,correctness_passed,wall_seconds,cpu_seconds,cost_complete,
rows:[.rows[]? | {task,status,error,
exact:(.value.exact_derivative | {rank,columns,full_rank,maximum_relative_error,correctness_passed}),
cases:[.value.cases[]? | {case,maximum_relative_error,components_before,components_after,audit}]}]}),
rows:[.rows[] | {task_id,status,assessment,routing,stage_costs,portfolio_triggered,trial_count,
additional_wall_seconds,additional_cpu_seconds,cost_complete,evaluation_seconds,
after_warm:(.evaluations.after_warm | {training_nmse,validation_nmse,coefficients_recovered,initials_recovered}),
after:(.evaluations.selected | {training_nmse,validation_nmse,coefficients_recovered,initials_recovered})}]}' "$AF_FITTING_ALLOCATION_ROOT/summary.json"
shopt -s nullglob
AF_BACKENDS=("$AF_FITTING_ALLOCATION_ROOT"/results/*/backend.json)
if ((${#AF_BACKENDS[@]})); then
  jq -s '[.[] | .value | {method:.identity.method,problem:.identity.problem_sha256,
precision_decisions,
conditional:[.operations[] | select(.operation | endswith("/conditional")) |
{operation,status,budget_charge_seconds,allocation:.value.allocation,
screened:(.value.candidates | length),calls:.value.calls,node_seconds:.value.node_seconds,
screening_seconds:.value.screening_seconds,
screen_failures:[.value.levels[]? | select(has("rollout_point"))],
screen_skips:[.value.levels[]? | select(has("screen_skip"))]}]}]' "${AF_BACKENDS[@]}"
fi
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json assessment.md results diagnostic-inputs.json diagnostics submission_manifest.json submission logs; do
  [[ ! -e "$AF_FITTING_ALLOCATION_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_FITTING_ALLOCATION_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_FITTING_ALLOCATION_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
