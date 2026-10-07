#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
AF_LARGER_COUPLED_ROOT=${1:-${AF_LARGER_COUPLED_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-larger-coupled-v1}}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT" PYTHONDONTWRITEBYTECODE=1
[[ -f "$AF_LARGER_COUPLED_ROOT/plan.json" ]] || { echo "No plan at $AF_LARGER_COUPLED_ROOT" >&2; exit 1; }
jq -e '.value.protocol == "phase-c-larger-coupled-1"' "$AF_LARGER_COUPLED_ROOT/plan.json" >/dev/null || { echo "Wrong campaign: $AF_LARGER_COUPLED_ROOT" >&2; exit 1; }
printf 'Campaign: %s\n' "$AF_LARGER_COUPLED_ROOT"
if [[ -f "$AF_LARGER_COUPLED_ROOT/submission_manifest.json" ]]; then
  AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_LARGER_COUPLED_ROOT/submission_manifest.json")
  sacct -j "$AF_IDS" --format=JobID,JobName%28,State,ExitCode,Elapsed
fi
"$AF_PYTHON" "$AF_CODE/scripts/phase_c_larger_coupled.py" report --root "$AF_LARGER_COUPLED_ROOT" >/dev/null
jq '{status,expected,recorded,status_counts,groups,
rows:[.rows[] | {task_id,case,seed,status,stop_reason,fit_seconds,
polishing:(.polishing | {polishing_attempted,strict_prediction_certified,actual_residual_calls,retained_stage}),
first_metrics:(.comparison.first_evaluation | {training_nmse,validation_nmse,coefficients_recovered,initials_recovered}),
final_metrics:(.evaluation | {training_nmse,validation_nmse,coefficient_recovery,coefficients_recovered,initials_recovered}),
timing:[.timing_operations[]? | {operation,status,elapsed_seconds,launch_to_entry_seconds,
worker_cpu_seconds:.timing.worker_cpu_seconds,
imports_seconds:.timing.groups.imports.wall_seconds,
maximum_write_seconds:.timing.groups.checkpoint_write.maximum_wall_seconds,
integration_wall_seconds:.timing.groups.trajectory_integration.wall_seconds,
integration_cpu_seconds:.timing.groups.trajectory_integration.cpu_seconds}]}]}' "$AF_LARGER_COUPLED_ROOT/summary.json"
AF_MEMBERS=()
for AF_ITEM in plan.json inputs.json summary.json qualification results submission_manifest.json submission logs; do
  [[ ! -e "$AF_LARGER_COUPLED_ROOT/$AF_ITEM" ]] || AF_MEMBERS+=("$AF_ITEM")
done
AF_REVIEW="$AF_LARGER_COUPLED_ROOT/review-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_LARGER_COUPLED_ROOT" "${AF_MEMBERS[@]}"
printf '\nDownload: %s\n' "$AF_REVIEW"
