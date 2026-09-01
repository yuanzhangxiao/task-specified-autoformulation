#!/bin/bash
#SBATCH --job-name=phase-b-baseline-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PYTHON_MODULE:=Python/3.11.5-GCCcore-13.2.0}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly task_plan="${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl"
readonly task="$(sed -n "$((AF_TASK_INDEX + 1))p" "${task_plan}")"
[[ -n "${task}" ]] || { echo "missing baseline task ${AF_TASK_INDEX}" >&2; exit 2; }
[[ "$(jq -r '.platform' <<<"${task}")" == "aces_cpu" ]] || {
  echo "task ${AF_TASK_INDEX} is not an ACES CPU task" >&2
  exit 2
}
readonly method="$(jq -r '.method' <<<"${task}")"
readonly benchmark_id="$(jq -r '.benchmark_id' <<<"${task}")"
readonly tier="$(jq -r '.tier' <<<"${task}")"
readonly repetition="$(jq -r '.repetition' <<<"${task}")"
readonly timeout="$(jq -r '.wall_timeout_seconds' <<<"${task}")"

module load "${AF_PYTHON_MODULE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runs"
cd "${AF_REPO_ROOT}"

arguments=(
  --data-root "${AF_PUBLIC_DATA_ROOT}"
  --benchmark-id "${benchmark_id}"
  --tier "${tier}"
  --method "${method}"
  --seed "${repetition}"
  --output-root "${AF_OUTPUT_ROOT}/runs"
  --wall-timeout-seconds "${timeout}"
  --development-only
)
if [[ "${method}" == "pysr" ]]; then
  arguments+=(
    --pysr-iterations "$(jq -r '.pysr_iterations' <<<"${task}")"
    --maximum-expression-size "$(jq -r '.maximum_expression_size' <<<"${task}")"
  )
fi
"${AF_PYTHON}" scripts/run_baseline.py "${arguments[@]}"
