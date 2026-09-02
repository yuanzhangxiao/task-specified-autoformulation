#!/bin/bash
#SBATCH --job-name=phase-b-baseline-cpu
#SBATCH --account=bibo-delta-cpu
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
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_JULIA_DEPOT:?AF_JULIA_DEPOT is required}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly task_plan="${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl"
readonly task="$(sed -n "$((AF_TASK_INDEX + 1))p" "${task_plan}")"
[[ -n "${task}" ]] || { echo "missing baseline task ${AF_TASK_INDEX}" >&2; exit 2; }
[[ "$(jq -r '.platform' <<<"${task}")" == "delta_cpu" ]] || {
  echo "task ${AF_TASK_INDEX} is not a Delta CPU task" >&2
  exit 2
}
readonly method="$(jq -r '.method' <<<"${task}")"
[[ "${method}" == "persistence" || "${method}" == "sindy" || "${method}" == "pysr" ]] || {
  echo "unsupported Delta CPU method: ${method}" >&2
  exit 2
}
readonly benchmark_id="$(jq -r '.benchmark_id' <<<"${task}")"
readonly tier="$(jq -r '.tier' <<<"${task}")"
readonly repetition="$(jq -r '.repetition' <<<"${task}")"
readonly timeout="$(jq -r '.wall_timeout_seconds' <<<"${task}")"

export JULIA_DEPOT_PATH="${AF_JULIA_DEPOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runs" \
  "${JULIA_DEPOT_PATH}"
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
if [[ "${method}" == "sindy" ]]; then
  mapfile -t sindy_thresholds < <(
    jq -r '.sindy_thresholds[]' <<<"${task}"
  )
  [[ "${#sindy_thresholds[@]}" -gt 0 ]] || {
    echo "SINDy task has no frozen threshold grid" >&2
    exit 2
  }
  arguments+=(--sindy-thresholds "${sindy_thresholds[@]}")
elif [[ "${method}" == "pysr" ]]; then
  (
    cd "${AF_OUTPUT_ROOT}/runtime"
    sha256sum -c pysr_runtime.json.sha256
  )
  arguments+=(
    --pysr-iterations "$(jq -r '.pysr_iterations' <<<"${task}")"
    --maximum-expression-size "$(
      jq -r '.maximum_expression_size' <<<"${task}"
    )"
  )
fi
"${AF_PYTHON}" scripts/run_baseline.py "${arguments[@]}"
