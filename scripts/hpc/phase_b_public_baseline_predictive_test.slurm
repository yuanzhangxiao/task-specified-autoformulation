#!/bin/bash
#SBATCH --job-name=phase-b-classical-test
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"
readonly task_index="${SLURM_ARRAY_TASK_ID:?submit as the frozen 0-359 array}"

cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_EVALUATOR_CODE_COMMIT}" ]] || {
  echo "evaluator checkout changed after submission" >&2
  exit 2
}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${AF_PYTHON}" scripts/evaluate_phase_b_public_baseline_predictive_test.py \
  --final-model-root "${AF_OUTPUT_ROOT}/final-model-freeze" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/predictive-test" \
  --task-index "${task_index}"
