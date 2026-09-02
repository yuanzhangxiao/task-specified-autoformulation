#!/bin/bash
#SBATCH --job-name=phase-b-classical-test-sum
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_EVALUATOR_CODE_COMMIT}" ]] || {
  echo "evaluator checkout changed after submission" >&2
  exit 2
}
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${AF_PYTHON}" scripts/summarize_phase_b_public_baseline_predictive_test.py \
  --final-model-root "${AF_OUTPUT_ROOT}/final-model-freeze" \
  --predictive-root "${AF_OUTPUT_ROOT}/predictive-test" \
  --output-root "${AF_OUTPUT_ROOT}/predictive-summary"
cat "${AF_OUTPUT_ROOT}/predictive-summary/predictive_test_summary.md"
