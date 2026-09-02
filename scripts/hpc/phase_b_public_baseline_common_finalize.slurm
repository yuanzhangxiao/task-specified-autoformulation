#!/bin/bash
#SBATCH --job-name=phase-b-classical-common
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
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"
readonly mechanism_root="${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1/specs"

cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_EVALUATOR_CODE_COMMIT}" ]] || {
  echo "evaluator checkout changed after submission" >&2
  exit 2
}
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${AF_PYTHON}" scripts/merge_phase_b_postfreeze.py \
  --subjects "${AF_OUTPUT_ROOT}/adapted/frozen_evaluation_subjects.jsonl" \
  --input-root "${AF_OUTPUT_ROOT}/common-postfreeze" \
  --output-root "${AF_OUTPUT_ROOT}/common-postfreeze"
"${AF_PYTHON}" scripts/assemble_phase_b_final_evaluation.py \
  --subjects "${AF_OUTPUT_ROOT}/common-postfreeze/postfreeze_subjects.jsonl" \
  --mechanism-config-root "${mechanism_root}" \
  --source-outcomes "${AF_OUTPUT_ROOT}/adapted/source_adapter_outcomes.jsonl" \
  --output-root "${AF_OUTPUT_ROOT}/common-evaluation"
cat "${AF_OUTPUT_ROOT}/common-evaluation/final_evaluation_summary.md"
