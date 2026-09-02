#!/bin/bash
#SBATCH --job-name=phase-b-classical-freeze
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_DEVELOPMENT_ROOT:?AF_DEVELOPMENT_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"

cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_EVALUATOR_CODE_COMMIT}" ]] || {
  echo "evaluator checkout changed after submission" >&2
  exit 2
}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${AF_PYTHON}" scripts/finalize_phase_b_public_baseline_models.py \
  --development-freeze-root "${AF_DEVELOPMENT_ROOT}/common-readiness-freeze" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/final-model-freeze"
"${AF_PYTHON}" scripts/export_phase_b_frozen_subjects.py \
  --requests "${AF_OUTPUT_ROOT}/final-model-freeze/source_adapter_requests.jsonl" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/adapted"
jq -e '
  .status == "frozen_before_test_or_private_evaluation" and
  .model_count == 360 and .symbolic_subject_request_count == 240 and
  .method_counts == {persistence: 120, sindy: 120, pysr: 120} and
  .test_data_opened == false and .private_reference_opened == false
' "${AF_OUTPUT_ROOT}/final-model-freeze/final_model_freeze.json" >/dev/null
jq -e '
  .status == "complete" and .request_count == 240 and
  .adapted_count == 240 and .failed_count == 0 and
  .test_data_opened == false
' "${AF_OUTPUT_ROOT}/adapted/source_adapter_manifest.json" >/dev/null
jq '{status,model_count,symbolic_subject_request_count,method_counts,test_data_opened,private_reference_opened}' \
  "${AF_OUTPUT_ROOT}/final-model-freeze/final_model_freeze.json"
jq '{status,request_count,adapted_count,failed_count,test_data_opened}' \
  "${AF_OUTPUT_ROOT}/adapted/source_adapter_manifest.json"
