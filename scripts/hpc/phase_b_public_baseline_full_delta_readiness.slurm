#!/bin/bash
#SBATCH --job-name=phase-b-baseline-ready
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_SOURCE_CODE_COMMIT:?AF_SOURCE_CODE_COMMIT is required}"

readonly task_plan="${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl"
jq -s -e '
  length == 360 and
  ([group_by(.method)[] | {key: .[0].method, value: length}] | from_entries)
    == {persistence: 120, pysr: 120, sindy: 120}
' "${task_plan}" >/dev/null || {
  echo "the common readiness freeze requires the frozen 40x3x3 task matrix" >&2
  exit 2
}

cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/freeze_phase_b_public_baseline_development_results.py \
  --experiment-root "${AF_OUTPUT_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/common-readiness-freeze" \
  --source-code-commit "${AF_SOURCE_CODE_COMMIT}"
(
  cd "${AF_OUTPUT_ROOT}/common-readiness-freeze"
  sha256sum -c development_result_freeze.json.sha256
)
jq '{status,task_count,methods,common_structural_evaluation_eligible_methods,predictive_reference_methods,test_data_opened,private_reference_opened}' \
  "${AF_OUTPUT_ROOT}/common-readiness-freeze/development_result_freeze.json"
