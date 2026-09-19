#!/bin/bash
#SBATCH --job-name=ext-baseline-test
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-23%12

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/external-baseline-evaluation-v1}"
: "${AF_MAX_TRAJECTORY_SECONDS:=300}"
: "${AF_SHARD_COUNT:=24}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"
: "${AF_CHAIN_INPUTS_DIGEST:?AF_CHAIN_INPUTS_DIGEST is required}"
readonly shard_index="${SLURM_ARRAY_TASK_ID:?submit as an array}"

cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_EVALUATOR_CODE_COMMIT}" ]] || {
  echo "evaluator checkout changed after submission" >&2
  exit 2
}
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
readonly actual_digest="$(bash scripts/hpc/external_baseline_inputs_digest.sh)"
[[ "${actual_digest}" == "${AF_CHAIN_INPUTS_DIGEST}" ]] || {
  echo "chain inputs changed after submission" >&2
  exit 2
}
jq -e '.execution_authorized == true' \
  "${AF_OUTPUT_ROOT}/frozen/external_baseline_freeze.json" >/dev/null

# Verify every sealed setting in Python against resolved paths. `sha256sum -c`
# would resolve the recorded basename against this working directory instead.
"${AF_PYTHON}" scripts/prepare_external_baseline_frozen_test_evaluation.py \
  --config "${AF_REPO_ROOT}/configs/external_baseline_frozen_test_evaluation_v1.json" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --verify-execution-record \
  --adapted-root "${AF_OUTPUT_ROOT}/adapted" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --evaluator-commit "${AF_EVALUATOR_CODE_COMMIT}" \
  --chain-inputs-digest "${AF_CHAIN_INPUTS_DIGEST}" \
  --maximum-trajectory-wall-time-seconds "${AF_MAX_TRAJECTORY_SECONDS}"

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

"${AF_PYTHON}" scripts/evaluate_phase_b_postfreeze.py \
  --subjects "${AF_OUTPUT_ROOT}/adapted/frozen_evaluation_subjects.jsonl" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/postfreeze" \
  --shard-index "${shard_index}" \
  --shard-count "${AF_SHARD_COUNT}" \
  --maximum-trajectory-wall-time-seconds "${AF_MAX_TRAJECTORY_SECONDS}"
