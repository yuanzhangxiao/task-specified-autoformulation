#!/bin/bash
#SBATCH --job-name=ext-baseline-merge
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/external-baseline-evaluation-v1}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"
: "${AF_CHAIN_INPUTS_DIGEST:?AF_CHAIN_INPUTS_DIGEST is required}"

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
"${AF_PYTHON}" scripts/merge_phase_b_postfreeze.py \
  --subjects "${AF_OUTPUT_ROOT}/adapted/frozen_evaluation_subjects.jsonl" \
  --input-root "${AF_OUTPUT_ROOT}/postfreeze" \
  --output-root "${AF_OUTPUT_ROOT}/postfreeze"
