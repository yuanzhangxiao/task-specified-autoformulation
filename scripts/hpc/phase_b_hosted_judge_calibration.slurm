#!/bin/bash
#SBATCH --job-name=phase-b-judge-hosted
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --output=logs/phase-b-judge-hosted-%A_%a.out
#SBATCH --error=logs/phase-b-judge-hosted-%A_%a.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_PROJECT}/phase_b/inputs/public}"
: "${AF_CALIBRATION_ROOT:=${AF_WORK}/phase_b/judge-v2-calibration-v2}"
: "${AF_CALIBRATION_PAIRS:=${AF_CALIBRATION_ROOT}/pairs.jsonl}"
: "${AF_HOSTED_OUTPUT_ROOT:=${AF_CALIBRATION_ROOT}/stronger-judge}"
: "${AF_JUDGE_MODEL_SPEC:?set AF_JUDGE_MODEL_SPEC to provider:model}"
: "${AF_REPETITIONS:=1}"
: "${AF_SHARD_COUNT:=5}"

readonly shard_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly shard_root="${AF_HOSTED_OUTPUT_ROOT}/shards/shard_${shard_index}"

[[ -f "${AF_CALIBRATION_PAIRS}" ]] || {
  echo "missing calibration pairs: ${AF_CALIBRATION_PAIRS}" >&2
  exit 2
}
mkdir -p "${shard_root}"
cd "${AF_REPO_ROOT}"

"${AF_PYTHON}" scripts/run_adversarial_judge.py \
  --pairs "${AF_CALIBRATION_PAIRS}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --judge-models "${AF_JUDGE_MODEL_SPEC}" \
  --baseline-label baseline \
  --repetitions "${AF_REPETITIONS}" \
  --output-root "${shard_root}" \
  --timeout-seconds 900 \
  --max-output-tokens 4096 \
  --shard-index "${shard_index}" \
  --shard-count "${AF_SHARD_COUNT}" \
  --shard-strategy contiguous
