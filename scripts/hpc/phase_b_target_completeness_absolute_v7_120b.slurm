#!/bin/bash
#SBATCH --job-name=phase-b-target-complete-v7-120b
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-target-complete-v7-120b-%j.out
#SBATCH --error=logs/phase-b-target-complete-v7-120b-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_SOURCE_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v6-fail-closed}"
: "${AF_EXPERIMENT_ROOT:=${AF_WORK}/phase_b/target-completeness-absolute-v7}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v4-clean-names/public}"
: "${AF_PROTOCOL_CONFIG:=${AF_REPO_ROOT}/configs/target_completeness_absolute_v7.json}"
: "${AF_CALIBRATION_PAIRS:=${AF_EXPERIMENT_ROOT}/pairs.jsonl}"
: "${AF_CALIBRATION_ROOT:=${AF_EXPERIMENT_ROOT}/gpt-oss-120b}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_SHARD_COUNT:=1}"
: "${AF_JUDGE_ENTRYPOINT:=target_completeness}"
: "${AF_PAIR_IDS:=}"

readonly source_pairs="${AF_SOURCE_ROOT}/pairs.jsonl"
readonly source_labels="${AF_SOURCE_ROOT}/hybrid_labels.jsonl"
[[ -f "${source_pairs}" && -f "${source_labels}" ]] || {
  echo "missing frozen v6 source pairs or labels under ${AF_SOURCE_ROOT}" >&2
  exit 2
}
mkdir -p "${AF_EXPERIMENT_ROOT}"
cp "${source_pairs}" "${AF_CALIBRATION_PAIRS}"
cp "${source_labels}" "${AF_EXPERIMENT_ROOT}/hybrid_labels.jsonl"
cp "${AF_PROTOCOL_CONFIG}" "${AF_EXPERIMENT_ROOT}/protocol_config.json"
cmp --silent "${source_pairs}" "${AF_CALIBRATION_PAIRS}"
cmp --silent "${source_labels}" "${AF_EXPERIMENT_ROOT}/hybrid_labels.jsonl"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_PYTHON AF_PUBLIC_DATA_ROOT
export AF_LOCAL_MODEL AF_TENSOR_PARALLEL_SIZE AF_SHARD_COUNT
export AF_CALIBRATION_ROOT AF_CALIBRATION_PAIRS AF_PAIR_IDS
export AF_JUDGE_ENTRYPOINT

bash "${AF_REPO_ROOT}/scripts/hpc/run_vllm_atomic_judge.sh"

readonly shard_root="${AF_CALIBRATION_ROOT}/shards/shard_0"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/analyze_target_completeness_judge.py" \
  --scores "${shard_root}/target_completeness_scores.csv" \
  --failures "${shard_root}/target_completeness_failures.jsonl" \
  --pairs "${AF_CALIBRATION_PAIRS}" \
  --config "${AF_EXPERIMENT_ROOT}/protocol_config.json" \
  --output-json "${AF_EXPERIMENT_ROOT}/target_completeness_validation.json" \
  --output-md "${AF_EXPERIMENT_ROOT}/target_completeness_validation.md"
