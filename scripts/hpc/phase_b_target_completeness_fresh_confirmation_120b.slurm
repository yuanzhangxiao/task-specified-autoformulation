#!/bin/bash
#SBATCH --job-name=phase-b-target-confirm-120b
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-target-confirm-120b-%j.out
#SBATCH --error=logs/phase-b-target-confirm-120b-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_DEVELOPMENT_ROOT:=${AF_WORK}/phase_b/target-completeness-absolute-v7}"
: "${AF_EXPERIMENT_ROOT:=${AF_WORK}/phase_b/target-completeness-fresh-confirmation-v1}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v4-clean-names/public}"
: "${AF_PROTOCOL_CONFIG:=${AF_EXPERIMENT_ROOT}/protocol_config.json}"
: "${AF_CALIBRATION_PAIRS:=${AF_EXPERIMENT_ROOT}/pairs.jsonl}"
: "${AF_CONFIRMATION_MANIFEST:=${AF_EXPERIMENT_ROOT}/target_completeness_confirmation_manifest.json}"
: "${AF_DEVELOPMENT_ANALYSIS:=${AF_DEVELOPMENT_ROOT}/target_completeness_validation.json}"
: "${AF_CALIBRATION_ROOT:=${AF_EXPERIMENT_ROOT}/gpt-oss-120b}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_SHARD_COUNT:=1}"
: "${AF_JUDGE_ENTRYPOINT:=target_completeness}"
: "${AF_PAIR_IDS:=}"

for af_required in \
  "${AF_PROTOCOL_CONFIG}" \
  "${AF_CALIBRATION_PAIRS}" \
  "${AF_CONFIRMATION_MANIFEST}" \
  "${AF_DEVELOPMENT_ANALYSIS}"; do
  [[ -f "${af_required}" ]] || {
    echo "missing frozen confirmation input: ${af_required}" >&2
    exit 2
  }
done

"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/verify_target_completeness_confirmation.py" \
  --pairs "${AF_CALIBRATION_PAIRS}" \
  --manifest "${AF_CONFIRMATION_MANIFEST}" \
  --config "${AF_PROTOCOL_CONFIG}" \
  --development-analysis "${AF_DEVELOPMENT_ANALYSIS}"

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
  --config "${AF_PROTOCOL_CONFIG}" \
  --output-json "${AF_EXPERIMENT_ROOT}/target_completeness_confirmation.json" \
  --output-md "${AF_EXPERIMENT_ROOT}/target_completeness_confirmation.md"

"${AF_PYTHON}" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("passed") is True else 1)' \
  "${AF_EXPERIMENT_ROOT}/target_completeness_confirmation.json"
