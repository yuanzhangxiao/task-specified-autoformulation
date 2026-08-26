#!/bin/bash
#SBATCH --job-name=phase-b-judge-target-map-v4-120b
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-judge-target-map-v4-120b-%j.out
#SBATCH --error=logs/phase-b-judge-target-map-v4-120b-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_SOURCE_PUBLIC_DATA_ROOT:=${AF_PROJECT}/phase_b/inputs/public}"
: "${AF_EXPERIMENT_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v4-clean-names}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_EXPERIMENT_ROOT}/public}"
: "${AF_V3_PROTOCOL_CONFIG:=${AF_REPO_ROOT}/configs/hybrid_judge_target_mapping_prompt_revision_v3.json}"
: "${AF_PROTOCOL_CONFIG:=${AF_REPO_ROOT}/configs/hybrid_judge_target_mapping_clean_names_v4.json}"
: "${AF_SOURCE_PAIRS:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v2/pairs.jsonl}"
: "${AF_CALIBRATION_PAIRS:=${AF_EXPERIMENT_ROOT}/pairs.jsonl}"
: "${AF_CALIBRATION_ROOT:=${AF_EXPERIMENT_ROOT}/gpt-oss-120b}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_SHARD_COUNT:=1}"
: "${AF_PAIR_IDS:=}"
: "${AF_TARGET_MAPPING_SEMANTIC_CONTRACT:=true}"
: "${AF_COMPARATIVE_INDETERMINATE_POLICY:=neutral_fixed_denominator}"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_PYTHON AF_PUBLIC_DATA_ROOT
export AF_LOCAL_MODEL AF_TENSOR_PARALLEL_SIZE AF_SHARD_COUNT
export AF_CALIBRATION_ROOT AF_CALIBRATION_PAIRS AF_PAIR_IDS
export AF_TARGET_MAPPING_SEMANTIC_CONTRACT
export AF_COMPARATIVE_INDETERMINATE_POLICY

"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/prepare_target_mapping_prompt_revision.py" \
  --source-data-root "${AF_SOURCE_PUBLIC_DATA_ROOT}" \
  --output-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --pairs "${AF_SOURCE_PAIRS}" \
  --protocol-config "${AF_V3_PROTOCOL_CONFIG}" \
  --manifest "${AF_EXPERIMENT_ROOT}/prompt_revision_manifest.json"

"${AF_PYTHON}" \
  "${AF_REPO_ROOT}/scripts/build_hybrid_judge_target_mapping_clean_names.py" \
  --input-pairs "${AF_SOURCE_PAIRS}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --protocol-config "${AF_PROTOCOL_CONFIG}" \
  --output "${AF_CALIBRATION_PAIRS}" \
  --manifest "${AF_EXPERIMENT_ROOT}/target_mapping_clean_names_manifest.json"

"${AF_PYTHON}" \
  "${AF_REPO_ROOT}/scripts/build_hybrid_judge_label_template.py" \
  --pairs "${AF_CALIBRATION_PAIRS}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-mapping-semantic-contract \
  --output "${AF_EXPERIMENT_ROOT}/hybrid_labels.jsonl"

cp "${AF_PROTOCOL_CONFIG}" "${AF_EXPERIMENT_ROOT}/protocol_config.json"

exec bash "${AF_REPO_ROOT}/scripts/hpc/run_vllm_atomic_judge.sh"
