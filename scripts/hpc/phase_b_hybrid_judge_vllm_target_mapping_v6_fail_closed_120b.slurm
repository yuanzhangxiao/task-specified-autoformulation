#!/bin/bash
#SBATCH --job-name=phase-b-judge-target-map-v6-120b
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-judge-target-map-v6-120b-%j.out
#SBATCH --error=logs/phase-b-judge-target-map-v6-120b-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_SOURCE_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v5-recursive-hard}"
: "${AF_EXPERIMENT_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v6-fail-closed}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v4-clean-names/public}"
: "${AF_PROTOCOL_CONFIG:=${AF_REPO_ROOT}/configs/hybrid_judge_target_mapping_fail_closed_v6.json}"
: "${AF_CALIBRATION_PAIRS:=${AF_EXPERIMENT_ROOT}/pairs.jsonl}"
: "${AF_CALIBRATION_ROOT:=${AF_EXPERIMENT_ROOT}/gpt-oss-120b}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_SHARD_COUNT:=1}"
: "${AF_PAIR_IDS:=}"
: "${AF_TARGET_MAPPING_SEMANTIC_CONTRACT:=true}"
: "${AF_RECURSIVE_TARGET_MAPPING_SEMANTICS:=true}"
: "${AF_TARGET_MAPPING_ENFORCEMENT:=hard}"
: "${AF_TARGET_MAPPING_CONSENSUS:=fail_dominant}"
: "${AF_REPAIR_MISSING_ATOMIC_UNITS:=true}"
: "${AF_COMPARATIVE_INDETERMINATE_POLICY:=neutral_fixed_denominator}"

readonly source_pairs="${AF_SOURCE_ROOT}/pairs.jsonl"
readonly source_labels="${AF_SOURCE_ROOT}/hybrid_labels.jsonl"
[[ -f "${source_pairs}" && -f "${source_labels}" ]] || {
  echo "missing frozen v5 source pairs or labels under ${AF_SOURCE_ROOT}" >&2
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
export AF_TARGET_MAPPING_SEMANTIC_CONTRACT
export AF_RECURSIVE_TARGET_MAPPING_SEMANTICS AF_TARGET_MAPPING_ENFORCEMENT
export AF_TARGET_MAPPING_CONSENSUS AF_REPAIR_MISSING_ATOMIC_UNITS
export AF_COMPARATIVE_INDETERMINATE_POLICY

exec bash "${AF_REPO_ROOT}/scripts/hpc/run_vllm_atomic_judge.sh"
