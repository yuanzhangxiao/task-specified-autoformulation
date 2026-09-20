#!/bin/bash
# Submit the Phase-B D3 campaign against a local gpt-oss-120b vLLM endpoint.
#
#   AF_TIER        easy | hard | all   (default easy — run easy first)
#   AF_BATCH_SIZE  tasks per GPU job   (default 10)
#   AF_D3_MODEL    served model id     (default openai/gpt-oss-120b)
#   AF_TASK_INDICES  explicit comma list, overrides AF_TIER (smoke: "0")
#   AF_BATCH_HOURS   walltime per batch (default 08:00:00)
#
# One GPU job per batch loads the model once and runs that batch's tasks.
# Smaller, shorter GPU requests queue faster than one long multi-GPU hold.

set -euo pipefail
readonly af_user="${USER:?}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
# Deliberately not AF_OUTPUT_ROOT: that name is used by other campaigns and
# a stale export would silently redirect this one into their directory.
: "${AF_D3_OUTPUT_ROOT:=${AF_WORK}/phase_b/d3-native-vllm-120b-v1}"
: "${AF_D3_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_d3_full_v1.json}"
: "${AF_D3_MODEL:=openai/gpt-oss-120b}"
: "${AF_TIER:=easy}"
: "${AF_BATCH_SIZE:=10}"
: "${AF_BATCH_HOURS:=08:00:00}"
# Cluster presets. `accounts` on the login node prints the project names,
# and the account must match the job type: Delta separates cpu from gpu.
: "${AF_CLUSTER:=delta}"
case "${AF_CLUSTER}" in
  delta)
    # No H100 on Delta; A40 is Ampere, so the MXFP4 weights are
    # dequantized and need four cards. Four GPU-hours per wall-hour.
    : "${AF_ACCOUNT:=bibo-delta-gpu}"
    : "${AF_GPU_PARTITION:=gpuA40x4}"
    : "${AF_GPU_REQUEST:=--gpus-per-node=4}"
    : "${AF_TENSOR_PARALLEL_SIZE:=4}"
    ;;
  aces)
    # H100 is Hopper: native MXFP4, one card, one GPU-hour per wall-hour.
    : "${AF_ACCOUNT:?set AF_ACCOUNT to the ACES GPU account from `accounts`}"
    : "${AF_GPU_PARTITION:=gpu}"
    : "${AF_GPU_REQUEST:=--gres=gpu:h100:1}"
    : "${AF_TENSOR_PARALLEL_SIZE:=1}"
    ;;
  *) echo "unknown AF_CLUSTER: ${AF_CLUSTER}" >&2; exit 2 ;;
esac
cd "${AF_REPO_ROOT}"
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs "${AF_D3_OUTPUT_ROOT}/logs"

indices="${AF_TASK_INDICES:-$("${AF_PYTHON}" scripts/list_phase_b_d3_task_indices.py \
  --config "${AF_D3_CONFIG}" --tier "${AF_TIER}")}"
[[ -n "${indices}" ]] || { echo "no tasks selected for tier ${AF_TIER}" >&2; exit 2; }
readonly task_count="$(tr ',' '\n' <<< "${indices}" | wc -l | tr -d ' ')"
readonly batches=$(( (task_count + AF_BATCH_SIZE - 1) / AF_BATCH_SIZE ))
echo "tier=${AF_TIER} tasks=${task_count} batch_size=${AF_BATCH_SIZE} batches=${batches}"
echo "output_root=${AF_D3_OUTPUT_ROOT}"
echo "cluster=${AF_CLUSTER} account=${AF_ACCOUNT} partition=${AF_GPU_PARTITION}"
echo "gpu=${AF_GPU_REQUEST} tensor_parallel=${AF_TENSOR_PARALLEL_SIZE}"

readonly common="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_D3_OUTPUT_ROOT=${AF_D3_OUTPUT_ROOT},AF_LOCAL_MODEL=${AF_D3_MODEL},AF_TASK_INDICES=${indices},AF_BATCH_SIZE=${AF_BATCH_SIZE},AF_D3_CONFIG=${AF_D3_CONFIG},AF_PUBLIC_ROOT=${AF_PUBLIC_ROOT},AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE}"

batch_job="$(sbatch --parsable --account="${AF_ACCOUNT}" \
  --partition="${AF_GPU_PARTITION}" --time="${AF_BATCH_HOURS}" \
  ${AF_GPU_REQUEST} \
  --array="0-$((batches - 1))%${AF_MAX_CONCURRENT_BATCHES:-2}" \
  --output="${AF_D3_OUTPUT_ROOT}/logs/batch-%A_%a.out" \
  --export="${common}" \
  scripts/hpc/phase_b_d3_vllm_batch.slurm)"
echo "BATCH_JOB=${batch_job}"

jq -n --arg tier "${AF_TIER}" --arg model "${AF_D3_MODEL}" --arg indices "${indices}" \
  --arg batch "${batch_job}" \
  --argjson tasks "${task_count}" --argjson batches "${batches}" \
  '{schema_version:"phase-b-d3-vllm-submission-1", tier:$tier, model:$model,
    task_count:$tasks, batches:$batches, task_indices:$indices,
    batch_job:$batch}' \
  | tee -a "${AF_D3_OUTPUT_ROOT}/submission_ledger.jsonl"
