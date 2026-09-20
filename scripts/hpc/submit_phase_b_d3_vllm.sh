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
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/d3-native-vllm-120b-v1}"
: "${AF_D3_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_d3_full_v1.json}"
: "${AF_D3_MODEL:=openai/gpt-oss-120b}"
: "${AF_TIER:=easy}"
: "${AF_BATCH_SIZE:=10}"
: "${AF_BATCH_HOURS:=08:00:00}"
# ACES defaults; `accounts` on the login node prints the project names.
: "${AF_ACCOUNT:=156264627414}"
: "${AF_GPU_PARTITION:=gpu}"
: "${AF_CPU_PARTITION:=cpu}"
# ACES: --gres=gpu:h100:N. Delta: --gpus-per-node=N on a gpuA40x4 partition.
: "${AF_GPU_REQUEST:=--gres=gpu:h100:1}"
cd "${AF_REPO_ROOT}"
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs "${AF_OUTPUT_ROOT}/logs"

indices="${AF_TASK_INDICES:-$("${AF_PYTHON}" scripts/list_phase_b_d3_task_indices.py \
  --config "${AF_D3_CONFIG}" --tier "${AF_TIER}")}"
[[ -n "${indices}" ]] || { echo "no tasks selected for tier ${AF_TIER}" >&2; exit 2; }
readonly task_count="$(tr ',' '\n' <<< "${indices}" | wc -l | tr -d ' ')"
readonly batches=$(( (task_count + AF_BATCH_SIZE - 1) / AF_BATCH_SIZE ))
echo "tier=${AF_TIER} tasks=${task_count} batch_size=${AF_BATCH_SIZE} batches=${batches}"
echo "account=${AF_ACCOUNT} partition=${AF_GPU_PARTITION} gpu=${AF_GPU_REQUEST}"

readonly common="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_LOCAL_MODEL=${AF_D3_MODEL},AF_TASK_INDICES=${indices},AF_BATCH_SIZE=${AF_BATCH_SIZE}"

# Preparation is CPU-only and idempotent: an existing sealed plan is reused.
prepare_job=""
if [[ ! -f "${AF_OUTPUT_ROOT}/plan.json" ]]; then
  prepare_job="$(sbatch --parsable --account="${AF_ACCOUNT}" \
    --partition="${AF_CPU_PARTITION}" --time=00:30:00 --cpus-per-task=2 --mem=16G \
    --job-name=d3-prepare --output="${AF_OUTPUT_ROOT}/logs/prepare-%j.out" \
    --export="${common}" --wrap "cd ${AF_REPO_ROOT} && \
      PYTHONPATH=${AF_REPO_ROOT}/src ${AF_PYTHON} scripts/phase_b_d3.py freeze \
      --config ${AF_D3_CONFIG} --public-root ${AF_PUBLIC_ROOT} \
      --root ${AF_OUTPUT_ROOT} --model ${AF_D3_MODEL} --provider vllm")"
  echo "PREPARE_JOB=${prepare_job}"
fi

depend=()
[[ -n "${prepare_job}" ]] && depend=(--dependency="afterok:${prepare_job}" --kill-on-invalid-dep=yes)
batch_job="$(sbatch --parsable --account="${AF_ACCOUNT}" \
  --partition="${AF_GPU_PARTITION}" --time="${AF_BATCH_HOURS}" \
  ${AF_GPU_REQUEST} \
  --array="0-$((batches - 1))%${AF_MAX_CONCURRENT_BATCHES:-2}" \
  --output="${AF_OUTPUT_ROOT}/logs/batch-%A_%a.out" \
  --export="${common}" "${depend[@]}" \
  scripts/hpc/phase_b_d3_vllm_batch.slurm)"
echo "BATCH_JOB=${batch_job}"

jq -n --arg tier "${AF_TIER}" --arg model "${AF_D3_MODEL}" --arg indices "${indices}" \
  --arg prepare "${prepare_job}" --arg batch "${batch_job}" \
  --argjson tasks "${task_count}" --argjson batches "${batches}" \
  '{schema_version:"phase-b-d3-vllm-submission-1", tier:$tier, model:$model,
    task_count:$tasks, batches:$batches, task_indices:$indices,
    prepare_job:$prepare, batch_job:$batch}' \
  | tee -a "${AF_OUTPUT_ROOT}/submission_ledger.jsonl"
