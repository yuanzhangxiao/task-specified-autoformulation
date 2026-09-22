#!/bin/bash
# Submit the Phase-B LLM-ODE campaign against a local gpt-oss-120b endpoint.
#
#   AF_TIER          easy | hard | all   (default easy)
#   AF_BATCH_SIZE    tasks per GPU job   (default 4)
#   AF_TASK_INDICES  explicit comma list, overrides AF_TIER (smoke: "0")
#   AF_BATCH_HOURS   walltime per batch  (default 12:00:00)
#   AF_LLM_ODE_ROOT  the pinned upstream checkout
#
# Each task is a full island search -- 200 iterations x 4 islands per target --
# so batches are small and walltime long, unlike D3's single-shot tasks.

set -euo pipefail
readonly af_user="${USER:?}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_LLM_ODE_OUTPUT_ROOT:=${AF_WORK}/phase_b/llm-ode-vllm-120b-v1}"
: "${AF_LLM_ODE_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_llm_ode_campaign_v1.json}"
: "${AF_LLM_ODE_ROOT:=${AF_PROJECT}/vendor/llm-ode}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TIER:=easy}"
: "${AF_BATCH_SIZE:=4}"
: "${AF_BATCH_HOURS:=12:00:00}"
: "${AF_CLUSTER:=delta}"
case "${AF_CLUSTER}" in
  delta)
    # No H100 on Delta; A40 is Ampere, so the MXFP4 weights are dequantized
    # and need four cards.
    : "${AF_ACCOUNT:=bibo-delta-gpu}"
    : "${AF_GPU_PARTITION:=gpuA40x4}"
    : "${AF_GPU_REQUEST:=--gpus-per-node=4}"
    : "${AF_TENSOR_PARALLEL_SIZE:=4}"
    ;;
  aces)
    # H100 is Hopper: native MXFP4, one card.
    : "${AF_ACCOUNT:?set AF_ACCOUNT to the ACES GPU account from `accounts`}"
    : "${AF_GPU_PARTITION:=gpu}"
    : "${AF_GPU_REQUEST:=--gres=gpu:h100:1}"
    : "${AF_TENSOR_PARALLEL_SIZE:=1}"
    ;;
  *) echo "unknown AF_CLUSTER: ${AF_CLUSTER}" >&2; exit 2 ;;
esac

# Fail here rather than in every batch job: a missing or wrong checkout makes
# the whole campaign unfaithful, and the queue wait would be wasted.
[[ -f "${AF_LLM_ODE_ROOT}/llmode/llmode.py" ]] || {
  echo "not an LLM-ODE checkout: ${AF_LLM_ODE_ROOT}" >&2
  echo "clone it with:" >&2
  echo "  git clone https://github.com/gryaklab/llm-ode ${AF_LLM_ODE_ROOT}" >&2
  exit 2
}
pinned="$("${AF_PYTHON}" -c "
import json, sys
sys.stdout.write(json.load(open('${AF_LLM_ODE_CONFIG}'))['upstream']['commit'])
")"
actual="$(git -C "${AF_LLM_ODE_ROOT}" rev-parse HEAD)"
[[ "${pinned}" == "${actual}" ]] || {
  echo "checkout is at ${actual}, not the pinned ${pinned}" >&2
  echo "  git -C ${AF_LLM_ODE_ROOT} checkout ${pinned}" >&2
  exit 2
}

cd "${AF_REPO_ROOT}"
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs "${AF_LLM_ODE_OUTPUT_ROOT}/logs"

indices="${AF_TASK_INDICES:-$("${AF_PYTHON}" scripts/list_phase_b_d3_task_indices.py \
  --config "${AF_LLM_ODE_CONFIG}" --tier "${AF_TIER}")}"
[[ -n "${indices}" ]] || { echo "no tasks selected for tier ${AF_TIER}" >&2; exit 2; }
readonly task_count="$(tr ',' '\n' <<< "${indices}" | wc -l | tr -d ' ')"
readonly batches=$(( (task_count + AF_BATCH_SIZE - 1) / AF_BATCH_SIZE ))
echo "tier=${AF_TIER} tasks=${task_count} batch_size=${AF_BATCH_SIZE} batches=${batches}"
echo "output_root=${AF_LLM_ODE_OUTPUT_ROOT}"
echo "upstream=${AF_LLM_ODE_ROOT} commit=${actual}"
echo "cluster=${AF_CLUSTER} account=${AF_ACCOUNT} partition=${AF_GPU_PARTITION}"
echo "gpu=${AF_GPU_REQUEST} tensor_parallel=${AF_TENSOR_PARALLEL_SIZE}"

readonly common="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_LLM_ODE_OUTPUT_ROOT=${AF_LLM_ODE_OUTPUT_ROOT},AF_LLM_ODE_ROOT=${AF_LLM_ODE_ROOT},AF_LOCAL_MODEL=${AF_LOCAL_MODEL},AF_TASK_INDICES=${indices},AF_BATCH_SIZE=${AF_BATCH_SIZE},AF_LLM_ODE_CONFIG=${AF_LLM_ODE_CONFIG},AF_PUBLIC_ROOT=${AF_PUBLIC_ROOT},AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE}"

batch_job="$(sbatch --parsable --account="${AF_ACCOUNT}" \
  --partition="${AF_GPU_PARTITION}" --time="${AF_BATCH_HOURS}" \
  ${AF_GPU_REQUEST} \
  --array="0-$((batches - 1))%${AF_MAX_CONCURRENT_BATCHES:-2}" \
  --output="${AF_LLM_ODE_OUTPUT_ROOT}/logs/batch-%A_%a.out" \
  --export="${common}" \
  scripts/hpc/phase_b_llm_ode_vllm_batch.slurm)"
echo "BATCH_JOB=${batch_job}"

jq -n --arg tier "${AF_TIER}" --arg model "${AF_LOCAL_MODEL}" --arg indices "${indices}" \
  --arg batch "${batch_job}" --arg commit "${actual}" \
  --argjson tasks "${task_count}" --argjson batches "${batches}" \
  '{schema_version:"phase-b-llm-ode-vllm-submission-1", tier:$tier, model:$model,
    task_count:$tasks, batches:$batches, task_indices:$indices,
    upstream_commit:$commit, batch_job:$batch}' \
  | tee -a "${AF_LLM_ODE_OUTPUT_ROOT}/submission_ledger.jsonl"
