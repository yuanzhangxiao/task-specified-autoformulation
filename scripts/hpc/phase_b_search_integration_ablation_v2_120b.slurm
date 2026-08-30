#!/bin/bash
#SBATCH --job-name=phase-b-search-v2
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-11%2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-search-v2-%A_%a.out
#SBATCH --error=logs/phase-b-search-v2-%A_%a.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_VLLM_IMAGE:=${AF_PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_HF_HOME:=${AF_PROJECT}/huggingface-cache}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v3}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_VLLM_PORT:=8000}"
: "${AF_APPTAINER_TMP_MIN_GIB:=40}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly frozen_root="${AF_OUTPUT_ROOT}/frozen"
readonly config_source="${AF_REPO_ROOT}/configs/phase_b_search_integration_ablation_v2.json"
readonly judge_protocol="${AF_REPO_ROOT}/configs/hybrid_judge_consensus_operating_point_v1.json"
readonly plan="${frozen_root}/plan.json"
readonly task_plan="${frozen_root}/task_plan.jsonl"
readonly freeze_manifest="${frozen_root}/freeze_manifest.json"
readonly task_line=$((AF_TASK_INDEX + 1))

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public overlay: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || { echo "missing target contracts: ${AF_TARGET_CONTRACT_ROOT}" >&2; exit 2; }
[[ -f "${config_source}" ]] || { echo "missing repository plan: ${config_source}" >&2; exit 2; }
[[ -f "${judge_protocol}" ]] || { echo "missing judge protocol: ${judge_protocol}" >&2; exit 2; }
[[ -f "${plan}" ]] || { echo "missing frozen plan: ${plan}" >&2; exit 2; }
[[ -f "${task_plan}" ]] || { echo "missing task plan: ${task_plan}" >&2; exit 2; }
[[ -x /usr/bin/time ]] || { echo "GNU time is unavailable" >&2; exit 2; }
cmp --silent "${config_source}" "${plan}" || {
  echo "repository plan differs from the frozen plan" >&2
  exit 2
}

(
  cd "${frozen_root}"
  sha256sum -c plan.json.sha256
  sha256sum -c task_plan.jsonl.sha256
  sha256sum -c freeze_manifest.json.sha256
)

readonly task_json="$(sed -n "${task_line}p" "${task_plan}")"
[[ -n "${task_json}" ]] || { echo "task index absent: ${AF_TASK_INDEX}" >&2; exit 2; }
readonly recorded_task_index="$(jq -r '.task_index' <<<"${task_json}")"
[[ "${recorded_task_index}" == "${AF_TASK_INDEX}" ]] || {
  echo "task ledger index mismatch" >&2
  exit 2
}

readonly arm_id="$(jq -r '.arm_id' <<<"${task_json}")"
readonly benchmark_id="$(jq -r '.benchmark_id' <<<"${task_json}")"
readonly tier="$(jq -r '.tier' <<<"${task_json}")"
readonly repetition="$(jq -r '.repetition' <<<"${task_json}")"
readonly selection_policy="$(jq -r '.selection_policy' <<<"${task_json}")"
readonly use_judge="$(jq -r '.use_judge' <<<"${task_json}")"
readonly science_weight="$(jq -r '.hybrid_science_weight // empty' <<<"${task_json}")"
readonly expected_prompt_sha256="$(jq -r '.public_prompt_sha256' <<<"${task_json}")"
readonly expected_contract_sha256="$(jq -r '.public_target_contract_sha256' <<<"${task_json}")"
readonly model="$(jq -r '.model_contract.proposer_model | sub("^vllm:"; "")' "${plan}")"
readonly judge_model="$(jq -r '.model_contract.judge_model | sub("^vllm:"; "")' "${plan}")"
readonly reasoning_effort="$(jq -r '.model_contract.reasoning_effort' "${plan}")"
readonly temperature="$(jq -r '.model_contract.temperature' "${plan}")"
readonly max_output_tokens="$(jq -r '.model_contract.proposer_max_output_tokens' "${plan}")"
readonly request_timeout="$(jq -r '.model_contract.request_timeout_seconds' "${plan}")"
readonly expected_judge_sha256="$(jq -r '.model_contract.judge_protocol_config_sha256' "${plan}")"
readonly iteration_budget="$(jq -r '.search_budget.iteration_budget' "${plan}")"
readonly beam_size="$(jq -r '.search_budget.beam_size' "${plan}")"
readonly fit_starts="$(jq -r '.search_budget.fit_starts' "${plan}")"
readonly fit_max_nfev="$(jq -r '.search_budget.fit_max_nfev' "${plan}")"
readonly fit_timeout="$(jq -r '.search_budget.fit_timeout_seconds' "${plan}")"
readonly final_fit_max_nfev="$(jq -r '.search_budget.final_fit_max_nfev' "${plan}")"
readonly final_fit_timeout="$(jq -r '.search_budget.final_fit_timeout_seconds' "${plan}")"

readonly prompt_path="${AF_PUBLIC_DATA_ROOT}/phase_b_v1/${benchmark_id}/proposer_prompt.txt"
readonly target_contract="${AF_TARGET_CONTRACT_ROOT}/specs/${benchmark_id}.json"
[[ -f "${prompt_path}" ]] || { echo "missing proposer prompt: ${prompt_path}" >&2; exit 2; }
[[ -f "${target_contract}" ]] || { echo "missing target contract: ${target_contract}" >&2; exit 2; }
[[ "$(sha256sum "${prompt_path}" | awk '{print $1}')" == "${expected_prompt_sha256}" ]] || {
  echo "public prompt differs from frozen task" >&2
  exit 2
}
[[ "$(sha256sum "${target_contract}" | awk '{print $1}')" == "${expected_contract_sha256}" ]] || {
  echo "public target contract differs from frozen task" >&2
  exit 2
}
[[ "$(sha256sum "${judge_protocol}" | awk '{print $1}')" == "${expected_judge_sha256}" ]] || {
  echo "judge protocol differs from frozen plan" >&2
  exit 2
}
readonly expected_overlay_manifest_sha256="$(jq -r '.public_input_validation.prompt_overlay_manifest_sha256' "${freeze_manifest}")"
readonly overlay_manifest="${AF_PUBLIC_DATA_ROOT}/prompt_overlay_manifest.json"
[[ "$(sha256sum "${overlay_manifest}" | awk '{print $1}')" == "${expected_overlay_manifest_sha256}" ]] || {
  echo "public overlay manifest differs from frozen preflight" >&2
  exit 2
}

readonly arm_output_root="${AF_OUTPUT_ROOT}/searches/${arm_id}"
readonly run_name="${benchmark_id}_${tier}_seed${repetition}"
readonly run_root="${arm_output_root}/runs/${run_name}"
readonly shared_cache="${AF_OUTPUT_ROOT}/shared-llm-cache/${run_name}"
readonly process_time="${run_root}/search_process_time.txt"
readonly gpu_inventory="${run_root}/gpu_inventory.txt"
readonly task_started_epoch_seconds="$(date +%s.%N)"
readonly endpoint_port="${AF_VLLM_PORT}"
readonly endpoint="http://127.0.0.1:${endpoint_port}"
readonly server_log="${AF_OUTPUT_ROOT}/vllm-${SLURM_JOB_ID}.log"
readonly apptainer_cache="${AF_PROJECT}/apptainer-cache"
readonly node_local_tmp_root="${SLURM_TMPDIR:-/tmp/${af_user}}"
readonly apptainer_tmp="${node_local_tmp_root}/apptainer-${SLURM_JOB_ID}"

server_pid=""
cleanup() {
  local status=$?
  trap - EXIT
  set +e
  "${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/write_phase_b_search_task_runtime.py" \
    --output "${run_root}/task_runtime.json" \
    --task-index "${AF_TASK_INDEX}" \
    --arm-id "${arm_id}" \
    --benchmark-id "${benchmark_id}" \
    --tier "${tier}" \
    --repetition "${repetition}" \
    --started-epoch-seconds "${task_started_epoch_seconds}" \
    --exit-code "${status}" \
    --allocated-cpus "${SLURM_CPUS_PER_TASK:-16}" \
    --allocated-gpus "${AF_TENSOR_PARALLEL_SIZE}" \
    --gpu-inventory "${gpu_inventory}" \
    --process-time "${process_time}" >/dev/null
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  rm -rf -- "${apptainer_tmp}"
  exit "${status}"
}
trap cleanup EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export APPTAINER_CACHEDIR="${apptainer_cache}"
export APPTAINER_TMPDIR="${apptainer_tmp}"

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_HF_HOME}" \
  "${apptainer_cache}" "${apptainer_tmp}" "${shared_cache}" "${run_root}" \
  "$(dirname "${AF_VLLM_IMAGE}")"
cd "${AF_REPO_ROOT}"

readonly available_tmp_kib="$(df -Pk "${apptainer_tmp}" | awk 'NR == 2 {print $4}')"
readonly required_tmp_kib=$((AF_APPTAINER_TMP_MIN_GIB * 1024 * 1024))
echo "task=${AF_TASK_INDEX} arm=${arm_id} benchmark=${benchmark_id} tier=${tier} repetition=${repetition}"
echo "model=${model} reasoning=${reasoning_effort} selection_policy=${selection_policy}"
if ((available_tmp_kib < required_tmp_kib)); then
  echo "Apptainer temp has less than ${AF_APPTAINER_TMP_MIN_GIB} GiB free" >&2
  exit 1
fi

runtime_image="${AF_VLLM_IMAGE}"
if [[ -f "${runtime_image}" ]]; then
  sha256sum "${runtime_image}"
else
  runtime_image="${apptainer_tmp}/vllm-openai-v0.27.1-sandbox"
  echo "Building job-local Apptainer sandbox from ${AF_VLLM_IMAGE_URI}"
  apptainer build --sandbox "${runtime_image}" "${AF_VLLM_IMAGE_URI}"
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "${gpu_inventory}"
apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${runtime_image}" \
  vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${endpoint_port}" \
    --max-model-len 32768 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size "${AF_TENSOR_PARALLEL_SIZE}" \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 1200); do
  if curl --silent --fail "${endpoint}/v1/models" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see ${server_log}" >&2
    tail -n 100 "${server_log}" >&2
    exit 1
  fi
  sleep 2
done
if [[ "${ready}" != true ]]; then
  echo "vLLM did not become ready; see ${server_log}" >&2
  exit 1
fi

resume_arguments=()
if [[ -f "${run_root}/checkpoints/run.json" ]]; then
  resume_arguments=(--resume)
  echo "Resuming checkpointed run: ${run_root}"
fi

selection_arguments=(--selection-policy "${selection_policy}")
if [[ "${use_judge}" == "true" ]]; then
  [[ -n "${science_weight}" ]] || { echo "judge arm has no science weight" >&2; exit 2; }
  selection_arguments+=(
    --judge-model "vllm:${judge_model}"
    --hybrid-science-weight "${science_weight}"
    --hybrid-judge-seed-base "$((12000 + 2 * repetition))"
  )
else
  selection_arguments+=(--no-judge --require-initial-proposer-cache-hit)
fi

/usr/bin/time -o "${process_time}" \
  -f $'elapsed_seconds=%e\nuser_cpu_seconds=%U\nsystem_cpu_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' \
  "${AF_PYTHON}" scripts/run_experiment.py \
    --data-root "${AF_PUBLIC_DATA_ROOT}" \
    --benchmark-id "${benchmark_id}" \
    --tier "${tier}" \
    --seed "${repetition}" \
    --proposer-model "vllm:${model}" \
    --vllm-base-url "${endpoint}" \
    --vllm-reasoning-effort "${reasoning_effort}" \
    --vllm-temperature "${temperature}" \
    --vllm-seed "${repetition}" \
    --development-only \
    --iteration-budget "${iteration_budget}" \
    --stagnation-iterations "${iteration_budget}" \
    --beam-size "${beam_size}" \
    --llm-timeout-seconds "${request_timeout}" \
    --llm-max-output-tokens "${max_output_tokens}" \
    --llm-cache-root "${shared_cache}" \
    --fit-starts "${fit_starts}" \
    --fit-max-nfev "${fit_max_nfev}" \
    --fit-timeout-seconds "${fit_timeout}" \
    --final-fit-max-nfev "${final_fit_max_nfev}" \
    --final-fit-timeout-seconds "${final_fit_timeout}" \
    --public-target-contract "${target_contract}" \
    --output-root "${arm_output_root}/runs" \
    "${selection_arguments[@]}" \
    "${resume_arguments[@]}"
