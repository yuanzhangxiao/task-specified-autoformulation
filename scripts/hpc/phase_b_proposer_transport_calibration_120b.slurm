#!/bin/bash
#SBATCH --job-name=phase-b-proposer-cal
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-5%2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-proposer-cal-%A_%a.out
#SBATCH --error=logs/phase-b-proposer-cal-%A_%a.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/proposer-transport-calibration-v1}"
: "${AF_CALIBRATION_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v1.json}"
: "${AF_VLLM_IMAGE:=${AF_PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_HF_HOME:=${AF_PROJECT}/huggingface-cache}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${AF_APPTAINER_TMP_MIN_GIB:=40}"

readonly frozen_root="${AF_OUTPUT_ROOT}/frozen"
readonly plan="${frozen_root}/plan.json"
readonly task_plan="${frozen_root}/task_plan.jsonl"
readonly freeze_manifest="${frozen_root}/freeze_manifest.json"
readonly endpoint_port="${AF_VLLM_PORT:-$((8100 + AF_TASK_INDEX))}"
readonly endpoint="http://127.0.0.1:${endpoint_port}"
readonly model="$(jq -r '.model_contract.model' "${plan}")"
readonly served_context="$(jq -r '.model_contract.served_context_tokens' "${plan}")"
readonly server_log="${AF_OUTPUT_ROOT}/vllm-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${AF_TASK_INDEX}.log"
readonly apptainer_cache="${AF_PROJECT}/apptainer-cache"
readonly node_local_tmp_root="${SLURM_TMPDIR:-/tmp/${af_user}}"
readonly apptainer_tmp="${node_local_tmp_root}/apptainer-${SLURM_JOB_ID}"
readonly gpu_inventory="${AF_OUTPUT_ROOT}/runtime/task_${AF_TASK_INDEX}_gpu_inventory.txt"

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || { echo "missing target contracts: ${AF_TARGET_CONTRACT_ROOT}" >&2; exit 2; }
[[ -f "${AF_CALIBRATION_CONFIG}" ]] || { echo "missing calibration config" >&2; exit 2; }
[[ -f "${plan}" && -f "${task_plan}" && -f "${freeze_manifest}" ]] || {
  echo "missing frozen calibration inputs" >&2
  exit 2
}
cmp --silent "${AF_CALIBRATION_CONFIG}" "${plan}" || {
  echo "repository calibration plan differs from frozen plan" >&2
  exit 2
}
(
  cd "${frozen_root}"
  sha256sum -c plan.json.sha256
  sha256sum -c task_plan.jsonl.sha256
  sha256sum -c freeze_manifest.json.sha256
)

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export APPTAINER_CACHEDIR="${apptainer_cache}"
export APPTAINER_TMPDIR="${apptainer_tmp}"

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runtime" "${AF_HF_HOME}" \
  "${apptainer_cache}" "${apptainer_tmp}" "$(dirname "${AF_VLLM_IMAGE}")"
cd "${AF_REPO_ROOT}"

readonly available_tmp_kib="$(df -Pk "${apptainer_tmp}" | awk 'NR == 2 {print $4}')"
readonly required_tmp_kib=$((AF_APPTAINER_TMP_MIN_GIB * 1024 * 1024))
if ((available_tmp_kib < required_tmp_kib)); then
  echo "Apptainer temp has less than ${AF_APPTAINER_TMP_MIN_GIB} GiB free" >&2
  exit 1
fi

runtime_image="${AF_VLLM_IMAGE}"
if [[ ! -f "${runtime_image}" ]]; then
  runtime_image="${apptainer_tmp}/vllm-openai-v0.27.1-sandbox"
  echo "Building job-local Apptainer sandbox from ${AF_VLLM_IMAGE_URI}"
  apptainer build --sandbox "${runtime_image}" "${AF_VLLM_IMAGE_URI}"
fi

server_pid=""
cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  rm -rf -- "${apptainer_tmp}"
  exit "${status}"
}
trap cleanup EXIT

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  | tee "${gpu_inventory}"
apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${runtime_image}" \
  vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${endpoint_port}" \
    --max-model-len "${served_context}" \
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

"${AF_PYTHON}" scripts/run_phase_b_proposer_transport_calibration.py \
  --plan "${plan}" \
  --task-index "${AF_TASK_INDEX}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}" \
  --vllm-base-url "${endpoint}"
