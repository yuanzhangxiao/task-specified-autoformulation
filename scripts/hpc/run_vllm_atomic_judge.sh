#!/bin/bash

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_PROJECT}/phase_b/inputs/public}"
: "${AF_VLLM_IMAGE:=${AF_PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_HF_HOME:=${AF_PROJECT}/huggingface-cache}"
: "${AF_LOCAL_MODEL:?set AF_LOCAL_MODEL in the Slurm wrapper}"
: "${AF_TENSOR_PARALLEL_SIZE:?set AF_TENSOR_PARALLEL_SIZE in the Slurm wrapper}"
: "${AF_CALIBRATION_PAIRS:=${AF_WORK}/phase_b/judge-hybrid-heldout-v1/pairs.jsonl}"
: "${AF_CALIBRATION_ROOT:?set AF_CALIBRATION_ROOT in the Slurm wrapper}"
: "${AF_REPETITIONS:=5}"
: "${AF_SHARD_COUNT:=1}"
: "${AF_MAX_ATTEMPTS:=10}"
: "${AF_JUDGE_SEED_BASE:=10000}"
: "${AF_PARTIAL_WEIGHT:=0.05}"
: "${AF_COMPARATIVE_WEIGHT:=0.25}"
: "${AF_TIE_THRESHOLD:=0.05}"
: "${AF_APPTAINER_TMP_MIN_GIB:=40}"

[[ "${AF_REPETITIONS}" == 5 && "${AF_MAX_ATTEMPTS}" == 10 ]] || {
  echo "atomic comparison requires five repetitions and ten attempts" >&2
  exit 2
}
[[ "${AF_JUDGE_SEED_BASE}" == 10000 ]] || {
  echo "atomic comparison requires seed base 10000" >&2
  exit 2
}

readonly shard_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly shard_root="${AF_CALIBRATION_ROOT}/shards/shard_${shard_index}"
readonly port=$((20000 + (SLURM_JOB_ID + shard_index) % 20000))
readonly endpoint="http://127.0.0.1:${port}"
readonly server_log="${shard_root}/vllm-${SLURM_JOB_ID}.log"
readonly apptainer_cache="${AF_PROJECT}/apptainer-cache"
readonly node_local_tmp_root="${SLURM_TMPDIR:-/tmp/${af_user}}"
readonly apptainer_tmp="${node_local_tmp_root}/apptainer-${SLURM_JOB_ID}"
readonly pair_ids=(
  heldout_55d8026028a90be5
  heldout_ee453d8cc6fcb7a2
  heldout_70b3222d4736ea1d
  heldout_cca8883e6ae1b33f
)

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  rm -rf -- "${apptainer_tmp}"
}
trap cleanup EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export APPTAINER_CACHEDIR="${apptainer_cache}"
export APPTAINER_TMPDIR="${apptainer_tmp}"

[[ -f "${AF_CALIBRATION_PAIRS}" ]] || {
  echo "missing calibration pairs: ${AF_CALIBRATION_PAIRS}" >&2
  exit 2
}
mkdir -p \
  "${AF_REPO_ROOT}/logs" \
  "${shard_root}" \
  "${AF_HF_HOME}" \
  "${apptainer_cache}" \
  "${apptainer_tmp}" \
  "$(dirname "${AF_VLLM_IMAGE}")"
cd "${AF_REPO_ROOT}"

readonly available_tmp_kib="$(
  df -Pk "${apptainer_tmp}" | awk 'NR == 2 {print $4}'
)"
readonly required_tmp_kib=$((AF_APPTAINER_TMP_MIN_GIB * 1024 * 1024))
echo "model=${AF_LOCAL_MODEL} reasoning=low atomic=true shard=${shard_index}/${AF_SHARD_COUNT}"
echo "Apptainer build temp: ${apptainer_tmp}"
df -hP "${apptainer_tmp}"
if ((available_tmp_kib < required_tmp_kib)); then
  echo "Apptainer build temp has less than ${AF_APPTAINER_TMP_MIN_GIB} GiB free" >&2
  exit 1
fi

runtime_image="${AF_VLLM_IMAGE}"
if [[ -f "${runtime_image}" ]]; then
  sha256sum "${runtime_image}"
else
  runtime_image="${apptainer_tmp}/vllm-openai-v0.27.1-sandbox"
  echo "Building job-local Apptainer sandbox from ${AF_VLLM_IMAGE_URI}"
  apptainer build --sandbox "${runtime_image}" "${AF_VLLM_IMAGE_URI}"
  du -sh "${runtime_image}"
fi

nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader

vllm_parallel_args=()
if ((AF_TENSOR_PARALLEL_SIZE > 1)); then
  vllm_parallel_args=(--tensor-parallel-size "${AF_TENSOR_PARALLEL_SIZE}")
fi

apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${runtime_image}" \
  vllm serve "${AF_LOCAL_MODEL}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --max-model-len 32768 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    "${vllm_parallel_args[@]}" \
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
  tail -n 100 "${server_log}" >&2
  exit 1
fi

flock --exclusive "${shard_root}/hybrid_judge.lock" \
  "${AF_PYTHON}" scripts/run_hybrid_judge.py \
  --pairs "${AF_CALIBRATION_PAIRS}" \
  --pair-ids "${pair_ids[@]}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --judge-models "vllm:${AF_LOCAL_MODEL}" \
  --repetitions "${AF_REPETITIONS}" \
  --output-root "${shard_root}" \
  --timeout-seconds 900 \
  --max-output-tokens 6144 \
  --max-attempts "${AF_MAX_ATTEMPTS}" \
  --vllm-base-url "${endpoint}" \
  --vllm-reasoning-effort low \
  --vllm-temperature 0.2 \
  --vllm-seed-base "${AF_JUDGE_SEED_BASE}" \
  --partial-tiebreak-weight "${AF_PARTIAL_WEIGHT}" \
  --comparative-weight "${AF_COMPARATIVE_WEIGHT}" \
  --tie-threshold "${AF_TIE_THRESHOLD}" \
  --atomic-signed-occurrences \
  --shard-index "${shard_index}" \
  --shard-count "${AF_SHARD_COUNT}" \
  --shard-strategy contiguous
