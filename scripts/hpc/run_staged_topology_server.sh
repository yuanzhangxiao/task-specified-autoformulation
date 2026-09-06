#!/bin/bash
# Shared one-server/multiple-task worker, without numerical data or fitting.
set -euo pipefail
: "${AF_REPO_ROOT:?required}"
: "${AF_PYTHON:?required}"
: "${AF_OUTPUT_ROOT:?required}"
: "${AF_VLLM_IMAGE:?required}"
: "${AF_HF_HOME:?required}"
: "${AF_COMPUTE_CACHE_ROOT:?required}"
: "${AF_IPC_TMP_ROOT:?required}"
readonly plan="${AF_OUTPUT_ROOT}/plan.json"
readonly model="$(jq -r '.config.model_settings.model' "${plan}")"
readonly model_revision="$(jq -r '.config.model_settings.model_revision' "${plan}")"
readonly expected_image_sha="$(jq -r '.config.serving_image_sha256' "${plan}")"
readonly context_tokens="$(jq -r '.config.served_context_tokens' "${plan}")"
readonly worker_seconds="$(jq -r '.config.wall_seconds' "${plan}")"
readonly runtime_root="${AF_OUTPUT_ROOT}/runtime"
readonly job_cache="${AF_COMPUTE_CACHE_ROOT}/${SLURM_JOB_ID}"
readonly ipc_tmp="${AF_IPC_TMP_ROOT}/${SLURM_JOB_ID}"
readonly endpoint="http://127.0.0.1:8100"
(( ${#ipc_tmp} <= 60 )) || { echo 'IPC path too long' >&2; exit 2; }
[[ -f "${AF_VLLM_IMAGE}" && -f "${plan}" && -x "${AF_PYTHON}" ]] || exit 2
mkdir -p "${runtime_root}" "${job_cache}" "${ipc_tmp}"
image_sha="$(sha256sum "${AF_VLLM_IMAGE}")"
[[ "${image_sha%% *}" == "${expected_image_sha}" ]] || {
  echo 'serving image differs from frozen campaign' >&2
  exit 2
}
printf '%s\n' "${image_sha}" >"${runtime_root}/image-${SLURM_JOB_ID}.sha256"
export PYTHONPATH="${AF_REPO_ROOT}/src"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
runtime="$(command -v apptainer || command -v singularity)"
server_pid=""
worker_pid=""
draining=false
drain() {
  draining=true
  if [[ -n "${worker_pid}" ]]; then
    kill -TERM "${worker_pid}" 2>/dev/null || true
  fi
}
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${worker_pid}" ]]; then
    kill -TERM "${worker_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]]; then
    kill -TERM "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap drain TERM INT
trap cleanup EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  >"${runtime_root}/gpu-${SLURM_JOB_ID}.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${runtime_root}/started-${SLURM_JOB_ID}.txt"
git -C "${AF_REPO_ROOT}" rev-parse HEAD >"${runtime_root}/commit-${SLURM_JOB_ID}.txt"
sha256sum "${AF_REPO_ROOT}/scripts/hpc/run_staged_topology_server.sh" \
  >"${runtime_root}/launcher-${SLURM_JOB_ID}.sha256"
"${runtime}" exec --nv \
  --bind "${AF_HF_HOME}:${AF_HF_HOME},${job_cache}:${job_cache},${ipc_tmp}:${ipc_tmp}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  --env "TRITON_CACHE_DIR=${job_cache}/triton" \
  --env "TORCHINDUCTOR_CACHE_DIR=${job_cache}/torchinductor" \
  --env "VLLM_CACHE_ROOT=${job_cache}/vllm" \
  --env "XDG_CACHE_HOME=${job_cache}/xdg" \
  --env "CUDA_CACHE_PATH=${job_cache}/cuda" \
  --env "TMPDIR=${ipc_tmp}" \
  "${AF_VLLM_IMAGE}" vllm serve "${model}" --revision "${model_revision}" \
  --host 127.0.0.1 --port 8100 --max-model-len "${context_tokens}" \
  --max-num-seqs 1 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size "${AF_TENSOR_PARALLEL_SIZE:-1}" \
  >"${runtime_root}/server-${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
ready=false
for _ in $(seq 1 300); do
  [[ "${draining}" == false ]] || exit 0
  if curl --noproxy '*' --silent --fail "${endpoint}/v1/models" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 60 "${runtime_root}/server-${SLURM_JOB_ID}.log" >&2
    exit 1
  fi
  sleep 2
done
[[ "${ready}" == true ]] || { echo 'server startup deadline exceeded' >&2; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ >"${runtime_root}/ready-${SLURM_JOB_ID}.txt"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/staged_topology_campaign.py" run \
  --plan "${plan}" --output "${AF_OUTPUT_ROOT}/results" \
  --base-url "${endpoint}" --wall-seconds "${worker_seconds}" \
  >"${runtime_root}/worker-${SLURM_JOB_ID}.log" 2>&1 &
worker_pid=$!
status=0
wait "${worker_pid}" || status=$?
if kill -0 "${worker_pid}" 2>/dev/null; then
  wait "${worker_pid}" || status=$?
fi
worker_pid=""
date -u +%Y-%m-%dT%H:%M:%SZ >"${runtime_root}/finished-${SLURM_JOB_ID}.txt"
exit "${status}"
