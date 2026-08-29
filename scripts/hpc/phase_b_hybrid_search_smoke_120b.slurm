#!/bin/bash
#SBATCH --job-name=phase-b-hybrid-search-smoke
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-hybrid-search-smoke-%j.out
#SBATCH --error=logs/phase-b-hybrid-search-smoke-%j.err

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
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/hybrid-search-smoke-v3}"
: "${AF_BENCHMARK_ID:=phase_b_dalla_man_t2_canonical_named_easy}"
: "${AF_TIER:=easy}"
: "${AF_SEED:=0}"
: "${AF_ITERATION_BUDGET:=8}"
: "${AF_HYBRID_SCIENCE_WEIGHT:=0.5}"
: "${AF_APPTAINER_TMP_MIN_GIB:=40}"

readonly benchmark_id="${AF_BENCHMARK_ID}"
readonly tier="${AF_TIER}"
readonly run_name="${benchmark_id}_${tier}_seed${AF_SEED}"
readonly run_root="${AF_OUTPUT_ROOT}/runs/${run_name}"
readonly config_source="${AF_REPO_ROOT}/configs/hybrid_search_objective_pilot_v2.json"
readonly frozen_config="${AF_OUTPUT_ROOT}/hybrid_search_objective_config.json"
readonly port=$((20000 + SLURM_JOB_ID % 20000))
readonly endpoint="http://127.0.0.1:${port}"
readonly server_log="${AF_OUTPUT_ROOT}/vllm-${SLURM_JOB_ID}.log"
readonly apptainer_cache="${AF_PROJECT}/apptainer-cache"
readonly node_local_tmp_root="${SLURM_TMPDIR:-/tmp/${af_user}}"
readonly apptainer_tmp="${node_local_tmp_root}/apptainer-${SLURM_JOB_ID}"

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

[[ -x "${AF_PYTHON}" ]] || {
  echo "missing experiment Python: ${AF_PYTHON}" >&2
  exit 2
}
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || {
  echo "missing public data root: ${AF_PUBLIC_DATA_ROOT}" >&2
  exit 2
}
[[ -f "${config_source}" ]] || {
  echo "missing frozen search config: ${config_source}" >&2
  exit 2
}

mkdir -p \
  "${AF_REPO_ROOT}/logs" \
  "${AF_OUTPUT_ROOT}" \
  "${AF_HF_HOME}" \
  "${apptainer_cache}" \
  "${apptainer_tmp}" \
  "$(dirname "${AF_VLLM_IMAGE}")"
cd "${AF_REPO_ROOT}"

if [[ -f "${frozen_config}" ]]; then
  cmp --silent "${config_source}" "${frozen_config}" || {
    echo "frozen search config differs from repository config" >&2
    exit 2
  }
else
  cp "${config_source}" "${frozen_config}"
fi
sha256sum "${frozen_config}"

readonly available_tmp_kib="$(
  df -Pk "${apptainer_tmp}" | awk 'NR == 2 {print $4}'
)"
readonly required_tmp_kib=$((AF_APPTAINER_TMP_MIN_GIB * 1024 * 1024))
echo "model=${AF_LOCAL_MODEL} reasoning=low search_seed=${AF_SEED}"
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
  tail -n 100 "${server_log}" >&2
  exit 1
fi

resume_arguments=()
if [[ -f "${run_root}/checkpoints/run.json" ]]; then
  resume_arguments=(--resume)
  echo "Resuming checkpointed run: ${run_root}"
fi

"${AF_PYTHON}" scripts/run_experiment.py \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --benchmark-id "${benchmark_id}" \
  --tier "${tier}" \
  --seed "${AF_SEED}" \
  --proposer-model "vllm:${AF_LOCAL_MODEL}" \
  --judge-model "vllm:${AF_LOCAL_MODEL}" \
  --vllm-base-url "${endpoint}" \
  --vllm-reasoning-effort low \
  --vllm-temperature 0.2 \
  --vllm-seed "${AF_SEED}" \
  --hybrid-judge-seed-base "$((12000 + 2 * AF_SEED))" \
  --development-only \
  --selection-policy incumbent_relative_hybrid \
  --hybrid-science-weight "${AF_HYBRID_SCIENCE_WEIGHT}" \
  --iteration-budget "${AF_ITERATION_BUDGET}" \
  --stagnation-iterations "${AF_ITERATION_BUDGET}" \
  --beam-size 1 \
  --llm-timeout-seconds 900 \
  --llm-max-output-tokens 4096 \
  --fit-starts 1 \
  --fit-max-nfev 10 \
  --fit-timeout-seconds 120 \
  --final-fit-max-nfev 10 \
  --final-fit-timeout-seconds 120 \
  --output-root "${AF_OUTPUT_ROOT}/runs" \
  "${resume_arguments[@]}"
