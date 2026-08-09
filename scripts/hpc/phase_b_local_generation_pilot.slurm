#!/bin/bash
#SBATCH --job-name=phase-b-local-pilot
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --output=logs/phase-b-local-pilot-%j.out
#SBATCH --error=logs/phase-b-local-pilot-%j.err

set -euo pipefail

: "${AF_PROJECT:?set AF_PROJECT to persistent project storage}"
: "${AF_WORK:?set AF_WORK to active work storage}"
: "${AF_REPO_ROOT:?set AF_REPO_ROOT to the repository checkout}"
: "${AF_PYTHON:?set AF_PYTHON to the project virtual-environment Python}"
: "${AF_PUBLIC_DATA_ROOT:?set AF_PUBLIC_DATA_ROOT to the released suite root}"
: "${AF_OLLAMA_IMAGE:?set AF_OLLAMA_IMAGE to a pinned Ollama SIF image}"
: "${AF_OLLAMA_MODELS:?set AF_OLLAMA_MODELS to the persistent model directory}"
: "${AF_LOCAL_MODEL:=gpt-oss:20b}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/local-generation-pilot-v5}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

readonly port=$((20000 + SLURM_JOB_ID % 20000))
readonly endpoint="http://127.0.0.1:${port}"
readonly server_log="${AF_OUTPUT_ROOT}/ollama-${SLURM_JOB_ID}.log"

mkdir -p "${AF_OUTPUT_ROOT}" "${AF_OLLAMA_MODELS}"
cd "${AF_REPO_ROOT}"

sha256sum "${AF_OLLAMA_IMAGE}"
apptainer exec "${AF_OLLAMA_IMAGE}" ollama --version

apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "OLLAMA_HOST=127.0.0.1:${port}" \
  --env "OLLAMA_MODELS=${AF_OLLAMA_MODELS}" \
  "${AF_OLLAMA_IMAGE}" ollama serve >"${server_log}" 2>&1 &
readonly server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 90); do
  if curl --silent --fail "${endpoint}/api/tags" >/dev/null; then
    ready=true
    break
  fi
  sleep 2
done
if [[ "${ready}" != true ]]; then
  echo "Ollama did not become ready; see ${server_log}" >&2
  exit 1
fi

apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "OLLAMA_HOST=127.0.0.1:${port}" \
  --env "OLLAMA_MODELS=${AF_OLLAMA_MODELS}" \
  "${AF_OLLAMA_IMAGE}" ollama show "${AF_LOCAL_MODEL}" >/dev/null

"${AF_PYTHON}" scripts/run_experiment.py \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --benchmark-id phase_b_dalla_man_t2_canonical_named_easy \
  --tier easy \
  --seed 0 \
  --proposer-model "ollama:${AF_LOCAL_MODEL}" \
  --judge-model "ollama:${AF_LOCAL_MODEL}" \
  --ollama-base-url "${endpoint}" \
  --ollama-thinking low \
  --development-only \
  --no-judge \
  --iteration-budget 6 \
  --beam-size 2 \
  --llm-timeout-seconds 900 \
  --llm-max-output-tokens 4096 \
  --fit-starts 1 \
  --fit-max-nfev 10 \
  --fit-timeout-seconds 120 \
  --final-fit-max-nfev 10 \
  --final-fit-timeout-seconds 120 \
  --output-root "${AF_OUTPUT_ROOT}/runs"
