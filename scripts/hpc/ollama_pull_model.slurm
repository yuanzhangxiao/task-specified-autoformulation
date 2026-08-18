#!/bin/bash
#SBATCH --job-name=ollama-pull-model
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/ollama-pull-model-%j.out
#SBATCH --error=logs/ollama-pull-model-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_OLLAMA_IMAGE:=${AF_PROJECT}/containers/ollama.sif}"
: "${AF_OLLAMA_MODELS:=${AF_PROJECT}/ollama-models}"
: "${AF_LOCAL_MODEL:?set AF_LOCAL_MODEL to the exact Ollama model tag}"

readonly port=$((20000 + SLURM_JOB_ID % 20000))
readonly endpoint="http://127.0.0.1:${port}"

mkdir -p "${AF_OLLAMA_MODELS}"
sha256sum "${AF_OLLAMA_IMAGE}"
apptainer exec "${AF_OLLAMA_IMAGE}" ollama --version

apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT}" \
  --env "OLLAMA_HOST=127.0.0.1:${port}" \
  --env "OLLAMA_MODELS=${AF_OLLAMA_MODELS}" \
  "${AF_OLLAMA_IMAGE}" ollama serve >"logs/ollama-pull-server-${SLURM_JOB_ID}.log" 2>&1 &
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
  echo "Ollama did not become ready" >&2
  exit 1
fi

apptainer exec \
  --bind "${AF_PROJECT}:${AF_PROJECT}" \
  --env "OLLAMA_HOST=127.0.0.1:${port}" \
  --env "OLLAMA_MODELS=${AF_OLLAMA_MODELS}" \
  "${AF_OLLAMA_IMAGE}" ollama pull "${AF_LOCAL_MODEL}"

apptainer exec \
  --bind "${AF_PROJECT}:${AF_PROJECT}" \
  --env "OLLAMA_HOST=127.0.0.1:${port}" \
  --env "OLLAMA_MODELS=${AF_OLLAMA_MODELS}" \
  "${AF_OLLAMA_IMAGE}" ollama show "${AF_LOCAL_MODEL}" >/dev/null

echo "download complete: ${AF_LOCAL_MODEL}"
du -sh "${AF_OLLAMA_MODELS}"
