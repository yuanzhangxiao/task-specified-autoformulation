#!/bin/bash
#SBATCH --job-name=proposer-cal-image
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/proposer-cal-image-%j.out
#SBATCH --error=logs/proposer-cal-image-%j.err

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_CALIBRATION_CONFIG:?AF_CALIBRATION_CONFIG is required}"
: "${AF_VLLM_IMAGE:?AF_VLLM_IMAGE is required}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_HF_HOME:=${SCRATCH:?SCRATCH is required}/huggingface-cache}"
: "${AF_APPTAINER_CACHEDIR:=${SCRATCH:?SCRATCH is required}/apptainer-cache}"

container_runtime=""
if command -v apptainer >/dev/null 2>&1; then
  container_runtime="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  container_runtime="singularity"
else
  echo "neither Apptainer nor Singularity is available" >&2
  exit 2
fi

module load WebProxy
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runtime" \
  "$(dirname "${AF_VLLM_IMAGE}")" "${AF_APPTAINER_CACHEDIR}" "${AF_HF_HOME}"

if [[ ! -s "${AF_VLLM_IMAGE}" ]]; then
  readonly temporary_image="${AF_VLLM_IMAGE}.partial-${SLURM_JOB_ID}"
  cleanup() {
    local status=$?
    trap - EXIT
    rm -f -- "${temporary_image}"
    exit "${status}"
  }
  trap cleanup EXIT
  export APPTAINER_CACHEDIR="${AF_APPTAINER_CACHEDIR}"
  export SINGULARITY_CACHEDIR="${AF_APPTAINER_CACHEDIR}"
  "${container_runtime}" build "${temporary_image}" "${AF_VLLM_IMAGE_URI}"
  mv "${temporary_image}" "${AF_VLLM_IMAGE}"
  trap - EXIT
fi

sha256sum "${AF_VLLM_IMAGE}" >"${AF_OUTPUT_ROOT}/runtime/vllm_image.sha256"
printf '%s\n' "${container_runtime}" \
  >"${AF_OUTPUT_ROOT}/runtime/container_runtime.txt"

readonly model="$(jq -r '.model_contract.model' "${AF_CALIBRATION_CONFIG}")"
"${container_runtime}" exec \
  --bind "${AF_HF_HOME}:${AF_HF_HOME}" \
  --env "AF_MODEL=${model},HF_HOME=${AF_HF_HOME}" \
  "${AF_VLLM_IMAGE}" \
  python -c \
  'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["AF_MODEL"])'
