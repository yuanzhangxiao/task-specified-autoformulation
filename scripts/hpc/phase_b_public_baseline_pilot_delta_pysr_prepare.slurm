#!/bin/bash
#SBATCH --job-name=phase-b-pysr-prepare
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_JULIA_DEPOT:?AF_JULIA_DEPOT is required}"

export JULIA_DEPOT_PATH="${AF_JULIA_DEPOT}"
export PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runtime" \
  "${JULIA_DEPOT_PATH}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/check_pysr_runtime.py \
  --output "${AF_OUTPUT_ROOT}/runtime/pysr_runtime.json"
sha256sum "${AF_OUTPUT_ROOT}/runtime/pysr_runtime.json" \
  >"${AF_OUTPUT_ROOT}/runtime/pysr_runtime.json.sha256"
