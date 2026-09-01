#!/bin/bash
#SBATCH --job-name=exact-derivative-fit
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/exact-derivative-fit-%j.out
#SBATCH --error=logs/exact-derivative-fit-%j.err

set -euo pipefail
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_REPO_ROOT}/artifacts/exact-derivative-fitting-benchmark-v1}"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/run_exact_derivative_fitting_benchmark.py \
  --config configs/exact_derivative_fitting_benchmark_v1.json \
  --output-root "${AF_OUTPUT_ROOT}"
cat "${AF_OUTPUT_ROOT}/benchmark_summary.md"
