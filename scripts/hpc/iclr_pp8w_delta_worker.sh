#!/bin/bash
set -euo pipefail
: "${AF_REPO_ROOT:?Submitted worker requires AF_REPO_ROOT}"
: "${AF_PYTHON:?Submitted worker requires AF_PYTHON}"
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_CASADI_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
unset AF_JETSTREAM_API_KEY OPENAI_API_KEY
cd "$AF_REPO_ROOT"
exec "$AF_PYTHON" -m scripts.iclr_pp8w_delta "${1:?worker stage}" --root "${2:?campaign root}"
