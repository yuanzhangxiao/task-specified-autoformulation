#!/bin/bash
# Prepare and submit the dedicated Delta services without hosted API credentials.
set -euo pipefail
AF_STAGE=${1:?Use prepare, submit, run, or status}
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PP8W_ROOT=${AF_PP8W_ROOT:-/work/hdd/bibo/yxiao2/phase_c/iclr-pp8w-20261118}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_CASADI_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset AF_JETSTREAM_API_KEY OPENAI_API_KEY
cd "$AF_REPO_ROOT"
case "$AF_STAGE" in
  prepare|run)
    "$AF_PYTHON" -m scripts.iclr_pp8w_delta prepare --root "$AF_PP8W_ROOT" \
      --source "${AF_PP8W_SOURCE:-$AF_REPO_ROOT/inputs/followup}"
    if [[ "$AF_STAGE" == run ]]; then
      exec "$AF_PYTHON" -m scripts.iclr_pp8w_delta submit --root "$AF_PP8W_ROOT"
    fi
    ;;
  submit|status)
    exec "$AF_PYTHON" -m scripts.iclr_pp8w_delta "$AF_STAGE" --root "$AF_PP8W_ROOT"
    ;;
  *) echo 'Use prepare, submit, run, or status.' >&2; exit 2 ;;
esac
