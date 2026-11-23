#!/bin/bash
# A single offline H100 serves all 18 paired proposer requests.
set -euo pipefail
AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_REPO_ROOT
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
AF_SOURCE=${AF_POINTWISE_SOURCE:-$AF_REPO_ROOT/inputs/source}
export AF_OUTPUT_ROOT=${AF_POINTWISE_ROOT:-$AF_GROUP/phase-c-pointwise-repair-20b-results-v1}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src"
if ! "$AF_PYTHON" -c 'import casadi, numpy, pandas, scipy' >/dev/null 2>&1; then
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src"
fi
case "${1:-}" in
  preflight|run|resume)
    "$AF_PYTHON" -m scripts.pointwise_repair_trial prepare --root "$AF_OUTPUT_ROOT" --source "$AF_SOURCE"
    if [[ "$1" == preflight ]]; then exit 0; fi
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      AF_HF_HOME="$AF_GROUP/huggingface-cache"
      AF_PIN=hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
      [[ -f "$AF_HF_HOME/$AF_PIN/config.json" ]] || AF_HF_HOME="/scratch/user/$USER/huggingface-cache"
      export AF_HF_HOME
    fi
    export AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
    AF_ARGS=(--root "$AF_OUTPUT_ROOT")
    if [[ "$1" == resume ]]; then AF_ARGS+=(--resume); fi
    exec "$AF_PYTHON" -m scripts.submit_t2_breadth "${AF_ARGS[@]}" ;;
  status)
    "$AF_PYTHON" -m scripts.submit_t2_breadth --status --root "$AF_OUTPUT_ROOT"
    exec "$AF_PYTHON" -m scripts.pointwise_repair_trial status --root "$AF_OUTPUT_ROOT" ;;
  inspect)
    exec "$AF_PYTHON" -m scripts.pointwise_repair_trial inspect --root "$AF_OUTPUT_ROOT" ;;
  *) echo 'Use preflight, run, resume, status or inspect.' >&2; exit 2 ;;
esac
