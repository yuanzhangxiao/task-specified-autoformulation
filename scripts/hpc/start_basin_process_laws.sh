#!/bin/bash
# Run from the committed bundle. Slurm CPU jobs may invoke this file by absolute path.
set -euo pipefail
AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_REPO_ROOT
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
AF_SOURCE=${AF_BASIN_SOURCE:-$AF_GROUP/phase-c-basin-function-review-results-v1}
export AF_OUTPUT_ROOT=${AF_BASIN_PROCESS_ROOT:-$AF_GROUP/phase-c-basin-process-proposals-v1}
AF_NUMERIC=${AF_BASIN_FIT_ROOT:-$AF_GROUP/phase-c-basin-process-fits-v1}
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
    "$AF_PYTHON" -m scripts.basin_process_proposals prepare --root "$AF_OUTPUT_ROOT" --campaign "$AF_SOURCE"
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
    exec "$AF_PYTHON" -m scripts.basin_process_proposals status --root "$AF_OUTPUT_ROOT" ;;
  fit)
    "$AF_PYTHON" -m scripts.fit_basin_process_proposals train --campaign "$AF_OUTPUT_ROOT" --root "$AF_NUMERIC"
    exec "$AF_PYTHON" -m scripts.fit_basin_process_proposals validate --campaign "$AF_OUTPUT_ROOT" --root "$AF_NUMERIC" --validation "$AF_REPO_ROOT/inputs/validation.json" ;;
  stability)
    AF_POLICY=${2:-original}
    case "$AF_POLICY" in original|geometric) ;; *) echo 'Use original or geometric' >&2; exit 2 ;; esac
    exec "$AF_PYTHON" -m scripts.basin_family_stability --start-policy "$AF_POLICY" --campaign "$AF_SOURCE" --root "$AF_GROUP/phase-c-basin-stability-$AF_POLICY-v1" ;;
  *) echo 'Use preflight, run, resume, status, fit or stability.' >&2; exit 2 ;;
esac
