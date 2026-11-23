#!/bin/bash
# Prepare the frozen train-only input, then queue generation, fits and validation.
set -euo pipefail
AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_REPO_ROOT
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
AF_PROFILE=${2:-fixed}
case "$AF_PROFILE" in
  fixed) AF_FOLDER=phase-c-trajectory-functions-results-v1 ;;
  source-review) AF_FOLDER=phase-c-function-source-review-results-v1 ;;
  basin-review) AF_FOLDER=phase-c-basin-function-review-results-v1 ;;
  *) echo "Choose fixed, source-review or basin-review." >&2; exit 2 ;;
esac
export AF_OUTPUT_ROOT=${AF_TRAJECTORY_FUNCTION_ROOT:-$AF_GROUP/$AF_FOLDER}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
AF_CLI="$AF_REPO_ROOT/scripts/trajectory_functions.py"
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
case "${1:-}" in
  run|resume|preflight)
    AF_PREPARE=(prepare --profile "$AF_PROFILE" --root "$AF_OUTPUT_ROOT" --source "$AF_REPO_ROOT/inputs/train.json")
    if [[ -n "${AF_POINTWISE_TOLERANCES:-}" ]]; then
      AF_PREPARE+=(--pointwise-tolerances "$AF_POINTWISE_TOLERANCES")
    fi
    "$AF_PYTHON" "$AF_CLI" "${AF_PREPARE[@]}" >/dev/null
    if [[ "$1" == preflight ]]; then exec "$AF_PYTHON" "$AF_CLI" preflight --root "$AF_OUTPUT_ROOT"; fi
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      AF_HF_HOME="$AF_GROUP/huggingface-cache"
      AF_PIN=hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
      [[ -f "$AF_HF_HOME/$AF_PIN/config.json" ]] || AF_HF_HOME="/scratch/user/$USER/huggingface-cache"
      export AF_HF_HOME
    fi
    export AF_HARMONY_ASSETS="${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}"
    AF_ARGS=(--root "$AF_OUTPUT_ROOT")
    [[ "$1" != resume ]] || AF_ARGS+=(--resume)
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_trajectory_functions.py" "${AF_ARGS[@]}" ;;
  status)
    if command -v squeue >/dev/null; then squeue -u "$USER" -n phasec-functions-20b,phasec-traj-fit,phasec-traj-val -o '%.20i %.24j %.10T %R'; fi
    exec "$AF_PYTHON" "$AF_CLI" status --root "$AF_OUTPUT_ROOT" ;;
  inspect) exec "$AF_PYTHON" "$AF_CLI" inspect --root "$AF_OUTPUT_ROOT" ;;
  *) echo "Use preflight, run, resume, status or inspect." >&2; exit 2 ;;
esac
