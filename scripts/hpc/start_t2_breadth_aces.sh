#!/bin/bash
# Submit once, report progress or package the three-origin T2 pilot.
set -euo pipefail
AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_REPO_ROOT
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
export AF_OUTPUT_ROOT=${AF_T2_BREADTH_ROOT:-$AF_GROUP/phase-c-t2-breadth-results-v1}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
AF_CLI="$AF_REPO_ROOT/scripts/t2_breadth.py"
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi
case "${1:-}" in
  run|resume)
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      AF_CACHE="$AF_GROUP/huggingface-cache"
      [[ -f "$AF_CACHE/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json" ]] || AF_CACHE=/scratch/user/u.yx126462/huggingface-cache
      export AF_HF_HOME="$AF_CACHE"
    fi
    export AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
    "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_OUTPUT_ROOT" --source "$AF_REPO_ROOT/inputs/intake.json" >/dev/null
    AF_ARGS=(--root "$AF_OUTPUT_ROOT")
    [[ "$1" != resume ]] || AF_ARGS+=(--resume)
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_t2_breadth.py" "${AF_ARGS[@]}" ;;
  status|inspect|preview)
    if [[ "$1" == preview && ! -f "$AF_OUTPUT_ROOT/plan.json" ]]; then
      "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_OUTPUT_ROOT" --source "$AF_REPO_ROOT/inputs/intake.json" >/dev/null
    fi
    if [[ "$1" == status ]] && command -v squeue >/dev/null; then squeue -u "$USER" -o '%.18i %.28j %.10T %R' || true; fi
    exec "$AF_PYTHON" "$AF_CLI" "$1" --root "$AF_OUTPUT_ROOT" ;;
  *) echo 'Use run, resume, status, inspect or preview.' >&2; exit 2 ;;
esac
