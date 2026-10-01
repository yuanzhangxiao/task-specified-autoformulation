#!/bin/bash
# New variable-only root; preserve the original 32-construction pilot.
set -euo pipefail
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" ]] || { echo 'Run on ACES.' >&2; exit 1; }
module load GCCcore/13.2.0 Python/3.11.5
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export AF_REPO_ROOT="$AF_TOOLS"
export AF_OUTPUT_ROOT=${AF_VARIABLE_ROOT:-$AF_GROUP/phase-c-variables-v1}
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
case "${1:?Use run or inspect}" in
  run)
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    export AF_HF_HOME=${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}
    export AF_COMPUTE_CACHE_ROOT="$AF_GROUP/phase-c-runtime-cache"
    export AF_IPC_TMP_ROOT=/tmp/phase-c-ipc-u.yx126462
    AF_SOURCE=${AF_VARIABLE_SOURCE_PLAN:-$AF_GROUP/phase-c-construction-v2/plan.json}
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/phase_c_variables.py" prepare \
      --source-plan "$AF_SOURCE" --root "$AF_OUTPUT_ROOT"
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_phase_c_baseline.py" \
      --root "$AF_OUTPUT_ROOT" --variables-only --wave "${AF_VARIABLE_WAVE:-variables-1}"
    ;;
  inspect)
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/phase_c_variables.py" report --root "$AF_OUTPUT_ROOT"
    cat "$AF_OUTPUT_ROOT/SUMMARY.md"
    tar -czf "$AF_OUTPUT_ROOT/inspection.tar.gz" -C "$AF_OUTPUT_ROOT" \
      plan.json summary.json SUMMARY.md VARIABLES.html results logs submissions
    printf '\nDownload: %s\n' "$AF_OUTPUT_ROOT/inspection.tar.gz"
    ;;
  *) echo 'Use run or inspect.' >&2; exit 2 ;;
esac
