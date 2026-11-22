#!/bin/bash
# Use one H100 for the selected bounded public-training topology experiment.
set -euo pipefail
AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_REPO_ROOT
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
case "${2:-repair}" in
  repair)
    export AF_OUTPUT_ROOT=${AF_TOPOLOGY_REPAIR_ROOT:-$AF_GROUP/phase-c-topology-repair-20b-results-v1}
    AF_CLI="$AF_REPO_ROOT/scripts/topology_repair_20b.py"
    AF_SOURCE="$AF_REPO_ROOT/inputs/case.json"
    AF_JOB_NAME=phasec-topo-edit-20b ;;
  selection)
    export AF_OUTPUT_ROOT=${AF_PREPROPOSAL_ROOT:-$AF_GROUP/phase-c-preproposal-selection-20b-results-v1}
    AF_CLI="$AF_REPO_ROOT/scripts/preproposal_selection_20b.py"
    AF_SOURCE="$AF_REPO_ROOT/inputs/preproposal.json"
    AF_JOB_NAME=phasec-preselect-20b ;;
  *) echo 'Unknown topology experiment.' >&2; exit 2 ;;
esac
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi
case "${1:-}" in
  run|resume|preview|preflight)
    if [[ "$1" == preflight && "${2:-repair}" != selection ]]; then echo "The preflight action is for screened selection." >&2; exit 2; fi
    "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_OUTPUT_ROOT" --source "$AF_SOURCE" >/dev/null
    if [[ "$1" == preview || "$1" == preflight ]]; then exec "$AF_PYTHON" "$AF_CLI" "$1" --root "$AF_OUTPUT_ROOT"; fi
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      AF_CACHE="$AF_GROUP/huggingface-cache"
      AF_PIN=hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
      [[ -f "$AF_CACHE/$AF_PIN/config.json" ]] || AF_CACHE="/scratch/user/$USER/huggingface-cache"
      export AF_HF_HOME="$AF_CACHE"
    fi
    export AF_HARMONY_ASSETS="${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}"
    AF_ARGS=(--root "$AF_OUTPUT_ROOT")
    [[ "$1" != resume ]] || AF_ARGS+=(--resume)
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_t2_breadth.py" "${AF_ARGS[@]}" ;;
  status|inspect)
    if [[ "$1" == status ]]; then
      if command -v squeue >/dev/null; then squeue -u "$USER" -n "$AF_JOB_NAME" -o '%.18i %.28j %.10T %R' || true; fi
      "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_t2_breadth.py" --status --root "$AF_OUTPUT_ROOT"
    fi
    exec "$AF_PYTHON" "$AF_CLI" "$1" --root "$AF_OUTPUT_ROOT" ;;
  *) echo 'Use run, resume, preview, status, inspect or selection preflight.' >&2; exit 2 ;;
esac
