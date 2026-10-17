#!/bin/bash
# Run on the ACES login node after extracting the reviewed bundle into group scratch.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" ]] || { echo 'Run this command on ACES, with AF_GROUP set to your writable scratch directory.' >&2; exit 2; }
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
AF_CLI="$AF_REPO_ROOT/scripts/saved_initialization_aces.py"
# Probe the real import chain: numpy can import while pandas/ctypes lacks libffi.
# --help imports the worker without reading a plan, submitting, or making calls.
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  echo 'Python worker imports failed; loading the Python module and its library dependencies.' >&2
  module load "${AF_PYTHON_MODULE:-Python/3.11.5}"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
export AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-$AF_GROUP/phase-c-initialization-focused-v3}
AF_INPUTS=${AF_INITIALIZATION_INPUTS:-$AF_REPO_ROOT/inputs/saved-initialization-v3}
case "${1:?Use run, status, or inspect}" in
  run)
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    AF_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
    AF_SNAPSHOT=hub/models--openai--gpt-oss-20b/snapshots/$AF_REVISION/config.json
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      if [[ -f "$AF_GROUP/huggingface-cache/$AF_SNAPSHOT" ]]; then
        export AF_HF_HOME="$AF_GROUP/huggingface-cache"
      else
        export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
      fi
    fi
    [[ -f "$AF_HF_HOME/$AF_SNAPSHOT" ]] || { echo 'Pinned model is not present in AF_HF_HOME; set AF_HF_HOME to its existing cache.' >&2; exit 2; }
    "$AF_PYTHON" "$AF_CLI" prepare --inputs "$AF_INPUTS" --root "$AF_OUTPUT_ROOT" --image "$AF_VLLM_IMAGE"
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_saved_initialization_aces.py" --root "$AF_OUTPUT_ROOT" --wave "${AF_WAVE:-focused-1}"
    ;;
  status)
    squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %R'
    if [[ -f "$AF_OUTPUT_ROOT/summary.json" ]]; then
      "$AF_PYTHON" - "$AF_OUTPUT_ROOT/summary.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(json.dumps({k: p.get(k) for k in ('planned_episodes', 'groups', 'physical_requests', 'observed_tokens', 'unmeasured_requests', 'initialization_complete', 'interaction_sources_available')}, indent=2))
PY
    else
      echo 'No report yet. Check the queued job and the logs directory below.'
    fi
    printf 'Logs: %s/logs\n' "$AF_OUTPUT_ROOT"
    ;;
  inspect)
    "$AF_PYTHON" "$AF_CLI" inspect --root "$AF_OUTPUT_ROOT"
    cat "$AF_OUTPUT_ROOT/SUMMARY.md"
    printf '\nDownload: %s/inspection.tar.gz\n' "$AF_OUTPUT_ROOT"
    ;;
  *) echo 'Use run, status, or inspect.' >&2; exit 2 ;;
esac
