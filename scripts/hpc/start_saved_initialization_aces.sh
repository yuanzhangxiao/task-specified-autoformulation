#!/bin/bash
# The run preflight needs an ACES compute node; status/inspect work on login nodes.
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
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
export AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-$AF_GROUP/phase-c-initialization-focused-v3}
AF_INPUTS=${AF_INITIALIZATION_INPUTS:-$AF_REPO_ROOT/inputs/saved-initialization-v3}
case "${1:?Use run, status, or inspect}" in
  run)
    if ! AF_CONTAINER=$(command -v apptainer || command -v singularity); then
      echo 'No Apptainer or Singularity executable is available on this node. Nothing was prepared or submitted.' >&2
      echo 'ACES provides container runtimes on compute nodes. Run this same bundle command in a CPU allocation, for example:' >&2
      printf 'srun --partition=cpu --account=%q --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 --export=ALL bash /path/to/run_recovery.sh run\n' "${AF_ACCOUNT:-156264627414}" >&2
      echo 'If already on a compute node, report the node name and missing runtime to HPRC support.' >&2
      exit 2
    fi
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
    export AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
    AF_HARMONY_CLI="$AF_REPO_ROOT/scripts/harmony_vocab.py"
    "$AF_PYTHON" "$AF_HARMONY_CLI" verify --directory "$AF_HARMONY_ASSETS" >/dev/null
    # This CPU-only render uses the pinned image and makes no generation request.
    "$AF_CONTAINER" exec \
      --bind "$AF_HARMONY_ASSETS:$AF_HARMONY_ASSETS:ro,$AF_HARMONY_CLI:$AF_HARMONY_CLI:ro" \
      --env "TIKTOKEN_ENCODINGS_BASE=$AF_HARMONY_ASSETS" --env PYTHONDONTWRITEBYTECODE=1 \
      "$AF_VLLM_IMAGE" python3 "$AF_HARMONY_CLI" probe --directory "$AF_HARMONY_ASSETS"
    "$AF_PYTHON" "$AF_CLI" prepare --inputs "$AF_INPUTS" --root "$AF_OUTPUT_ROOT" --image "$AF_VLLM_IMAGE"
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_saved_initialization_aces.py" --root "$AF_OUTPUT_ROOT" --wave "${AF_WAVE:-focused-1}"
    ;;
  status)
    squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %R'
    if [[ -f "$AF_OUTPUT_ROOT/summary.json" ]]; then
      "$AF_PYTHON" - "$AF_OUTPUT_ROOT/summary.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
keys = ('planned_episodes', 'groups', 'physical_requests', 'observed_tokens', 'unmeasured_requests', 'initialization_complete', 'interaction_sources_available')
keys += ('provider_recovery_policy', 'inherited_cost', 'new_cost', 'cumulative_cost')
print(json.dumps({k: p[k] for k in keys if k in p}, indent=2))
PY
    else
      echo 'No report yet. Check the queued job and the logs directory below.'
    fi
    printf 'Logs: %s/logs\n' "$AF_OUTPUT_ROOT"
    ;;
  inspect)
    [[ -f "$AF_OUTPUT_ROOT/plan.json" ]] || {
      printf 'No prepared plan: %s/plan.json\nRun the bundle with run and check its launch output before inspecting.\n' "$AF_OUTPUT_ROOT" >&2
      exit 2
    }
    "$AF_PYTHON" "$AF_CLI" inspect --root "$AF_OUTPUT_ROOT"
    cat "$AF_OUTPUT_ROOT/SUMMARY.md"
    printf '\nDownload: %s/inspection.tar.gz\n' "$AF_OUTPUT_ROOT"
    ;;
  *) echo 'Use run, status, or inspect.' >&2; exit 2 ;;
esac
