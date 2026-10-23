#!/bin/bash
# Explicitly submit the bounded construction revision; preserve the original campaign.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" ]] || { echo 'Run on ACES with AF_GROUP set to writable scratch.' >&2; exit 2; }
# Generic AF_OUTPUT_ROOT may still name a previous fitting campaign in the
# interactive shell. Only the task-specific override selects this campaign.
export AF_REVISION_PROTOCOL=${AF_REVISION_PROTOCOL:-construction-feedback-revision-1}
case "$AF_REVISION_PROTOCOL" in
  construction-feedback-revision-1) AF_REVISION_DEFAULT=phase-c-construction-revision-v1 ;;
  construction-feedback-revision-2) AF_REVISION_DEFAULT=phase-c-construction-revision-checks-v2 ;;
  construction-feedback-revision-3) AF_REVISION_DEFAULT=phase-c-function-science-checks-v1 ;;
  *) echo 'Unsupported AF_REVISION_PROTOCOL.' >&2; exit 2 ;;
esac
export AF_REVISION_ROOT=${AF_REVISION_ROOT:-$AF_GROUP/$AF_REVISION_DEFAULT}
if [[ -n "${AF_OUTPUT_ROOT:-}" && "$AF_OUTPUT_ROOT" != "$AF_REVISION_ROOT" ]]; then
  printf 'Ignoring inherited AF_OUTPUT_ROOT=%s; revision output is %s.\n' "$AF_OUTPUT_ROOT" "$AF_REVISION_ROOT"
fi
export AF_OUTPUT_ROOT="$AF_REVISION_ROOT"
printf 'Construction revision output: %s\n' "$AF_OUTPUT_ROOT"
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
AF_CLI="$AF_REPO_ROOT/scripts/construction_feedback_revision.py"
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
case "${1:?Use run, status, report, or inspect}" in
  run)
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    AF_SNAPSHOT=hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      if [[ -f "$AF_GROUP/huggingface-cache/$AF_SNAPSHOT" ]]; then
        export AF_HF_HOME="$AF_GROUP/huggingface-cache"
      else
        export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
      fi
    fi
    [[ -f "$AF_HF_HOME/$AF_SNAPSHOT" && -f "$AF_VLLM_IMAGE" ]] || {
      echo 'Set AF_HF_HOME and AF_VLLM_IMAGE to the existing model cache and image.' >&2; exit 2;
    }
    export AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/harmony_vocab.py" verify --directory "$AF_HARMONY_ASSETS"
    AF_PREPARE_OPTIONS=(--protocol "$AF_REVISION_PROTOCOL")
    if [[ "$AF_REVISION_PROTOCOL" != construction-feedback-revision-1 ]]; then
      AF_PREPARE_OPTIONS+=(--consistency-evidence "${AF_REVISION_CONSISTENCY:-$AF_REPO_ROOT/inputs/consistency-evidence.json}")
    fi
    "$AF_PYTHON" "$AF_CLI" prepare --source "${AF_REVISION_INPUTS:-$AF_REPO_ROOT/inputs/revision-inputs.json}" --root "$AF_OUTPUT_ROOT" "${AF_PREPARE_OPTIONS[@]}"
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_construction_revision_aces.py" --root "$AF_OUTPUT_ROOT" --wave "${AF_WAVE:-revision-1}"
    ;;
  status)
    squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %R'
    if [[ -f "$AF_OUTPUT_ROOT/summary.json" ]]; then
      "$AF_PYTHON" - "$AF_OUTPUT_ROOT/summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(json.dumps({key: value.get(key) for key in ('status', 'status_counts', 'cost')}, indent=2))
PY
    else
      echo 'No summary yet; inspect the queued job and logs.'
    fi
    printf 'Last saved summary; refreshed when the worker exits. Logs: %s/logs\n' "$AF_OUTPUT_ROOT"
    ;;
  report)
    exec "$AF_PYTHON" "$AF_CLI" report --root "$AF_OUTPUT_ROOT"
    ;;
  inspect)
    "$AF_PYTHON" "$AF_CLI" inspect --root "$AF_OUTPUT_ROOT"
    printf 'Download: %s/inspection.tar.gz\n' "$AF_OUTPUT_ROOT"
    ;;
  *) echo 'Use run, status, report, or inspect.' >&2; exit 2 ;;
esac
