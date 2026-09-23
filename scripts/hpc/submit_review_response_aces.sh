#!/bin/bash
# Fresh, bounded recovery pilot. Keep existing experiments pinned and untouched.
set -euo pipefail
: "${AF_REPO_ROOT:=$(git rev-parse --show-toplevel)}"
: "${AF_PYTHON:=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
: "${AF_SOURCE_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-continuation-v2}"
: "${AF_OUTPUT_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/dalla-response-feedback-v1}"
: "${AF_COMPUTE_CACHE_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/runtime-cache/review}"
: "${AF_IPC_TMP_ROOT:=/tmp/af-ipc-u.yx126462}"
: "${AF_ADDITIONAL_VISITS:=3}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src"
if [[ -z "${AF_SOURCE_ROUND:-}" ]]; then
  if [[ -f "$AF_OUTPUT_ROOT/plan.json" ]]; then
    AF_SOURCE_ROUND="$(jq -er '.continuation.source_round' "$AF_OUTPUT_ROOT/plan.json")"
  else
    AF_SOURCE_ROUND="$("$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_response.py" latest-source --root "$AF_SOURCE_ROOT")"
  fi
fi
export AF_REPO_ROOT AF_PYTHON AF_SOURCE_ROOT AF_OUTPUT_ROOT AF_COMPUTE_CACHE_ROOT AF_IPC_TMP_ROOT
export AF_SOURCE_ROUND AF_ADDITIONAL_VISITS AF_CONTINUATION_PROTOCOL=review-deadline-7
mkdir -p "$AF_COMPUTE_CACHE_ROOT"
probe="$(mktemp "$AF_COMPUTE_CACHE_ROOT/write-check.XXXXXX")"
printf 'cache write check\n' > "$probe"
rm -- "$probe"
exec bash "$AF_REPO_ROOT/scripts/hpc/submit_review_continuation_aces.sh"
