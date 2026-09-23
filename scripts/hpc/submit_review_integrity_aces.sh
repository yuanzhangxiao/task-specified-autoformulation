#!/bin/bash
# Import the completed response-feedback checkpoint into a fresh v8 phase.
set -euo pipefail
: "${AF_REPO_ROOT:=$(git rev-parse --show-toplevel)}"
: "${AF_SOURCE_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/dalla-response-feedback-r14-v1}"
: "${AF_OUTPUT_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/dalla-review-integrity-r17-v1}"
: "${AF_SOURCE_ROUND:=17}" "${AF_ADDITIONAL_VISITS:=3}"
: "${AF_COMPUTE_CACHE_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/runtime-cache/review}"
: "${AF_IPC_TMP_ROOT:=/tmp/af-ipc-u.yx126462}"
export AF_REPO_ROOT AF_SOURCE_ROOT AF_OUTPUT_ROOT AF_SOURCE_ROUND AF_ADDITIONAL_VISITS
export AF_COMPUTE_CACHE_ROOT AF_IPC_TMP_ROOT AF_CONTINUATION_PROTOCOL=review-deadline-8
exec bash "$AF_REPO_ROOT/scripts/hpc/submit_review_continuation_aces.sh"
