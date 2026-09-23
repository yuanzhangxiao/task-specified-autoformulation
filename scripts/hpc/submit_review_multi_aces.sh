#!/bin/bash
# Separate twelve-visit continuation; never update a running scientific checkout.
set -euo pipefail
: "${AF_REPO_ROOT:=$(git rev-parse --show-toplevel)}"
: "${AF_SOURCE_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-pilots-v1}"
: "${AF_OUTPUT_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-continuation-v2}"
: "${AF_COMPUTE_CACHE_ROOT:=/scratch/group/p.nairr260351.000/u.yx126462/runtime-cache/review}"
: "${AF_IPC_TMP_ROOT:=/tmp/af-ipc-u.yx126462}"
: "${AF_SOURCE_ROUND:=2}" "${AF_ADDITIONAL_VISITS:=12}"
export AF_REPO_ROOT AF_SOURCE_ROOT AF_OUTPUT_ROOT AF_COMPUTE_CACHE_ROOT AF_IPC_TMP_ROOT
export AF_SOURCE_ROUND AF_ADDITIONAL_VISITS AF_CONTINUATION_PROTOCOL=review-deadline-6
mkdir -p "$AF_COMPUTE_CACHE_ROOT"
probe="$(mktemp "$AF_COMPUTE_CACHE_ROOT/write-check.XXXXXX")"
printf 'cache write check\n' > "$probe"
rm -- "$probe"
exec bash "$AF_REPO_ROOT/scripts/hpc/submit_review_continuation_aces.sh"
