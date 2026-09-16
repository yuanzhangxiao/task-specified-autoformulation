#!/bin/bash
# Import retained round two and queue a fresh bounded development phase.
set -euo pipefail
: "${AF_REPO_ROOT:=$(git rev-parse --show-toplevel)}"
: "${AF_PYTHON:=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
: "${AF_SOURCE_ROOT:=/scratch/user/u.yx126462/phase_b/review-deadline-v2}"
: "${AF_OUTPUT_ROOT:=/scratch/user/u.yx126462/phase_b/review-continuation-v3}"
: "${AF_SOURCE_ROUND:=2}" "${AF_ADDITIONAL_VISITS:=5}"
: "${AF_VLLM_IMAGE:=/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=/scratch/user/u.yx126462/huggingface-cache}"
: "${AF_COMPUTE_CACHE_ROOT:=/scratch/user/u.yx126462/autoformalism-runtime-cache/review}"
: "${AF_IPC_TMP_ROOT:=/scratch/user/u.yx126462/af-ipc}"
module load GCCcore/13.2.0 Python/3.11.5
[[ -x "$AF_PYTHON" && -f "$AF_VLLM_IMAGE" ]] || { echo 'Missing Python or image' >&2; exit 2; }
[[ -z "$(git -C "$AF_REPO_ROOT" status --porcelain)" ]] || { echo 'Use a clean pinned checkout' >&2; exit 2; }
export AF_REPO_ROOT AF_PYTHON AF_OUTPUT_ROOT AF_VLLM_IMAGE AF_HF_HOME
export AF_COMPUTE_CACHE_ROOT AF_IPC_TMP_ROOT
export AF_COMMIT="$(git -C "$AF_REPO_ROOT" rev-parse HEAD)"
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" prepare-continuation \
  --source "$AF_SOURCE_ROOT" --source-round "$AF_SOURCE_ROUND" \
  --visits "$AF_ADDITIONAL_VISITS" --root "$AF_OUTPUT_ROOT"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_review_continuation.py" \
  --root "$AF_OUTPUT_ROOT" --round 1
