#!/bin/bash
# Submit the public R4 sign pilot from an immutable upload archive on Delta.
set -euo pipefail
export AF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/projects/bibo/yxiao2/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-/projects/bibo/yxiao2/huggingface-cache}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/dalla-sign-repair-delta-v2-r4}"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/dalla-sign-cache}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-/tmp/af-sign-$(id -u)}"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$AF_PYTHON" ]] || { echo "Missing Python: $AF_PYTHON" >&2; exit 2; }
[[ -s "$AF_VLLM_IMAGE" ]] || { echo "Missing vLLM 0.27.1 SIF: $AF_VLLM_IMAGE" >&2; exit 2; }
[[ -s "$AF_REPO_ROOT/inputs/rescue-models.json" ]] || { echo 'Missing original rescue packet in upload archive' >&2; exit 2; }
command -v sbatch >/dev/null
command -v jq >/dev/null
command -v apptainer >/dev/null || command -v singularity >/dev/null
echo 'Hashing the existing container, freezing the public R4 input, and submitting four jobs.'
exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_dalla_sign_repair.py" \
  --delta --config "$AF_REPO_ROOT/configs/dalla_sign_repair_v2.json" \
  --inputs "$AF_REPO_ROOT/inputs/rescue-models.json" --root "$AF_OUTPUT_ROOT"
