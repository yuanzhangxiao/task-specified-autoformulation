#!/bin/bash
# Invoke from an existing ACES shell; all new artifacts use group scratch.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_COMMIT:?}"
group=/scratch/group/p.nairr260351.000/u.yx126462
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-$group/process-pruning-v1}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-$group/general-critic-v1}"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-$group/huggingface-cache}"
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Keep the revision pinned on repeat submissions; prefer an existing cached model.
if [[ -f "$AF_OUTPUT_ROOT/submission_manifest.json" ]]; then
  export AF_JUDGE_REVISION="$(jq -er '.identity.judge_revision' "$AF_OUTPUT_ROOT/submission_manifest.json")"
elif [[ -z "${AF_JUDGE_REVISION:-}" ]]; then
  for cache in "$AF_HF_HOME" /scratch/user/u.yx126462/huggingface-cache; do
    if [[ -f "$cache/hub/models--openai--gpt-oss-120b/refs/main" ]]; then
      export AF_JUDGE_REVISION="$(cat "$cache/hub/models--openai--gpt-oss-120b/refs/main")"
      break
    fi
  done
fi
if [[ -z "${AF_JUDGE_REVISION:-}" ]]; then
  export AF_JUDGE_REVISION="$(curl --fail --silent --show-error https://huggingface.co/api/models/openai/gpt-oss-120b | jq -er '.sha')"
fi
[[ "$AF_JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo 'Set a full AF_JUDGE_REVISION' >&2; exit 2; }
exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_general_critic.py" --source "$AF_SOURCE_ROOT" --root "$AF_OUTPUT_ROOT"
