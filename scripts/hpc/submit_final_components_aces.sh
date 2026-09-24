#!/bin/bash
# Run on ACES, with the supplied pinned source archive and completed API gate.
set -euo pipefail
: "${AF_REPO_ROOT:?Set the pinned source directory}"
: "${AF_JUDGE_GATE_ROOT:?Upload the completed Jetstream gate plan.json and summary.json}"
group=/scratch/group/p.nairr260351.000/u.yx126462
export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-$group/final-components-v1}"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-$group/component-runtime-cache}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-/tmp/af-ipc-u.yx126462}"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
if [[ -z "${AF_PUBLIC_ROOT:-}" ]]; then
  export AF_PUBLIC_ROOT="$group/final-component-inputs-v1"
  "$AF_PYTHON" scripts/component_campaign.py stage-public --root "$AF_PUBLIC_ROOT" \
    --source /scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1 \
    --source "$group/dalla-response-feedback-r14-v1/public"
fi
options=()
[[ -z "${AF_BLOCKS:-}" ]] || options+=(--blocks "$AF_BLOCKS")
"$AF_PYTHON" scripts/component_campaign.py prepare --root "$AF_OUTPUT_ROOT" \
  --public-root "$AF_PUBLIC_ROOT" "${options[@]}"
"$AF_PYTHON" scripts/component_campaign.py authorize-critic --root "$AF_OUTPUT_ROOT" \
  --calibration-plan "$AF_JUDGE_GATE_ROOT/plan.json" \
  --calibration-summary "$AF_JUDGE_GATE_ROOT/summary.json"
# Read on the submitting terminal. The key is inherited, never printed or saved.
if [[ -z "${AF_JETSTREAM_API_KEY:-}" ]]; then
  read -r -s -p 'Jetstream API key: ' AF_JETSTREAM_API_KEY
  printf '\n'
  export AF_JETSTREAM_API_KEY
fi
exec "$AF_PYTHON" scripts/submit_component_campaign.py --root "$AF_OUTPUT_ROOT" \
  --wave "${AF_WAVE:-wave1}" --proposers "${AF_PROPOSERS:-8}" \
  --critics "${AF_CRITICS:-4}" --fits "${AF_FIT_WORKERS:-32}" \
  --pruning "${AF_PRUNING_WORKERS:-8}"
