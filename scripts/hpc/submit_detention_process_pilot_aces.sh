#!/bin/bash
# Run from an ACES login shell; source release is the qualified public basin data.
set -euo pipefail
: "${AF_REPO_ROOT:?set the pinned source directory}"
group=/scratch/group/p.nairr260351.000/u.yx126462
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-$group/detention-process-pilot-v1}"
export AF_PUBLIC_ROOT="${AF_PUBLIC_ROOT:-$group/detention-development-v1}"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-$group/shared-runtime-cache}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-$group/af-ipc}"
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
[[ -x "$AF_PYTHON" && -f "$AF_VLLM_IMAGE" ]] || { echo 'Missing Python or serving image' >&2; exit 2; }
bash scripts/hpc/run_staged_topology_server.sh --check-config configs/detention_process_pilot_v1.json
"$AF_PYTHON" scripts/detention_process_pilot.py prepare --source "$AF_PUBLIC_ROOT" --root "$AF_OUTPUT_ROOT"
exec "$AF_PYTHON" scripts/submit_detention_process_pilot.py --root "$AF_OUTPUT_ROOT"
