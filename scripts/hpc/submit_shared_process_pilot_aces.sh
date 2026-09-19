#!/bin/bash
# Invoke from the ACES login shell, using a pinned checkout/archive in group scratch.
set -euo pipefail
round="${1:-0}"
[[ "$#" -le 1 && "$round" =~ ^[01]$ ]] || { echo 'Usage: submit_shared_process_pilot_aces.sh [0|1]' >&2; exit 2; }
: "${AF_REPO_ROOT:?set the pinned source directory}"
group="/scratch/group/p.nairr260351.000/u.yx126462"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-$group/shared-process-pilot-v1}"
export AF_PUBLIC_ROOT="${AF_PUBLIC_ROOT:-/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1}"
export AF_CONFIG="$AF_REPO_ROOT/configs/shared_process_pilot_v1.json"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-$group/shared-runtime-cache}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-$group/af-ipc}"
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
[[ -x "$AF_PYTHON" && -f "$AF_VLLM_IMAGE" ]] || { echo 'Missing Python or serving image' >&2; exit 2; }
bash scripts/hpc/run_staged_topology_server.sh --check-config "$AF_CONFIG"
# Fail before copying any assets or creating submission intent.
"$AF_PYTHON" - "$AF_CONFIG" "$AF_PUBLIC_ROOT" <<'PY'
import json, sys
from pathlib import Path
from autoformalism.rebuttal import review_deadline_io as io
config = io.DeadlineConfig.model_validate_json(Path(sys.argv[1]).read_text())
assert config.protocol == io.SHARED_PROTOCOL
missing = [str(Path(sys.argv[2])/'phase_b_v1'/cell/name)
           for cell in config.public_cells for name in io.FILES
           if not (Path(sys.argv[2])/'phase_b_v1'/cell/name).is_file()]
if missing:
    raise SystemExit('Missing public development files; set AF_PUBLIC_ROOT:\n'+'\n'.join(missing))
print(json.dumps({'lineages':len(io.tasks(config)), 'rounds':config.rounds,
                  'fit_profile':config.fit_profile}))
PY
"$AF_PYTHON" scripts/review_deadline.py prepare --config "$AF_CONFIG" --public-root "$AF_PUBLIC_ROOT" --root "$AF_OUTPUT_ROOT"
exec "$AF_PYTHON" scripts/submit_shared_process_pilot.py --root "$AF_OUTPUT_ROOT" --round "$round"
