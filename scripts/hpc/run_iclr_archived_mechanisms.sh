#!/usr/bin/env bash
# One ACES array element per original search visit. Run in a CPU allocation.
set -euo pipefail
: "${AF_PYTHON:?Set the project interpreter}"
: "${AF_REPO_ROOT:?Set the checkout path}"
: "${AF_ORIGINAL:?Set the original final-components-v1 directory}"
: "${AF_ASSESSMENTS:?Set a new writable assessment root}"
: "${AF_VISIT:=${SLURM_ARRAY_TASK_ID:?Use a 0-14 array or set AF_VISIT}}"
if [[ ! "$AF_VISIT" =~ ^([0-9]|1[0-4])$ ]]; then exit 2; fi
cd "$AF_REPO_ROOT"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
printf -v lane '%s/visit_%02d' "$AF_ASSESSMENTS" "$AF_VISIT"
"$AF_PYTHON" scripts/export_component_mechanisms.py --source "$AF_ORIGINAL" \
    --output "$lane/models" --round "$AF_VISIT"
"$AF_PYTHON" scripts/assess_functional_mechanisms.py prepare --bundle "$lane/models/models.json" \
    --public-root "$AF_ORIGINAL/public" --root "$lane/functional"
"$AF_PYTHON" - "$lane/functional" "${SLURM_CPUS_PER_TASK:-1}" <<'PY'
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import sys
from autoformalism.rebuttal import mechanism_functional_campaign as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read
root = Path(sys.argv[1])
count = len(sealed_read(root / "plan.json")["rows"])
with ProcessPoolExecutor(max_workers=int(sys.argv[2])) as pool:
    for _ in pool.map(campaign.run, repeat(root), range(count)):
        pass
PY
"$AF_PYTHON" scripts/assess_functional_mechanisms.py report --root "$lane/functional"
