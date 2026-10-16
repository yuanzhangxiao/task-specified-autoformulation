#!/bin/bash
# M40: import preserved M39 results and run only the nine unfinished diagnostics.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
printf 'M40 ACES: loading the existing scientific environment\n'
module load GCCcore/13.2.0 Python/3.11.5
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_CODE/src:$AF_CODE${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_PROFILE_ROOT=${AF_PROFILE_ROOT:-$AF_GROUP/phase_c/practical-uncertainty-supplement-aces-v1}
: "${AF_PROFILE_SOURCE:?Set AF_PROFILE_SOURCE to the imported M39 results directory}"
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to the existing fitting environment.' >&2; exit 1; }
[[ -f "$AF_PROFILE_SOURCE/plan.json" ]] || { echo "No M39 plan at $AF_PROFILE_SOURCE" >&2; exit 1; }
cd "$AF_CODE"
printf 'M40 ACES: checking code integrity in %s\n' "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
printf 'M40 ACES: checking numerical imports\n'
"$AF_PYTHON" -u -c 'import numpy, pydantic, pytest, scipy; print("SciPy", scipy.__version__, "NumPy", numpy.__version__, flush=True)'
printf 'M40 ACES: source=%s; output=%s\n' "$AF_PROFILE_SOURCE" "$AF_PROFILE_ROOT"
"$AF_PYTHON" -u scripts/submit_phase_c_practical_uncertainty_supplement.py \
  --platform aces --root "$AF_PROFILE_ROOT" --source "$AF_PROFILE_SOURCE" \
  --config "${AF_PROFILE_CONFIG:-$AF_CODE/configs/phase_c_practical_uncertainty_supplement_v1.json}" \
  --account "${AF_ACCOUNT:-156264627414}" --concurrency "${AF_CONCURRENCY:-2}"
printf '\nResults: %s/summary.json\n' "$AF_PROFILE_ROOT"
