#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_PROFILE_ROOT=${AF_PROFILE_ROOT:-/work/hdd/bibo/yxiao2/phase_c/practical-uncertainty-supplement-v1}
AF_PROFILE_SOURCE=${AF_PROFILE_SOURCE:-/work/hdd/bibo/yxiao2/phase_c/practical-uncertainty-fitting-v1}
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to the existing fitting environment.' >&2; exit 1; }
[[ -f "$AF_PROFILE_SOURCE/plan.json" ]] || { echo "No M39 plan at $AF_PROFILE_SOURCE" >&2; exit 1; }
cd "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
"$AF_PYTHON" scripts/submit_phase_c_practical_uncertainty_supplement.py \
  --root "$AF_PROFILE_ROOT" --source "$AF_PROFILE_SOURCE" \
  --config "${AF_PROFILE_CONFIG:-$AF_CODE/configs/phase_c_practical_uncertainty_supplement_v1.json}" \
  --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-2}"
printf '\nResults: %s/summary.json\n' "$AF_PROFILE_ROOT"
