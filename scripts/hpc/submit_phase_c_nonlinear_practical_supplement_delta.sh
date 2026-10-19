#!/bin/bash
# M42: additional diagnostics at retained M41 fitted parameters; CPU only.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_NONLINEAR_SUPPLEMENT_ROOT=${AF_NONLINEAR_SUPPLEMENT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-supplement-v1}
AF_NONLINEAR_SOURCE=${AF_NONLINEAR_SOURCE:-/work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-uncertainty-v1}
printf 'M42 code: %s\nSource: %s\nOutput: %s\n' "$AF_CODE" "$AF_NONLINEAR_SOURCE" "$AF_NONLINEAR_SUPPLEMENT_ROOT"
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to the existing fitting environment.' >&2; exit 1; }
printf 'M42: checking source plan and code integrity...\n'
[[ -f "$AF_NONLINEAR_SOURCE/plan.json" ]] || { echo "No M41 plan at $AF_NONLINEAR_SOURCE" >&2; exit 1; }
cd "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
printf 'M42: checking numerical imports...\n'
"$AF_PYTHON" -u -c 'import casadi, scipy; print("CasADi", casadi.__version__, "SciPy", scipy.__version__)'
"$AF_PYTHON" -u scripts/submit_phase_c_nonlinear_practical_supplement.py \
  --root "$AF_NONLINEAR_SUPPLEMENT_ROOT" --source "$AF_NONLINEAR_SOURCE" \
  --config "${AF_NONLINEAR_SUPPLEMENT_CONFIG:-$AF_CODE/configs/phase_c_nonlinear_practical_supplement_v1.json}" \
  --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-2}"
printf '\nResults: %s/summary.json\n' "$AF_NONLINEAR_SUPPLEMENT_ROOT"
