#!/bin/bash
# Run on Delta; reuse existing Python/dependency installs without GPU allocation.
set -euo pipefail
export AF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-m1}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to an existing project Python.' >&2; exit 2; }
if [[ -z "${AF_CASADI_ROOT:-}" && -d /projects/bibo/yxiao2/venvs/fitter-methods-v1-deps ]]; then
  export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
fi
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
"$AF_PYTHON" - <<'PY'
import inspect
import casadi, numpy, scipy, pydantic, pytest
from scipy.optimize import least_squares
if 'callback' not in inspect.signature(least_squares).parameters:
    raise SystemExit('This fitter requires SciPy >= 1.16 with least_squares callbacks.')
print(f'Numerical dependencies available: SciPy {scipy.__version__}, CasADi {casadi.__version__}')
PY
[[ -f "$AF_REPO_ROOT/inputs/fitting-inputs.json" ]] || { echo 'Missing sealed input bundle.' >&2; exit 2; }
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_phase_c_fitting.py" \
  --root "$AF_OUTPUT_ROOT" --inputs "$AF_REPO_ROOT/inputs/fitting-inputs.json" \
  --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-2}"
