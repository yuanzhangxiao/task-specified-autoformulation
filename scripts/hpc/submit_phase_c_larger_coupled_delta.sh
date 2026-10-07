#!/bin/bash
# Use the frozen larger linear controls shipped with the portable bundle.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_LARGER_COUPLED_ROOT=${AF_LARGER_COUPLED_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-larger-coupled-v1}
[[ -x "$AF_PYTHON" ]] || { echo "Set AF_PYTHON to the existing fitting environment." >&2; exit 1; }
cd "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
"$AF_PYTHON" -c 'import casadi, scipy; print("CasADi", casadi.__version__, "SciPy", scipy.__version__)'
"$AF_PYTHON" scripts/submit_phase_c_larger_coupled.py --root "$AF_LARGER_COUPLED_ROOT" --inputs "${AF_LARGER_COUPLED_INPUTS:-$AF_CODE/larger-coupled-inputs.json}" --config "$AF_CODE/configs/phase_c_larger_coupled_v1.json" --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-6}"
printf '\nResults: %s/summary.json\n' "$AF_LARGER_COUPLED_ROOT"
