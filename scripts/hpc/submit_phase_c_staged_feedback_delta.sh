#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_STAGED_FEEDBACK_ROOT=${AF_STAGED_FEEDBACK_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-staged-feedback-v1}
AF_STAGED_FEEDBACK_SOURCE=${AF_STAGED_FEEDBACK_SOURCE:-/work/hdd/bibo/yxiao2/phase_c/fitting-feedback-v1}
[[ -x "$AF_PYTHON" ]] || { echo "Set AF_PYTHON to the existing fitting environment." >&2; exit 1; }
cd "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
"$AF_PYTHON" -c 'import casadi, scipy; print("CasADi", casadi.__version__, "SciPy", scipy.__version__)'
if [[ -n "${AF_STAGED_FEEDBACK_INPUTS:-}" ]]; then
  [[ -f "$AF_STAGED_FEEDBACK_INPUTS" ]] || { echo "AF_STAGED_FEEDBACK_INPUTS must name an existing sealed export." >&2; exit 1; }
elif [[ -f "$AF_CODE/staged-feedback-inputs.json" ]]; then
  AF_STAGED_FEEDBACK_INPUTS="$AF_CODE/staged-feedback-inputs.json"
else
  AF_STAGED_FEEDBACK_INPUTS="$AF_STAGED_FEEDBACK_ROOT/exported-inputs.json"
  if [[ ! -f "$AF_STAGED_FEEDBACK_INPUTS" ]]; then
    "$AF_PYTHON" scripts/phase_c_staged_feedback.py export --source "$AF_STAGED_FEEDBACK_SOURCE" --output "$AF_STAGED_FEEDBACK_INPUTS"
  fi
fi
"$AF_PYTHON" scripts/submit_phase_c_staged_feedback.py --root "$AF_STAGED_FEEDBACK_ROOT" --inputs "$AF_STAGED_FEEDBACK_INPUTS" --config "${AF_STAGED_FEEDBACK_CONFIG:-$AF_CODE/configs/phase_c_staged_feedback_v1.json}" --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-3}"
printf '\nResults: %s/summary.json\n' "$AF_STAGED_FEEDBACK_ROOT"
