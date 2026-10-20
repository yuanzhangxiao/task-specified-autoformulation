#!/bin/bash
# M45: an independent CPU supplement for incomplete M44 numerical operations.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_PREFIX_WINDOW_SUPPLEMENT_ROOT=${AF_PREFIX_WINDOW_SUPPLEMENT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/prefix-window-supplement-v1}
AF_PREFIX_WINDOW_SOURCE=${AF_PREFIX_WINDOW_SOURCE:-/work/hdd/bibo/yxiao2/phase_c/prefix-window-selection-v1}
printf 'M45 code: %s\nSource: %s\nOutput: %s\n' "$AF_CODE" "$AF_PREFIX_WINDOW_SOURCE" "$AF_PREFIX_WINDOW_SUPPLEMENT_ROOT"
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to the existing fitting environment.' >&2; exit 1; }
printf 'M45: checking source plan and code integrity...\n'
[[ -f "$AF_PREFIX_WINDOW_SOURCE/plan.json" ]] || { echo "No M44 plan at $AF_PREFIX_WINDOW_SOURCE" >&2; exit 1; }
cd "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
printf 'M45: checking numerical imports...\n'
"$AF_PYTHON" -u -c 'import casadi, scipy; print("CasADi", casadi.__version__, "SciPy", scipy.__version__)'
"$AF_PYTHON" -u scripts/submit_phase_c_prefix_window_supplement.py \
  --root "$AF_PREFIX_WINDOW_SUPPLEMENT_ROOT" --source "$AF_PREFIX_WINDOW_SOURCE" \
  --config "${AF_PREFIX_WINDOW_SUPPLEMENT_CONFIG:-$AF_CODE/configs/phase_c_prefix_window_supplement_v1.json}" \
  --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-2}"
printf '\nFull report: %s/summary.json\n' "$AF_PREFIX_WINDOW_SUPPLEMENT_ROOT"
