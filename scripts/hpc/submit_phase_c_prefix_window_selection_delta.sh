#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE:$AF_CASADI_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_PREFIX_WINDOW_ROOT=${AF_PREFIX_WINDOW_ROOT:-/work/hdd/bibo/yxiao2/phase_c/prefix-window-selection-v1}
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to the existing fitting environment.' >&2; exit 1; }
cd "$AF_CODE"
printf 'M44: checking source integrity in %s\n' "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
"$AF_PYTHON" -u scripts/submit_phase_c_prefix_window_selection.py \
  --root "$AF_PREFIX_WINDOW_ROOT" \
  --config "${AF_PREFIX_WINDOW_CONFIG:-$AF_CODE/configs/phase_c_prefix_window_selection_v1.json}" \
  --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-2}"
printf '\nFull report: %s/summary.json\n' "$AF_PREFIX_WINDOW_ROOT"
