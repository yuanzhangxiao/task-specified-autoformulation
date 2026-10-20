#!/bin/bash
# M47: independent CPU experiment; no previous campaign is required.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_CODE/src:$AF_CODE${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
AF_PREFIX_NOISE_ROOT=${AF_PREFIX_NOISE_ROOT:-/work/hdd/bibo/yxiao2/phase_c/prefix-noise-reliability-v1}
AF_PREFIX_NOISE_CONFIG=${AF_PREFIX_NOISE_CONFIG:-$AF_CODE/configs/phase_c_prefix_noise_reliability_v1.json}
printf 'M47 code: %s\nOutput: %s\n' "$AF_CODE" "$AF_PREFIX_NOISE_ROOT"
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to the existing fitting environment.' >&2; exit 1; }
[[ -f "$AF_PREFIX_NOISE_CONFIG" ]] || { echo "No M47 config at $AF_PREFIX_NOISE_CONFIG" >&2; exit 1; }
printf 'M47: checking code integrity...\n'
cd "$AF_CODE"
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
printf 'M47: checking numerical imports...\n'
"$AF_PYTHON" -u -c 'import casadi, scipy; print("CasADi", casadi.__version__, "SciPy", scipy.__version__)'
"$AF_PYTHON" -u scripts/submit_phase_c_prefix_noise_reliability.py \
  --root "$AF_PREFIX_NOISE_ROOT" --config "$AF_PREFIX_NOISE_CONFIG" \
  --account "${AF_ACCOUNT:-bibo-delta-cpu}" --concurrency "${AF_CONCURRENCY:-2}"
printf '\nFull report: %s/summary.json\n' "$AF_PREFIX_NOISE_ROOT"
