#!/bin/bash
# M47: isolated prefix noise-reliability qualification with frozen synthetic inputs.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]] || {
    echo 'M47 source commit differs from the submitted identity.' >&2; exit 1;
  }
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]] || {
    echo 'M47 checkout commit differs from the submitted identity.' >&2; exit 1;
  }
fi
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:?prepare, run or report}" in
  prepare)
    "$AF_PYTHON" -u scripts/phase_c_prefix_noise_reliability.py verify --root "$AF_OUTPUT_ROOT"
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider \
      tests/test_prefix_initialization.py \
      tests/test_prefix_multiroot_controls.py \
      tests/test_prefix_execution.py \
      tests/test_prefix_multiroot_campaign.py \
      tests/test_prefix_portable_comparison.py \
      tests/test_prefix_predictive_support.py \
      tests/test_prefix_noise_reliability.py \
      tests/test_submit_prefix_noise_reliability.py
    ;;
  run)
    "$AF_PYTHON" -u scripts/phase_c_prefix_noise_reliability.py run \
      --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    "$AF_PYTHON" -u scripts/phase_c_prefix_noise_reliability.py report --root "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
