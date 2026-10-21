#!/bin/bash
# Reporting-only sidecar: scientific reconstruction uses the frozen M47 source.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
: "${AF_REPORT_HELPER_ROOT:?}"
printf 'M47 report stage=%s host=%s time=%s\n' "${1:?check or reduce}" "$(hostname)" "$(date -Is)"
cd "$AF_REPORT_HELPER_ROOT"
printf 'Checking reporting helper checksums...\n'
sha256sum -c SHA256SUMS --quiet
cd "$AF_REPO_ROOT"
[[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
# The Python reporter verifies the complete original source inventory once.
# Rehashing it here would duplicate shared-filesystem reads for every shard.
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$1" in
  check)
    "$AF_PYTHON" -u "$AF_REPORT_HELPER_ROOT/report_prefix_noise_sharded.py" \
      check --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  reduce)
    "$AF_PYTHON" -u "$AF_REPORT_HELPER_ROOT/report_prefix_noise_sharded.py" \
      reduce --root "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
