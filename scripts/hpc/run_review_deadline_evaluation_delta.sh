#!/bin/bash
# Frozen models only; no fitting or proposer work on this evaluation worker.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PUBLIC_ROOT:?}" "${AF_COMMIT:?}"
[[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" == "$AF_COMMIT" ]] || exit 2
export PYTHONPATH="$AF_REPO_ROOT/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${1:-evaluate}" in
 evaluate)
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" evaluate \
      --root "$AF_OUTPUT_ROOT" --public-root "$AF_PUBLIC_ROOT" \
      --shard "${SLURM_ARRAY_TASK_ID:?}" --shards "${AF_EVAL_SHARDS:?}"
    ;;
 report) exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" evaluation-report --root "$AF_OUTPUT_ROOT" ;;
 *) exit 2 ;;
esac
