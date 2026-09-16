#!/bin/bash
# All phases are CPU-only and never request test access or model generation.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
[[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" == "$AF_COMMIT" ]] || exit 2
git -C "$AF_REPO_ROOT" diff --quiet
git -C "$AF_REPO_ROOT" diff --cached --quiet
export PYTHONPATH="$AF_REPO_ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cli="$AF_REPO_ROOT/scripts/baseline_validation.py"
case "${1:-run}" in
  prepare)
    "$AF_PYTHON" "$cli" inventory \
      --root "$AF_CLASSICAL_ROOT" --root "$AF_RAW_ROOT" \
      --root "$AF_REFRESH_ROOT" --root "$AF_D3_ROOT" \
      --output "$AF_OUTPUT_ROOT/inventory.json"
    "$AF_PYTHON" "$cli" prepare --inventory "$AF_OUTPUT_ROOT/inventory.json" \
      --roster "${AF_ROSTER:-$AF_REPO_ROOT/configs/baseline_validation_v1.json}" \
      --public-root "$AF_PUBLIC_ROOT" --legacy-root "$AF_LEGACY_ROOT" \
      --output-root "$AF_OUTPUT_ROOT"
    ;;
  run)
    "$AF_PYTHON" "$cli" run --root "$AF_OUTPUT_ROOT" \
      --shard "${SLURM_ARRAY_TASK_ID:?}" --shards 8
    ;;
  report) "$AF_PYTHON" "$cli" report --root "$AF_OUTPUT_ROOT" ;;
  *) exit 2 ;;
esac
