#!/bin/bash
# Saved fitted models only: no optimizer, provider calls or test data.
set -euo pipefail
repo="${AF_REPO_ROOT:?set AF_REPO_ROOT}"
root="${AF_OUTPUT_ROOT:?set AF_OUTPUT_ROOT}"
source_root="${AF_SOURCE_ROOT:?set AF_SOURCE_ROOT}"
python="${AF_PYTHON:?set AF_PYTHON}"
if [[ -f "$repo/SOURCE_COMMIT" ]]; then
  actual_commit=$(cat "$repo/SOURCE_COMMIT")
else
  actual_commit=$(git -C "$repo" rev-parse HEAD)
fi
[[ "$actual_commit" == "${AF_COMMIT:?set AF_COMMIT}" ]]
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:?stage required}" in
  prepare)
    choice=(--latest-retained)
    if [[ -n "${AF_SNAPSHOT_ROUND:-}" ]]; then
      choice=(--round "$AF_SNAPSHOT_ROUND")
    fi
    "$python" "$repo/scripts/export_component_mechanisms.py" \
      --source "$source_root" --output "$root/inputs" "${choice[@]}"
    "$python" "$repo/scripts/assess_functional_mechanisms.py" prepare \
      --bundle "$root/inputs/models.json" --public-root "$source_root/public" \
      --root "$root"
    ;;
  assess)
    "$python" "$repo/scripts/assess_functional_mechanisms.py" run \
      --root "$root" --index "${SLURM_ARRAY_TASK_ID:?array index required}"
    ;;
  assess-pool)
    "$python" "$repo/scripts/run_component_mechanism_pool.py" \
      --root "$root" --worker "${SLURM_ARRAY_TASK_ID:?array index required}" \
      --workers "${AF_MECHANISM_WORKERS:?worker count required}"
    ;;
  report)
    "$python" "$repo/scripts/assess_functional_mechanisms.py" report --root "$root"
    "$python" "$repo/scripts/summarize_functional_mechanisms.py" \
      --root "$root" --output "$root/mean-sd"
    cat "$root/mean-sd/SUMMARY.md"
    ;;
  *) echo 'Unknown stage' >&2; exit 2 ;;
esac
