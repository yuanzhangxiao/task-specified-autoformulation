#!/bin/bash
# Corrected-development pilot only; no historical edits or automatic resubmission.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
"$AF_PYTHON" scripts/phase_c_baseline.py verify --root "$AF_OUTPUT_ROOT"
case "${1:?stage}" in
  propose)
    module load WebProxy
    exec bash scripts/hpc/run_staged_topology_server.sh ;;
  fit|assess)
    exec "$AF_PYTHON" scripts/phase_c_baseline.py "$1" --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  fit-assess)
    "$AF_PYTHON" scripts/phase_c_baseline.py fit --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    exec "$AF_PYTHON" scripts/phase_c_baseline.py assess --root "$AF_OUTPUT_ROOT" --index "$SLURM_ARRAY_TASK_ID" ;;
  report)
    exec "$AF_PYTHON" scripts/phase_c_baseline.py report --root "$AF_OUTPUT_ROOT" ;;
  *) exit 2 ;;
esac
