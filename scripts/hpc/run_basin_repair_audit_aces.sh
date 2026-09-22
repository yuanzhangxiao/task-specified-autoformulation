#!/bin/bash
#SBATCH --job-name=basin-repair-audit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_SOURCE_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
"$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_basin_repair_v2.py tests/test_basin_repair_audit.py
exec "$AF_PYTHON" scripts/audit_basin_repairs.py --source "$AF_SOURCE_ROOT" --output "$AF_OUTPUT_ROOT"
