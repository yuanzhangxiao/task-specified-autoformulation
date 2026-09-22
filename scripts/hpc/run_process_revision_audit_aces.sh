#!/bin/bash
#SBATCH --job-name=process-revision-audit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
: "${AF_REPO_ROOT:?set pinned source directory}"
: "${AF_SOURCE_ROOT:?set completed v6 pilot directory}"
: "${AF_OUTPUT_ROOT:?set separate audit directory}"
: "${AF_PYTHON:?set existing Python environment}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
exec "$AF_PYTHON" scripts/audit_process_revisions.py \
  --source "$AF_SOURCE_ROOT" --output "$AF_OUTPUT_ROOT"
