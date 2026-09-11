#!/bin/bash
# Run or summarize one frozen collocation-node comparison on ACES CPUs.
#SBATCH --job-name=collocation-nodes
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_CODE_COMMIT:?}"
module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_CODE_COMMIT}" && -z "$(git status --porcelain)" ]] || {
  echo 'Checkout changed after submission.' >&2
  exit 2
}
export PYTHONPATH="${AF_REPO_ROOT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
if [[ "${1:-run}" == summarize ]]; then
  "${AF_PYTHON}" scripts/run_collocation_node_campaign.py summarize --output "${AF_OUTPUT_ROOT}"
else
  "${AF_PYTHON}" scripts/run_collocation_node_campaign.py run --output "${AF_OUTPUT_ROOT}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
fi
