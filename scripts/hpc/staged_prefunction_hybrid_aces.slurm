#!/bin/bash
# One warm 20B server for the six-task hybrid pre-function readiness campaign.
#SBATCH --job-name=prefunction-hybrid
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --signal=B:TERM@300
#SBATCH --requeue

set -euo pipefail
module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}" WebProxy
exec bash "${AF_REPO_ROOT:?required}/scripts/hpc/run_staged_topology_server.sh"
