#!/bin/bash
# One warm GPT-OSS-20B server plus bounded CPU fitting between revision calls.
#SBATCH --job-name=multiround-feedback
#SBATCH --partition=gpu_debug
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --signal=B:TERM@300
#SBATCH --requeue

set -euo pipefail
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
exec bash "${AF_REPO_ROOT:?required}/scripts/hpc/run_staged_topology_server.sh"
