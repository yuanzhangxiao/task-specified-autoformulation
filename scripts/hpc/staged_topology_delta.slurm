#!/bin/bash
# Same frozen worker on one Delta A40.
#SBATCH --job-name=staged-topology-probe
#SBATCH --partition=gpuA40x4
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62G
#SBATCH --gpus-per-node=1
#SBATCH --signal=B:TERM@240
#SBATCH --requeue
set -euo pipefail
exec bash "${AF_REPO_ROOT:?required}/scripts/hpc/run_staged_topology_server.sh"
