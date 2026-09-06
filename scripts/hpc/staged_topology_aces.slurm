#!/bin/bash
# One warm server for the bounded implementation probe.
#SBATCH --job-name=staged-topology-probe
#SBATCH --partition=gpu_debug
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --signal=B:TERM@240
#SBATCH --requeue
set -euo pipefail
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
exec bash "${AF_REPO_ROOT:?required}/scripts/hpc/run_staged_topology_server.sh"
