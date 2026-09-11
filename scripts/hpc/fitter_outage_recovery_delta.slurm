#!/bin/bash
#SBATCH --job-name=fitter-recovery
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
set -euo pipefail
# This marker precedes any Python, /projects access or numerical imports.
echo "recovery shell-start job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} stage=${1:-run}"
: "${AF_BOOTSTRAP_PYTHON:?}" "${AF_RECOVERY_DRIVER:?}" "${AF_RECOVERY_OUTPUT:?}"
stage="${1:-run}"
args=("${stage}" --output "${AF_RECOVERY_OUTPUT}")
seconds=3450
if [[ "${stage}" == run ]]; then
  args+=(--task-index "${SLURM_ARRAY_TASK_ID:?}")
elif [[ "${stage}" == smoke ]]; then
  seconds=540
elif [[ "${stage}" == summarize ]]; then
  seconds=1140
else
  echo 'Unknown recovery stage.' >&2; exit 2
fi
# The driver is on /work and needs only the system standard library. Its child
# owns the original /projects Python and numerical source, with a shorter timeout.
timeout --signal=TERM --kill-after=15s "${seconds}s" \
  "${AF_BOOTSTRAP_PYTHON}" "${AF_RECOVERY_DRIVER}" "${args[@]}"
