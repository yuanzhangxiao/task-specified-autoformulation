#!/usr/bin/env bash
#SBATCH --job-name=iclr-search
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
set -euo pipefail
umask 077
: "${AF_REMAINDER_OPS:?}" "${AF_SOLVER_BACKUP:?}" "${AF_REPO_ROOT:?}" "${AF_SEARCH_ROOT:?}"
case ${SLURM_ARRAY_TASK_ID:?} in 2[4-9]|[34][0-9]|5[0-6]) ;; *) exit 2 ;; esac
(cd "$AF_REMAINDER_OPS" && sha256sum --quiet -c SHA256SUMS)
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REMAINDER_OPS:$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
env -u AF_JETSTREAM_API_KEY "$AF_PYTHON" "$AF_REMAINDER_OPS/scripts/iclr_stress_remainder.py" worker-start \
    --root "$AF_SEARCH_ROOT" --mirror "$AF_SOLVER_BACKUP" --ops "$AF_REMAINDER_OPS" --account "$AF_GPU_ACCOUNT"
bash "$AF_REPO_ROOT/scripts/hpc/run_iclr_controlled_search_aces.sh" "$@"
unset AF_JETSTREAM_API_KEY
"$AF_PYTHON" "$AF_REMAINDER_OPS/scripts/iclr_stress_solver_archive.py" \
    --root "$AF_SEARCH_ROOT" --mirror "$AF_SOLVER_BACKUP" --blocks "$SLURM_ARRAY_TASK_ID"
