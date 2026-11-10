#!/usr/bin/env bash
#SBATCH --job-name=iclr-pilot-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set +x
set -euo pipefail
umask 077
: "${AF_CAMPAIGN:?Source the campaign activation file in your ACES login shell}"
: "${AF_DELIVERY:?Set by the campaign activation file}"
: "${AF_PYTHON:?Set by the campaign activation file}"
export AF_REPO_ROOT="$AF_DELIVERY/code"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-submit}" in
    submit)
        helper="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/iclr_aces_pilot.py"
        cd "$AF_REPO_ROOT"
        if [[ -z ${AF_JETSTREAM_API_KEY:-} ]]; then
            read -r -s -p 'Jetstream critic key: ' AF_JETSTREAM_API_KEY </dev/tty
            printf '\n' >/dev/tty
            export AF_JETSTREAM_API_KEY
        fi
        "$AF_PYTHON" "$helper" submit --cached-pilot --root "$AF_CAMPAIGN/pilot" \
            --wave "${2:-cpu-recovery1}" --concurrency 1 \
            --account "${AF_CPU_ACCOUNT:-${AF_ACCOUNT:-156264627414}}"
        ;;
    run)
        : "${SLURM_JOB_ID:?Submit this worker through the ACES scheduler}"
        : "${AF_PILOT_RESUME_CLI:?Missing submission helper}"
        [[ ${SLURM_ARRAY_TASK_ID:?} == 0 && ${AF_SEARCH_ROOT:?} == "$AF_CAMPAIGN/pilot" ]]
        cd "$AF_REPO_ROOT"
        module load WebProxy
        "$AF_PYTHON" "$AF_PILOT_RESUME_CLI" run-cached --root "$AF_SEARCH_ROOT" --seconds 9900
        ;;
    *) echo 'Use submit [WAVE] from the login shell.' >&2; exit 2 ;;
esac
