#!/usr/bin/env bash
# Optional CPU controls: canonical CSTR equations fitted to the perturbed public data.
set -eo pipefail
umask 077
action=${1:-}
platform=${AF_PLATFORM:-delta}
if [[ $action == worker ]]; then
    : "${AF_REPO_ROOT:?missing source directory}"
    refit_code=$AF_REPO_ROOT
else
    refit_code=$(cd "$(dirname "$0")/../.." && pwd -P)
fi
case "$platform" in
    delta)
        root=${AF_REFERENCE_ROOT:-/work/hdd/bibo/yxiao2/phase_c/iclr-targeted-reference-v1}
        export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
        export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
        account=${AF_CPU_ACCOUNT:-bibo-delta-cpu}
        ;;
    aces)
        group=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
        root=${AF_REFERENCE_ROOT:-$group/iclr-targeted-reference-v1}
        if [[ $action != status ]]; then
            source "${AF_ANCHOR_ACTIVATION:-$group/iclr-pilot-fit-r1-2334957/activate.sh}"
            module load GCCcore/13.2.0 Python/3.11.5
            group=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
            root=${AF_REFERENCE_ROOT:-$group/iclr-targeted-reference-v1}
        fi
        export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
        export AF_CASADI_ROOT=${AF_CASADI_ROOT:-$group/iclr-python311-casadi-3.7.2}
        account=${AF_CPU_ACCOUNT:-156264627414}
        ;;
    *) echo 'AF_PLATFORM must be aces or delta.' >&2; exit 2 ;;
esac
if [[ $action == status ]]; then
    squeue -u "$USER" -n iclr-cstr-refit -o '%.18i %.24j %.9T %.10M %R'
    shopt -s nullglob
    fits=("$root"/fits/task-*/result.json)
    printf 'Reference-fit results: %s/9.\n' "${#fits[@]}"
    ids=("$root"/submissions/*/array.id)
    if (( ${#ids[@]} )); then
        job_ids=$(cat "${ids[@]}" | sort -u | paste -sd, -)
        sacct -X --array -j "$job_ids" --format=JobID%24,State,ExitCode,Elapsed
    fi
    exit 0
fi
case "$action" in
    prepare|run|worker|finish) ;;
    *) echo "Use: AF_PLATFORM=delta bash $0 run [wave1] | status | finish" >&2; exit 2 ;;
esac
set -u
cd "$refit_code"
export AF_REPO_ROOT="$refit_code" AF_REFERENCE_ROOT="$root" AF_PLATFORM="$platform"
export PYTHONPATH="$refit_code/src:$refit_code${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset AF_JETSTREAM_API_KEY
"$AF_PYTHON" -c 'import casadi, scipy, numpy, pandas, pydantic, sympy'
if [[ $action == prepare || $action == run ]]; then
    sha256sum --quiet -c SHA256SUMS
    "$AF_PYTHON" -m scripts.iclr_perturbed_cstr_refits prepare --root "$root" --public "$refit_code/public"
    if [[ $action == run ]]; then
        "$AF_PYTHON" -m scripts.iclr_perturbed_cstr_refits submit --root "$root" --wave "${2:-wave1}" \
            --account "$account" --platform "$platform"
    fi
elif [[ $action == worker ]]; then
    "$AF_PYTHON" -m scripts.iclr_perturbed_cstr_refits run --root "$root" --index "${SLURM_ARRAY_TASK_ID:?}"
else
    "$AF_PYTHON" -m scripts.iclr_perturbed_cstr_refits finish --root "$root"
fi
