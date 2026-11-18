#!/usr/bin/env bash
# Six fresh 20B Full lineages, using the anchor numerical environment and critic.
set -eo pipefail
umask 077
targeted_code=$(cd "$(dirname "$0")/../.." && pwd -P)
group=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
root=${AF_TARGETED_ROOT:-$group/iclr-targeted-perturbations-v1}
action=${1:-}

if [[ $action == status ]]; then
    squeue -u "$USER" -n iclr-targeted-20b -o '%.18i %.24j %.9T %.10M %R'
    shopt -s nullglob
    visits=("$root"/results/*/round_*/result.json)
    endpoints=("$root"/pruning/*/result.json)
    printf 'Visit records: %s/90; pruning dispositions: %s/6.\n' "${#visits[@]}" "${#endpoints[@]}"
    ids=("$root"/submissions/*/array.id)
    if (( ${#ids[@]} )); then
        job_ids=$(cat "${ids[@]}" | sort -u | paste -sd, -)
        sacct -X --array -j "$job_ids" --format=JobID%24,State,ExitCode,Elapsed
    fi
    if [[ -s "$root/targeted-perturbation-results.zip" ]]; then
        printf 'Frozen results: %s/targeted-perturbation-results.zip\n' "$root"
    fi
    exit 0
fi
case "$action" in
    prepare|run|finish) ;;
    *) echo "Use: bash $0 run [wave1] | prepare | status | finish" >&2; exit 2 ;;
esac
activation=${AF_ANCHOR_ACTIVATION:-$group/iclr-pilot-fit-r1-2334957/activate.sh}
[[ -r "$activation" ]] || { echo "Missing anchor environment: $activation" >&2; exit 1; }
source "$activation"
set -u
# An activation file may define generic shell names such as root/code/action.
targeted_code=$(cd "$(dirname "$0")/../.." && pwd -P)
group=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
root=${AF_TARGETED_ROOT:-$group/iclr-targeted-perturbations-v1}
action=${1:-}
export AF_REPO_ROOT="$targeted_code" AF_HARMONY_ASSETS="$targeted_code/assets/harmony"
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-$group/iclr-python311-casadi-3.7.2}
export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
export AF_COMPUTE_CACHE_ROOT=job-local AF_BLOCK_SECONDS=36000
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$targeted_code/src:$targeted_code:$AF_CASADI_ROOT"
cd "$targeted_code"

if [[ $action == prepare || $action == run ]]; then
    sha256sum --quiet -c SHA256SUMS
    # Imports alone do not exercise the fitter's dynamically read identity file.
    "$AF_PYTHON" -c 'from autoformalism.rebuttal.fitter_diagnostic import runtime_identity; runtime_identity()'
    "$AF_PYTHON" scripts/harmony_vocab.py verify --directory "$AF_HARMONY_ASSETS" >/dev/null
    snap=hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json
    for cache in "${AF_HF_HOME:-}" "$group/huggingface-cache" /scratch/user/u.yx126462/huggingface-cache; do
        if [[ -n $cache && -s "$cache/$snap" ]]; then export AF_HF_HOME="$cache"; break; fi
    done
    [[ -s "${AF_HF_HOME:-}/$snap" ]] || { echo 'Set AF_HF_HOME to the pinned 20B model cache.' >&2; exit 1; }
    [[ -s "$AF_VLLM_IMAGE" ]] || { echo "Missing vLLM image: $AF_VLLM_IMAGE" >&2; exit 1; }
    if [[ -n ${AF_TARGETED_RECOVER_FROM:-} ]]; then
        "$AF_PYTHON" -m scripts.recover_iclr_targeted_perturbations \
            --source "$AF_TARGETED_RECOVER_FROM" --destination "$root"
    else
        "$AF_PYTHON" -m scripts.iclr_targeted_perturbations prepare --root "$root" \
            --public "$targeted_code/public" --anchor "${AF_ANCHOR_REFERENCE:-$group/iclr-pilot-fit-r1-2334957/anchors}" \
            --gate "${AF_JUDGE_GATE_ROOT:-$group/judge-jetstream-known-cases-v1}"
    fi
    if [[ $action == run ]]; then
        module load WebProxy
        "$AF_PYTHON" -m scripts.iclr_aces_pilot submit --root "$root" --wave "${2:-wave1}" \
            --account "${AF_GPU_ACCOUNT:-156264627414}" --concurrency 6 --confirm-critic-access
    fi
    exit 0
fi

unset AF_JETSTREAM_API_KEY
shopt -s nullglob
ids=("$root"/submissions/*/array.id)
if (( ${#ids[@]} )); then
    job_ids=$(cat "${ids[@]}" | sort -u | paste -sd, -)
    if sacct --array -j "$job_ids" -P --format=JobID%45,ElapsedRaw,TotalCPU,AllocCPUS,AllocTRES%120,State >"$root/sacct.psv" &&
        "$AF_PYTHON" -m scripts.iclr_stress_resources --plan "$root/plan.json" --sacct "$root/sacct.psv" \
            --output "$root/scheduler-cost.csv"; then
        :
    else
        echo 'Slurm accounting unavailable; rerun finish later to include GPU costs.' >&2
    fi
fi
"$AF_PYTHON" -m scripts.iclr_targeted_perturbations finish --root "$root"
