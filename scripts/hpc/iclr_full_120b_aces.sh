#!/usr/bin/env bash
# Fresh Full lineages paired by cell and seed with ACES anchor array 2337694.
set -eo pipefail
umask 077
full_code=$(cd "$(dirname "$0")/../.." && pwd -P)
if [[ ${1:-} != status ]]; then
    full_env="${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}/iclr-pilot-fit-r1-2334957/activate.sh"
    if [[ -r "$full_env" ]]; then source "$full_env"; fi
fi
group=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
reference=${AF_ANCHOR_REFERENCE:-$group/iclr-pilot-fit-r1-2334957/anchors}
root=${AF_FULL_120B_ROOT:-$group/iclr-anchors-full-120b-v1}
action=${1:-}

if [[ $action == status ]]; then
    squeue -u "$USER" -n iclr-full-120b -o '%.18i %.18j %.9T %.10M %R'
    shopt -s nullglob
    visits=("$root"/results/*/round_*/result.json)
    endpoints=("$root"/pruning/*/result.json)
    printf 'Recorded visits %s/90, pruning dispositions %s/6\n' "${#visits[@]}" "${#endpoints[@]}"
    ids=("$root"/submissions/*/array.id)
    if (( ${#ids[@]} )); then
        job_ids=$(cat "${ids[@]}" | sort -u | paste -sd, -)
        sacct -X --array -j "$job_ids" --format=JobID%24,State,ExitCode,Elapsed
    fi
    exit 0
fi
case "$action" in
    run|prepare|finish) ;;
    *) echo "Use: bash $0 prepare | run [wave] | status | finish" >&2; exit 2 ;;
esac
set -u
cd "$full_code"
export AF_REPO_ROOT="$full_code" AF_HARMONY_ASSETS="$full_code/assets/harmony"
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-$group/iclr-python311-casadi-3.7.2}
export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
export AF_COMPUTE_CACHE_ROOT=job-local
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$full_code/src:$full_code:$AF_CASADI_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ $action == prepare || $action == run ]]; then
    (cd "$full_code/.." && sha256sum --quiet -c SHA256SUMS)
    snap=hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json
    for cache in "${AF_HF_HOME:-}" "$group/huggingface-cache" /scratch/user/u.yx126462/huggingface-cache; do
        if [[ -n $cache && -s "$cache/$snap" ]]; then export AF_HF_HOME="$cache"; break; fi
    done
    [[ -s "${AF_HF_HOME:-}/$snap" ]] || { echo "120B weights unavailable; set AF_HF_HOME to the pinned HF cache." >&2; exit 1; }
    [[ -s "$AF_VLLM_IMAGE" ]] || { echo "Missing vLLM image: $AF_VLLM_IMAGE" >&2; exit 1; }
    "$AF_PYTHON" -m scripts.iclr_full_120b prepare --root "$root" --reference "$reference" \
        --gate "${AF_JUDGE_GATE_ROOT:-$group/judge-jetstream-known-cases-v1}" --hf-home "$AF_HF_HOME"
    if [[ $action == run ]]; then
        module load WebProxy
        "$AF_PYTHON" -m scripts.iclr_aces_pilot submit --root "$root" --wave "${2:-wave1}" \
            --account "${AF_GPU_ACCOUNT:-156264627414}" --concurrency 1 --confirm-critic-access
    fi
    exit 0
fi

unset AF_JETSTREAM_API_KEY
"$AF_PYTHON" -m scripts.iclr_full_120b finish --root "$root" --reference "$reference"
cd "$root"
shopt -s nullglob
ids=(submissions/*/array.id)
files=(plan.json spec_variants.json ablation_switch_manifest.json critic_authorization.json
    frozen_models.json evaluation_freeze.json paired_ablation_results.csv paired_proposer_validation.csv
    search_visits.csv endpoint-audit runtime/service-*.json)
if (( ${#ids[@]} )); then
    files+=("${ids[@]}")
    job_ids=$(cat "${ids[@]}" | sort -u | paste -sd, -)
    if sacct --array -j "$job_ids" -P --format=JobID%45,ElapsedRaw,TotalCPU,AllocCPUS,AllocTRES%120,State >sacct.psv &&
        "$AF_PYTHON" -m scripts.iclr_stress_resources --plan plan.json --sacct sacct.psv --output scheduler-cost.csv; then
        files+=(scheduler-cost.csv sacct.psv)
    else
        echo "Slurm accounting unavailable; run finish again to include GPU costs." >&2
    fi
fi
archive=$(mktemp "$root/.inspection.XXXXXXXX")
trap 'rm -f -- "$archive"' EXIT
tar -czf "$archive" "${files[@]}" -C "$full_code" SOURCE_COMMIT
mv "$archive" "$root/inspection.tar.gz"
printf 'Inspection package: %s/inspection.tar.gz\n' "$root"
