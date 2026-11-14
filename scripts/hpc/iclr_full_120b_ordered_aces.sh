#!/usr/bin/env bash
# Each three-seed case can be frozen before the next case is launched.
set -euo pipefail
umask 077
ordered_code=$(cd "$(dirname "$0")/../.." && pwd -P)
group=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
source "$group/iclr-pilot-fit-r1-2334957/activate.sh"
base=${AF_ORDERED_120B_ROOT:-$group/iclr-full-120b-ordered-v1}
action=${1:?Use prepare CASE, run CASE [WAVE], status CASE, or finish CASE}
case_name=${2:-cstr-hard}
case "$action" in prepare|run|status|finish) ;; *) exit 2 ;; esac
cd "$ordered_code"
module load GCCcore/13.2.0 Python/3.11.5
export AF_REPO_ROOT="$ordered_code" AF_HARMONY_ASSETS="$ordered_code/assets/harmony"
export AF_COMPUTE_CACHE_ROOT=job-local
export PYTHONPATH="$ordered_code/src:$ordered_code:$AF_CASADI_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
args=("$action" "$case_name" --base "$base")
if [[ $action == prepare || $action == run ]]; then
    (cd "$ordered_code/.." && sha256sum --quiet -c SHA256SUMS)
    snap=hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json
    for cache in "${AF_HF_HOME:-}" "$group/huggingface-cache" /scratch/user/u.yx126462/huggingface-cache; do
        if [[ -n $cache && -s "$cache/$snap" ]]; then export AF_HF_HOME="$cache"; break; fi
    done
    [[ -s "${AF_HF_HOME:-}/$snap" && -s "$AF_VLLM_IMAGE" ]] || {
        echo 'Pinned 120B weights or vLLM image are missing.' >&2; exit 1;
    }
    "$AF_PYTHON" -c 'import os; from pathlib import Path; from scripts.submit_structural_revision_aces import snapshot_identity; snapshot_identity(Path(os.environ["AF_HF_HOME"]), "120b")'
fi
if [[ $action == prepare ]]; then
    args+=(--anchor "$group/iclr-pilot-fit-r1-2334957/anchors"
        --t1 "$group/iclr-t1-perturbed-full-v1"
        --gate "$AF_JUDGE_GATE_ROOT" --public "$ordered_code/../inputs/public")
elif [[ $action == run ]]; then
    module load WebProxy
    args+=(--wave "${3:-wave1}" --account "${AF_GPU_ACCOUNT:-156264627414}")
fi
"$AF_PYTHON" -m scripts.iclr_full_120b_ordered "${args[@]}"
