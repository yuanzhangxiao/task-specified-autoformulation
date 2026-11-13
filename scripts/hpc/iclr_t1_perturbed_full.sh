#!/usr/bin/env bash
# ACES launcher staged beside iclr_t1_perturbed_full.py and the four public files.
set -eo pipefail
umask 077
ops=$(cd "$(dirname "$0")" && pwd -P)
group=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
root=${AF_T1_FULL_ROOT:-$group/iclr-t1-perturbed-full-v1}
[[ $ops =~ ^/[[:alnum:]_./-]+$ && $root =~ ^/[[:alnum:]_./-]+$ && $group =~ ^/[[:alnum:]_./-]+$ ]]
source "$group/iclr-pilot-fit-r1-2334957/activate.sh"
set -u
export AF_ACCOUNT=${AF_ACCOUNT:-156264627414}
export AF_GPU_ACCOUNT=${AF_GPU_ACCOUNT:-$AF_ACCOUNT} AF_CPU_ACCOUNT=${AF_CPU_ACCOUNT:-$AF_ACCOUNT}
code="$group/iclr-anchors-full-120b-a0b1501/code"
[[ $(cat "$code/SOURCE_COMMIT") == a0b15013021f351adb5658b3a7bfe8372d1aa7f3 ]]
export AF_REPO_ROOT="$code" AF_HARMONY_ASSETS="$code/assets/harmony" AF_COMPUTE_CACHE_ROOT=job-local
export PYTHONPATH="$code/src:$code${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load GCCcore/13.2.0 Python/3.11.5
cd "$code"
case "${1:-}" in
  prepare)
    (cd "$ops" && sha256sum --quiet -c SHA256SUMS)
    (cd "$code/.." && sha256sum --quiet -c SHA256SUMS)
    test -s "$AF_VLLM_IMAGE"
    test -s "$AF_HF_HOME/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json"
    "$AF_PYTHON" "$ops/iclr_t1_perturbed_full.py" prepare --root "$root" --public "$ops/public" \
        --anchor "$group/iclr-pilot-fit-r1-2334957/anchors" --gate "$AF_JUDGE_GATE_ROOT"
    ;;
  run)
    wave=${2:-wave1}
    [[ $wave =~ ^wave[1-9][0-9]*$ ]] || { echo 'Expected wave1, wave2, etc.' >&2; exit 2; }
    (cd "$ops" && sha256sum --quiet -c SHA256SUMS)
    test -s "$root/critic_authorization.json"
    module load WebProxy
    job=$("$AF_PYTHON" -m scripts.iclr_aces_pilot submit --root "$root" --wave "$wave" \
        --account "$AF_GPU_ACCOUNT" --concurrency 2 --confirm-critic-access)
    printf 'T1 Full search: %s\n' "$job"
    [[ $job =~ ^[0-9]+$ ]] || exit 0
    # Schedule the seal/export independently of the search credential. An incomplete
    # search leaves the exporter without a bundle and can resume under a new wave.
    folder="$root/submissions/$wave"
    (
      flock -x 9
      if [[ -s "$folder/finish.id" ]]; then cat "$folder/finish.id"; exit 0; fi
      [[ ! -e "$folder/finish.intent" ]] || { echo 'Reconcile existing finish.intent in Slurm before scheduling another exporter.' >&2; exit 1; }
      printf '%s\n' "$job" > "$folder/finish.intent"
      reply=$(env -u AF_JETSTREAM_API_KEY sbatch --parsable --export=NIL --dependency="afterany:$job" \
          --account="$AF_CPU_ACCOUNT" --partition=cpu --job-name=t1-full-export --time=01:00:00 \
          --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --output="$root/logs/$wave-export-%j.out" \
          --error="$root/logs/$wave-export-%j.err" "$ops/finish.sbatch" "$ops" "$group" "$root")
      id=${reply%%;*}
      [[ $id =~ ^[0-9]+$ ]]
      printf '%s\n' "$id" > "$folder/finish.id"
      printf 'Dependent model export: %s\n' "$id"
    ) 9>"$folder/.finish.lock"
    ;;
  status)
    shopt -s nullglob
    visits=("$root"/results/block*_full/round_*/result.json)
    prunes=("$root"/pruning/block*_full/result.json)
    printf 'Visit records: %s/45. Pruning dispositions: %s/3.\n' "${#visits[@]}" "${#prunes[@]}"
    for f in "$root"/submissions/*/{array,finish}.id; do printf '%s: ' "$f"; cat "$f"; done
    if [[ -s "$root/t1-perturbed-full.zip" ]]; then printf 'Frozen export: %s/t1-perturbed-full.zip\n' "$root"; fi
    ;;
  finish)
    "$AF_PYTHON" "$ops/iclr_t1_perturbed_full.py" finish --root "$root"
    ;;
  *) echo "Use: bash $0 prepare | run [wave1] | status | finish" >&2; exit 2 ;;
esac
