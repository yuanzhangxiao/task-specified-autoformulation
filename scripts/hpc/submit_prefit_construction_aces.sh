#!/bin/bash
# Freeze twelve tasks, prepare on CPU, construct on one H100, then fit on CPU.
set -euo pipefail
repo="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
root="${AF_OUTPUT_ROOT:?set a new experiment output root}"
public="${AF_PUBLIC_ROOT:-/scratch/user/u.yx126462/phase_b/staged-fitter-rescue-v1-c1754fe/frozen/public}"
config="${AF_CONFIG:-$repo/configs/prefit_matched_construction_v1.json}"
image="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
hf="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
account="${AF_ACCOUNT:-156264627414}"
resume="${AF_RESUME:-0}"
[[ "$resume" == 0 || "$resume" == 1 ]] || { echo 'AF_RESUME must be 0 or 1' >&2; exit 2; }
manifest="$root/submission_manifest.json"
worker="$repo/scripts/hpc/run_prefit_construction_aces.sh"
[[ -x "$python" && -f "$image" && -f "$config" && -f "$worker" ]] || { echo 'Missing Python, config, image, or worker' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned experiment checkout' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ -f "$root/plan.json" ]]; then
  "$python" "$repo/scripts/prefit_construction_campaign.py" verify --root "$root"
else
  [[ "$resume" == 0 ]] || { echo 'Cannot resume without a frozen plan' >&2; exit 2; }
  "$python" "$repo/scripts/prefit_construction_campaign.py" freeze --config "$config" --public-root "$public" --output "$root"
fi
plan_hash="$(jq -er '.artifact_sha256' "$root/plan.json")"
if [[ -f "$manifest" && "$resume" == 0 ]]; then jq . "$manifest"; exit 0; fi
mkdir -p "$root/logs" "$root/submissions" "$hf"
export AF_REPO_ROOT="$repo" AF_PYTHON="$python" AF_OUTPUT_ROOT="$root" AF_VLLM_IMAGE="$image" AF_HF_HOME="$hf"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-/scratch/user/u.yx126462/autoformalism-runtime-cache/prefit-construction}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-/scratch/user/u.yx126462/af-ipc}"
prior_construct='' prior_fit='' construct_needed=true
if [[ "$resume" == 1 ]]; then
  [[ -f "$manifest" ]] || { echo 'Missing submission manifest; inspect any existing intent before retrying' >&2; exit 2; }
  [[ "$(jq -er '.plan_sha256' "$manifest")" == "$plan_hash" ]] || { echo 'Submission plan differs' >&2; exit 2; }
  prepare="$(jq -er '.prepare_job' "$manifest")"
  prior_construct="$(jq -er '.construct_job' "$manifest")"
  prior_fit="$(jq -er '.fit_job' "$manifest")"
  for job in "$prepare" "$prior_construct" "$prior_fit"; do [[ "$job" =~ ^[0-9]+$ ]] || exit 2; done
  [[ -z "$(squeue -h -j "$prepare,$prior_construct,$prior_fit" -o '%A')" ]] || { echo 'A prior job remains queued/running; no duplicate submission' >&2; exit 2; }
  [[ "$(sacct -n -X -j "$prepare" -o State --parsable2 | head -1 | cut -d'|' -f1)" == COMPLETED ]] || { echo 'Preparation did not complete; inspect its log and dependencies' >&2; exit 2; }
  for job in "$prior_construct" "$prior_fit"; do
    state="$(sacct -n -X -j "$job" -o State --parsable2 | head -1 | cut -d'|' -f1)"
    [[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) ]] || { echo 'Prior job state is uncertain; inspect accounting' >&2; exit 2; }
  done
  "$python" "$repo/scripts/prefit_construction_campaign.py" summary --root "$root" > "$root/submissions/resume-summary.json"
  [[ "$(jq -r '.status' "$root/submissions/resume-summary.json")" != complete ]] || { echo 'All tasks are terminal; no new jobs needed'; exit 0; }
  [[ "$(jq -r '.construction_complete' "$root/submissions/resume-summary.json")" != true ]] || construct_needed=false
  intent="$root/submissions/resume-$prior_construct-$prior_fit"
else
  intent="$root/submissions/initial"
fi
mkdir "$intent" || { echo 'Submission intent exists; scheduler outcome may be uncertain. Inspect it before retrying.' >&2; exit 2; }
submit_job() {
  local stage="$1" result
  shift
  if ! result="$(sbatch --parsable --account="$account" --nodes=1 --ntasks=1 --export=ALL --output="$root/logs/$stage-%j.out" --error="$root/logs/$stage-%j.err" "$@" "$worker" "$stage")"; then
    echo 'Scheduler submission failed or is uncertain; inspect the intent and queue' >&2
    return 2
  fi
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || { echo 'Scheduler returned an invalid job ID; inspect the submission intent' >&2; return 2; }
  printf '%s\n' "$result" > "$intent/$stage-job.txt"
  printf '%s\n' "$result"
}
if [[ "$resume" == 0 ]]; then
  prepare="$(submit_job prepare --job-name=prefit-prepare --partition=cpu --time=01:00:00 --cpus-per-task=4 --mem=16G)"
fi
if [[ "$construct_needed" == true ]]; then
  construct="$(submit_job construct --dependency="afterok:$prepare" --job-name=prefit-construct --partition=gpu --time=04:00:00 --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --signal=B:TERM@300)"
  fit="$(submit_job fit --dependency="afterany:$construct" --job-name=prefit-fit --partition=cpu --time=02:00:00 --cpus-per-task=8 --mem=32G --signal=B:TERM@300)"
else
  construct="$prior_construct"
  fit="$(submit_job fit --job-name=prefit-fit --partition=cpu --time=02:00:00 --cpus-per-task=8 --mem=32G --signal=B:TERM@300)"
fi
jq -n --arg prepare_job "$prepare" --arg construct_job "$construct" --arg fit_job "$fit" --arg prior_construct "$prior_construct" --arg prior_fit "$prior_fit" --arg commit "$(git -C "$repo" rev-parse HEAD)" --arg plan_sha256 "$plan_hash" '{prepare_job:$prepare_job,construct_job:$construct_job,fit_job:$fit_job,prior_construct:$prior_construct,prior_fit:$prior_fit,commit:$commit,plan_sha256:$plan_sha256}' > "$intent/manifest.json"
cp "$intent/manifest.json" "$manifest.tmp"
mv "$manifest.tmp" "$manifest"
printf 'ACES_PREFIT_PREPARE_JOB=%s\nACES_PREFIT_CONSTRUCTION_JOB=%s\nACES_PREFIT_FIT_JOB=%s\nACES_PREFIT_ROOT=%s\n' "$prepare" "$construct" "$fit" "$root"
