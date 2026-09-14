#!/bin/bash
# Idempotent two-job ACES pilot: CPU replay/preflight, then one 20B H100 server.
set -euo pipefail
repo="${AF_REPO_ROOT:?set the pinned checkout path}"
root="${AF_OUTPUT_ROOT:?set a new output root}"
python="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
source_root="${AF_SOURCE_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1-e7ffd12}"
config="${AF_CONFIG:-$repo/configs/prefit_matched_feedback_v1.json}"
resume="${AF_RESUME:-0}"
manifest="$root/submission_manifest.json"
[[ "$resume" == 0 || "$resume" == 1 ]] || { echo 'AF_RESUME must be 0 or 1' >&2; exit 2; }
[[ -x "$python" && -f "$source_root/plan.json" && -f "$config" ]] || { echo 'Missing Python, historical source, or config' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned experiment checkout' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
export AF_REPO_ROOT="$repo" AF_PYTHON="$python" AF_OUTPUT_ROOT="$root" AF_SOURCE_ROOT="$source_root" AF_CONFIG="$config" PYTHONPATH="$repo/src"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-/scratch/user/u.yx126462/autoformalism-runtime-cache/prefit-feedback}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-/scratch/user/u.yx126462/af-ipc}"
[[ -f "$AF_VLLM_IMAGE" ]] || { echo 'Missing pinned image' >&2; exit 2; }
"$python" - "$root" "$source_root" <<'PY'
import sys
from pathlib import Path
if Path(sys.argv[1]).resolve().is_relative_to(Path(sys.argv[2]).resolve()):
    raise SystemExit('New output root must be outside the historical source')
PY
if [[ -f "$manifest" ]]; then
  [[ "$(jq -r '.commit' "$manifest")" == "$(git -C "$repo" rev-parse HEAD)" && "$(jq -r '.source_root' "$manifest")" == "$source_root" && "$(jq -r '.config_sha256' "$manifest")" == "$(sha256sum "$config" | cut -d' ' -f1)" ]] || { echo 'Submission identity differs; use the original pinned checkout and inputs' >&2; exit 2; }
  if [[ "$resume" == 0 ]]; then cat "$manifest"; exit 0; fi
fi
mkdir -p "$root/logs" "$root/submissions" "$AF_HF_HOME"
worker="$repo/scripts/hpc/run_prefit_feedback_aces.sh"
if [[ "$resume" == 1 ]]; then
  [[ -f "$manifest" && -f "$root/plan.json" ]] || { echo 'Resume requires a submitted, frozen experiment' >&2; exit 2; }
  "$python" "$repo/scripts/prefit_feedback_campaign.py" verify --root "$root"
  prepare="$(jq -er '.prepare_job' "$manifest")"
  prior="$(jq -er '.repair_job' "$manifest")"
  [[ "$prepare" =~ ^[0-9]+$ && "$prior" =~ ^[0-9]+$ ]] || exit 2
  [[ -z "$(squeue -h -j "$prepare,$prior" -o '%A')" ]] || { echo 'Prior job still queued/running; no duplicate submission' >&2; exit 2; }
  [[ "$(sacct -n -X -j "$prepare" -o State --parsable2 | head -1 | cut -d'|' -f1)" == COMPLETED ]] || { echo 'Preparation did not complete; inspect logs before a new run' >&2; exit 2; }
  state="$(sacct -n -X -j "$prior" -o State --parsable2 | head -1 | cut -d'|' -f1)"
  [[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) ]] || { echo 'Prior scheduler outcome is uncertain' >&2; exit 2; }
  "$python" "$repo/scripts/prefit_feedback_campaign.py" summary --root "$root" > "$root/submissions/resume-summary.json"
  [[ "$(jq -r '.status' "$root/submissions/resume-summary.json")" != complete ]] || { echo 'Every episode is terminal; no new job needed'; exit 0; }
  intent="$root/submissions/resume-$prior"
else
  [[ ! -e "$root/plan.json" ]] || { echo 'A frozen plan already exists; inspect it before submitting' >&2; exit 2; }
  intent="$root/submissions/initial"
fi
mkdir "$intent" || { echo 'Submission intent exists; inspect scheduler outcome before retrying' >&2; exit 2; }
submit() {
  local stage="$1" job
  shift
  if ! job="$(sbatch --parsable --account="${AF_ACCOUNT:-156264627414}" --nodes=1 --ntasks=1 --export=ALL --output="$root/logs/$stage-%j.out" --error="$root/logs/$stage-%j.err" "$@" "$worker" "$stage")"; then
    echo 'Submission failed or is uncertain; inspect the saved intent and queue' >&2
    return 2
  fi
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid scheduler job ID' >&2; return 2; }
  printf '%s\n' "$job" > "$intent/$stage-job.txt"
  printf '%s\n' "$job"
}
if [[ "$resume" == 0 ]]; then
  prepare="$(submit prepare --job-name=prefit-replay --partition=cpu --cpus-per-task=4 --mem=16G --time=01:00:00)"
fi
repair="$(submit repair --dependency="afterok:$prepare" --job-name=prefit-feedback --partition=gpu --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --signal=B:TERM@300)"
jq -n --arg prepare_job "$prepare" --arg repair_job "$repair" --arg commit "$(git -C "$repo" rev-parse HEAD)" --arg source_root "$source_root" --arg config_sha256 "$(sha256sum "$config" | cut -d' ' -f1)" '{prepare_job:$prepare_job,repair_job:$repair_job,commit:$commit,source_root:$source_root,config_sha256:$config_sha256}' > "$intent/manifest.json"
cp "$intent/manifest.json" "$manifest.tmp"
mv "$manifest.tmp" "$manifest"
cat "$manifest"
