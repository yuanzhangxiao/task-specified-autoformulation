#!/bin/bash
# Freeze both arms once; submit one independent, resumable arm per invocation.
set -euo pipefail
repo="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
root="${AF_OUTPUT_ROOT:?set a new experiment output root}"
source_root="${AF_SOURCE_RESCUE_ROOT:-/scratch/user/u.yx126462/phase_b/staged-fitter-rescue-v1-c1754fe}"
image="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
hf="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
account="${AF_ACCOUNT:-156264627414}"
resume="${AF_RESUME:-0}"
arm="${AF_ARM:-redesigned_runtime}"
case "$arm" in
  redesigned_runtime) gpus=1; hours=02:00:00; seconds=6900; name=repair-runtime ;;
  redesigned_prefit_judge) gpus=2; hours=04:00:00; seconds=14100; name=repair-judge ;;
  *) echo 'AF_ARM must be redesigned_runtime or redesigned_prefit_judge' >&2; exit 2 ;;
esac
submission="$root/submissions/$arm"
manifest="$submission/manifest.json"
[[ -f "$image" ]] || { echo "Missing image: $image (set AF_VLLM_IMAGE explicitly)" >&2; exit 2; }
[[ -x "$python" && -f "$source_root/summary/summary.json" ]] || { echo 'Missing Python or frozen source rescue summary' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned worktree' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
export PYTHONPATH="$repo/src"
if [[ -f "$manifest" && "$resume" != 1 ]]; then jq . "$manifest"; exit 0; fi
if [[ -f "$root/plan.json" ]]; then revision="$(jq -er '.config.judge_revision' "$root/plan.json")"; else revision="${AF_JUDGE_REVISION:-}"; fi
if [[ -z "$revision" && -f "$hf/hub/models--openai--gpt-oss-120b/refs/main" ]]; then revision="$(<"$hf/hub/models--openai--gpt-oss-120b/refs/main")"; fi
if [[ -z "$revision" ]]; then revision="$(curl --fail --silent --show-error https://huggingface.co/api/models/openai/gpt-oss-120b | jq -er '.sha')"; fi
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Cannot resolve judge revision; set AF_JUDGE_REVISION' >&2; exit 2; }
"$python" "$repo/scripts/repair_feedback_comparison.py" freeze --source-rescue-root "$source_root" --output "$root" --judge-revision "$revision"
expected="$(jq -er '.serving_image_sha256' "$root/plan.json")"
actual="$(sha256sum "$image")"
[[ "${actual%% *}" == "$expected" ]] || { echo 'Image differs from frozen protocol; do not override the hash' >&2; exit 2; }
mkdir -p "$root/logs" "$hf" "$submission"
export AF_REPO_ROOT="$repo" AF_PYTHON="$python" AF_OUTPUT_ROOT="$root" AF_VLLM_IMAGE="$image" AF_HF_HOME="$hf"
export AF_ARM="$arm" AF_WORKER_SECONDS="$seconds"
prior=""
if [[ "$resume" == 1 ]]; then
  prior="$(jq -er '.job_id' "$manifest")"
  prepare="$(jq -er '.prepare_job' "$manifest")"
  [[ "$prior" =~ ^[0-9]+$ && "$prepare" =~ ^[0-9]+$ ]] || exit 2
  [[ -z "$(squeue -h -j "$prior,$prepare" -o '%A')" ]] || { echo 'Previous job remains queued; not resubmitting' >&2; exit 2; }
  state="$(sacct -n -X -j "$prior" -o State --parsable2 | head -1 | cut -d'|' -f1)"
  [[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) ]] || { echo 'Prior job state is uncertain; inspect accounting' >&2; exit 2; }
  [[ "$(sacct -n -X -j "$prepare" -o State --parsable2 | head -1 | cut -d'|' -f1)" == COMPLETED ]] || { echo 'CPU preparation did not complete; inspect its log' >&2; exit 2; }
  intent="$submission/resume-intent-$prior"
  mkdir "$intent" || { echo 'Resume submission may already exist; inspect the queue' >&2; exit 2; }
else
  intent="$submission/submission-intent"
  mkdir "$intent" || { echo 'Submission intent exists: inspect squeue/sacct before resubmitting' >&2; exit 2; }
  prepare=$(sbatch --parsable --account="$account" --job-name="$name-cache" --partition=cpu --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --output="$root/logs/prepare-%j.out" --error="$root/logs/prepare-%j.err" --export=ALL "$repo/scripts/hpc/run_repair_comparison_aces.sh" prepare)
  prepare="${prepare%%;*}"
  [[ "$prepare" =~ ^[0-9]+$ ]] || exit 2
  printf '%s\n' "$prepare" > "$intent/prepare_job.txt"
fi
job=$(sbatch --parsable --account="$account" --dependency="afterok:$prepare" --job-name="$name" --partition="${AF_PARTITION:-gpu}" --time="$hours" --nodes=1 --ntasks=1 --gres="gpu:h100:$gpus" --cpus-per-task=16 --mem=128G --signal=B:TERM@300 --output="$root/logs/$name-%j.out" --error="$root/logs/$name-%j.err" --export=ALL "$repo/scripts/hpc/run_repair_comparison_aces.sh")
job="${job%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$job" > "$intent/job_id.txt"
jq -n --arg arm "$arm" --arg prepare_job "$prepare" --arg job_id "$job" --arg prior_job "$prior" --arg commit "$(git -C "$repo" rev-parse HEAD)" --arg plan_sha256 "$(jq -r '.plan_sha256' "$root/plan.json")" '{arm:$arm,prepare_job:$prepare_job,job_id:$job_id,prior_job:$prior_job,commit:$commit,plan_sha256:$plan_sha256}' > "$intent/manifest.json"
cp "$intent/manifest.json" "$manifest"
printf 'ACES_REPAIR_ARM=%s\nACES_REPAIR_PREPARE_JOB=%s\nACES_REPAIR_COMPARISON_JOB=%s\nACES_REPAIR_COMPARISON_ROOT=%s\n' "$arm" "$prepare" "$job" "$root"
