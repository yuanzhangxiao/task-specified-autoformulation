#!/bin/bash
# Requires a globally frozen campaign copied from ACES, and the complete release.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_PUBLIC_ROOT="${AF_PUBLIC_ROOT:-/work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3}"
export AF_COMMIT="$(git -C "$AF_REPO_ROOT" rev-parse HEAD)"
export AF_EVAL_SHARDS=44
[[ -f "$AF_OUTPUT_ROOT/evaluation_freeze.json" && -x "$AF_PYTHON" ]] || { echo 'Missing global freeze or Python' >&2; exit 2; }
[[ -z "$(git -C "$AF_REPO_ROOT" status --porcelain)" ]] || { echo 'Use the pinned clean source' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$AF_PYTHON" - "$AF_OUTPUT_ROOT" <<'PY'
import hashlib,sys
from pathlib import Path
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read
root=Path(sys.argv[1]); plan=sealed_read(root/'plan.json'); freeze=sealed_read(root/'evaluation_freeze.json')
if plan['source_sha256']!=public._source_identity(): raise SystemExit('Source differs from frozen campaign')
if hashlib.sha256((root/'subjects.jsonl').read_bytes()).hexdigest()!=freeze['subjects_sha256']: raise SystemExit('Subjects differ from global freeze')
PY
intent="$AF_OUTPUT_ROOT/evaluation-submission"
if [[ -f "$intent/manifest.json" ]]; then cat "$intent/manifest.json"; exit 0; fi
mkdir "$intent" || { echo 'Evaluation submission uncertain: inspect recorded job IDs and squeue' >&2; exit 2; }
mkdir -p "$AF_OUTPUT_ROOT/evaluation-logs"
worker="$AF_REPO_ROOT/scripts/hpc/run_review_deadline_evaluation_delta.sh"
common=(--parsable --account="${AF_ACCOUNT:-bibo-delta-cpu}" --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --export=ALL)
fit="$(sbatch "${common[@]}" --job-name=review-heldout --mem=8G --time=04:00:00 --array="0-$((AF_EVAL_SHARDS-1))%16" --output="$AF_OUTPUT_ROOT/evaluation-logs/eval-%A_%a.out" --error="$AF_OUTPUT_ROOT/evaluation-logs/eval-%A_%a.err" "$worker" evaluate)"
fit="${fit%%;*}"; [[ "$fit" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$fit" > "$intent/evaluation.id"
summary="$(sbatch "${common[@]}" --job-name=review-eval-summary --mem=4G --time=00:15:00 --dependency="afterany:$fit" --output="$AF_OUTPUT_ROOT/evaluation-logs/summary-%j.out" --error="$AF_OUTPUT_ROOT/evaluation-logs/summary-%j.err" "$worker" report)"
summary="${summary%%;*}"; [[ "$summary" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$summary" > "$intent/summary.id"
jq -n --arg evaluation "$fit" --arg summary "$summary" --arg commit "$AF_COMMIT" '{evaluation_job_id:$evaluation,summary_job_id:$summary,commit:$commit,fitting_jobs:0,gpus:0}' > "$intent/manifest.json"
cat "$intent/manifest.json"
