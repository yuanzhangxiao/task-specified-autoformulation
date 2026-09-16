#!/bin/bash
# A fresh output root freezes one inventory/roster. Reusing it never resubmits jobs.
set -euo pipefail
: "${AF_REPO_ROOT:?Set AF_REPO_ROOT to a clean pinned checkout}"
export AF_REPO_ROOT
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/$USER/venvs/autoformalism-v21/bin/python}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/$USER/phase_b/baseline-validation-v1}"
export AF_PUBLIC_ROOT="${AF_PUBLIC_ROOT:-/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v3}"
export AF_LEGACY_ROOT="${AF_LEGACY_ROOT:-/projects/bibo/$USER/repos/autoformalism-v21/data_raw}"
export AF_CLASSICAL_ROOT="${AF_CLASSICAL_ROOT:-/work/hdd/bibo/$USER/phase_b/public-baselines-full-v1}"
export AF_RAW_ROOT="${AF_RAW_ROOT:-/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-v1}"
export AF_REFRESH_ROOT="${AF_REFRESH_ROOT:-/work/hdd/bibo/$USER/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1}"
export AF_D3_ROOT="${AF_D3_ROOT:-/projects/bibo/$USER/repos/autoformalism-v21/artifacts/rebuttal/consolidated_inputs/vm/vm2-results/artifacts/baselines}"
export AF_ROSTER="${AF_ROSTER:-$AF_REPO_ROOT/configs/baseline_validation_v1.json}"
export AF_COMMIT="$(git -C "$AF_REPO_ROOT" rev-parse HEAD)"
[[ -x "$AF_PYTHON" ]] || { echo "Missing Python: $AF_PYTHON" >&2; exit 2; }
git -C "$AF_REPO_ROOT" diff --quiet
git -C "$AF_REPO_ROOT" diff --cached --quiet
mkdir -p "$AF_OUTPUT_ROOT/logs"
intent="$AF_OUTPUT_ROOT/submission"
if [[ -f "$intent/manifest.json" ]]; then cat "$intent/manifest.json"; exit 0; fi
mkdir "$intent" || { echo 'Submission incomplete: inspect receipts; do not resubmit blindly.' >&2; exit 2; }
worker="$AF_REPO_ROOT/scripts/hpc/run_baseline_validation_delta.sh"
common=(--parsable --account="${AF_ACCOUNT:-bibo-delta-cpu}" --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --export=ALL)
submit() {
    local label="$1"; shift
    sbatch "${common[@]}" "$@" > "$intent/$label.receipt" 2> "$intent/$label.err" || return $?
    local identifier
    identifier="$(cat "$intent/$label.receipt")"; identifier="${identifier%%;*}"
    [[ "$identifier" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch receipt: $label" >&2; return 2; }
    printf '%s\n' "$identifier" > "$intent/$label.id"
    printf '%s\n' "$identifier"
}
prepare="$(submit prepare --job-name=baseline-val-prep --time=00:30:00 --output="$AF_OUTPUT_ROOT/logs/prepare-%j.out" --error="$AF_OUTPUT_ROOT/logs/prepare-%j.err" "$worker" prepare)"
replay="$(submit replay --job-name=baseline-val --time=12:00:00 --array=0-7%8 --dependency="afterok:$prepare" --output="$AF_OUTPUT_ROOT/logs/replay-%A_%a.out" --error="$AF_OUTPUT_ROOT/logs/replay-%A_%a.err" "$worker" run)"
summary="$(submit report --job-name=baseline-val-report --time=00:10:00 --dependency="afterany:$replay" --output="$AF_OUTPUT_ROOT/logs/report-%j.out" --error="$AF_OUTPUT_ROOT/logs/report-%j.err" "$worker" report)"
jq -n --arg prepare "$prepare" --arg replay "$replay" --arg report "$summary" --arg commit "$AF_COMMIT" \
  '{prepare_job:$prepare,replay_job:$replay,report_job:$report,commit:$commit,gpus:0,live_llm_calls:0,test_data_opened:false}' > "$intent/manifest.json"
cat "$intent/manifest.json"
