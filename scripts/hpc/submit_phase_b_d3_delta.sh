#!/bin/bash
# Fresh native D3 discovery: paid GPT API calls, CPU fitting, validation only.
set -euo pipefail
: "${AF_REPO_ROOT:?Set a clean pinned checkout}" "${AF_D3_MODEL:?Set the OpenAI API model ID}"
: "${OPENAI_API_KEY:?Load OPENAI_API_KEY into the environment without printing it}"
export AF_REPO_ROOT AF_D3_MODEL
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/$USER/venvs/autoformalism-v21/bin/python}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/$USER/phase_b/d3-native-pilot-v1}"
export AF_PUBLIC_ROOT="${AF_PUBLIC_ROOT:-/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v3}"
export AF_D3_CONFIG="${AF_D3_CONFIG:-$AF_REPO_ROOT/configs/phase_b_d3_pilot_v1.json}"
export AF_COMMIT="$(git -C "$AF_REPO_ROOT" rev-parse HEAD)"
export PYTHONPATH="$AF_REPO_ROOT/src"
[[ -x "$AF_PYTHON" ]] || { echo "Missing Python: $AF_PYTHON" >&2; exit 2; }
git -C "$AF_REPO_ROOT" diff --quiet
git -C "$AF_REPO_ROOT" diff --cached --quiet
# Dependency/config checks happen before job submission or provider calls.
"$AF_PYTHON" -c 'import torch, openai; print("D3 dependencies available")'
count="$("$AF_PYTHON" -c 'import sys; from pathlib import Path; from autoformalism.rebuttal.phase_b_d3 import Campaign, read_json; c=Campaign.model_validate(read_json(Path(sys.argv[1]))); print(len(c.cells)*len(c.repetitions))' "$AF_D3_CONFIG")"
mkdir -p "$AF_OUTPUT_ROOT/logs"
intent="$AF_OUTPUT_ROOT/submission"
if [[ -f "$intent/manifest.json" ]]; then cat "$intent/manifest.json"; exit 0; fi
mkdir "$intent" || { echo 'Incomplete submission: inspect receipts before resubmitting.' >&2; exit 2; }
worker="$AF_REPO_ROOT/scripts/hpc/run_phase_b_d3_delta.sh"
common=(--parsable --account="${AF_ACCOUNT:-bibo-delta-cpu}" --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --export=ALL)
submit() {
    local label="$1"; shift
    sbatch "${common[@]}" "$@" > "$intent/$label.receipt" 2> "$intent/$label.err" || return $?
    local identifier
    identifier="$(cat "$intent/$label.receipt")"; identifier="${identifier%%;*}"
    [[ "$identifier" =~ ^[0-9]+$ ]] || { echo "Invalid receipt: $label" >&2; return 2; }
    printf '%s\n' "$identifier" > "$intent/$label.id"
    printf '%s\n' "$identifier"
}
prepare="$(submit prepare --job-name=d3-phaseb-prepare --time=00:20:00 --output="$AF_OUTPUT_ROOT/logs/prepare-%j.out" --error="$AF_OUTPUT_ROOT/logs/prepare-%j.err" "$worker" prepare)"
discover="$(submit discover --job-name=d3-phaseb --time=02:00:00 --array="0-$((count-1))%${AF_CONCURRENCY:-2}" --dependency="afterok:$prepare" --kill-on-invalid-dep=yes --output="$AF_OUTPUT_ROOT/logs/discover-%A_%a.out" --error="$AF_OUTPUT_ROOT/logs/discover-%A_%a.err" "$worker" run)"
summary="$(submit report --job-name=d3-phaseb-report --time=00:10:00 --dependency="afterany:$discover" --output="$AF_OUTPUT_ROOT/logs/report-%j.out" --error="$AF_OUTPUT_ROOT/logs/report-%j.err" "$worker" report)"
jq -n --arg prepare "$prepare" --arg discover "$discover" --arg report "$summary" --arg commit "$AF_COMMIT" --arg model "$AF_D3_MODEL" --argjson count "$count" \
  '{prepare_job:$prepare,discovery_job:$discover,report_job:$report,commit:$commit,model:$model,tasks:$count,gpus:0,live_llm_calls:true,test_data_opened:false}' > "$intent/manifest.json"
cat "$intent/manifest.json"
