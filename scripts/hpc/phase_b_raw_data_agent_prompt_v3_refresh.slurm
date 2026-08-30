#!/bin/bash
#SBATCH --job-name=phase-b-raw-v3-refresh
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-29%4
#SBATCH --output=logs/phase-b-raw-v3-refresh-%A_%a.out
#SBATCH --error=logs/phase-b-raw-v3-refresh-%A_%a.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly config_source="${AF_REPO_ROOT}/configs/raw_data_agent_fitted_model_prompt_v3_refresh_v1.json"
readonly frozen_root="${AF_OUTPUT_ROOT}/frozen"
readonly plan="${frozen_root}/plan.json"
readonly task_plan="${frozen_root}/task_plan.jsonl"
readonly task_line=$((AF_TASK_INDEX + 1))

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public prompt v3 root" >&2; exit 2; }
[[ -f "${config_source}" ]] || { echo "missing refresh config" >&2; exit 2; }
[[ -f "${plan}" ]] || { echo "missing frozen refresh plan" >&2; exit 2; }
[[ -f "${task_plan}" ]] || { echo "missing frozen refresh tasks" >&2; exit 2; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY is unset" >&2; exit 2; }
cmp --silent "${config_source}" "${plan}" || {
  echo "repository refresh config differs from frozen plan" >&2
  exit 2
}
(
  cd "${frozen_root}"
  sha256sum -c plan.json.sha256
  sha256sum -c task_plan.jsonl.sha256
  sha256sum -c freeze_manifest.json.sha256
)

readonly task_json="$(sed -n "${task_line}p" "${task_plan}")"
[[ -n "${task_json}" ]] || { echo "task index absent: ${AF_TASK_INDEX}" >&2; exit 2; }
[[ "$(jq -r '.task_id' <<<"${task_json}")" == "${AF_TASK_INDEX}" ]] || {
  echo "task ledger index mismatch" >&2
  exit 2
}
readonly benchmark_id="$(jq -r '.benchmark_id' <<<"${task_json}")"
readonly tier="$(jq -r '.tier' <<<"${task_json}")"
readonly repetition="$(jq -r '.repetition' <<<"${task_json}")"
readonly cell_root="${AF_PUBLIC_DATA_ROOT}/phase_b_v1/${benchmark_id}"

for specification in \
  "proposer_prompt.txt:public_prompt_sha256" \
  "train.csv:train_sha256" \
  "validation.csv:validation_sha256"
do
  filename="${specification%%:*}"
  digest_field="${specification##*:}"
  expected_digest="$(jq -r ".${digest_field}" <<<"${task_json}")"
  path="${cell_root}/${filename}"
  [[ -f "${path}" ]] || { echo "missing frozen public input: ${path}" >&2; exit 2; }
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_digest}" ]] || {
    echo "public input differs from frozen task: ${path}" >&2
    exit 2
  }
done

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runs"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/run_raw_data_agent.py \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --benchmark-id "${benchmark_id}" \
  --tier "${tier}" \
  --provider openai \
  --model gpt-5.6-sol \
  --repetition "${repetition}" \
  --output-root "${AF_OUTPUT_ROOT}/runs" \
  --output-contract fitted_model \
  --reasoning-effort high \
  --max-tool-calls 12 \
  --max-output-tokens 30000 \
  --llm-timeout-seconds 1200 \
  --llm-max-attempts 2 \
  --contract-repair-attempts 1
