#!/bin/bash
#SBATCH --job-name=phase-b-raw-agent
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/phase-b-raw-agent-%A_%a.out
#SBATCH --error=logs/phase-b-raw-agent-%A_%a.err

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?submit with --array=0-11%6}"
user_name="${SLURM_JOB_USER:-${USER:?}}"
project_root="${AF_PROJECT:-/projects/bibo/${user_name}}"
work_root="${AF_WORK:-/work/hdd/bibo/${user_name}}"
repo_root="${AF_REPO_ROOT:-${project_root}/repos/autoformalism-v21}"
python_bin="${AF_PYTHON:-${project_root}/venvs/autoformalism-v21/bin/python}"
data_root="${AF_PUBLIC_DATA_ROOT:-${project_root}/phase_b/inputs/public}"
output_root="${AF_RAW_AGENT_ROOT:-${work_root}/phase_b/raw-data-agent-pilot-v1}"
protocol_config="${AF_RAW_AGENT_PROTOCOL_CONFIG:-${repo_root}/configs/raw_data_agent_pilot_v1.json}"
max_tool_calls="${AF_RAW_AGENT_MAX_TOOL_CALLS:-12}"

test -d "$repo_root" || { echo "missing repository: $repo_root" >&2; exit 2; }
test -x "$python_bin" || { echo "missing Python: $python_bin" >&2; exit 2; }
test -d "$data_root" || { echo "missing public data root: $data_root" >&2; exit 2; }
test -f "$protocol_config" || { echo "missing protocol config: $protocol_config" >&2; exit 2; }
mkdir -p "$repo_root/logs" "$output_root"
cd "$repo_root"

provider_index=$((task_id % 2))
within_provider=$((task_id / 2))
benchmark_index=$((within_provider / 3))
repetition=$((within_provider % 3))

providers=(openai gemini)
gemini_model="${AF_GEMINI_MODEL:-gemini-3.1-pro-preview}"
models=(gpt-5.6-sol "$gemini_model")
benchmarks=(
  phase_b_dalla_man_t2_canonical_named_easy
  phase_b_anonymous_system_task_canonical_opaque_hard
)
tiers=(easy hard)

provider="${providers[$provider_index]}"
model="${models[$provider_index]}"
benchmark_id="${benchmarks[$benchmark_index]}"
tier="${tiers[$benchmark_index]}"

if [[ "$provider" == openai ]]; then
  test -n "${OPENAI_API_KEY:-}" || { echo "OPENAI_API_KEY is unset" >&2; exit 2; }
else
  if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "GEMINI_API_KEY or GOOGLE_API_KEY is unset" >&2
    exit 2
  fi
  "$python_bin" -c 'from google import genai' || {
    echo "google-genai is missing; install the project gemini extra before submission" >&2
    exit 2
  }
fi

if [[ "$task_id" -eq 0 ]]; then
  cp -f "$protocol_config" "$output_root/protocol_config.json"
fi
"$python_bin" scripts/run_raw_data_agent.py \
  --data-root "$data_root" \
  --benchmark-id "$benchmark_id" \
  --tier "$tier" \
  --provider "$provider" \
  --model "$model" \
  --repetition "$repetition" \
  --output-root "$output_root" \
  --reasoning-effort high \
  --max-tool-calls "$max_tool_calls" \
  --max-output-tokens 30000 \
  --llm-timeout-seconds 1200 \
  --llm-max-attempts 2 \
  --contract-repair-attempts 1 \
  --fit-max-nfev 50 \
  --fit-timeout-seconds 300
