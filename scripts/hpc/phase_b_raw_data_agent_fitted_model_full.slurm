#!/bin/bash
#SBATCH --job-name=phase-b-raw-fitted-full
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-119%12
#SBATCH --export=ALL
#SBATCH --output=logs/phase-b-raw-fitted-full-%A_%a.out
#SBATCH --error=logs/phase-b-raw-fitted-full-%A_%a.err

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?submit as the declared 0-119 array}"
user_name="${SLURM_JOB_USER:-${USER:?}}"
project_root="${AF_PROJECT:-/projects/bibo/${user_name}}"
work_root="${AF_WORK:-/work/hdd/bibo/${user_name}}"
repo_root="${AF_REPO_ROOT:-${project_root}/repos/autoformalism-v21}"
python_bin="${AF_PYTHON:-${project_root}/venvs/autoformalism-v21/bin/python}"
data_root="${AF_PUBLIC_DATA_ROOT:-${project_root}/phase_b/inputs/public}"
output_root="${AF_RAW_FITTED_ROOT:-${work_root}/phase_b/raw-data-agent-fitted-v1}"
protocol_config="${repo_root}/configs/raw_data_agent_fitted_model_full_v1.json"

test -d "$repo_root" || { echo "missing repository: $repo_root" >&2; exit 2; }
test -x "$python_bin" || { echo "missing Python: $python_bin" >&2; exit 2; }
test -d "$data_root" || { echo "missing public data root: $data_root" >&2; exit 2; }
test -f "$protocol_config" || { echo "missing protocol: $protocol_config" >&2; exit 2; }
test -n "${OPENAI_API_KEY:-}" || { echo "OPENAI_API_KEY is unset" >&2; exit 2; }
mkdir -p "$repo_root/logs" "$output_root"
cd "$repo_root"

mapfile -t resolved_task < <(
  "$python_bin" scripts/resolve_raw_data_agent_matrix_task.py \
    --config "$protocol_config" \
    --task-id "$task_id"
)
[[ "${#resolved_task[@]}" -eq 3 ]] || {
  echo "matrix resolver did not return exactly three fields" >&2
  exit 2
}
benchmark_id="${resolved_task[0]}"
tier="${resolved_task[1]}"
repetition="${resolved_task[2]}"

if [[ "$task_id" -eq 0 ]]; then
  cp -f "$protocol_config" "$output_root/full_protocol_config.json"
fi

"$python_bin" scripts/run_raw_data_agent.py \
  --data-root "$data_root" \
  --benchmark-id "$benchmark_id" \
  --tier "$tier" \
  --provider openai \
  --model gpt-5.6-sol \
  --repetition "$repetition" \
  --output-root "$output_root" \
  --output-contract fitted_model \
  --reasoning-effort high \
  --max-tool-calls 12 \
  --max-output-tokens 30000 \
  --llm-timeout-seconds 1200 \
  --llm-max-attempts 2 \
  --contract-repair-attempts 1
