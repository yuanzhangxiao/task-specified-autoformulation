#!/bin/bash
#SBATCH --job-name=phase-b-raw-refit
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --export=ALL
#SBATCH --output=logs/phase-b-raw-refit-%A_%a.out
#SBATCH --error=logs/phase-b-raw-refit-%A_%a.err

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?submit with --array=0-5%6}"
user_name="${SLURM_JOB_USER:-${USER:?}}"
project_root="${AF_PROJECT:-/projects/bibo/${user_name}}"
work_root="${AF_WORK:-/work/hdd/bibo/${user_name}}"
repo_root="${AF_REPO_ROOT:-${project_root}/repos/autoformalism-v21}"
python_bin="${AF_PYTHON:-${project_root}/venvs/autoformalism-v21/bin/python}"
data_root="${AF_PUBLIC_DATA_ROOT:-${project_root}/phase_b/inputs/public}"
source_root="${AF_RAW_AGENT_SOURCE_ROOT:-${work_root}/phase_b/raw-data-agent-pilot-v1}"
output_root="${AF_COMMON_REFIT_ROOT:-${work_root}/phase_b/raw-data-agent-common-refit-v2}"
protocol_config="${AF_COMMON_REFIT_CONFIG:-${repo_root}/configs/raw_data_agent_common_refit_v2.json}"

test -d "$repo_root" || { echo "missing repository: $repo_root" >&2; exit 2; }
test -x "$python_bin" || { echo "missing Python: $python_bin" >&2; exit 2; }
test -d "$data_root" || { echo "missing public data root: $data_root" >&2; exit 2; }
test -d "$source_root" || { echo "missing source root: $source_root" >&2; exit 2; }
test -f "$protocol_config" || { echo "missing protocol config: $protocol_config" >&2; exit 2; }
mkdir -p "$repo_root/logs" "$output_root"
cd "$repo_root"

mapfile -t source_runs < <(
  find "$source_root" -mindepth 1 -maxdepth 1 -type d \
    -name 'openai_gpt-5-6-sol_*' -print | sort
)
[[ "${#source_runs[@]}" -eq 6 ]] || {
  echo "expected 6 GPT source runs, found ${#source_runs[@]}" >&2
  exit 2
}
[[ "$task_id" -lt "${#source_runs[@]}" ]] || {
  echo "array task $task_id exceeds source-run count" >&2
  exit 2
}

"$python_bin" scripts/refit_raw_data_agent_candidate.py \
  --source-run "${source_runs[$task_id]}" \
  --data-root "$data_root" \
  --protocol-config "$protocol_config" \
  --output-root "$output_root"
