#!/bin/bash
# Submit the frozen development-only fit-recovery diagnostic.

set -euo pipefail

readonly script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd "${script_root}/../.." && pwd)"
mkdir -p "${repo_root}/logs"
cd "${repo_root}"

readonly recovery_job="$(sbatch --parsable --account=bibo-delta-cpu \
  scripts/hpc/phase_b_search_fit_recovery_v1.slurm)"
readonly summary_job="$(sbatch --parsable --account=bibo-delta-cpu \
  --dependency="afterok:${recovery_job}" \
  scripts/hpc/phase_b_search_fit_recovery_summary_v1.slurm)"

echo "FIT_RECOVERY_JOB=${recovery_job}"
echo "FIT_RECOVERY_SUMMARY_JOB=${summary_job}"
