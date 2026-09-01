#!/bin/bash
# Submit the frozen full GPT-5.6 deterministic evaluation chain.

set -euo pipefail
readonly af_user="${USER:?}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/raw-agent-deterministic-evaluation-v1}"
cd "${AF_REPO_ROOT}"
mkdir -p logs "${AF_OUTPUT_ROOT}"

prep="$(sbatch --parsable --account=bibo-delta-cpu \
  --export=ALL,AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT}" \
  scripts/hpc/phase_b_raw_agent_eval_prepare.slurm)"
post="$(sbatch --parsable --account=bibo-delta-cpu \
  --dependency="afterok:${prep}" \
  --export=ALL,AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT}" \
  scripts/hpc/phase_b_raw_agent_eval_postfreeze.slurm)"
post_merge="$(sbatch --parsable --account=bibo-delta-cpu \
  --dependency="afterok:${post}" \
  --export=ALL,AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT}" \
  scripts/hpc/phase_b_raw_agent_eval_postfreeze_merge.slurm)"
hidden="$(sbatch --parsable --account=bibo-delta-cpu \
  --dependency="afterok:${post_merge}" \
  --export=ALL,AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT}" \
  scripts/hpc/phase_b_raw_agent_eval_hidden.slurm)"
final="$(sbatch --parsable --account=bibo-delta-cpu \
  --dependency="afterok:${hidden}" \
  --export=ALL,AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT}" \
  scripts/hpc/phase_b_raw_agent_eval_finalize.slurm)"

printf 'RAW_EVAL_PREP_JOB=%s\n' "${prep}"
printf 'RAW_EVAL_POSTFREEZE_JOB=%s\n' "${post}"
printf 'RAW_EVAL_POSTFREEZE_MERGE_JOB=%s\n' "${post_merge}"
printf 'RAW_EVAL_HIDDEN_JOB=%s\n' "${hidden}"
printf 'RAW_EVAL_FINAL_JOB=%s\n' "${final}"
