#!/bin/bash
# Submit the high-proposer/low-judge fit-retry integration ablation.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v4}"
: "${AF_SEARCH_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_search_integration_ablation_v3.json}"
: "${AF_SEARCH_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_search_integration_ablation_v3_120b.slurm}"
: "${AF_EVALUATION_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_search_ablation_evaluation_v3.slurm}"
: "${AF_SUBMISSION_SCHEMA:=phase-b-search-integration-submission-3}"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_OUTPUT_ROOT AF_SEARCH_CONFIG AF_SEARCH_JOB
export AF_EVALUATION_JOB AF_SUBMISSION_SCHEMA
exec "${AF_REPO_ROOT}/scripts/hpc/submit_phase_b_search_integration_ablation_v2.sh"
