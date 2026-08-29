#!/bin/bash
# Submit the frozen judge integration ablation and its sealed evaluation.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v1}"

readonly config="${AF_REPO_ROOT}/configs/phase_b_search_integration_ablation_v1.json"
readonly search_job="${AF_REPO_ROOT}/scripts/hpc/phase_b_search_integration_ablation_120b.slurm"
readonly evaluation_job="${AF_REPO_ROOT}/scripts/hpc/phase_b_search_ablation_evaluation_v1.slurm"

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -f "${config}" ]] || { echo "missing config: ${config}" >&2; exit 2; }
[[ -f "${search_job}" ]] || { echo "missing search job: ${search_job}" >&2; exit 2; }
[[ -f "${evaluation_job}" ]] || { echo "missing evaluation job: ${evaluation_job}" >&2; exit 2; }

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"

"${AF_PYTHON}" scripts/prepare_phase_b_search_integration_ablation.py \
  --config "${config}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen"

judge_submission="$(
  sbatch --parsable \
    --array=0-5%2 \
    --export="ALL,AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${search_job}"
)"
readonly judge_job_id="${judge_submission%%;*}"

no_judge_submission="$(
  sbatch --parsable \
    --dependency="afterany:${judge_job_id}" \
    --array=6-11%2 \
    --export="ALL,AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${search_job}"
)"
readonly no_judge_job_id="${no_judge_submission%%;*}"

evaluation_submission="$(
  sbatch --parsable \
    --dependency="afterany:${no_judge_job_id}" \
    --export="ALL,AF_SEARCH_ROOT=${AF_OUTPUT_ROOT}" \
    "${evaluation_job}"
)"
readonly evaluation_job_id="${evaluation_submission%%;*}"

echo "JUDGE_SEARCH_JOB=${judge_job_id}"
echo "NO_JUDGE_SEARCH_JOB=${no_judge_job_id}"
echo "ABLATION_EVALUATION_JOB=${evaluation_job_id}"
