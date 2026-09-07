#!/bin/bash
# Freeze and submit the matched atomic-versus-equation function experiment.
set -euo pipefail
: "${AF_SOURCE_FUNCTION_PLAN:?set AF_SOURCE_FUNCTION_PLAN to the completed staged-function-v2 plan.json}"
readonly repository="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
readonly python="${AF_PYTHON:-${repository}/.venv/bin/python}"
readonly scratch_root="${SCRATCH:-/scratch/user/${USER}}"
readonly project_root="${PROJECT:-${scratch_root}}"
readonly output_root="${AF_OUTPUT_ROOT:-${scratch_root}/phase_b/staged-function-granularity-v1-aces-h100x1}"
readonly image="${AF_VLLM_IMAGE:-${project_root}/containers/vllm-openai-v0.27.1.sif}"
readonly account="${AF_ACCOUNT:-156264627414}"
readonly config="${repository}/configs/staged_function_granularity_v1.json"
readonly plan="${output_root}/plan.json"
readonly manifest="${output_root}/submission_manifest.json"
git -C "${repository}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "missing repository: ${repository}" >&2
  exit 2
}
[[ -x "${python}" ]] || { echo "missing Python: ${python}" >&2; exit 2; }
[[ -f "${AF_SOURCE_FUNCTION_PLAN}" ]] || { echo "missing source function plan: ${AF_SOURCE_FUNCTION_PLAN}" >&2; exit 2; }
[[ -f "${image}" ]] || { echo "missing vLLM image: ${image}" >&2; exit 2; }
mkdir -p "${output_root}/logs"
export PYTHONPATH="${repository}/src"
"${python}" "${repository}/scripts/staged_function_granularity_campaign.py" freeze --config "${config}" --source-function-plan "${AF_SOURCE_FUNCTION_PLAN}" --output "${plan}"
if [[ -f "${manifest}" ]]; then
  cat "${manifest}"
  exit 0
fi
mkdir "${output_root}/submission.intent" 2>/dev/null || {
  echo 'Submission may already exist; reconcile the queue before retrying.' >&2
  exit 3
}
job_id="$(
  sbatch --parsable --account="${account}" --open-mode=append \
    --output="${output_root}/logs/staged-function-granularity-%j.out" \
    --error="${output_root}/logs/staged-function-granularity-%j.err" \
    --export=ALL,AF_REPO_ROOT="${repository}",AF_PYTHON="${python}",AF_OUTPUT_ROOT="${output_root}",AF_VLLM_IMAGE="${image}",AF_HF_HOME="${AF_HF_HOME:-${scratch_root}/huggingface-cache}",AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-${scratch_root}/compute-cache}",AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-${scratch_root}/ipc}" \
    "${repository}/scripts/hpc/staged_function_granularity_aces.slurm"
)"
job_id="${job_id%%;*}"
commit="$(git -C "${repository}" rev-parse HEAD)"
"${python}" - "${manifest}" "${job_id}" "${plan}" "${AF_SOURCE_FUNCTION_PLAN}" "${commit}" <<'PY'
import hashlib
import sys
from pathlib import Path
from autoformalism.llm.staged_topology import atomic_json
destination = Path(sys.argv[1])
job_id = sys.argv[2]
plan_path = Path(sys.argv[3])
source_path = Path(sys.argv[4])
commit = sys.argv[5]
payload = {
    "schema_version": "scientific-staged-function-granularity-submission-1",
    "job_id": job_id,
    "commit": commit,
    "plan_path": str(plan_path),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "source_function_plan": str(source_path),
    "source_function_plan_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
}
atomic_json(destination, payload)
PY
printf 'ACES_FUNCTION_GRANULARITY_JOB=%s\nACES_FUNCTION_GRANULARITY_ROOT=%s\n' "${job_id}" "${output_root}"
