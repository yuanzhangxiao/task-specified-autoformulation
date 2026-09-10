#!/bin/bash
# Freeze and submit the public function-first two-round feedback pilot.

set -euo pipefail
: "${AF_SOURCE_RESCUE_ROOT:?set AF_SOURCE_RESCUE_ROOT to the completed staged-fitter-rescue root}"
readonly repository="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
readonly python="${AF_PYTHON:-${repository}/.venv/bin/python}"
readonly scratch_root="${SCRATCH:-/scratch/user/${USER}}"
readonly project_root="${PROJECT:-${scratch_root}}"
readonly output_root="${AF_OUTPUT_ROOT:-${scratch_root}/phase_b/staged-multiround-feedback-v2-aces-h100x1}"
readonly account="${AF_ACCOUNT:-156264627414}"
readonly config="${AF_CONFIG:-${repository}/configs/staged_multiround_feedback_v2.json}"
readonly plan="${output_root}/plan.json"
readonly manifest="${output_root}/submission_manifest.json"
git -C "${repository}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "missing repository: ${repository}" >&2
  exit 2
}
[[ -x "${python}" ]] || { echo "missing Python: ${python}" >&2; exit 2; }
[[ -f "${config}" ]] || { echo "missing config: ${config}" >&2; exit 2; }
[[ -f "${AF_SOURCE_RESCUE_ROOT}/plan.json" ]] || { echo "missing source rescue plan" >&2; exit 2; }
[[ -f "${AF_SOURCE_RESCUE_ROOT}/summary/summary.json" ]] || { echo "missing source rescue summary" >&2; exit 2; }
expected_image_sha="$(jq -r '.serving_image_sha256' "${config}")"
if [[ -n "${AF_VLLM_IMAGE:-}" ]]; then
  image="${AF_VLLM_IMAGE}"
else
  image=""
  while IFS= read -r -d '' candidate; do
    candidate_sha="$(sha256sum "${candidate}")"
    if [[ "${candidate_sha%% *}" == "${expected_image_sha}" ]]; then
      image="${candidate}"
      break
    fi
  done < <(find "${scratch_root}" "${project_root}" -maxdepth 4 -type f -name '*.sif' -print0 2>/dev/null)
fi
[[ -f "${image}" ]] || { echo "missing vLLM image: ${image}" >&2; exit 2; }
actual_image_sha="$(sha256sum "${image}")"
[[ "${actual_image_sha%% *}" == "${expected_image_sha}" ]] || {
  echo "vLLM image does not match frozen config SHA" >&2
  exit 2
}
readonly image
[[ -z "$(git -C "${repository}" status --porcelain)" ]] || {
  echo "multiround checkout must be clean before freezing" >&2
  exit 2
}
mkdir -p "${output_root}/logs"
export PYTHONPATH="${repository}/src"
"${python}" "${repository}/scripts/staged_multiround_feedback_campaign.py" freeze --config "${config}" --source-rescue-root "${AF_SOURCE_RESCUE_ROOT}" --output "${output_root}"
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
    --output="${output_root}/logs/multiround-feedback-%j.out" \
    --error="${output_root}/logs/multiround-feedback-%j.err" \
    --export=ALL,AF_REPO_ROOT="${repository}",AF_PYTHON="${python}",AF_OUTPUT_ROOT="${output_root}",AF_VLLM_IMAGE="${image}",AF_HF_HOME="${AF_HF_HOME:-${scratch_root}/huggingface-cache}",AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-${scratch_root}/compute-cache}",AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-${scratch_root}/ipc}" \
    "${repository}/scripts/hpc/staged_multiround_feedback_aces.slurm"
)"
job_id="${job_id%%;*}"
commit="$(git -C "${repository}" rev-parse HEAD)"
"${python}" - "${manifest}" "${job_id}" "${plan}" "${AF_SOURCE_RESCUE_ROOT}" "${commit}" <<'PY'
import hashlib
import sys
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json

destination = Path(sys.argv[1])
job_id = sys.argv[2]
plan_path = Path(sys.argv[3])
source_root = Path(sys.argv[4])
commit = sys.argv[5]
payload = {
    "schema_version": "scientific-staged-multiround-feedback-submission-2",
    "job_id": job_id,
    "commit": commit,
    "plan_path": str(plan_path),
    "plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "source_rescue_root": str(source_root),
    "source_rescue_plan_file_sha256": hashlib.sha256(
        (source_root / "plan.json").read_bytes()
    ).hexdigest(),
}
atomic_json(destination, payload)
PY
printf 'ACES_MULTIROUND_FEEDBACK_JOB=%s\nACES_MULTIROUND_FEEDBACK_ROOT=%s\n' "${job_id}" "${output_root}"
