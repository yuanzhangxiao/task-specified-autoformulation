#!/bin/bash
# Freeze, audit, and submit the public-only pre-fitting feedback experiment.
set -euo pipefail
: "${AF_SOURCE_HYBRID_ROOT:?set AF_SOURCE_HYBRID_ROOT to the completed hybrid-v2 root}"
readonly repository="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
readonly python="${AF_PYTHON:-${repository}/.venv/bin/python}"
readonly scratch_root="${SCRATCH:-/scratch/user/${USER}}"
readonly project_root="${PROJECT:-${scratch_root}}"
readonly output_root="${AF_OUTPUT_ROOT:-${scratch_root}/phase_b/staged-prefit-feedback-v1-aces-h100x1}"
readonly image="${AF_VLLM_IMAGE:-${project_root}/containers/vllm-openai-v0.27.1.sif}"
readonly account="${AF_ACCOUNT:-156264627414}"
readonly config="${AF_CONFIG:-${repository}/configs/staged_prefit_feedback_v1.json}"
readonly source_plan="${AF_SOURCE_HYBRID_ROOT}/plan.json"
readonly source_results="${AF_SOURCE_HYBRID_ROOT}/results"
readonly plan="${output_root}/plan.json"
readonly audit="${output_root}/source_audit.json"
readonly manifest="${output_root}/submission_manifest.json"
git -C "${repository}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "missing repository: ${repository}" >&2; exit 2; }
[[ -x "${python}" ]] || { echo "missing Python: ${python}" >&2; exit 2; }
[[ -f "${source_plan}" ]] || { echo "missing source hybrid plan: ${source_plan}" >&2; exit 2; }
[[ -f "${source_results}/summary.json" ]] || { echo "missing source hybrid summary" >&2; exit 2; }
[[ -f "${image}" ]] || { echo "missing vLLM image: ${image}" >&2; exit 2; }
mkdir -p "${output_root}/logs"
export PYTHONPATH="${repository}/src"
"${python}" "${repository}/scripts/staged_prefit_feedback_campaign.py" freeze --config "${config}" --source-hybrid-plan "${source_plan}" --source-results "${source_results}" --output "${plan}"
"${python}" "${repository}/scripts/staged_prefit_feedback_campaign.py" audit --plan "${plan}" --output "${audit}" >/dev/null
jq -e '.deterministic_prefit_pass_rate == 1' "${audit}" >/dev/null || { echo "source deterministic prefit audit failed" >&2; exit 2; }
if [[ -f "${manifest}" ]]; then cat "${manifest}"; exit 0; fi
mkdir "${output_root}/submission.intent" 2>/dev/null || { echo 'Submission may already exist; reconcile the queue before retrying.' >&2; exit 3; }
job_id="$(sbatch --parsable --account="${account}" --open-mode=append --output="${output_root}/logs/staged-prefit-feedback-%j.out" --error="${output_root}/logs/staged-prefit-feedback-%j.err" --export=ALL,AF_REPO_ROOT="${repository}",AF_PYTHON="${python}",AF_OUTPUT_ROOT="${output_root}",AF_VLLM_IMAGE="${image}",AF_HF_HOME="${AF_HF_HOME:-${scratch_root}/huggingface-cache}",AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-${scratch_root}/compute-cache}",AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-${scratch_root}/ipc}" "${repository}/scripts/hpc/staged_prefit_feedback_aces.slurm")"
job_id="${job_id%%;*}"
commit="$(git -C "${repository}" rev-parse HEAD)"
"${python}" - "${manifest}" "${job_id}" "${plan}" "${audit}" "${source_plan}" "${commit}" <<'PY'
import hashlib
import sys
from pathlib import Path
from autoformalism.llm.staged_topology import atomic_json
destination, job_id, plan, audit, source, commit = sys.argv[1:]
paths = [Path(plan), Path(audit), Path(source)]
atomic_json(Path(destination), {
    "schema_version": "scientific-staged-prefit-feedback-submission-1",
    "job_id": job_id,
    "commit": commit,
    "plan_path": plan,
    "audit_path": audit,
    "source_hybrid_plan": source,
    "file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
})
PY
printf 'ACES_PREFIT_FEEDBACK_JOB=%s\nACES_PREFIT_FEEDBACK_ROOT=%s\n' "${job_id}" "${output_root}"
