#!/bin/bash
# Submit the Phase-B LLM-SR campaign against a local gpt-oss-120b endpoint.
#
#   AF_TIER          easy | hard | all   (default easy)
#   AF_BATCH_SIZE    tasks per GPU job   (default 4)
#   AF_TASK_INDICES  explicit comma list, overrides AF_TIER (smoke: "0")
#                    The list reaches the job as a file, never through
#                    --export, which is itself comma separated.
#   AF_BATCH_HOURS   walltime per batch  (default 12:00:00)
#   AF_LLM_SR_ROOT  the pinned upstream checkout
#
# Each task is a full island search -- 200 iterations x 4 islands per target --
# so batches are small and walltime long, unlike D3's single-shot tasks.

set -euo pipefail
# These are produced by this script and handed to the job. Slurm's
# --export=ALL also forwards the submitting shell, so a stale value left over
# from an earlier run could reach the job instead of the one resolved here.
# Refuse rather than rely on the caller remembering `env -u`.
for internal in AF_SUBMISSION_DIR AF_TASK_INDEX_FILE; do
  if [[ -n "${!internal:-}" ]]; then
    echo "${internal} is set in the environment; unset it and resubmit" >&2
    exit 2
  fi
done
readonly af_user="${USER:?}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_LLM_SR_OUTPUT_ROOT:=${AF_WORK}/phase_b/llm-sr-vllm-120b-v1}"
: "${AF_LLM_SR_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_llm_sr_six_cell_v1.json}"
: "${AF_LLM_SR_ROOT:=${AF_PROJECT}/vendor/LLM-SR}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TIER:=easy}"
: "${AF_BATCH_SIZE:=4}"
: "${AF_BATCH_HOURS:=12:00:00}"
: "${AF_CLUSTER:=delta}"
case "${AF_CLUSTER}" in
  delta)
    # No H100 on Delta; A40 is Ampere, so the MXFP4 weights are dequantized
    # and need four cards.
    : "${AF_ACCOUNT:=bibo-delta-gpu}"
    : "${AF_GPU_PARTITION:=gpuA40x4}"
    : "${AF_GPU_REQUEST:=--gpus-per-node=4}"
    : "${AF_TENSOR_PARALLEL_SIZE:=4}"
    ;;
  aces)
    # H100 is Hopper: native MXFP4, one card.
    : "${AF_ACCOUNT:?set AF_ACCOUNT to the ACES GPU account from `accounts`}"
    : "${AF_GPU_PARTITION:=gpu}"
    : "${AF_GPU_REQUEST:=--gres=gpu:h100:1}"
    : "${AF_TENSOR_PARALLEL_SIZE:=1}"
    ;;
  *) echo "unknown AF_CLUSTER: ${AF_CLUSTER}" >&2; exit 2 ;;
esac

# Fail here rather than in every batch job: a missing or wrong checkout makes
# the whole campaign unfaithful, and the queue wait would be wasted.
[[ -f "${AF_LLM_SR_ROOT}/llmsr/pipeline.py" ]] || {
  echo "not an LLM-SR checkout: ${AF_LLM_SR_ROOT}" >&2
  echo "clone it with:" >&2
  echo "  git clone https://github.com/deep-symbolic-mathematics/LLM-SR ${AF_LLM_SR_ROOT}" >&2
  exit 2
}
pinned="$("${AF_PYTHON}" -c "
import json, sys
sys.stdout.write(json.load(open('${AF_LLM_SR_CONFIG}'))['upstream']['commit'])
")"
actual="$(git -C "${AF_LLM_SR_ROOT}" rev-parse HEAD)"
[[ "${pinned}" == "${actual}" ]] || {
  echo "checkout is at ${actual}, not the pinned ${pinned}" >&2
  echo "  git -C ${AF_LLM_SR_ROOT} checkout ${pinned}" >&2
  exit 2
}

cd "${AF_REPO_ROOT}"
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs "${AF_LLM_SR_OUTPUT_ROOT}/logs"

indices="${AF_TASK_INDICES:-$("${AF_PYTHON}" scripts/list_phase_b_d3_task_indices.py \
  --config "${AF_LLM_SR_CONFIG}" --tier "${AF_TIER}")}"
[[ -n "${indices}" ]] || { echo "no tasks selected for tier ${AF_TIER}" >&2; exit 2; }
readonly task_count="$(tr ',' '\n' <<< "${indices}" | wc -l | tr -d ' ')"
readonly batches=$(( (task_count + AF_BATCH_SIZE - 1) / AF_BATCH_SIZE ))
echo "tier=${AF_TIER} tasks=${task_count} batch_size=${AF_BATCH_SIZE} batches=${batches}"
echo "output_root=${AF_LLM_SR_OUTPUT_ROOT}"
echo "upstream=${AF_LLM_SR_ROOT} commit=${actual}"
echo "cluster=${AF_CLUSTER} account=${AF_ACCOUNT} partition=${AF_GPU_PARTITION}"
echo "gpu=${AF_GPU_REQUEST} tensor_parallel=${AF_TENSOR_PARALLEL_SIZE}"

# Slurm splits --export on commas, so a comma-separated value arrives
# truncated at its first element. Hand over a path instead.
#
# One directory per submission, never reused: a retry or smoke run for the
# same tier must not rewrite the list an already-queued array will read when
# it finally starts. That would reproduce wrong-task execution with no comma
# involved at all.
readonly submission_id="${AF_TIER}-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
readonly submission_dir="${AF_LLM_SR_OUTPUT_ROOT}/submissions/${submission_id}"
[[ -e "${submission_dir}" ]] && {
  echo "submission directory already exists: ${submission_dir}" >&2
  exit 2
}
mkdir -p "${submission_dir}"
readonly index_file="${submission_dir}/tasks.txt"
tr ',' '\n' <<< "${indices}" | grep -v '^$' > "${index_file}"
readonly tasks_sha256="$("${AF_PYTHON}" -c "
import hashlib, sys
sys.stdout.write(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
" "${index_file}")"
echo "submission=${submission_id}"
echo "index_file=${index_file} ($(wc -l < "${index_file}" | tr -d ' ') entries)"

readonly common="ALL,AF_MODULES=${AF_MODULES:-},AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_LLM_SR_OUTPUT_ROOT=${AF_LLM_SR_OUTPUT_ROOT},AF_LLM_SR_ROOT=${AF_LLM_SR_ROOT},AF_LOCAL_MODEL=${AF_LOCAL_MODEL},AF_SUBMISSION_DIR=${submission_dir},AF_BATCH_SIZE=${AF_BATCH_SIZE},AF_LLM_SR_CONFIG=${AF_LLM_SR_CONFIG},AF_PUBLIC_ROOT=${AF_PUBLIC_ROOT},AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE}"

export AF_SUBMISSION_ID="${submission_id}"
export AF_SUBMISSION_BATCHES="${batches}"
export AF_SUBMISSION_MODEL="${AF_LOCAL_MODEL}"
export AF_SUBMISSION_CONFIG="${AF_LLM_SR_CONFIG}"
export AF_SUBMISSION_OUTPUT_ROOT="${AF_LLM_SR_OUTPUT_ROOT}"
export AF_TIER AF_BATCH_SIZE AF_REPO_ROOT AF_ACCOUNT AF_GPU_PARTITION AF_GPU_REQUEST
# The manifest is what the job validates against before it loads a model.
"${AF_PYTHON}" - "${submission_dir}" <<'PYEOF'
import hashlib, json, os, sys
from pathlib import Path

directory = Path(sys.argv[1])
tasks = [line for line in (directory / "tasks.txt").read_text().split() if line]
manifest = {
    "schema_version": "phase-b-submission-manifest-1",
    "submission_id": os.environ["AF_SUBMISSION_ID"],
    "tier": os.environ["AF_TIER"],
    "task_count": len(tasks),
    "tasks_sha256": hashlib.sha256(
        (directory / "tasks.txt").read_bytes()
    ).hexdigest(),
    "batch_size": int(os.environ["AF_BATCH_SIZE"]),
    "batches": int(os.environ["AF_SUBMISSION_BATCHES"]),
    "model": os.environ["AF_SUBMISSION_MODEL"],
    "config": os.environ["AF_SUBMISSION_CONFIG"],
    "config_sha256": hashlib.sha256(
        Path(os.environ["AF_SUBMISSION_CONFIG"]).read_bytes()
    ).hexdigest(),
    "repo_root": os.environ["AF_REPO_ROOT"],
    "output_root": os.environ["AF_SUBMISSION_OUTPUT_ROOT"],
    "account": os.environ["AF_ACCOUNT"],
    "partition": os.environ["AF_GPU_PARTITION"],
    "gpu_request": os.environ["AF_GPU_REQUEST"],
}
(directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PYEOF

batch_job="$(sbatch --parsable --account="${AF_ACCOUNT}" \
  --partition="${AF_GPU_PARTITION}" --time="${AF_BATCH_HOURS}" \
  ${AF_GPU_REQUEST} \
  --array="0-$((batches - 1))%${AF_MAX_CONCURRENT_BATCHES:-2}" \
  --output="${AF_LLM_SR_OUTPUT_ROOT}/logs/batch-%A_%a.out" \
  --export="${common}" \
  scripts/hpc/phase_b_llm_sr_vllm_batch.slurm)"
echo "BATCH_JOB=${batch_job}"

jq -n --arg tier "${AF_TIER}" --arg model "${AF_LOCAL_MODEL}" --arg indices "${indices}" \
  --arg batch "${batch_job}" --arg commit "${actual}" \
  --argjson tasks "${task_count}" --argjson batches "${batches}" \
  '{schema_version:"phase-b-llm-sr-vllm-submission-1", tier:$tier, model:$model,
    task_count:$tasks, batches:$batches, task_indices:$indices,
    upstream_commit:$commit, batch_job:$batch}' \
  | tee -a "${AF_LLM_SR_OUTPUT_ROOT}/submission_ledger.jsonl"
