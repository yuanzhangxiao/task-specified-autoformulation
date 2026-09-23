#!/bin/bash
#SBATCH --job-name=llmsr-phaseb-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
# No GPU directive here: ACES wants --gres=gpu:h100:N and Delta wants
# --gpus-per-node=N, so the submitter passes AF_GPU_REQUEST on the
# command line, where it overrides any directive.

# One GPU job per batch: load gpt-oss-120b once, then run this batch's LLM-SR
# tasks against the local endpoint. Each task is a long island search, so the
# batches are smaller and the walltime longer than D3's.

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_LLM_SR_OUTPUT_ROOT:?AF_LLM_SR_OUTPUT_ROOT is required}"
: "${AF_LLM_SR_ROOT:?AF_LLM_SR_ROOT must point at the pinned checkout}"
# Slurm's --export is a comma-separated list, so a comma-separated *value*
# is silently truncated at its first element: a 60-task list arrived as one
# task and the array reported success having run almost nothing. The list
# travels as a file, whose path holds no commas.
: "${AF_SUBMISSION_DIR:=}"
: "${AF_TASK_INDEX_FILE:=}"
: "${AF_BATCH_SIZE:?AF_BATCH_SIZE is required}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=1}"
# Their prompts carry the whole specification plus in-context example
# functions, so they are far longer than an equation list; 16k leaves room
# without making the KV cache expensive.
: "${AF_VLLM_MAX_MODEL_LEN:=16384}"
# Every sampled program is executed in its own process with a 30 second
# timeout, so a task is CPU-bound between calls; keep few tasks in flight.
: "${AF_TASK_CONCURRENCY:=4}"
: "${AF_LLM_SR_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_llm_sr_six_cell_v1.json}"
: "${AF_PUBLIC_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_HF_HOME:=${AF_WORK}/hf}"
: "${AF_VLLM_IMAGE:=${AF_WORK}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_APPTAINER_TMP_MIN_GIB:=40}"

readonly batch="${SLURM_ARRAY_TASK_ID:?submit as an array over batches}"
# Assign these paths, never mktemp: the parent may not exist yet, and under
# set -e that kills the job before it can log why.
readonly apptainer_cache="${AF_PROJECT}/apptainer-cache"
readonly node_local_tmp_root="${SLURM_TMPDIR:-/tmp/${af_user}}"
readonly apptainer_tmp="${node_local_tmp_root}/apptainer-${SLURM_JOB_ID}"
cleanup() {
  [[ -n "${server_pid:-}" ]] && kill "${server_pid}" 2>/dev/null || true
  rm -rf -- "${apptainer_tmp}"
}
trap cleanup EXIT

# Site initialization, opt-in. A venv built from a module-provided Python
# generally needs that module loaded to find its runtime, and a batch script
# does not inherit the submitting shell's module environment. `module` is a
# shell function from the site profile, which a non-login shell may not have.
if [[ -n "${AF_MODULES:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                /usr/share/lmod/lmod/init/bash; do
      [[ -r "${init}" ]] && source "${init}" && break
    done
  fi
  command -v module >/dev/null 2>&1 || {
    echo "AF_MODULES is set but 'module' is unavailable in this shell" >&2
    exit 1
  }
  # shellcheck disable=SC2086
  module load ${AF_MODULES}
  echo "modules=${AF_MODULES}"
fi

echo "batch job starting: host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "output_root=${AF_LLM_SR_OUTPUT_ROOT} repo=${AF_REPO_ROOT}"
export APPTAINER_CACHEDIR="${apptainer_cache}" APPTAINER_TMPDIR="${apptainer_tmp}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
# The driver imports the checkout by path; upstream is not installed.
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_LLM_SR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${AF_HF_HOME}" "${apptainer_cache}" "${apptainer_tmp}" \
  "$(dirname "${AF_VLLM_IMAGE}")" "${AF_REPO_ROOT}/logs" \
  "${AF_LLM_SR_OUTPUT_ROOT}/logs"
cd "${AF_REPO_ROOT}"

# Faithfulness is a claim about a specific revision, so check it rather than
# trusting whatever happens to be checked out.
readonly pinned="$(${AF_PYTHON} -c "
import json, sys
config = json.load(open('${AF_LLM_SR_CONFIG}'))
sys.stdout.write(config['upstream']['commit'])
")"
readonly actual="$(git -C "${AF_LLM_SR_ROOT}" rev-parse HEAD)"
if [[ "${pinned}" != "${actual}" ]]; then
  echo "LLM-SR checkout is at ${actual}, not the pinned ${pinned}" >&2
  exit 1
fi
echo "llm_ode_checkout=${AF_LLM_SR_ROOT} commit=${actual}"

# This batch's slice of the task list.
if [[ -n "${AF_SUBMISSION_DIR}" ]]; then
  # Validate the submission before any GPU work: the membership a job acts on
  # must be the membership that was submitted, and a mismatch is far cheaper
  # to find here than after a model load.
  "${AF_PYTHON}" - "${AF_SUBMISSION_DIR}" "${AF_BATCH_SIZE}" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

directory = Path(sys.argv[1])
manifest = json.loads((directory / "manifest.json").read_text())
raw = (directory / "tasks.txt").read_bytes()
if hashlib.sha256(raw).hexdigest() != manifest["tasks_sha256"]:
    sys.exit("task list does not match the manifest it was submitted with")
tasks = [item for item in raw.decode().split() if item]
if len(tasks) != manifest["task_count"]:
    sys.exit(f"expected {manifest['task_count']} tasks, found {len(tasks)}")
if not tasks:
    sys.exit("submission carries no tasks")
if len(set(tasks)) != len(tasks):
    sys.exit("submission assigns the same task twice")
if not all(item.isdigit() for item in tasks):
    sys.exit(f"malformed task ids: {[i for i in tasks if not i.isdigit()][:5]}")
if int(sys.argv[2]) != manifest["batch_size"]:
    sys.exit(
        f"batch size {sys.argv[2]} differs from the submitted "
        f"{manifest['batch_size']}"
    )
PYEOF
  echo "submission=$(basename "${AF_SUBMISSION_DIR}") verified"
  AF_TASK_INDEX_FILE="${AF_SUBMISSION_DIR}/tasks.txt"
fi
if [[ -n "${AF_TASK_INDEX_FILE}" ]]; then
  [[ -f "${AF_TASK_INDEX_FILE}" ]] || {
    echo "AF_TASK_INDEX_FILE does not exist: ${AF_TASK_INDEX_FILE}" >&2
    exit 1
  }
  # read, not mapfile: bash 3.2 has no mapfile and the local harness uses it
  all_indices=()
  while IFS= read -r index_line; do
    [[ -n "${index_line}" ]] && all_indices+=("${index_line}")
  done < "${AF_TASK_INDEX_FILE}"
else
  : "${AF_TASK_INDICES:?set AF_TASK_INDEX_FILE or AF_TASK_INDICES}"
  IFS=',' read -r -a all_indices <<< "${AF_TASK_INDICES}"
fi
((${#all_indices[@]})) || { echo "no task indices supplied" >&2; exit 1; }
echo "task_list_size=${#all_indices[@]}"
readonly slice_start=$((batch * AF_BATCH_SIZE))
batch_indices=("${all_indices[@]:${slice_start}:${AF_BATCH_SIZE}}")
((${#batch_indices[@]})) || { echo "batch ${batch} is empty"; exit 0; }
echo "batch=${batch} tasks=${batch_indices[*]}"
echo "model=${AF_LOCAL_MODEL} tp=${AF_TENSOR_PARALLEL_SIZE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

readonly available_tmp_kib="$(df -Pk "${apptainer_tmp}" | awk 'NR == 2 {print $4}')"
if ((available_tmp_kib < AF_APPTAINER_TMP_MIN_GIB * 1024 * 1024)); then
  echo "Apptainer temp has less than ${AF_APPTAINER_TMP_MIN_GIB} GiB free" >&2
  exit 1
fi

runtime_image="${AF_VLLM_IMAGE}"
if [[ ! -f "${runtime_image}" ]]; then
  runtime_image="${apptainer_tmp}/vllm-openai-sandbox"
  echo "Building job-local Apptainer sandbox from ${AF_VLLM_IMAGE_URI}"
  apptainer build --sandbox "${runtime_image}" "${AF_VLLM_IMAGE_URI}"
fi

# The lock runs a new shell, and ":=" defaults are not exported, so the inner
# shell would see them unset under set -u.
export AF_LLM_SR_OUTPUT_ROOT AF_PYTHON AF_LLM_SR_CONFIG AF_PUBLIC_ROOT

# Freeze the plan here rather than in a separate CPU job: ACES rejects CPU jobs
# submitted under a GPU account, and prepare is idempotent.
flock "${AF_LLM_SR_OUTPUT_ROOT}/prepare.shlock" bash -c '
  set -euo pipefail
  if [[ ! -f "${AF_LLM_SR_OUTPUT_ROOT}/plan.json" ]]; then
    echo "freezing the campaign plan"
    "${AF_PYTHON}" scripts/phase_b_llm_sr.py prepare \
      --config "${AF_LLM_SR_CONFIG}" --public-root "${AF_PUBLIC_ROOT}" \
      --root "${AF_LLM_SR_OUTPUT_ROOT}"
  else
    echo "reusing the sealed plan"
  fi
'

readonly port=$((20000 + RANDOM % 20000))
readonly endpoint="http://127.0.0.1:${port}"
# Include the job id: two tiers submitted against one output root both number
# their batches from zero and would otherwise share this file.
readonly server_log="${AF_LLM_SR_OUTPUT_ROOT}/logs/vllm-${SLURM_JOB_ID}-batch-${batch}.log"

vllm_parallel_args=()
if ((AF_TENSOR_PARALLEL_SIZE > 1)); then
  vllm_parallel_args=(--tensor-parallel-size "${AF_TENSOR_PARALLEL_SIZE}")
fi
# Upstream queries the Responses API and reads the first served model id, so
# exactly one model is served here.
apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${runtime_image}" \
  vllm serve "${AF_LOCAL_MODEL}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --max-model-len "${AF_VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs $((AF_TASK_CONCURRENCY * 4)) \
    --gpu-memory-utilization 0.90 \
    "${vllm_parallel_args[@]}" \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 450); do
  if curl --silent --fail --max-time 5 "${endpoint}/v1/models" >/dev/null; then ready=true; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see ${server_log}" >&2
    tail -n 100 "${server_log}" >&2
    exit 1
  fi
  sleep 2
done
[[ "${ready}" == true ]] || {
  echo "vLLM did not become ready; see ${server_log}" >&2
  tail -n 100 "${server_log}" >&2
  exit 1
}
export AF_VLLM_BASE_URL="${endpoint}"
export AF_LLM_SR_ROOT

# Completed tasks resume without a call; a failed task must not abandon the batch.
echo "task_concurrency=${AF_TASK_CONCURRENCY}"
failures=0
position=0
while ((position < ${#batch_indices[@]})); do
  group=("${batch_indices[@]:position:AF_TASK_CONCURRENCY}")
  pids=()
  for index in "${group[@]}"; do
    echo "== starting task ${index}"
    "${AF_PYTHON}" scripts/phase_b_llm_sr.py run \
      --root "${AF_LLM_SR_OUTPUT_ROOT}" --index "${index}" \
      >"${AF_LLM_SR_OUTPUT_ROOT}/logs/task-${index}.out" 2>&1 &
    pids+=("$!")
  done
  for slot in "${!pids[@]}"; do
    # run() records a terminal failure as a result rather than raising, so a
    # zero exit status does not mean the task discovered anything.
    if wait "${pids[slot]}"; then
      status="$(jq -r '.status // "missing"' \
        "${AF_LLM_SR_OUTPUT_ROOT}/results/${group[slot]}/result.json" 2>/dev/null \
        || echo missing)"
      if [[ "${status}" == complete ]]; then
        echo "task ${group[slot]} complete"
      else
        echo "task ${group[slot]} recorded ${status}" >&2
        failures=$((failures + 1))
      fi
    else
      echo "task ${group[slot]} failed; see logs/task-${group[slot]}.out" >&2
      failures=$((failures + 1))
    fi
  done
  position=$((position + AF_TASK_CONCURRENCY))
done
echo "batch=${batch} completed with ${failures} failed task(s) of ${#batch_indices[@]}"
# Tolerate a partial failure so one bad task cannot abandon its batch, but a
# batch where nothing succeeded must not report success.
if ((failures == ${#batch_indices[@]})); then
  echo "every task in batch ${batch} failed" >&2
  exit 1
fi
exit 0
