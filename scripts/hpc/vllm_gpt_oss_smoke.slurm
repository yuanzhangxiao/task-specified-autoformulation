#!/bin/bash
#SBATCH --job-name=vllm-gpt-oss-smoke
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/vllm-gpt-oss-smoke-%j.out
#SBATCH --error=logs/vllm-gpt-oss-smoke-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_VLLM_IMAGE:=${AF_PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_HF_HOME:=${AF_PROJECT}/huggingface-cache}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-20b}"
: "${AF_VLLM_SMOKE_ROOT:=${AF_WORK}/phase_b/vllm-gpt-oss-smoke-v1}"
: "${AF_APPTAINER_TMP_MIN_GIB:=40}"
: "${AF_APPTAINER_MKSQUASHFS_PROCESSORS:=2}"

readonly port=$((20000 + SLURM_JOB_ID % 20000))
readonly endpoint="http://127.0.0.1:${port}"
readonly server_log="${AF_VLLM_SMOKE_ROOT}/vllm-${SLURM_JOB_ID}.log"
readonly response_path="${AF_VLLM_SMOKE_ROOT}/response-${SLURM_JOB_ID}.json"
readonly apptainer_cache="${AF_PROJECT}/apptainer-cache"
readonly node_local_tmp_root="${SLURM_TMPDIR:-/tmp/${af_user}}"
readonly apptainer_tmp="${node_local_tmp_root}/apptainer-${SLURM_JOB_ID}"

server_pid=""
partial_image=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  if [[ -n "${partial_image}" ]]; then
    rm -f -- "${partial_image}"
  fi
  rm -rf -- "${apptainer_tmp}"
}
trap cleanup EXIT

mkdir -p \
  "${AF_REPO_ROOT}/logs" \
  "${AF_VLLM_SMOKE_ROOT}" \
  "${AF_HF_HOME}" \
  "${apptainer_cache}" \
  "${apptainer_tmp}" \
  "$(dirname "${AF_VLLM_IMAGE}")"
cd "${AF_REPO_ROOT}"

export APPTAINER_CACHEDIR="${apptainer_cache}"
export APPTAINER_TMPDIR="${apptainer_tmp}"
export APPTAINER_MKSQUASHFS_ARGS="-processors ${AF_APPTAINER_MKSQUASHFS_PROCESSORS}"

readonly available_tmp_kib="$(
  df -Pk "${apptainer_tmp}" | awk 'NR == 2 {print $4}'
)"
readonly required_tmp_kib=$((AF_APPTAINER_TMP_MIN_GIB * 1024 * 1024))
echo "Apptainer build temp: ${apptainer_tmp}"
df -hP "${apptainer_tmp}"
if ((available_tmp_kib < required_tmp_kib)); then
  echo \
    "Apptainer build temp has less than ${AF_APPTAINER_TMP_MIN_GIB} GiB free" \
    >&2
  exit 1
fi
echo \
  "Apptainer SquashFS processors: ${AF_APPTAINER_MKSQUASHFS_PROCESSORS}"

if [[ ! -f "${AF_VLLM_IMAGE}" ]]; then
  partial_image="${AF_VLLM_IMAGE}.partial-${SLURM_JOB_ID}"
  apptainer pull "${partial_image}" "${AF_VLLM_IMAGE_URI}"
  mv "${partial_image}" "${AF_VLLM_IMAGE}"
  partial_image=""
fi

sha256sum "${AF_VLLM_IMAGE}"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader

apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${AF_VLLM_IMAGE}" \
  vllm serve "${AF_LOCAL_MODEL}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --max-model-len 16384 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 900); do
  if curl --silent --fail "${endpoint}/v1/models" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see ${server_log}" >&2
    tail -n 100 "${server_log}" >&2
    exit 1
  fi
  sleep 2
done
if [[ "${ready}" != true ]]; then
  echo "vLLM did not become ready; see ${server_log}" >&2
  tail -n 100 "${server_log}" >&2
  exit 1
fi

curl --silent --show-error --fail \
  "${endpoint}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(
    "${AF_PYTHON}" - "${AF_LOCAL_MODEL}" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [
        {
            "role": "system",
            "content": "Return one schema-conforming scientific assessment.",
        },
        {
            "role": "user",
            "content": (
                "A meal input adds material to a positive balance. Is subtracting "
                "the absolute meal input source-role consistent?"
            ),
        },
    ],
    "stream": False,
    "reasoning_effort": "low",
    "temperature": 0.2,
    "seed": 9000,
    "max_tokens": 512,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "smoke_assessment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "evidence": {"type": "string", "minLength": 1},
                },
                "required": ["verdict", "evidence"],
                "additionalProperties": False,
            },
        },
    },
}))
PY
  )" \
  --output "${response_path}"

"${AF_PYTHON}" - "${response_path}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choice = payload["choices"][0]
message = choice["message"]
content = message.get("content")
if not isinstance(content, str) or not content.strip():
    raise SystemExit(f"empty vLLM final content: {message}")
assessment = json.loads(content)
if assessment.get("verdict") not in {"pass", "fail"}:
    raise SystemExit(f"invalid verdict: {assessment}")
if not isinstance(assessment.get("evidence"), str) or not assessment["evidence"]:
    raise SystemExit(f"invalid evidence: {assessment}")
print(json.dumps({
    "finish_reason": choice.get("finish_reason"),
    "reasoning_present": bool(
        message.get("reasoning") or message.get("reasoning_content")
    ),
    "assessment": assessment,
}, indent=2, sort_keys=True))
PY
