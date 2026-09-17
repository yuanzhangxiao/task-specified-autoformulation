#!/bin/bash
# Run only saved-model equation reviews. No search, fitting, or test access.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]] || { echo 'Pinned commit changed' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$AF_PYTHON" scripts/mechanism_audit.py report --root "$AF_OUTPUT_ROOT"
[[ "${1:-judge}" != report ]] || exit 0
: "${AF_VLLM_IMAGE:=/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=/scratch/user/u.yx126462/huggingface-cache}"
plan="$AF_OUTPUT_ROOT/plan.json"
actual="$(sha256sum "$AF_VLLM_IMAGE")"
[[ "${actual%% *}" == "$(jq -r .serving_image_sha256 "$plan")" ]] || { echo 'Image differs' >&2; exit 2; }
model="$(jq -r .settings.model "$plan")"
revision="$(jq -r .settings.model_revision "$plan")"
tp="$(jq -r .tensor_parallel_size "$plan")"
cache="$(mktemp -d /tmp/af-ma-XXXXXX)"
port="$("$AF_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
runtime="$(command -v apptainer || command -v singularity)"
server_pid='' worker_pid=''
cleanup() {
  status=$?
  trap - EXIT
  [[ -z "$worker_pid" ]] || kill -TERM "$worker_pid" 2>/dev/null || true
  [[ -z "$server_pid" ]] || kill -TERM "$server_pid" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
trap '[[ -z "$worker_pid" ]] || kill -TERM "$worker_pid" 2>/dev/null || true' TERM INT
mkdir -p "$AF_OUTPUT_ROOT/runtime"
log="$AF_OUTPUT_ROOT/runtime/server-${SLURM_JOB_ID:-local}.log"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
"$runtime" exec --nv --bind "$AF_HF_HOME:$AF_HF_HOME,$cache:$cache" \
  --env "HF_HOME=$AF_HF_HOME,HF_HUB_OFFLINE=1,VLLM_CACHE_ROOT=$cache/vllm,TRITON_CACHE_DIR=$cache/triton,TMPDIR=$cache" \
  "$AF_VLLM_IMAGE" vllm serve "$model" --revision "$revision" \
  --host 127.0.0.1 --port "$port" --tensor-parallel-size "$tp" \
  --max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.90 >"$log" 2>&1 &
server_pid=$!
ready=false
for _ in $(seq 1 300); do
  if curl --noproxy '*' --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null; then ready=true; break; fi
  kill -0 "$server_pid" 2>/dev/null || { tail -n 60 "$log"; exit 1; }
  sleep 2
done
[[ "$ready" == true ]] || { echo 'Server startup deadline exceeded' >&2; exit 1; }
"$AF_PYTHON" scripts/mechanism_audit.py judge --root "$AF_OUTPUT_ROOT" \
  --base-url "http://127.0.0.1:$port" --wall-seconds "${AF_WORKER_SECONDS:-19800}" &
worker_pid=$!
wait "$worker_pid"
worker_pid=''
"$AF_PYTHON" scripts/mechanism_audit.py report --root "$AF_OUTPUT_ROOT"
