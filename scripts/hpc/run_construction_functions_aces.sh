#!/bin/bash
# One local H100 server; function construction only, with no fitting or follow-up job.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_VLLM_IMAGE:?}" "${AF_HF_HOME:?}" "${SLURM_JOB_ID:?}"
[[ "${1:-propose}" == propose ]] || exit 2
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
AF_CLI="$AF_REPO_ROOT/scripts/construction_functions.py"
AF_SUBMIT_CLI="$AF_REPO_ROOT/scripts/submit_construction_functions_aces.py"
# Check imports and frozen source before starting any server or touching CUDA.
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  echo 'Python worker imports failed; loading the Python module and its library dependencies.' >&2
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=8
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost"
"$AF_PYTHON" "$AF_CLI" verify --root "$AF_OUTPUT_ROOT"
"$AF_PYTHON" "$AF_SUBMIT_CLI" --verify-only --root "$AF_OUTPUT_ROOT"
if ! AF_CONTAINER=$(command -v apptainer || command -v singularity); then
  printf 'No Apptainer or Singularity executable on compute node %s; server and model requests were not started.\n' "${SLURMD_NODENAME:-unknown}" >&2
  echo 'Check the node runtime with HPRC support before retrying.' >&2
  exit 2
fi
AF_EXPECTED_IMAGE=$("$AF_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_sha256"])' "$AF_OUTPUT_ROOT/execution.json")
AF_ACTUAL_IMAGE=$(sha256sum "$AF_VLLM_IMAGE")
[[ "${AF_ACTUAL_IMAGE%% *}" == "$AF_EXPECTED_IMAGE" ]] || { echo 'Serving image changed.' >&2; exit 2; }
AF_SNAPSHOT="$AF_HF_HOME/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json"
[[ -f "$AF_SNAPSHOT" ]] || { echo 'Pinned offline model snapshot is missing.' >&2; exit 2; }
AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
AF_HARMONY_CLI="$AF_REPO_ROOT/scripts/harmony_vocab.py"
"$AF_PYTHON" "$AF_HARMONY_CLI" verify --directory "$AF_HARMONY_ASSETS" >/dev/null
AF_RUNTIME="$AF_OUTPUT_ROOT/runtime"
mkdir -p "$AF_RUNTIME"
# Compiler caches stay on node-local temporary storage, outside scratch inode quotas.
AF_JOB_TMP=$(mktemp -d "/tmp/af-functions-${SLURM_JOB_ID}.XXXXXX")
AF_SERVER_PID=''
AF_WORKER_PID=''
AF_DRAINING=false
cleanup() {
  AF_EXIT=$?
  trap - EXIT
  [[ -z "$AF_WORKER_PID" ]] || kill -TERM "$AF_WORKER_PID" 2>/dev/null || true
  [[ -z "$AF_SERVER_PID" ]] || kill -TERM "$AF_SERVER_PID" 2>/dev/null || true
  rm -rf -- "$AF_JOB_TMP"
  exit "$AF_EXIT"
}
drain() {
  AF_DRAINING=true
  [[ -z "$AF_WORKER_PID" ]] || kill -TERM "$AF_WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap drain TERM INT
AF_CONTAINER_ARGS=(exec --bind "$AF_HF_HOME:$AF_HF_HOME,$AF_JOB_TMP:$AF_JOB_TMP,$AF_HARMONY_ASSETS:$AF_HARMONY_ASSETS:ro,$AF_HARMONY_CLI:$AF_HARMONY_CLI:ro"
  --env "HF_HOME=$AF_HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env "TIKTOKEN_ENCODINGS_BASE=$AF_HARMONY_ASSETS" --env PYTHONDONTWRITEBYTECODE=1
  --env "TRITON_CACHE_DIR=$AF_JOB_TMP/triton" --env "TORCHINDUCTOR_CACHE_DIR=$AF_JOB_TMP/torchinductor"
  --env "VLLM_CACHE_ROOT=$AF_JOB_TMP/vllm" --env "XDG_CACHE_HOME=$AF_JOB_TMP/xdg"
  --env "CUDA_CACHE_PATH=$AF_JOB_TMP/cuda" --env "TMPDIR=$AF_JOB_TMP")
AF_HARMONY_RECORD="$AF_RUNTIME/harmony-${SLURM_JOB_ID}.json"
"$AF_CONTAINER" "${AF_CONTAINER_ARGS[@]}" "$AF_VLLM_IMAGE" python3 "$AF_HARMONY_CLI" probe --directory "$AF_HARMONY_ASSETS" >"$AF_HARMONY_RECORD"
"$AF_CONTAINER" "${AF_CONTAINER_ARGS[@]}" --nv "$AF_VLLM_IMAGE" python3 - <<'PY' >"$AF_RUNTIME/probe-${SLURM_JOB_ID}.json"
import json, subprocess, torch, vllm
assert torch.cuda.device_count() == 1, 'Expected one allocated GPU'
assert 'H100' in torch.cuda.get_device_name(0), 'Expected one H100'
assert vllm.__version__ == '0.27.1', 'Expected pinned vLLM 0.27.1'
x = torch.ones(8, device='cuda')
assert x.sum().item() == 8
print(json.dumps({'visible_gpus': torch.cuda.device_count(), 'gpu_name': torch.cuda.get_device_name(0), 'cuda_kernel_passed': True, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'vllm': vllm.__version__, 'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip().splitlines()[0]}))
PY
# Save the tested CPU renderer and frozen server identity alongside CUDA evidence.
"$AF_PYTHON" - "$AF_RUNTIME/probe-${SLURM_JOB_ID}.json" "$AF_HARMONY_RECORD" "$AF_OUTPUT_ROOT/execution.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
record = json.loads(path.read_text())
record['harmony'] = json.loads(Path(sys.argv[2]).read_text())
execution = json.loads(Path(sys.argv[3]).read_text())
record['serving_identity'] = {key: execution[key] for key in (
    'artifact_sha256', 'model', 'model_revision', 'image_sha256',
    'served_context_tokens', 'tensor_parallel_size', 'maximum_sequences')}
path.write_text(json.dumps(record) + '\n')
PY
AF_PORT=$("$AF_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
AF_ENDPOINT="http://127.0.0.1:$AF_PORT"
"$AF_CONTAINER" "${AF_CONTAINER_ARGS[@]}" --nv "$AF_VLLM_IMAGE" vllm serve openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee --host 127.0.0.1 --port "$AF_PORT" \
  --max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  >"$AF_RUNTIME/server-${SLURM_JOB_ID}.log" 2>&1 &
AF_SERVER_PID=$!
AF_READY=false
for AF_ATTEMPT in $(seq 1 450); do
  [[ "$AF_DRAINING" == false ]] || exit 0
  if curl --noproxy '*' --silent --fail "$AF_ENDPOINT/v1/models" >/dev/null; then AF_READY=true; break; fi
  if ! kill -0 "$AF_SERVER_PID" 2>/dev/null; then tail -n 60 "$AF_RUNTIME/server-${SLURM_JOB_ID}.log" >&2; exit 1; fi
  sleep 2
done
[[ "$AF_READY" == true ]] || { echo 'Server startup deadline exceeded.' >&2; exit 1; }
[[ "$AF_DRAINING" == false ]] || exit 0
"$AF_PYTHON" "$AF_CLI" run --root "$AF_OUTPUT_ROOT" --base-url "$AF_ENDPOINT" --runtime-record "$AF_RUNTIME/probe-${SLURM_JOB_ID}.json" --seconds 5400 &
AF_WORKER_PID=$!
AF_STATUS=0
wait "$AF_WORKER_PID" || AF_STATUS=$?
# TERM can interrupt wait before the child has durably drained its current call.
if kill -0 "$AF_WORKER_PID" 2>/dev/null; then wait "$AF_WORKER_PID" || AF_STATUS=$?; fi
AF_WORKER_PID=''
printf '%s\n' "$AF_STATUS" >"$AF_RUNTIME/worker-exit-${SLURM_JOB_ID}.txt"
exit "$AF_STATUS"
