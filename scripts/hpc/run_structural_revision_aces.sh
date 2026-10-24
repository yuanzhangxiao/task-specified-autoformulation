#!/bin/bash
# Local offline vLLM, with resources selected by the sealed 20B/120B arm.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_VLLM_IMAGE:?}" "${AF_HF_HOME:?}" "${SLURM_JOB_ID:?}"
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
AF_CLI="$AF_REPO_ROOT/scripts/structural_revision_campaign.py"
AF_SUBMIT="$AF_REPO_ROOT/scripts/submit_structural_revision_aces.py"
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  echo 'Loading scientific Python and its compiler/library hierarchy.' >&2
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
# Source, image and all offline shard/tokenizer paths are checked before CUDA.
"$AF_PYTHON" "$AF_SUBMIT" --verify-only --check-assets --root "$AF_OUTPUT_ROOT" >/dev/null
if ! AF_CONTAINER=$(command -v apptainer || command -v singularity); then
  echo 'No container runtime on this compute node; no model requests started.' >&2
  exit 2
fi
read -r AF_MODEL AF_REVISION AF_GPUS AF_CONTEXT AF_WORKER_SECONDS < <("$AF_PYTHON" - "$AF_OUTPUT_ROOT/execution.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
print(v['model'], v['model_revision'], v['tensor_parallel_size'], v['served_context_tokens'], v['worker_seconds'])
PY
)
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=8
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost"
AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
AF_HARMONY_CLI="$AF_REPO_ROOT/scripts/harmony_vocab.py"
AF_RUNTIME="$AF_OUTPUT_ROOT/runtime"
mkdir -p "$AF_RUNTIME"
AF_JOB_TMP=$(mktemp -d "/tmp/af-structural-${SLURM_JOB_ID}.XXXXXX")
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
"$AF_CONTAINER" "${AF_CONTAINER_ARGS[@]}" --nv "$AF_VLLM_IMAGE" python3 - "$AF_GPUS" <<'PY' >"$AF_RUNTIME/probe-${SLURM_JOB_ID}.json"
import json, subprocess, sys, torch, vllm
expected = int(sys.argv[1])
assert torch.cuda.device_count() == expected, f'Expected {expected} allocated GPUs'
assert vllm.__version__ == '0.27.1', 'Expected pinned vLLM 0.27.1'
names, memory = [], []
for i in range(expected):
    name = torch.cuda.get_device_name(i)
    assert 'H100' in name, 'Expected H100 GPUs'
    assert torch.ones(8, device=f'cuda:{i}').sum().item() == 8
    names.append(name)
    memory.append(torch.cuda.get_device_properties(i).total_memory)
print(json.dumps({'visible_gpus': expected, 'gpu_names': names, 'gpu_memory_bytes': memory, 'cuda_kernel_passed': True, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'vllm': vllm.__version__, 'gpu_driver': sorted(set(subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip().splitlines()))}))
PY
"$AF_PYTHON" - "$AF_RUNTIME/probe-${SLURM_JOB_ID}.json" "$AF_HARMONY_RECORD" "$AF_OUTPUT_ROOT/execution.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
path = Path(sys.argv[1])
record = json.loads(path.read_text())
record['harmony'] = json.loads(Path(sys.argv[2]).read_text())
execution = json.loads(Path(sys.argv[3]).read_text())
record['serving_identity'] = {key: execution[key] for key in ('artifact_sha256', 'model_size', 'model', 'model_revision', 'image_sha256', 'served_context_tokens', 'tensor_parallel_size', 'maximum_sequences')}
record['artifact_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
path.write_text(json.dumps(record, indent=2) + '\n')
PY
AF_PORT=$("$AF_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
AF_ENDPOINT="http://127.0.0.1:$AF_PORT"
AF_SERVER_ARGS=(vllm serve "$AF_MODEL" --revision "$AF_REVISION" --host 127.0.0.1 --port "$AF_PORT"
  --max-model-len "$AF_CONTEXT" --max-num-seqs 1 --gpu-memory-utilization 0.90 --tensor-parallel-size "$AF_GPUS")
"$AF_PYTHON" - "$AF_RUNTIME/server-argv-${SLURM_JOB_ID}.json" "${AF_SERVER_ARGS[@]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:], indent=2) + '\n')
PY
"$AF_CONTAINER" "${AF_CONTAINER_ARGS[@]}" --nv "$AF_VLLM_IMAGE" "${AF_SERVER_ARGS[@]}" >"$AF_RUNTIME/server-${SLURM_JOB_ID}.log" 2>&1 &
AF_SERVER_PID=$!
AF_READY=false
for AF_ATTEMPT in $(seq 1 600); do
  [[ "$AF_DRAINING" == false ]] || exit 0
  if curl --noproxy '*' --silent --fail "$AF_ENDPOINT/v1/models" >"$AF_RUNTIME/models-${SLURM_JOB_ID}.json"; then AF_READY=true; break; fi
  if ! kill -0 "$AF_SERVER_PID" 2>/dev/null; then tail -n 60 "$AF_RUNTIME/server-${SLURM_JOB_ID}.log" >&2; exit 1; fi
  sleep 2
done
[[ "$AF_READY" == true ]] || { echo 'Server startup deadline exceeded.' >&2; exit 1; }
"$AF_PYTHON" - "$AF_RUNTIME/models-${SLURM_JOB_ID}.json" "$AF_MODEL" <<'PY'
import json, sys
models = json.load(open(sys.argv[1]))['data']
assert [v['id'] for v in models] == [sys.argv[2]], 'Served model alias differs'
PY
[[ "$AF_DRAINING" == false ]] || exit 0
"$AF_PYTHON" "$AF_CLI" run --root "$AF_OUTPUT_ROOT" --endpoint "$AF_ENDPOINT" --runtime-record "$AF_RUNTIME/probe-${SLURM_JOB_ID}.json" --seconds "$AF_WORKER_SECONDS" &
AF_WORKER_PID=$!
AF_STATUS=0
wait "$AF_WORKER_PID" || AF_STATUS=$?
if kill -0 "$AF_WORKER_PID" 2>/dev/null; then wait "$AF_WORKER_PID" || AF_STATUS=$?; fi
AF_WORKER_PID=''
printf '%s\n' "$AF_STATUS" >"$AF_RUNTIME/worker-exit-${SLURM_JOB_ID}.txt"
exit "$AF_STATUS"
