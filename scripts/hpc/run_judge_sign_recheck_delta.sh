#!/bin/bash
# Same bounded judge requests; an explicitly separate Delta execution identity.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]; else [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]; fi
stage="${1:?stage}"
cli=("$AF_PYTHON" "$AF_REPO_ROOT/scripts/judge_sign_delta.py")
case "$stage" in
 image) exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/prepare_vllm_image.py" \
   --image "${AF_VLLM_IMAGE:?}" --scratch "${AF_IMAGE_TMP_ROOT:?}" ;;
 report) exec "${cli[@]}" report --root "$AF_OUTPUT_ROOT" ;;
 prepare|review) ;;
 *) echo 'Unknown stage' >&2; exit 2 ;;
esac
: "${AF_VLLM_IMAGE:?}" "${AF_HF_HOME:?}"
actual="$(sha256sum "$AF_VLLM_IMAGE")"
actual="${actual%% *}"
if [[ "$stage" == prepare ]]; then
  "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_judge_sign_delta.py tests/test_judge_sign_recheck.py tests/test_judge_sign_correction.py
  "$AF_PYTHON" scripts/smoke_judge_sign_delta.py
  "$AF_PYTHON" - <<'PY'
import os
from pathlib import Path
from autoformalism.rebuttal.judge_sign_delta import imported_plan
assert imported_plan(Path(os.environ['AF_SOURCE_PLAN']))['artifact_sha256'] == os.environ['AF_SIGN_ORIGIN_SHA256'], 'source plan changed after submission'
PY
  "${cli[@]}" freeze --source-plan "${AF_SOURCE_PLAN:?}" --root "$AF_OUTPUT_ROOT" --image-sha256 "$actual"
fi
"${cli[@]}" verify --root "$AF_OUTPUT_ROOT"
plan="$AF_OUTPUT_ROOT/plan.json"
[[ "$actual" == "$(jq -er '.execution.serving_image_sha256' "$plan")" ]] || { echo 'Pinned Delta image SHA differs' >&2; exit 2; }
if [[ "$(jq '[.reviews[] | select(.affected)] | length' "$plan")" == 0 ]]; then
  exec "${cli[@]}" report --root "$AF_OUTPUT_ROOT"
fi
runtime="$(command -v apptainer || command -v singularity)"
revision="$(jq -er '.judge_revision' "$plan")"
mkdir -p "$AF_OUTPUT_ROOT/runtime" "$AF_HF_HOME"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
if [[ "$stage" == prepare ]]; then
  container_python="$("$runtime" exec "$AF_VLLM_IMAGE" sh -c 'command -v python3 || command -v python')"
  [[ "$container_python" == /* && "$container_python" != *$'\n'* ]]
  exec "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME,PYTHONDONTWRITEBYTECODE=1" \
    "$AF_VLLM_IMAGE" "$container_python" -c 'from importlib.metadata import version; from huggingface_hub import snapshot_download; import sys; assert version("vllm") == "0.27.1", "requires pinned vLLM 0.27.1"; snapshot_download("openai/gpt-oss-120b", revision=sys.argv[1])' "$revision"
fi
cache="$(mktemp -d /tmp/af-sign-delta-XXXXXX)"
server_pid=''
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n "$server_pid" ]]; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" || true; fi
  rm -rf -- "$cache"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 75' TERM INT
port="$("$AF_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
log="$AF_OUTPUT_ROOT/runtime/server-${SLURM_JOB_ID}.log"
nvidia-smi > "$AF_OUTPUT_ROOT/runtime/gpu-${SLURM_JOB_ID}.txt"
"$runtime" exec --nv --bind "$AF_HF_HOME:$AF_HF_HOME,$cache:$cache" \
  --env "HF_HOME=$AF_HF_HOME,HF_HUB_OFFLINE=1,VLLM_CACHE_ROOT=$cache/vllm,TRITON_CACHE_DIR=$cache/triton,TORCHINDUCTOR_CACHE_DIR=$cache/torchinductor,XDG_CACHE_HOME=$cache/xdg,CUDA_CACHE_PATH=$cache/cuda,TMPDIR=$cache,PYTHONDONTWRITEBYTECODE=1" \
  "$AF_VLLM_IMAGE" vllm serve openai/gpt-oss-120b --revision "$revision" --host 127.0.0.1 --port "$port" \
  --tensor-parallel-size 4 --max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.90 > "$log" 2>&1 &
server_pid=$!
ready=false
for _ in $(seq 1 1200); do
  if curl --noproxy '*' --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null; then ready=true; break; fi
  kill -0 "$server_pid" 2>/dev/null || { tail -n 80 "$log"; exit 1; }
  sleep 2
done
[[ "$ready" == true ]] || { tail -n 80 "$log"; exit 1; }
"${cli[@]}" run --root "$AF_OUTPUT_ROOT" --base-url "http://127.0.0.1:$port"
