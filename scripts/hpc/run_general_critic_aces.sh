#!/bin/bash
# Separate CPU replay/fits from the two established GPU model configurations.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]; else [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]; fi
stage="${1:?stage}"
cli=("$AF_PYTHON" "$AF_REPO_ROOT/scripts/general_critic.py")
case "$stage" in
 fit|report) exec "${cli[@]}" "$stage" --root "$AF_OUTPUT_ROOT" ;;
 prepare|parent-review|propose|child-review) ;;
 *) echo 'Unknown stage' >&2; exit 2 ;;
esac
: "${AF_VLLM_IMAGE:?}" "${AF_HF_HOME:?}" "${AF_JUDGE_REVISION:?}"
module load WebProxy
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
runtime="$(command -v apptainer || command -v singularity)"
mkdir -p "$AF_OUTPUT_ROOT/runtime" "$AF_HF_HOME"
if [[ "$stage" == prepare ]]; then
  "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_general_critic.py tests/test_search_hybrid_pair.py
  "$AF_PYTHON" scripts/smoke_general_critic.py
  "${cli[@]}" freeze --source "${AF_SOURCE_ROOT:?}" --root "$AF_OUTPUT_ROOT" --judge-revision "$AF_JUDGE_REVISION"
fi
"${cli[@]}" verify --root "$AF_OUTPUT_ROOT"
plan="$AF_OUTPUT_ROOT/plan.json"
expected="$(jq -er '.serving_image_sha256' "$plan")"
actual="$(sha256sum "$AF_VLLM_IMAGE")"
[[ "${actual%% *}" == "$expected" ]] || { echo 'Frozen image SHA differs' >&2; exit 2; }
if [[ "$stage" == prepare ]]; then
  container_python="$("$runtime" exec "$AF_VLLM_IMAGE" sh -c 'command -v python3 || command -v python')"
  [[ "$container_python" == /* && "$container_python" != *$'\n'* ]]
  proposer_revision="$(jq -er '.proposer_settings.model_revision' "$plan")"
  "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME,PYTHONDONTWRITEBYTECODE=1" \
    "$AF_VLLM_IMAGE" "$container_python" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download("openai/gpt-oss-20b", revision=sys.argv[1]); snapshot_download("openai/gpt-oss-120b", revision=sys.argv[2])' "$proposer_revision" "$AF_JUDGE_REVISION"
  exec "${cli[@]}" evidence --root "$AF_OUTPUT_ROOT"
fi
if [[ "$stage" == propose ]]; then
  model=openai/gpt-oss-20b; tp=1
  revision="$(jq -er '.proposer_settings.model_revision' "$plan")"
else
  model=openai/gpt-oss-120b; tp=2
  revision="$(jq -er '.judge_revision' "$plan")"
fi
cache="$(mktemp -d /tmp/af-critic-XXXXXX)"
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
log="$AF_OUTPUT_ROOT/runtime/server-${SLURM_JOB_ID}-${stage}.log"
nvidia-smi > "$AF_OUTPUT_ROOT/runtime/gpu-${SLURM_JOB_ID}.txt"
"$runtime" exec --nv --bind "$AF_HF_HOME:$AF_HF_HOME,$cache:$cache" \
  --env "HF_HOME=$AF_HF_HOME,HF_HUB_OFFLINE=1,VLLM_CACHE_ROOT=$cache/vllm,TRITON_CACHE_DIR=$cache/triton,TORCHINDUCTOR_CACHE_DIR=$cache/torchinductor,XDG_CACHE_HOME=$cache/xdg,CUDA_CACHE_PATH=$cache/cuda,TMPDIR=$cache,PYTHONDONTWRITEBYTECODE=1" \
  "$AF_VLLM_IMAGE" vllm serve "$model" --revision "$revision" --host 127.0.0.1 --port "$port" \
  --tensor-parallel-size "$tp" --max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.90 > "$log" 2>&1 &
server_pid=$!
ready=false
for _ in $(seq 1 300); do
  if curl --noproxy '*' --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null; then ready=true; break; fi
  kill -0 "$server_pid" 2>/dev/null || { tail -n 80 "$log"; exit 1; }
  sleep 2
done
[[ "$ready" == true ]] || { tail -n 80 "$log"; exit 1; }
"${cli[@]}" "$stage" --root "$AF_OUTPUT_ROOT" --base-url "http://127.0.0.1:$port"
