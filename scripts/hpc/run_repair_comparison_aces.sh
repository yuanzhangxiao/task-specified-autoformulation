#!/bin/bash
# The runtime arm serves only 20B on one H100; the judge arm swaps on two H100s.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_OUTPUT_ROOT:?required}" "${AF_PYTHON:?required}"
: "${AF_VLLM_IMAGE:?required}" "${AF_HF_HOME:?required}"
: "${AF_ARM:?required}" "${AF_WORKER_SECONDS:?required}"
case "$AF_ARM" in
  redesigned_runtime) roles=(proposer) ;;
  redesigned_prefit_judge) roles=(proposer judge) ;;
  *) echo 'Unknown comparison arm' >&2; exit 2 ;;
esac
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
export PYTHONPATH="${AF_REPO_ROOT}/src" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
runtime="$(command -v apptainer || command -v singularity)"
plan="${AF_OUTPUT_ROOT}/plan.json"
expected="$(jq -er '.serving_image_sha256' "$plan")"
actual="$(sha256sum "$AF_VLLM_IMAGE")"
[[ "${actual%% *}" == "$expected" ]] || { echo 'Frozen image SHA differs' >&2; exit 2; }
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/repair_feedback_comparison.py" verify --root "$AF_OUTPUT_ROOT"
mkdir -p "$AF_OUTPUT_ROOT/runtime"
git -C "$AF_REPO_ROOT" rev-parse HEAD > "$AF_OUTPUT_ROOT/runtime/commit-${SLURM_JOB_ID}.txt"
printf '%s\n' "$actual" > "$AF_OUTPUT_ROOT/runtime/image-${SLURM_JOB_ID}.sha256"
if [[ "${1:-}" == prepare ]]; then
  # Match the image-build launcher: the image may expose python3 without python.
  if ! container_python="$("$runtime" exec "$AF_VLLM_IMAGE" sh -c 'command -v python3 || command -v python')"; then
    echo 'Container exposes neither python3 nor python; inspect the pinned image' >&2
    exit 2
  fi
  [[ "$container_python" == /* && "$container_python" != *$'\n'* ]] || { echo 'Container Python discovery did not return one absolute path' >&2; exit 2; }
  printf '%s\n' "$container_python" > "$AF_OUTPUT_ROOT/runtime/container-python-${SLURM_JOB_ID}.txt"
  cd "$AF_REPO_ROOT"
  "$AF_PYTHON" -m pytest -q tests/test_repair_comparison.py > "$AF_OUTPUT_ROOT/runtime/preflight-tests-${SLURM_JOB_ID}.log" 2>&1
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/smoke_repair_feedback_comparison.py" > "$AF_OUTPUT_ROOT/runtime/preflight-smoke-${SLURM_JOB_ID}.json"
  judge_revision="$(jq -er '.config.judge_revision' "$plan")"
  proposer_revision="$(jq -er '.proposer_settings.model_revision' "$plan")"
  "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME" "$AF_VLLM_IMAGE" "$container_python" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download("openai/gpt-oss-20b", revision=sys.argv[1]); sys.argv[3] == "redesigned_prefit_judge" and snapshot_download("openai/gpt-oss-120b", revision=sys.argv[2])' "$proposer_revision" "$judge_revision" "$AF_ARM"
  exit 0
fi
nvidia-smi > "$AF_OUTPUT_ROOT/runtime/gpu-${SLURM_JOB_ID}.txt"
server_pid=''
worker_pid=''
cleanup() {
  status=$?
  trap - EXIT
  [[ -z "$worker_pid" ]] || kill -TERM "$worker_pid" 2>/dev/null || true
  [[ -z "$server_pid" ]] || kill -TERM "$server_pid" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 75' TERM INT
cache="$(mktemp -d /tmp/af-repair-XXXXXX)"
end=$((SECONDS+AF_WORKER_SECONDS))
for cycle in $(seq 1 10); do
  for role in "${roles[@]}"; do
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/repair_feedback_comparison.py" summary --root "$AF_OUTPUT_ROOT" --arm "$AF_ARM" > "$AF_OUTPUT_ROOT/runtime/summary-${SLURM_JOB_ID}.json"
    status="$(jq -r '.status' "$AF_OUTPUT_ROOT/runtime/summary-${SLURM_JOB_ID}.json")"
    [[ "$status" != complete ]] || exit 0
    if [[ "$role" == judge ]] && [[ "$(jq -r '.pending_scientific_reviews' "$AF_OUTPUT_ROOT/runtime/summary-${SLURM_JOB_ID}.json")" == 0 ]]; then continue; fi
    (( SECONDS < end-900 )) || exit 75
    if [[ "$role" == proposer ]]; then
      model=openai/gpt-oss-20b
      revision="$(jq -er '.proposer_settings.model_revision' "$plan")"
      tp=1
    else
      model=openai/gpt-oss-120b
      revision="$(jq -er '.config.judge_revision' "$plan")"
      tp=2
    fi
    port="$("$AF_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
    log="$AF_OUTPUT_ROOT/runtime/server-${SLURM_JOB_ID}-${cycle}-${role}.log"
    "$runtime" exec --nv --bind "$AF_HF_HOME:$AF_HF_HOME,$cache:$cache" \
      --env "HF_HOME=$AF_HF_HOME,HF_HUB_OFFLINE=1,VLLM_CACHE_ROOT=$cache/vllm,TRITON_CACHE_DIR=$cache/triton,TMPDIR=$cache" \
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
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/repair_feedback_comparison.py" run --root "$AF_OUTPUT_ROOT" --arm "$AF_ARM" --role "$role" --base-url "http://127.0.0.1:$port" --wall-seconds "$((end-SECONDS))" > "$AF_OUTPUT_ROOT/runtime/worker-${SLURM_JOB_ID}-${cycle}-${role}.log" 2>&1 &
    worker_pid=$!
    wait "$worker_pid"
    worker_pid=''
    kill -TERM "$server_pid"
    wait "$server_pid" || true
    server_pid=''
  done
done
exit 75
