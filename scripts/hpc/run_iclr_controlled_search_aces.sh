#!/usr/bin/env bash
#SBATCH --job-name=iclr-search
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
set -euo pipefail
module load GCCcore/13.2.0 Python/3.11.5 WebProxy
: "${AF_REPO_ROOT:?set the checked-out project directory}"
: "${AF_SEARCH_ROOT:?set the prepared controlled search root}"
: "${SLURM_ARRAY_TASK_ID:?submit a cell/seed block array}"
: "${AF_PYTHON:=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
: "${AF_VLLM_IMAGE:=/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=/scratch/user/u.yx126462/huggingface-cache}"
: "${AF_BLOCK_SECONDS:=36000}"
cd "$AF_REPO_ROOT"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
umask 077
plan="$AF_SEARCH_ROOT/plan.json"
test "$(jq -r .protocol "$plan")" = iclr-controlled-search-1
if [[ "$(jq -r .critic_enabled "$plan")" = true ]]; then
    : "${AF_JETSTREAM_API_KEY:?inherit the critic API key from the submission environment}"
    test -s "$AF_SEARCH_ROOT/critic_authorization.json"
fi
# Existing plan files can still be readable after the project runs out of inodes.
if ! mkdir -p "$AF_SEARCH_ROOT/runtime" || ! probe=$(mktemp "$AF_SEARCH_ROOT/runtime/.write-check.XXXXXXXX"); then
    echo "Cannot create a durable checkpoint file. Check ACES file-count and byte quotas before retrying." >&2
    exit 1
fi
if ! printf 'ready\n' >"$probe"; then
    rm -f -- "$probe"
    echo "Cannot write a durable checkpoint. Check the ACES byte quota before retrying." >&2
    exit 1
fi
rm -- "$probe"
df -Pi "$AF_SEARCH_ROOT"
image_digest="$(sha256sum "$AF_VLLM_IMAGE")"
test "${image_digest%% *}" = "$(jq -r .serving_image_sha256 "$plan")"
model="$(jq -r .model_settings.model "$plan")"
revision="$(jq -r .model_settings.model_revision "$plan")"
port="$("$AF_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
endpoint="http://127.0.0.1:$port"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,localhost,${no_proxy:-}"
: "${SLURM_JOB_ID:?}"
ipc="$(mktemp -d "${TMPDIR:-/tmp}/ics.XXXXXX")"
pid=""
cleanup() {
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -rf -- "$ipc"
}
trap cleanup EXIT
cache="$ipc"
mounts="$AF_HF_HOME:$AF_HF_HOME,$ipc:$ipc"
if [[ ${AF_COMPUTE_CACHE_ROOT:-job-local} != job-local ]]; then
    cache="$AF_COMPUTE_CACHE_ROOT/$SLURM_JOB_ID"
    mkdir -p "$cache"
    mounts+=",$cache:$cache"
fi
runtime="$(command -v apptainer || command -v singularity)"
server_log="$AF_SEARCH_ROOT/runtime/server-${SLURM_JOB_ID}.log"
vocab=()
if [[ -n "${AF_HARMONY_ASSETS:-}" ]]; then
    "$AF_PYTHON" scripts/harmony_vocab.py verify --directory "$AF_HARMONY_ASSETS" >/dev/null
    vocab=(--bind "$AF_HARMONY_ASSETS:$AF_HARMONY_ASSETS:ro" --env "TIKTOKEN_ENCODINGS_BASE=$AF_HARMONY_ASSETS")
fi
env -u AF_JETSTREAM_API_KEY "$runtime" exec --nv \
    "${vocab[@]}" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --bind "$mounts" \
    --env "HF_HOME=$AF_HF_HOME" --env "TMPDIR=$ipc" \
    --env "VLLM_CACHE_ROOT=$cache/vllm" --env "TRITON_CACHE_DIR=$cache/triton" \
    --env "TORCHINDUCTOR_CACHE_DIR=$cache/torchinductor" \
    --env "XDG_CACHE_HOME=$cache/xdg" --env "CUDA_CACHE_PATH=$cache/cuda" \
    "$AF_VLLM_IMAGE" vllm serve "$model" --revision "$revision" \
    --host 127.0.0.1 --port "$port" --max-model-len 32768 --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 --tensor-parallel-size 1 >"$server_log" 2>&1 &
pid=$!
for _ in $(seq 1 300); do
    kill -0 "$pid" || { echo "vLLM exited; inspect $server_log" >&2; exit 1; }
    if curl --noproxy '*' --silent --fail "$endpoint/v1/models" >/dev/null; then break; fi
    sleep 5
done
curl --noproxy '*' --silent --fail "$endpoint/v1/models" |
    jq -e --arg model "$model" 'any(.data[]; .id == $model)' >/dev/null
runtime_receipt="$AF_SEARCH_ROOT/runtime/service-${SLURM_JOB_ID}.json"
jq -n --arg job "$SLURM_JOB_ID" --arg host "$(hostname)" --arg endpoint "$endpoint" \
    --arg model "$model" --arg revision "$revision" --arg image "${image_digest%% *}" \
    --arg plan "$(jq -r .artifact_sha256 "$plan")" --arg cache "$cache" --argjson block "$SLURM_ARRAY_TASK_ID" \
    '{job:$job,host:$host,endpoint:$endpoint,model:$model,revision:$revision,
      image_sha256:$image,plan_sha256:$plan,block:$block,compute_cache:$cache}' >"$runtime_receipt.tmp"
mv "$runtime_receipt.tmp" "$runtime_receipt"
"$AF_PYTHON" -m scripts.iclr_controlled_search run --root "$AF_SEARCH_ROOT" \
    --block "$SLURM_ARRAY_TASK_ID" --base-url "$endpoint" --seconds "$AF_BLOCK_SECONDS"
