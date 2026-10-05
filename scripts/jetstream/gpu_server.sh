#!/bin/bash
# Serve gpt-oss on this Jetstream2 GPU VM for the vendored LLM baselines.
#
#   gpu_server.sh setup [20b] [120b]  prepare the search environment, check the
#                                     GPU, pull the pinned vLLM image, and
#                                     download and verify the pinned weights
#                                     (both models when none is named)
#   gpu_server.sh start MODEL TASKS   serve MODEL (20b or 120b) for TASKS tasks
#                                     running at once
#   gpu_server.sh stop                stop the server
#   gpu_server.sh status              what is served, if anything
#
# The server listens on this VM's loopback address only; campaigns reach it as
# their vm_local_vllm endpoint. The weights live on an Exosphere volume, which
# outlives the VM: attached to a new VM, `setup` checks them against the
# published hashes instead of downloading them again.
#
#   AF_HOME    the search environment (default ~/af)
#   AF_MODELS  the mounted volume that holds the weights
#              (default /media/volume/af-models)

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
: "${AF_MODELS:=/media/volume/af-models}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly hf_home="${AF_MODELS}/hf"
readonly record="${AF_HOME}/server.json"
readonly container=af-vllm
readonly port=8000
# The vLLM release our method's model is served with
# (src/autoformalism/rebuttal/vllm_image_bootstrap.py), pinned to the image the
# v0.27.1 tag named on 2026-10-04. It is built on CUDA 13.0.
readonly image="vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
readonly vllm_version=0.27.1
# The context our method's server allows (configs/phase_c_construction_v1.json).
readonly max_model_len=32768

step() { printf '\n== %s\n' "$*"; }
fail() {
  echo "$*" >&2
  exit 2
}

model_repo() {
  case "$1" in
    20b) echo openai/gpt-oss-20b ;;
    120b) echo openai/gpt-oss-120b ;;
    *) fail "unknown model '$1'; use 20b or 120b" ;;
  esac
}
model_revision() {
  case "$1" in
    # The revision our method's model is served at.
    20b)
      python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model_settings"]["model_revision"])' \
        "${repo}/configs/phase_c_construction_v1.json"
      ;;
    # The revision this project already pinned for gpt-oss-120b
    # (src/autoformalism/rebuttal/component_critic.py).
    120b) echo b5c939de8f754692c1647ca79fbf85e8c1e70f8a ;;
  esac
}
manifest_of() { echo "${AF_MODELS}/manifests/gpt-oss-$1.json"; }
running() { docker container inspect "${container}" >/dev/null 2>&1; }
served_ids() {
  curl -sf --max-time 5 "http://127.0.0.1:${port}/v1/models" |
    python3 -c 'import json, sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))'
}

install_toolkit() {
  # NVIDIA's documented apt installation of the container toolkit.
  local keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    sudo gpg --dearmor --yes -o "${keyring}"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed "s#deb https://#deb [signed-by=${keyring}] https://#g" |
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
}

setup() {
  local sizes=("$@")
  ((${#sizes[@]})) || sizes=(20b 120b)
  local size
  for size in "${sizes[@]}"; do model_repo "${size}" >/dev/null; done

  step "search environment"
  AF_CHECK_HOSTED=0 bash "${repo}/scripts/jetstream/setup_vm.sh"

  step "GPU"
  command -v nvidia-smi >/dev/null ||
    fail "no NVIDIA driver here; create the VM with a GPU size (g3.xl or g5.xl)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi | grep -o 'CUDA Version: [0-9.]*' || true

  step "container runtime"
  docker info >/dev/null 2>&1 || fail "Docker is not usable by $(whoami)"
  if ! command -v nvidia-ctk >/dev/null; then
    echo "installing the NVIDIA container toolkit, so containers can use the GPU"
    install_toolkit
  fi
  if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
  docker --version

  step "vLLM image"
  docker pull --quiet "${image}"
  local found
  found="$(docker run --rm --entrypoint python3 "${image}" -c \
    'from importlib.metadata import version; print(version("vllm"))')"
  [[ "${found}" == "${vllm_version}" ]] ||
    fail "the image reports vLLM ${found}, not ${vllm_version}"
  echo "vLLM ${found}"
  # CUDA 13.0 needs a driver from the 580 series or later, or one of the older
  # branches the image runs on through NVIDIA's forward compatibility.
  docker run --rm --gpus all --entrypoint python3 "${image}" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA", torch.version.cuda, "on", torch.cuda.get_device_name(0))' ||
    fail "the image cannot use this GPU; the driver may be too old for CUDA 13.0"

  step "weights"
  mountpoint -q "${AF_MODELS}" ||
    fail "${AF_MODELS} is not a mounted volume; attach the af-models volume in Exosphere"
  sudo install -d -o "$(id -u)" -g "$(id -g)" "${AF_MODELS}/manifests"
  sudo install -d "${hf_home}"
  for size in "${sizes[@]}"; do
    local repo_id revision
    repo_id="$(model_repo "${size}")"
    revision="$(model_revision "${size}")"
    echo "${repo_id} at ${revision}"
    # Downloads only what is missing; the folders vLLM does not load are skipped.
    docker run --rm -v "${hf_home}:/root/.cache/huggingface" \
      --entrypoint python3 "${image}" -c '
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1], revision=sys.argv[2], ignore_patterns=["original/*", "metal/*"]
)
' "${repo_id}" "${revision}"
    python3 "${repo}/scripts/jetstream/verify_weights.py" --hf-home "${hf_home}" \
      --repo "${repo_id}" --revision "${revision}" --manifest "$(manifest_of "${size}")"
  done
  df -h "${AF_MODELS}" | tail -n 1

  echo
  echo "GPU server setup complete"
}

start() {
  local size="${1:-}" tasks="${2:-}"
  [[ "${tasks}" =~ ^[1-9][0-9]*$ ]] || fail "usage: gpu_server.sh start 20b|120b TASKS"
  local repo_id revision
  repo_id="$(model_repo "${size}")"
  revision="$(model_revision "${size}")"
  [[ -f "$(manifest_of "${size}")" ]] || fail "run: gpu_server.sh setup ${size}"
  if running; then
    fail "a server is already running ($(served_ids || echo loading)); stop it first"
  fi
  # LLM-SR asks for four programs per request, so each task can hold four
  # sequences, as in the Phase B jobs.
  local sequences=$((tasks * 4))
  docker run -d --name "${container}" --gpus all --ipc=host \
    -p "127.0.0.1:${port}:8000" \
    -v "${hf_home}:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 \
    "${image}" "${repo_id}" --revision "${revision}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len "${max_model_len}" \
    --max-num-seqs "${sequences}" \
    --gpu-memory-utilization 0.90 >/dev/null
  echo "loading ${repo_id}; this takes a few minutes"
  local waited=0
  until served_ids >/dev/null 2>&1; do
    if [[ "$(docker container inspect -f '{{.State.Running}}' "${container}")" != true ]]; then
      docker logs --tail 60 "${container}" >&2
      docker rm -f "${container}" >/dev/null
      fail "the server stopped while loading; its log is above"
    fi
    if ((waited >= 1800)); then
      docker logs --tail 60 "${container}" >&2
      fail "the server did not answer within 30 minutes; it is still running"
    fi
    sleep 10
    waited=$((waited + 10))
  done
  local served
  served="$(served_ids)"
  [[ "${served}" == "${repo_id}" ]] || fail "the server lists '${served}', not ${repo_id}"

  python3 - "${record}" <<EOF
import hashlib, json, subprocess, sys, time
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True, check=True,
).stdout.strip()
manifest = open("$(manifest_of "${size}")", "rb").read()
record = {
    "model": "${repo_id}",
    "revision": "${revision}",
    "weights_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    "image": "${image}",
    "vllm_version": "${vllm_version}",
    "base_url": "http://127.0.0.1:${port}",
    "max_model_len": ${max_model_len},
    "max_num_seqs": ${sequences},
    "gpu_memory_utilization": 0.90,
    "tasks": ${tasks},
    "gpu": gpu,
    "source_commit": "$(git -C "${repo}" rev-parse HEAD)",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(sys.argv[1], "w") as handle:
    json.dump(record, handle, indent=1)
    handle.write("\n")
EOF
  echo "serving ${repo_id} at ${revision} on 127.0.0.1:${port} for ${tasks} tasks"
}

stop() {
  if running; then
    docker rm -f "${container}" >/dev/null
    echo "stopped the server"
  else
    echo "no server is running"
  fi
  rm -f "${record}"
}

status() {
  if ! running; then
    echo "no server is running"
    return
  fi
  docker ps --filter "name=^${container}$" --format '{{.Status}}'
  served_ids || echo "not answering yet"
  [[ ! -f "${record}" ]] || cat "${record}"
}

case "${1:-}" in
  setup)
    shift
    setup "$@"
    ;;
  start)
    shift
    start "$@"
    ;;
  stop) stop ;;
  status) status ;;
  *)
    sed -n '2,20p' "$0"
    exit 2
    ;;
esac
