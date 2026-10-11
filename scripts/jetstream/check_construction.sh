#!/bin/bash
# Isolated, bounded H100 compatibility check. Never modifies an ACES plan.
set -euo pipefail
AF_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_WORK=${AF_JS2_ROOT:-$HOME/af-construction}
AF_IMAGE=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
AF_MODEL=openai/gpt-oss-20b
AF_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
AF_CONTAINER=af-construction-vllm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AF_REPO:$AF_REPO/src"

fail() { echo "$*" >&2; exit 2; }
step() { printf '\n%s\n' "$*"; }
docker_cmd() { sudo -n docker "$@"; }

case "${1:-}" in
  run)
    [[ -f "$AF_REPO/probe-requests.json" ]] || fail 'Use the delivered compatibility bundle.'
    sudo -n true || fail 'Run sudo -v first; this setup needs local administrator access.'
    mkdir -p "$AF_WORK"
    if [[ -f "$AF_WORK/worker.lock" ]] && ! flock -n "$AF_WORK/worker.lock" true; then
      echo "A worker is already running. Progress: tail -f $AF_WORK/launcher.log"
      exit 0
    fi
    # The worker lock prevents concurrent setup/server changes and requests.
    AF_LAUNCH_LOG="launcher-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    nohup bash "$0" _worker >"$AF_WORK/$AF_LAUNCH_LOG" 2>&1 </dev/null &
    echo "Worker PID: $!"
    ln -sfn "$AF_LAUNCH_LOG" "$AF_WORK/launcher.log"
    echo "Progress: tail -f $AF_WORK/launcher.log"
    echo "Results: $AF_WORK/probe/summary.json"
    ;;
  status)
    [[ -f "$AF_WORK/launcher.log" ]] || fail 'No compatibility worker has started in this directory.'
    echo 'Worker state / exit code:'
    [[ ! -f "$AF_WORK/exit-code.txt" ]] || cat "$AF_WORK/exit-code.txt"
    [[ ! -f "$AF_WORK/probe/summary.json" ]] || cat "$AF_WORK/probe/summary.json"
    [[ ! -f "$AF_WORK/launcher.log" ]] || tail -n 30 "$AF_WORK/launcher.log"
    ;;
  inspect)
    AF_FILES=()
    for AF_PART in probe cuda.json server.json weights.json models.json version.json python-packages.txt server.log exit-code.txt; do
      [[ ! -e "$AF_WORK/$AF_PART" ]] || AF_FILES+=("$AF_PART")
    done
    for AF_LOG in "$AF_WORK"/launcher-*.log; do
      [[ ! -f "$AF_LOG" ]] || AF_FILES+=("$(basename "$AF_LOG")")
    done
    ((${#AF_FILES[@]})) || fail 'No compatibility evidence has been saved yet.'
    tar -czf "$AF_WORK/inspection.tar.gz" -C "$AF_WORK" "${AF_FILES[@]}"
    echo "Inspection package: $AF_WORK/inspection.tar.gz"
    ;;
  stop)
    [[ "$(docker_cmd inspect -f '{{index .Config.Labels "autoformalism.role"}}' "$AF_CONTAINER")" == construction-compatibility ]] || fail 'Not our compatibility server.'
    docker_cmd logs "$AF_CONTAINER" >"$AF_WORK/server.log" 2>&1
    docker_cmd stop "$AF_CONTAINER"
    ;;
  _worker)
    exec 9>"$AF_WORK/worker.lock"
    flock -n 9 || fail 'A compatibility worker is already running.'
    printf 'running\n' >"$AF_WORK/exit-code.txt"
    trap 'AF_EXIT=$?; printf "%s\n" "$AF_EXIT" >"$AF_WORK/exit-code.txt"' EXIT
    [[ "$(uname -s)" == Linux ]] || fail 'Run this on the Jetstream2 H100 VM.'
    [[ "$(uname -m)" == x86_64 ]] || fail 'Expected the x86_64 GPU VM.'
    step 'Checking GPU, storage, and local runtime'
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    df -h "$AF_WORK"
    python3 - "$AF_WORK" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 65 * 1024**3:
    raise SystemExit('Need at least 65 GiB free for initial image/weights setup.')
PY
    sudo -n apt-get update -qq
    sudo -n env NEEDRESTART_MODE=l apt-get install -y -qq python3-venv curl ca-certificates gnupg
    if ! command -v docker >/dev/null; then
      sudo -n env NEEDRESTART_MODE=l apt-get install -y -qq docker.io
      sudo -n systemctl start docker
    fi
    docker_cmd info >/dev/null
    if ! command -v nvidia-ctk >/dev/null; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
        sudo -n gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
        sudo -n tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
      sudo -n apt-get update -qq
      sudo -n env NEEDRESTART_MODE=l apt-get install -y -qq nvidia-container-toolkit
    fi
    if ! docker_cmd info --format '{{json .Runtimes}}' | python3 -c 'import json,sys; sys.exit("nvidia" not in json.load(sys.stdin))'; then
      sudo -n nvidia-ctk runtime configure --runtime=docker
      # Reload runtime registration without stopping Exosphere's web containers.
      sudo -n systemctl kill --kill-who=main --signal=HUP docker
      sleep 2
    fi
    docker_cmd info --format '{{json .Runtimes}}' | python3 -c 'import json,sys; assert "nvidia" in json.load(sys.stdin), "NVIDIA runtime reload failed"'
    python3 -m venv "$AF_WORK/venv"
    "$AF_WORK/venv/bin/pip" install --disable-pip-version-check "${AF_REPO}[casadi]"
    "$AF_WORK/venv/bin/pip" freeze >"$AF_WORK/python-packages.txt"
    step 'Pulling the pinned vLLM image'
    docker_cmd pull "$AF_IMAGE"
    mkdir -p "$AF_WORK/hf" "$AF_WORK/cache"
    step 'Testing a CUDA matrix multiplication inside that image'
    docker_cmd run --rm --runtime=nvidia --gpus all --entrypoint python3 "$AF_IMAGE" -c '
import json, torch
from importlib.metadata import version
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.ones((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
assert bool(torch.all(y == 1024))
print(json.dumps({"vllm": version("vllm"), "torch": torch.__version__,
 "cuda": torch.version.cuda, "device_count": torch.cuda.device_count(),
 "gpu": torch.cuda.get_device_name(0), "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
 "kernel": "passed"}))' >"$AF_WORK/cuda.json"
    cat "$AF_WORK/cuda.json"
    step 'Downloading and verifying only the pinned 20B weights'
    docker_cmd run --rm --user "$(id -u):$(id -g)" \
      -v "$AF_WORK/hf:/af/hf" -e HF_HOME=/af/hf --entrypoint python3 "$AF_IMAGE" -c '
from huggingface_hub import snapshot_download
snapshot_download(repo_id="openai/gpt-oss-20b", revision="6cee5e81ee83917806bbde320786a8fb61efebee", ignore_patterns=["original/*", "metal/*"])
'
    python3 "$AF_REPO/scripts/jetstream/verify_weights.py" --hf-home "$AF_WORK/hf" \
      --repo "$AF_MODEL" --revision "$AF_REVISION" --manifest "$AF_WORK/weights.json"
    step 'Starting the private, single-sequence 32K server'
    if docker_cmd container inspect "$AF_CONTAINER" >/dev/null 2>&1; then
      [[ "$(docker_cmd inspect -f '{{index .Config.Labels "autoformalism.role"}}' "$AF_CONTAINER")" == construction-compatibility ]] || fail 'Container name belongs to another service.'
      # Validate configuration below before using or restarting this container.
    else
      docker_cmd run -d --name "$AF_CONTAINER" --runtime=nvidia --gpus all \
        --label autoformalism.role=construction-compatibility \
        --ipc=host -p 127.0.0.1:8000:8000 \
        -v "$AF_WORK/hf:/af/hf:ro" -v "$AF_WORK/cache:/af/cache" \
        -e HF_HOME=/af/hf -e HF_HUB_OFFLINE=1 \
        -e XDG_CACHE_HOME=/af/cache -e VLLM_CACHE_ROOT=/af/cache/vllm \
        -e TORCHINDUCTOR_CACHE_DIR=/af/cache/torch -e TRITON_CACHE_DIR=/af/cache/triton \
        "$AF_IMAGE" --model "$AF_MODEL" --revision "$AF_REVISION" \
        --host 0.0.0.0 --port 8000 --max-model-len 32768 \
        --max-num-seqs 1 --gpu-memory-utilization 0.90 --tensor-parallel-size 1
    fi
    docker_cmd inspect "$AF_CONTAINER" | "$AF_WORK/venv/bin/python" \
      "$AF_REPO/scripts/jetstream_server_record.py" --work "$AF_WORK" \
      --image "$AF_IMAGE"
    docker_cmd start "$AF_CONTAINER" >/dev/null
    AF_READY=false
    for ((AF_TRY=0; AF_TRY<180; AF_TRY++)); do
      if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        AF_READY=true
        break
      fi
      if [[ "$(docker_cmd inspect -f '{{.State.Running}}' "$AF_CONTAINER")" != true ]]; then
        docker_cmd logs --tail 80 "$AF_CONTAINER"
        fail 'The model server exited; its logs are retained.'
      fi
      sleep 10
    done
    docker_cmd logs "$AF_CONTAINER" >"$AF_WORK/server.log" 2>&1
    [[ "$AF_READY" == true ]] || fail 'Server startup exceeded 30 minutes; inspect server.log.'
    curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models >"$AF_WORK/models.json"
    curl -fsS --max-time 10 http://127.0.0.1:8000/version >"$AF_WORK/version.json"
    python3 - "$AF_WORK" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
assert json.loads((p/'version.json').read_text())['version'] == '0.27.1'
assert [m['id'] for m in json.loads((p/'models.json').read_text())['data']] == ['openai/gpt-oss-20b']
PY
    step 'Running four independent stage/schema requests, with durable caching'
    "$AF_WORK/venv/bin/python" "$AF_REPO/scripts/jetstream_construction_probe.py" run \
      --requests "$AF_REPO/probe-requests.json" --server "$AF_WORK/server.json" --root "$AF_WORK/probe"
    docker_cmd logs "$AF_CONTAINER" >"$AF_WORK/server.log" 2>&1
    step 'Compatibility check finished. Server remains available on localhost only.'
    ;;
  *) fail 'Use run, status, inspect, or stop.' ;;
esac
