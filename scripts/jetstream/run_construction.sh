#!/bin/bash
# Reuse the verified local server; no Slurm, installation, or model downloads.
set -euo pipefail
AF_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_WORK=${AF_JS2_ROOT:-$HOME/af-construction}
AF_ROOT=${AF_JS2_CAMPAIGN_ROOT:-$AF_WORK/focused-seed0-v1}
AF_PYTHON="$AF_WORK/venv/bin/python"
AF_CONTAINER=af-construction-vllm
AF_IMAGE=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
export PYTHONPATH="$AF_REPO:$AF_REPO/src"
export PYTHONDONTWRITEBYTECODE=1

fail() { echo "$*" >&2; exit 2; }
package() {
  local AF_FILES=() AF_ITEM
  for AF_ITEM in plan.json execution.json summary.json SUMMARY.md TOPOLOGY.html VARIABLES.json VARIABLES.md SHARED_PROCESSES.json SHARED_PROCESSES.md results runtime exit-code.txt; do
    [[ ! -e "$AF_ROOT/$AF_ITEM" ]] || AF_FILES+=("$AF_ITEM")
  done
  for AF_ITEM in "$AF_ROOT"/launcher-*.log; do
    [[ ! -f "$AF_ITEM" ]] || AF_FILES+=("$(basename "$AF_ITEM")")
  done
  ((${#AF_FILES[@]})) || fail 'No saved campaign evidence.'
  tar -czf "$AF_ROOT/inspection.tar.gz.tmp" -C "$AF_ROOT" "${AF_FILES[@]}"
  mv "$AF_ROOT/inspection.tar.gz.tmp" "$AF_ROOT/inspection.tar.gz"
  echo "Inspection package: $AF_ROOT/inspection.tar.gz"
}

case "${1:-}" in
  run)
    [[ -x "$AF_PYTHON" && -f "$AF_WORK/probe/plan.json" ]] || fail 'Complete the H100 compatibility check first.'
    [[ -f "$AF_REPO/inputs/topology-source/plan.json" && -f "$AF_REPO/probe-requests.json" ]] || fail 'Use the delivered pilot bundle.'
    sudo -n true || fail 'Run sudo -v first; reading the local server needs Docker access.'
    mkdir -p "$AF_ROOT"
    for AF_LOCK in "$AF_WORK/worker.lock" "$AF_ROOT/worker.lock"; do
      if [[ -f "$AF_LOCK" ]] && ! flock -n "$AF_LOCK" true; then
        fail 'A compatibility or construction worker is still running.'
      fi
    done
    AF_LOG="launcher-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    nohup bash "$0" _worker >"$AF_ROOT/$AF_LOG" 2>&1 </dev/null &
    echo "Worker PID: $!"
    ln -sfn "$AF_LOG" "$AF_ROOT/launcher.log"
    echo "Progress: tail -f $AF_ROOT/launcher.log"
    ;;
  _worker)
    exec 9>"$AF_WORK/worker.lock"
    flock -n 9 || fail 'The local server is already in use by another worker.'
    exec 8>"$AF_ROOT/worker.lock"
    flock -n 8 || fail 'This campaign already has a worker.'
    printf 'running\n' >"$AF_ROOT/exit-code.txt"
    trap 'AF_EXIT=$?; printf "%s\n" "$AF_EXIT" >"$AF_ROOT/exit-code.txt"' EXIT
    mkdir -p "$AF_ROOT/runtime"
    [[ "$(sudo -n docker inspect -f '{{index .Config.Labels "autoformalism.role"}}' "$AF_CONTAINER")" == construction-compatibility ]] || fail 'The expected owned model server is absent.'
    [[ "$(sudo -n docker inspect -f '{{.State.Running}}' "$AF_CONTAINER")" == true ]] || fail 'Server is stopped. Run the original compatibility launcher with run to restart it.'
    sudo -n docker inspect "$AF_CONTAINER" | "$AF_PYTHON" \
      "$AF_REPO/scripts/jetstream_server_record.py" --work "$AF_WORK" --image "$AF_IMAGE"
    cp "$AF_WORK/server.json" "$AF_ROOT/runtime/server.json"
    cp "$AF_WORK/weights.json" "$AF_ROOT/runtime/weights.json"
    cp "$AF_WORK/probe/plan.json" "$AF_ROOT/runtime/compatibility-plan.json"
    "$AF_PYTHON" - "$AF_ROOT" <<'PY'
import os, shutil, sys
p = sys.argv[1]
if shutil.disk_usage(p).free < 2 * 1024**3:
    raise SystemExit('Need 2 GiB free before starting this construction pilot.')
s = os.statvfs(p)
if s.f_files and s.f_favail < 10000:
    raise SystemExit('Need 10,000 free filesystem entries before this pilot.')
PY
    AF_EXIT=0
    "$AF_PYTHON" "$AF_REPO/scripts/jetstream_construction_campaign.py" run \
      --root "$AF_ROOT" --work "$AF_WORK" \
      --source "$AF_REPO/inputs/topology-source" --requests "$AF_REPO/probe-requests.json" || AF_EXIT=$?
    sudo -n docker logs "$AF_CONTAINER" >"$AF_ROOT/runtime/server.log" 2>&1 || true
    # Package only after this still-open log is closed, via inspect.
    echo "Worker exit: $AF_EXIT. Results: $AF_ROOT/SUMMARY.md"
    echo 'Use inspect to render the latest reports and create the download archive.'
    exit "$AF_EXIT"
    ;;
  status)
    if [[ -f "$AF_ROOT/worker.lock" ]] && ! flock -n "$AF_ROOT/worker.lock" true; then
      echo 'Worker: running'
    else
      echo 'Worker: stopped'
    fi
    [[ ! -f "$AF_ROOT/exit-code.txt" ]] || cat "$AF_ROOT/exit-code.txt"
    [[ ! -f "$AF_ROOT/SUMMARY.md" ]] || cat "$AF_ROOT/SUMMARY.md"
    [[ ! -f "$AF_ROOT/launcher.log" ]] || tail -n 20 "$AF_ROOT/launcher.log"
    ;;
  inspect)
    mkdir -p "$AF_ROOT"
    exec 8>"$AF_ROOT/worker.lock"
    flock -n 8 || fail 'Wait for the worker to stop before creating a consistent archive.'
    if [[ -f "$AF_ROOT/execution.json" ]]; then
      "$AF_PYTHON" "$AF_REPO/scripts/jetstream_construction_campaign.py" report --root "$AF_ROOT"
    fi
    package
    ;;
  *) fail 'Use run, status, or inspect.' ;;
esac
