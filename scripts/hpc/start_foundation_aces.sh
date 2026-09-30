#!/bin/bash
# Separate historical closeout and corrected-data pilot.
set -euo pipefail
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" ]] || { echo 'Run this command on ACES.' >&2; exit 1; }
module load GCCcore/13.2.0 Python/3.11.5
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1
case "${1:?Use closeout, pilot, inspect-closeout, or inspect-pilot}" in
  closeout)
    export AF_REPO_ROOT="$AF_GROUP/repos/final-components-34378d1"
    export AF_OUTPUT_ROOT="$AF_GROUP/final-components-v1"
    export AF_COMMIT=34378d1f1da5eefb2c1f977b440346afab29a33f
    [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
    export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
    if [[ -z "${AF_JETSTREAM_API_KEY:-}" ]]; then
      read -r -s -p 'Jetstream API key (hidden): ' AF_JETSTREAM_API_KEY
      printf '\n'
      export AF_JETSTREAM_API_KEY
    fi
    exec "$AF_PYTHON" "$AF_TOOLS/scripts/submit_component_closeout.py" \
      --repo "$AF_REPO_ROOT" --root "$AF_OUTPUT_ROOT" --wave endpoint-closeout-1 --pruning 4
    ;;
  pilot)
    export AF_REPO_ROOT="$AF_TOOLS"
    export AF_OUTPUT_ROOT="$AF_GROUP/phase-c-construction-v1"
    export PYTHONPATH="$AF_REPO_ROOT/src"
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    export AF_HF_HOME=${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}
    export AF_COMPUTE_CACHE_ROOT="$AF_GROUP/phase-c-runtime-cache"
    export AF_IPC_TMP_ROOT=/tmp/phase-c-ipc-u.yx126462
    AF_RELEASE=${AF_PHASE_C_RELEASE:-$AF_TOOLS/public-release}
    [[ -f "$AF_VLLM_IMAGE" && -d "$AF_HF_HOME" ]]
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/phase_c_baseline.py" prepare \
      --release "$AF_RELEASE" --root "$AF_OUTPUT_ROOT"
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_phase_c_baseline.py" \
      --root "$AF_OUTPUT_ROOT" --wave construction-1
    ;;
  inspect-closeout)
    AF_ROOT="$AF_GROUP/final-components-v1"
    AF_CAPTURE=$(mktemp -d "$AF_GROUP/phase-b-closeout-inspection.XXXXXX")
    "$AF_PYTHON" "$AF_TOOLS/scripts/audit_component_campaign.py" collect \
      --root "$AF_ROOT" --output "$AF_CAPTURE/campaign"
    AF_JOBS=$(paste -sd, "$AF_CAPTURE/campaign/job_ids.txt")
    sacct -j "$AF_JOBS" --parsable2 \
      --format=JobID%40,JobIDRaw%40,JobName%40,State%30,ExitCode,Elapsed,Submit,Start,End \
      > "$AF_CAPTURE/scheduler.psv"
    "$AF_PYTHON" "$AF_TOOLS/scripts/audit_component_campaign.py" report \
      --input "$AF_CAPTURE/campaign/audit.json" --scheduler "$AF_CAPTURE/scheduler.psv" \
      --output "$AF_CAPTURE/reconciled"
    cat "$AF_CAPTURE/reconciled/summary.json"
    tar -czf "$AF_CAPTURE/closeout-results.tar.gz" -C "$AF_CAPTURE" reconciled scheduler.psv
    printf '\nDownload: %s\n' "$AF_CAPTURE/closeout-results.tar.gz"
    ;;
  inspect-pilot)
    export PYTHONPATH="$AF_TOOLS/src"
    AF_ROOT="$AF_GROUP/phase-c-construction-v1"
    "$AF_PYTHON" "$AF_TOOLS/scripts/phase_c_baseline.py" report --root "$AF_ROOT"
    cat "$AF_ROOT/SUMMARY.md"
    tar -czf "$AF_ROOT/inspection.tar.gz" -C "$AF_ROOT" \
      plan.json summary.json SUMMARY.md EQUATIONS.md results logs submissions
    printf '\nDownload: %s\n' "$AF_ROOT/inspection.tar.gz"
    ;;
  *) echo 'Unknown action' >&2; exit 2 ;;
esac
