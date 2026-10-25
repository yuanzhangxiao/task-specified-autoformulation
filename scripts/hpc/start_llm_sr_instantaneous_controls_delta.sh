#!/bin/bash
# A distinct root prevents an inherited AF_OUTPUT_ROOT from mixing campaigns.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_LLMSR_INSTANTANEOUS_ROOT=${AF_LLMSR_INSTANTANEOUS_ROOT:-/work/hdd/bibo/yxiao2/phase_c/llm-sr-instantaneous-controls-v1}
AF_WORKER="$AF_REPO_ROOT/scripts/hpc/run_llm_sr_instantaneous_controls_delta.sh"
case "${1:?Use run, status, report, or inspect}" in
  run)
    bash "$AF_WORKER" prepare --inputs "${AF_LLMSR_INSTANTANEOUS_INPUTS:-$AF_REPO_ROOT/inputs/llm-sr-instantaneous-controls-input.json}"
    exec bash "$AF_WORKER" submit
    ;;
  status|report|inspect) exec bash "$AF_WORKER" "$1" ;;
  *) echo 'Use run, status, report, or inspect.' >&2; exit 2 ;;
esac
