#!/bin/bash
# Continue the approved returned endpoints in a separate, nonrenewable ledger.
set -euo pipefail
AF_CONTINUATION_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_LLMSR_DETENTION_STAGED_POLISHING_ROOT=${AF_LLMSR_DETENTION_STAGED_CONTINUATION_ROOT:-/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-staged-continuation-v1}
export AF_LLMSR_DETENTION_STAGED_POLISHING_INPUTS=${AF_LLMSR_DETENTION_STAGED_CONTINUATION_INPUTS:-$AF_CONTINUATION_REPO/inputs/llm-sr-detention-staged-continuation-input.json}
exec bash "$AF_CONTINUATION_REPO/scripts/hpc/start_llm_sr_detention_staged_polishing_delta.sh" "$@"
