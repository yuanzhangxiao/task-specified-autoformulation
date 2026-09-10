#!/bin/bash
# Shared diagnostics and an equal-budget start-portfolio comparison on Delta.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/fitter_methods_v3.json}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-methods-v3}"
exec bash "${AF_REPO_ROOT}/scripts/hpc/submit_fitter_methods_delta.sh"
