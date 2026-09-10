#!/bin/bash
# User-run matched joint/alternating collocation comparison on Delta CPUs.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/fitter_methods_v4.json}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-methods-v4}"
exec bash "${AF_REPO_ROOT}/scripts/hpc/submit_fitter_methods_delta.sh"
