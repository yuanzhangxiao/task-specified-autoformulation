#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_CAMPAIGN=${AF_CAMPAIGN:-/work/hdd/bibo/yxiao2/phase_c/fitting-reuse-diagnostic-v1}
bash "$AF_CODE/scripts/hpc/inspect_phase_c_fitting_strategies_delta.sh"
