#!/bin/bash
# Separate M4 campaign; the original M3 allocation/results stay at their paths.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_FITTING_CONFIG="$AF_CODE/configs/phase_c_fitting_strategies_v2.json"
export AF_MATCHED_SOURCE=${AF_MATCHED_SOURCE:-$AF_CODE/matched-m3-inputs}
export AF_CAMPAIGN=${AF_CAMPAIGN:-/work/hdd/bibo/yxiao2/phase_c/fitting-budget-reuse-v1}
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_strategies_delta.sh"
