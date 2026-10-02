#!/bin/bash
# M5 isolates formulation caching from primal warm starts. CPU only.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_FITTING_CONFIG="$AF_CODE/configs/phase_c_fitting_reuse_diagnostic_v1.json"
export AF_MATCHED_SOURCE=${AF_MATCHED_SOURCE:-$AF_CODE/matched-m3-inputs}
export AF_CAMPAIGN=${AF_CAMPAIGN:-/work/hdd/bibo/yxiao2/phase_c/fitting-reuse-diagnostic-v1}
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_strategies_delta.sh"
