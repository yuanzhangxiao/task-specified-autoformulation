#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_STAGNATION_CONFIG="$AF_CODE/configs/phase_c_stagnation_v2.json"
export AF_STAGNATION_ROOT=${AF_STAGNATION_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-stagnation-v2}
exec bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_stagnation_delta.sh"
