#!/bin/bash
# M6: two fixed-equation cases, three starts, six arms, one CPU per task.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_FITTING_CONFIG="$AF_CODE/configs/phase_c_fitting_challenging_v1.json"
export AF_FITTING_INPUTS=${AF_FITTING_INPUTS:-$AF_CODE/challenging-inputs.json}
export AF_CAMPAIGN=${AF_CAMPAIGN:-/work/hdd/bibo/yxiao2/phase_c/fitting-challenging-v1}
[[ -z "${AF_MATCHED_SOURCE:-}" ]] || { echo "M6 uses fresh starts; unset AF_MATCHED_SOURCE." >&2; exit 1; }
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_strategies_delta.sh"
