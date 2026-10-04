#!/bin/bash
# M8: three fixed methods, two checkpoint policies, two cases, three starts; CPU only.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_FITTING_CONFIG="$AF_CODE/configs/phase_c_fitting_checkpoint_diagnostic_v1.json"
export AF_FITTING_INPUTS=${AF_FITTING_INPUTS:-$AF_CODE/challenging-inputs.json}
export AF_CAMPAIGN=${AF_CAMPAIGN:-/work/hdd/bibo/yxiao2/phase_c/fitting-checkpoint-diagnostic-v1}
[[ -z "${AF_MATCHED_SOURCE:-}" ]] || { echo "M8 uses generic starts; unset AF_MATCHED_SOURCE." >&2; exit 1; }
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_strategies_delta.sh"
