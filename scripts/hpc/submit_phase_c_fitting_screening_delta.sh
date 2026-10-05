#!/bin/bash
# M9: bounded screens and training-fitted assisted collocation; CPU only.
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_FITTING_CONFIG="$AF_CODE/configs/phase_c_fitting_screening_assistance_v1.json"
export AF_FITTING_INPUTS=${AF_FITTING_INPUTS:-$AF_CODE/assisted-inputs.json}
export AF_CAMPAIGN=${AF_CAMPAIGN:-/work/hdd/bibo/yxiao2/phase_c/fitting-screening-assistance-v1}
[[ -z "${AF_MATCHED_SOURCE:-}" ]] || { echo "M9 freezes both generic and explicitly assisted starts; unset AF_MATCHED_SOURCE." >&2; exit 1; }
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_strategies_delta.sh"
