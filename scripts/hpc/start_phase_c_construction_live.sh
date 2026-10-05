#!/bin/bash
# One seed/prompt across every case and policy; reuse the comparison runtime.
set -euo pipefail
export AF_COMPARISON_STUDY=live_confirmation
export AF_COMPARISON_WAVE=${AF_COMPARISON_WAVE:-live-1}
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -f "$AF_TOOLS/inputs/topology-source/plan.json" ]]; then
  export AF_COMPARISON_SOURCE=${AF_COMPARISON_SOURCE:-$AF_TOOLS/inputs/topology-source}
fi
exec bash "$AF_TOOLS/scripts/hpc/start_phase_c_construction_comparison.sh" "$@"
