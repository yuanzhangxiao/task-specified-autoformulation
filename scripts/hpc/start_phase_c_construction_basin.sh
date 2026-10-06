#!/bin/bash
# Both basins, one seed/prompt across three policies; reuse the comparison runtime.
set -euo pipefail
export AF_COMPARISON_STUDY=basin_confirmation
export AF_COMPARISON_WAVE=${AF_COMPARISON_WAVE:-basins-1}
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -f "$AF_TOOLS/inputs/topology-source/plan.json" ]]; then
  export AF_COMPARISON_SOURCE=${AF_COMPARISON_SOURCE:-$AF_TOOLS/inputs/topology-source}
fi
exec bash "$AF_TOOLS/scripts/hpc/start_phase_c_construction_comparison.sh" "$@"
