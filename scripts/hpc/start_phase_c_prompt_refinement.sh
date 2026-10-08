#!/bin/bash
# Eight fresh minimal constructions plus three current-prompt saved repair episodes.
set -euo pipefail
export AF_COMPARISON_STUDY=refinement_confirmation
export AF_COMPARISON_WAVE=${AF_COMPARISON_WAVE:-prompt-refinement-1}
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -f "$AF_TOOLS/inputs/topology-source/plan.json" ]]; then
  export AF_COMPARISON_SOURCE=${AF_COMPARISON_SOURCE:-$AF_TOOLS/inputs/topology-source}
fi
exec bash "$AF_TOOLS/scripts/hpc/start_phase_c_construction_comparison.sh" "$@"
