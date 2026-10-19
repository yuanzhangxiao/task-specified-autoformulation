#!/bin/bash
# Same saved models and seeds, with isolated scope-aware feedback and new budgets.
set -euo pipefail
AF_SCOPE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
export AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-$AF_GROUP/phase-c-initialization-scope-repair-seeds5-v1}
export AF_GRAPH_REPAIR_FEEDBACK_POLICY=scope-aware-2
export AF_WAVE=${AF_WAVE:-scope-1}
exec bash "$AF_SCOPE_REPO/scripts/hpc/start_initialization_repair_aces.sh" "$@"
