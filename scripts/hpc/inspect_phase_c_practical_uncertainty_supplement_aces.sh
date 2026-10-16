#!/bin/bash
set -euo pipefail
AF_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
module load GCCcore/13.2.0 Python/3.11.5
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
AF_PROFILE_ROOT=${1:-${AF_PROFILE_ROOT:-$AF_GROUP/phase_c/practical-uncertainty-supplement-aces-v1}}
# An explicitly empty path prevents the shared inspector's Delta dependency default.
export AF_CASADI_ROOT=${AF_CASADI_ROOT-}
exec bash "$AF_CODE/scripts/hpc/inspect_phase_c_practical_uncertainty_supplement_delta.sh" "$AF_PROFILE_ROOT"
