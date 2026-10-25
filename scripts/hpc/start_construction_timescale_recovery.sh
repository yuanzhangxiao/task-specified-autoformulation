#!/bin/bash
# V2 uses its own root/input/config variables so stale v1 exports cannot redirect it.
set -euo pipefail
AF_RECOVERY_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_PLATFORM=${AF_PLATFORM:-delta}
case "$AF_PLATFORM" in
  delta)
    AF_RECOVERY_DEFAULT=/work/hdd/bibo/yxiao2/phase_c/construction-timescale-controls-v2
    ;;
  aces)
    AF_RECOVERY_DEFAULT=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}/phase-c-construction-timescale-controls-v2
    ;;
  *) echo 'AF_PLATFORM must be delta or aces.' >&2; exit 2 ;;
esac
export AF_TIMESCALE_CONTROLS_ROOT=${AF_TIMESCALE_RECOVERY_ROOT:-$AF_RECOVERY_DEFAULT}
export AF_TIMESCALE_CONTROLS_INPUTS=${AF_TIMESCALE_RECOVERY_INPUTS:-$AF_RECOVERY_REPO/inputs/timescale-recovery-inputs.json}
export AF_TIMESCALE_CONTROLS_CONFIG="$AF_RECOVERY_REPO/configs/construction_timescale_controls_v2.json"
exec bash "$AF_RECOVERY_REPO/scripts/hpc/start_construction_timescale_controls.sh" "$@"
