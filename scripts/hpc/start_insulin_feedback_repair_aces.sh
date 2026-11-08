#!/bin/bash
# One fresh Candidate 15 patch against the fitted parent and prior feedback.
set -euo pipefail
AF_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
export AF_MEAL_INSULIN_REVISION_ROOT=${AF_INSULIN_FEEDBACK_REPAIR_ROOT:-$AF_GROUP/phase-c-insulin-feedback-120b-v1}
export AF_MEAL_INSULIN_REVISION_SOURCE="$AF_CODE/inputs/insulin-feedback-intake.json"
exec bash "$AF_CODE/scripts/hpc/start_meal_insulin_revision_aces.sh" "$@"
