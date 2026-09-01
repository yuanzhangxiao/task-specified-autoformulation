#!/bin/bash
# Submit the v2 high-budget proposer continuation on ACES H100s.

set -euo pipefail

: "${PROJECT:?PROJECT is unset}"
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PREREQUISITE_ANALYSIS:?AF_PREREQUISITE_ANALYSIS must name the copied Delta v1 analysis}"

export AF_REPO_ROOT
export AF_PREREQUISITE_ANALYSIS
export AF_CALIBRATION_CONFIG="${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v2.json"
export AF_OUTPUT_ROOT="${SCRATCH}/phase_b/proposer-transport-calibration-v2-aces-h100x2"

exec bash "${AF_REPO_ROOT}/scripts/hpc/submit_phase_b_proposer_transport_calibration_aces.sh"
