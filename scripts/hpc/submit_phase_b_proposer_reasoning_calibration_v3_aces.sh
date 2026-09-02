#!/bin/bash
# Submit the matched low/medium proposer reasoning calibration on ACES H100s.

set -euo pipefail

: "${PROJECT:?PROJECT is unset}"
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"

export AF_REPO_ROOT
export AF_CALIBRATION_CONFIG="${AF_REPO_ROOT}/configs/phase_b_proposer_reasoning_calibration_v3.json"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/proposer-reasoning-calibration-v3-aces-h100x2}"
export AF_OUTPUT_ROOT
export AF_PREREQUISITE_ANALYSIS=""

exec bash "${AF_REPO_ROOT}/scripts/hpc/submit_phase_b_proposer_transport_calibration_aces.sh"
