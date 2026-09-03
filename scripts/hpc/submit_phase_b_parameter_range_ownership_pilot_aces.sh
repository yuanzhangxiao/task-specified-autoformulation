#!/bin/bash
# Submit the 2-cell x 3-seed runtime-owned parameter-range pilot on ACES CPUs.

set -euo pipefail
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_parameter_range_ownership_pilot_v1.json}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/parameter-range-ownership-pilot-v1-aces-cpu}"

export AF_REPO_ROOT AF_CONFIG AF_OUTPUT_ROOT
exec bash "${AF_REPO_ROOT}/scripts/hpc/submit_phase_b_reciprocal_fitting_pilot_aces.sh"
