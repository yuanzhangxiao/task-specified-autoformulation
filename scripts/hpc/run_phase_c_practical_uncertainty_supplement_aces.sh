#!/bin/bash
# Same numerical worker; ACES requires its Python runtime modules in each job.
set -euo pipefail
module load GCCcore/13.2.0 Python/3.11.5
: "${AF_REPO_ROOT:?}"
exec bash "$AF_REPO_ROOT/scripts/hpc/run_phase_c_practical_uncertainty_supplement_delta.sh" "$@"
