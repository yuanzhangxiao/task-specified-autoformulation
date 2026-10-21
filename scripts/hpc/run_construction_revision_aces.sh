#!/bin/bash
# One bounded construction-revision worker using the shared verified local H100 server.
set -euo pipefail
: "${AF_REPO_ROOT:?}"
export AF_FUNCTION_CLI="$AF_REPO_ROOT/scripts/construction_feedback_revision.py"
export AF_FUNCTION_SUBMIT_CLI="$AF_REPO_ROOT/scripts/submit_construction_revision_aces.py"
export AF_FUNCTION_WORKER_SECONDS=5400
exec bash "$AF_REPO_ROOT/scripts/hpc/run_construction_functions_aces.sh" "$@"
