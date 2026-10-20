#!/bin/bash
# One bounded initializer-only worker using the shared verified local H100 server.
set -euo pipefail
: "${AF_REPO_ROOT:?}"
export AF_FUNCTION_CLI="$AF_REPO_ROOT/scripts/repair_function_handoff.py"
export AF_FUNCTION_SUBMIT_CLI="$AF_REPO_ROOT/scripts/submit_function_handoff_repair_aces.py"
export AF_FUNCTION_WORKER_SECONDS=1800
exec bash "$AF_REPO_ROOT/scripts/hpc/run_construction_functions_aces.sh" "$@"
