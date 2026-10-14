#!/bin/bash
# Fresh five-seed prompt/repair pilot; topology diagnostics remain offline.
set -euo pipefail
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AF_JS2_STUDY=protocol_clarity_confirmation
export AF_JS2_SEEDS=${AF_JS2_SEEDS:-0 1 2 3 4}
export AF_JS2_CAMPAIGN_ROOT=${AF_JS2_CAMPAIGN_ROOT:-${AF_JS2_ROOT:-$HOME/af-construction}/protocol-clarity-seeds5-v1}
exec bash "$AF_TOOLS/run_construction.sh" "${1:?Use run, status, or inspect}"
