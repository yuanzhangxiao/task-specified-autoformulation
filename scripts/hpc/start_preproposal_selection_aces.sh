#!/bin/bash
# Run the screened scientific portfolio and downstream topology test.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/start_topology_repair_20b_aces.sh" "${1:-}" selection
