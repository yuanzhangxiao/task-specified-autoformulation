#!/bin/bash
# Run the matched 20B selection experiment with a separate result directory.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/start_topology_repair_20b_aces.sh" "${1:-}" matched-selection
