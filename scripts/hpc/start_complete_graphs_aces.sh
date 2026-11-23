#!/bin/bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/start_topology_repair_20b_aces.sh" "${1:-}" complete-graphs
