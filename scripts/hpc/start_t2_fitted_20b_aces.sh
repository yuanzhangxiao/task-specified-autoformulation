#!/bin/bash
# Independently sample revisions of each fitted T2-easy parent.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/start_t2_breadth_aces.sh" "${1:-}" fitted-20b
