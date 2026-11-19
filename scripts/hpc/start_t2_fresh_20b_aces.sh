#!/bin/bash
# Five fresh T2-easy seeds using the 20B proposer and literal public output keys.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/start_t2_breadth_aces.sh" "${1:-}" fresh-20b
