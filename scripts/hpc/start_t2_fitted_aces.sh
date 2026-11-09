#!/bin/bash
# Two five-option batches against the fitted T2 meal-memory and sigmoid parents.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/start_t2_breadth_aces.sh" "${1:-}" fitted
