#!/bin/bash
# Fresh five-seed repair-protocol confirmation using the existing verified server.
set -euo pipefail
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF_ACTION=${1:?Use run, status, or inspect}
AF_REPO=$(cd "$AF_TOOLS/../.." && pwd)
export AF_JS2_TRAINING_EVIDENCE=${AF_JS2_TRAINING_EVIDENCE:-$AF_REPO/inputs/training-evidence.json}
if [[ "$AF_ACTION" == run && ! -f "$AF_JS2_TRAINING_EVIDENCE" ]]; then
  echo 'Set AF_JS2_TRAINING_EVIDENCE to the sealed complete training evidence bundle.' >&2
  exit 2
fi
export AF_JS2_STUDY=evidence_repair_confirmation
export AF_JS2_SEEDS=${AF_JS2_SEEDS:-0 1 2 3 4}
export AF_JS2_CAMPAIGN_ROOT=${AF_JS2_CAMPAIGN_ROOT:-${AF_JS2_ROOT:-$HOME/af-construction}/evidence-repair-seeds5-v2}
exec bash "$AF_TOOLS/run_construction.sh" "$AF_ACTION"
