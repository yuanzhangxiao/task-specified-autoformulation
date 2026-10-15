#!/bin/bash
# Explicit complete training evidence; fresh five-seed study on the existing server.
set -euo pipefail
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF_ACTION=${1:?Use run, status, or inspect}
if [[ "$AF_ACTION" == run && -z "${AF_JS2_TRAINING_EVIDENCE:-}" ]]; then
  echo 'Set AF_JS2_TRAINING_EVIDENCE to the sealed complete training evidence bundle.' >&2
  exit 2
fi
export AF_JS2_STUDY=evidence_feedback_confirmation
export AF_JS2_SEEDS=${AF_JS2_SEEDS:-0 1 2 3 4}
export AF_JS2_CAMPAIGN_ROOT=${AF_JS2_CAMPAIGN_ROOT:-${AF_JS2_ROOT:-$HOME/af-construction}/evidence-feedback-seeds5-v1}
exec bash "$AF_TOOLS/run_construction.sh" "$AF_ACTION"
