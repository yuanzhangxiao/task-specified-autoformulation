#!/bin/bash
# Offline pre-checks for the external-baseline frozen test evaluation.
#
# Runs the Sol budget audit, the 12-vs-24 contract check, the public-data
# identity pre-check, and the source inventory. Opens no test or private data
# and makes no provider call. It deliberately does NOT submit the sealed chain:
# the inventory must be reviewed first.

set -euo pipefail
readonly af_user="${USER:?}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_SYMBOLIC_FREEZE:=${AF_WORK}/phase_b/public-baselines-full-v1/common-readiness-freeze}"
: "${AF_D3_CAMPAIGN_ROOT:=${AF_WORK}/phase_b/d3-native-full-v1}"
: "${AF_HIDDEN_AUDIT:=${AF_WORK}/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json}"
: "${AF_SOL_FREEZE_MANIFEST:=${AF_WORK}/phase_b/raw-agent-deterministic-evaluation-v1/frozen/raw_agent_freeze_manifest.json}"
: "${AF_INVENTORY_ROOT:=${AF_WORK}/phase_b/external-baseline-inventory-v1/frozen}"
: "${AF_BUDGET24_ROOT:=${AF_WORK}/phase_b/raw-data-agent-budget24-v1}"

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
cd "${AF_REPO_ROOT}"
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
readonly plan="${AF_REPO_ROOT}/configs/external_baseline_frozen_test_evaluation_v1.json"

banner() { printf '\n========== %s ==========\n' "$1"; }

banner "0. checkout"
git rev-parse HEAD
"${AF_PYTHON}" -c "
import json;d=json.load(open('${plan}'))
print('plan status :', d['status'])
print('roster      :', d['reporting_roster'], '|', len(d['cells']), 'cells x',
      len(d['repetitions']), 'reps x', len(d['methods']), 'methods')
"

# Informational: a missing root is reported, not fatal.
banner "1. Sol tool-budget audit (no API call)"
for name in raw-data-agent-pilot-v1 raw-data-agent-fitted-v1 \
            raw-data-agent-fitted-prompt-v3-refresh-v1; do
  root="${AF_WORK}/phase_b/${name}"
  if [[ -d "${root}" ]]; then
    echo "-- ${name}"
    "${AF_PYTHON}" scripts/audit_raw_data_agent_budget.py --root "${root}" || \
      echo "   audit failed for ${name}"
  else
    echo "-- ${name}: absent"
  fi
done

banner "2. 12-versus-24 sensitivity (no API call)"
if [[ -d "${AF_BUDGET24_ROOT}" ]]; then
  "${AF_PYTHON}" scripts/audit_raw_data_agent_budget.py \
    --root "${AF_BUDGET24_ROOT}" || echo "   audit failed"
  echo "-- recorded output contracts:"
  jq -r '.output_contract // "unrecorded"' "${AF_BUDGET24_ROOT}"/*/run_config.json \
    2>/dev/null | sort | uniq -c || echo "   no run_config.json found"
else
  echo "no 24-call sensitivity run exists at ${AF_BUDGET24_ROOT}"
fi

# Fatal from here: these gate the sealed run.
banner "3. public-data identity pre-check (development splits only)"
AF_PUBLIC_DATA_ROOT="${AF_PUBLIC_DATA_ROOT}" "${AF_PYTHON}" -c "
import os
from pathlib import Path
from autoformalism.rebuttal.external_baseline_freeze import (
    load_external_baseline_plan, public_development_identity)
plan = load_external_baseline_plan(Path('${plan}'))
root = Path(os.environ['AF_PUBLIC_DATA_ROOT'])
print('public identity:', public_development_identity(root, plan))
"

banner "4. source inventory (opens nothing)"
"${AF_PYTHON}" scripts/prepare_external_baseline_frozen_test_evaluation.py \
  --config "${plan}" \
  --symbolic-development-freeze "${AF_SYMBOLIC_FREEZE}" \
  --d3-campaign-root "${AF_D3_CAMPAIGN_ROOT}" \
  --hidden-audit "${AF_HIDDEN_AUDIT}" \
  --raw-agent-freeze-manifest "${AF_SOL_FREEZE_MANIFEST}" \
  --output-root "${AF_INVENTORY_ROOT}"

jq '{expected_source_count, available_source_count, missing_source_count,
     evaluator_unsupported_count, adapter_request_count, counts_by_method}' \
  "${AF_INVENTORY_ROOT}/external_baseline_freeze.json"

banner "offline checks complete"
cat <<'NOTE'
No test or private data was opened.

Review the counts above before the sealed run. Then, separately:

  export AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/external-baseline-evaluation-v1
  bash scripts/hpc/submit_external_baseline_evaluation_delta.sh
NOTE
