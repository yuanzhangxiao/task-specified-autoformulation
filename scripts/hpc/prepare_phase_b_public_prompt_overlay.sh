#!/bin/bash
set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_SOURCE_PUBLIC_DATA_ROOT:=${AF_PROJECT}/phase_b/inputs/public}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v2}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_PROMPT_OVERLAY_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_prompt_overlay_v2.json}"

"${AF_PYTHON}" \
  "${AF_REPO_ROOT}/scripts/prepare_phase_b_public_prompt_overlay.py" \
  --source-data-root "${AF_SOURCE_PUBLIC_DATA_ROOT}" \
  --output-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --config "${AF_PROMPT_OVERLAY_CONFIG}"

echo "AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT}"
echo "AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT}"
