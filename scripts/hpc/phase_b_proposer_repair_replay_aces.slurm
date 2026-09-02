#!/bin/bash
#SBATCH --job-name=proposer-repair-replay
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_REPLAY_PLAN:?AF_REPLAY_PLAN is required}"
: "${AF_SOURCE_OUTPUT_ROOT:?AF_SOURCE_OUTPUT_ROOT is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_TARGET_CONTRACT_ROOT:?AF_TARGET_CONTRACT_ROOT is required}"
: "${AF_REPLAY_OUTPUT_ROOT:?AF_REPLAY_OUTPUT_ROOT is required}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/replay_phase_b_proposer_first_attempt_repairs.py \
  --replay-plan "${AF_REPLAY_PLAN}" \
  --experiment-root "${AF_SOURCE_OUTPUT_ROOT}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --output-root "${AF_REPLAY_OUTPUT_ROOT}"

cd "${AF_REPLAY_OUTPUT_ROOT}"
sha256sum -c proposer_repair_replay.json.sha256
jq -r '"\(.sha256)  \(.path)"' artifact_ledger.jsonl | sha256sum -c -
