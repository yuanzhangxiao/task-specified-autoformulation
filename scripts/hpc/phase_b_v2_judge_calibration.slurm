#!/bin/bash
#SBATCH --job-name=phase-b-judge-v2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --output=logs/phase-b-judge-v2-%A_%a.out
#SBATCH --error=logs/phase-b-judge-v2-%A_%a.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_PROJECT}/phase_b/inputs/public}"
: "${AF_OLLAMA_IMAGE:=${AF_PROJECT}/containers/ollama.sif}"
: "${AF_OLLAMA_MODELS:=${AF_PROJECT}/ollama-models}"
: "${AF_LOCAL_MODEL:=gpt-oss:20b}"
: "${AF_CALIBRATION_ROOT:=${AF_WORK}/phase_b/judge-v2-calibration-v1}"
: "${AF_CALIBRATION_PAIRS:=${AF_CALIBRATION_ROOT}/pairs.jsonl}"
: "${AF_REPETITIONS:=5}"
: "${AF_SHARD_COUNT:=5}"
: "${AF_JUDGE_SEED_BASE:=1000}"

readonly shard_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly shard_root="${AF_CALIBRATION_ROOT}/shards/shard_${shard_index}"
readonly port=$((20000 + SLURM_JOB_ID % 20000))
readonly endpoint="http://127.0.0.1:${port}"
readonly server_log="${shard_root}/ollama-${SLURM_JOB_ID}.log"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

[[ -f "${AF_CALIBRATION_PAIRS}" ]] || {
  echo "missing calibration pairs: ${AF_CALIBRATION_PAIRS}" >&2
  exit 2
}
mkdir -p "${shard_root}" "${AF_OLLAMA_MODELS}"
cd "${AF_REPO_ROOT}"

sha256sum "${AF_OLLAMA_IMAGE}"
apptainer exec "${AF_OLLAMA_IMAGE}" ollama --version
apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
  --env "OLLAMA_HOST=127.0.0.1:${port}" \
  --env "OLLAMA_MODELS=${AF_OLLAMA_MODELS}" \
  "${AF_OLLAMA_IMAGE}" ollama serve >"${server_log}" 2>&1 &
readonly server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 90); do
  if curl --silent --fail "${endpoint}/api/tags" >/dev/null; then
    ready=true
    break
  fi
  sleep 2
done
if [[ "${ready}" != true ]]; then
  echo "Ollama did not become ready; see ${server_log}" >&2
  exit 1
fi

"${AF_PYTHON}" scripts/run_adversarial_judge.py \
  --pairs "${AF_CALIBRATION_PAIRS}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --judge-models "ollama:${AF_LOCAL_MODEL}" \
  --baseline-label baseline \
  --repetitions "${AF_REPETITIONS}" \
  --output-root "${shard_root}" \
  --timeout-seconds 900 \
  --max-output-tokens 4096 \
  --ollama-base-url "${endpoint}" \
  --ollama-thinking low \
  --ollama-temperature 0.2 \
  --ollama-seed-base "${AF_JUDGE_SEED_BASE}" \
  --shard-index "${shard_index}" \
  --shard-count "${AF_SHARD_COUNT}" \
  --shard-strategy contiguous
