#!/bin/bash
#SBATCH --job-name=phase-b-screen
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-39%40
#SBATCH --output=logs/phase-b-screen-%A_%a.out
#SBATCH --error=logs/phase-b-screen-%A_%a.err

set -euo pipefail

: "${AF_REPO_ROOT:?set AF_REPO_ROOT to the repository checkout}"
: "${AF_PYTHON:?set AF_PYTHON to the project virtual-environment Python}"
: "${AF_PUBLIC_DATA_ROOT:?set AF_PUBLIC_DATA_ROOT to the released suite root}"
: "${AF_CANDIDATE_POOL:?set AF_CANDIDATE_POOL to the portable JSONL pool}"
: "${AF_OUTPUT_ROOT:?set AF_OUTPUT_ROOT to persistent result storage}"

readonly shards_per_cell=10
readonly cell_index=$((SLURM_ARRAY_TASK_ID / shards_per_cell))
readonly shard_index=$((SLURM_ARRAY_TASK_ID % shards_per_cell))

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

case "${cell_index}" in
  0)
    source_benchmark=original_b1
    destination_benchmark=phase_b_dalla_man_t1_canonical_named_easy
    tier=easy
    label=canonical_named_easy
    ;;
  1)
    source_benchmark=original_b1
    destination_benchmark=phase_b_dalla_man_t1_canonical_named_hard
    tier=hard
    label=canonical_named_hard
    ;;
  2)
    source_benchmark=perturbed_b1
    destination_benchmark=phase_b_dalla_man_t1_perturbed_named_easy
    tier=easy
    label=perturbed_named_easy
    ;;
  3)
    source_benchmark=perturbed_b1
    destination_benchmark=phase_b_dalla_man_t1_perturbed_named_hard
    tier=hard
    label=perturbed_named_hard
    ;;
  *)
    echo "unsupported cell index: ${cell_index}" >&2
    exit 2
    ;;
esac

cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/replay_phase_b_frozen_candidates.py \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --candidate-pool "${AF_CANDIDATE_POOL}" \
  --source-benchmark "${source_benchmark}" \
  --destination-benchmark "${destination_benchmark}" \
  --tier "${tier}" \
  --output-root "${AF_OUTPUT_ROOT}/screen/${label}/shard-$(printf '%03d' "${shard_index}")" \
  --maximum-candidates 100 \
  --max-nfev 3 \
  --timeout-seconds 60 \
  --shard-index "${shard_index}" \
  --shard-count "${shards_per_cell}" \
  --screen-only
