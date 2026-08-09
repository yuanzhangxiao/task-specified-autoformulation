#!/bin/bash
#SBATCH --job-name=phase-b-merge
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-3%4
#SBATCH --output=logs/phase-b-merge-%A_%a.out
#SBATCH --error=logs/phase-b-merge-%A_%a.err

set -euo pipefail

: "${AF_REPO_ROOT:?set AF_REPO_ROOT to the repository checkout}"
: "${AF_PYTHON:?set AF_PYTHON to the project virtual-environment Python}"
: "${AF_PUBLIC_DATA_ROOT:?set AF_PUBLIC_DATA_ROOT to the released suite root}"
: "${AF_CANDIDATE_POOL:?set AF_CANDIDATE_POOL to the portable JSONL pool}"
: "${AF_OUTPUT_ROOT:?set AF_OUTPUT_ROOT to persistent result storage}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

case "${SLURM_ARRAY_TASK_ID}" in
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
    echo "unsupported array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/merge_phase_b_replay_shards.py \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --candidate-pool "${AF_CANDIDATE_POOL}" \
  --source-benchmark "${source_benchmark}" \
  --destination-benchmark "${destination_benchmark}" \
  --tier "${tier}" \
  --shards-root "${AF_OUTPUT_ROOT}/screen/${label}" \
  --output-root "${AF_OUTPUT_ROOT}/final/${label}" \
  --maximum-candidates 100 \
  --refine-top-k 5 \
  --refine-max-nfev 10 \
  --refine-timeout-seconds 120
