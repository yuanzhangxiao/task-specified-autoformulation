#!/bin/bash
# One replay array and one finalizer. Never resubmit an uncertain scheduler reply.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_RECOVERY_ROOT:?}" "${AF_RECOVERY_CODE:?}"
: "${AF_SOURCE_ROOT:?}"
module load GCCcore/13.2.0 Python/3.11.5
[[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" = dfc6f81a613e186ddffdd0b2406feca583a58944 ]]
export AF_REPO_ROOT AF_PYTHON AF_RECOVERY_ROOT AF_RECOVERY_CODE AF_SOURCE_ROOT
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# The original finalizer/dispatcher are dead. Refuse to race any surviving job.
queued="$(squeue -h -u "$USER" -o '%i %T')"
live="$(awk '$1 ~ /^(2139573|2139574|2139575|2139576)(_|$)/' <<< "$queued")"
[[ -z "$live" ]] || { echo "Original jobs still active: $live" >&2; exit 2; }
"$AF_PYTHON" "$AF_RECOVERY_CODE/recover_review_quota.py" prepare \
  --source "$AF_SOURCE_ROOT" --root "$AF_RECOVERY_ROOT" --round 2 \
  --indices 26,33,34,36,38,39,41,43

intent="$AF_RECOVERY_ROOT/quota_submission"
if ! mkdir "$intent"; then
  echo "Submission already attempted. Inspect $intent; do not duplicate jobs." >&2
  exit 2
fi
mkdir -p "$AF_RECOVERY_ROOT/quota_logs"
worker="$AF_RECOVERY_CODE/run_review_quota_recovery_aces.sh"
submit() {
  local name="$1"; shift
  printf '%q ' sbatch --parsable "$@" > "$intent/$name.command.txt"
  if ! sbatch --parsable "$@" > "$intent/$name.out" 2> "$intent/$name.err"; then
    cat "$intent/$name.err" >&2
    echo "Submission unconfirmed: inspect $intent and squeue before any retry." >&2
    exit 2
  fi
  local reply
  reply="$(cat "$intent/$name.out")"
  reply="${reply%%;*}"
  [[ "$reply" =~ ^[0-9]+$ ]] || {
    echo "Uncertain scheduler response retained in $intent/$name.out" >&2; exit 2;
  }
  printf '%s\n' "$reply" > "$intent/$name.id"
}
submit replay --account=156264627414 --nodes=1 --ntasks=1 --export=ALL \
  --partition=cpu --job-name=review-quota-replay \
  --cpus-per-task=1 --mem=16G --time=00:20:00 --array=0-7%8 \
  --output="$AF_RECOVERY_ROOT/quota_logs/replay-%A_%a.out" \
  --error="$AF_RECOVERY_ROOT/quota_logs/replay-%A_%a.err" "$worker" worker
replay="$(cat "$intent/replay.id")"
submit finalize --account=156264627414 --nodes=1 --ntasks=1 --export=ALL \
  --kill-on-invalid-dep=yes --partition=cpu --job-name=review-quota-finalize \
  --cpus-per-task=1 --mem=4G --time=00:15:00 --dependency="afterany:$replay" \
  --output="$AF_RECOVERY_ROOT/quota_logs/finalize-%j.out" \
  --error="$AF_RECOVERY_ROOT/quota_logs/finalize-%j.err" "$worker" finalize
printf 'Replay job: %s\nFinalize job: %s\n' "$replay" "$(cat "$intent/finalize.id")"
