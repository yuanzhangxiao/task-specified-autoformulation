#!/bin/bash
# CPU-only diagnostics; durable receipts prevent blind duplicate submission.
set -euo pipefail
export AF_AUDIT_SITE="${1:?Use aces, delta, or jetstream2}"
export AF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$AF_REPO_ROOT/SOURCE_COMMIT" ]]; then
  AF_COMMIT=$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")
else
  AF_COMMIT=$(git -C "$AF_REPO_ROOT" rev-parse HEAD)
fi
export AF_COMMIT
export AF_REFERENCE_MODE=${AF_REFERENCE_MODE:-audit}
case "$AF_REFERENCE_MODE" in
  audit) af_label=continuous-inputs; af_job=continuous-input-audit ;;
  qualify)
    : "${AF_AUDIT_ROOT:?Set AF_AUDIT_ROOT to the completed audit directory}"
    [[ -f "$AF_AUDIT_ROOT/summary.json" ]] || { echo "Missing audit summary" >&2; exit 2; }
    export AF_AUDIT_ROOT
    af_label=phase-c-reference; af_job=reference-qualification ;;
  refine)
    : "${AF_QUALIFICATION_ROOT:?Set AF_QUALIFICATION_ROOT to the original qualification}"
    [[ -f "$AF_QUALIFICATION_ROOT/tasks.json" ]] || { echo "Missing qualification tasks" >&2; exit 2; }
    export AF_QUALIFICATION_ROOT
    af_label=phase-c-refinement; af_job=cstr-refinement ;;
  *) echo "Unknown AF_REFERENCE_MODE" >&2; exit 2 ;;
esac
case "$AF_AUDIT_SITE" in
  aces)
    AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
    AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-/scratch/group/p.nairr260351.000/u.yx126462/${af_label}-${AF_COMMIT:0:7}}
    af_account=${AF_ACCOUNT:-156264627414}
    ;;
  delta)
    AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
    AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/${af_label}-${AF_COMMIT:0:7}}
    af_account=${AF_ACCOUNT:-bibo-delta-cpu}
    ;;
  jetstream2)
    AF_PYTHON=${AF_PYTHON:-$AF_REPO_ROOT/.venv/bin/python}
    AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT:-$AF_REPO_ROOT/artifacts/${af_label}-${AF_COMMIT:0:7}}
    ;;
  *) echo "Unknown site: $AF_AUDIT_SITE" >&2; exit 2 ;;
esac
export AF_PYTHON AF_OUTPUT_ROOT
[[ -x "$AF_PYTHON" ]] || { echo "Set AF_PYTHON to an existing project Python." >&2; exit 2; }
if [[ "$AF_REFERENCE_MODE" != refine ]]; then
for af_spec in \
  data_raw/benchmark5_anonymous_nonlinear_process/private/system_specification.json \
  data_raw/benchmark6_alien_device/private/selected_system_spec.json; do
  [[ -f "$AF_REPO_ROOT/$af_spec" ]] || { echo "Missing private audit asset: $af_spec" >&2; exit 2; }
done
fi
mkdir -p "$AF_OUTPUT_ROOT/logs"
printf '%s\n' "Output: $AF_OUTPUT_ROOT" "Commit: $AF_COMMIT"
if [[ "$AF_AUDIT_SITE" == jetstream2 ]]; then
  exec bash "$AF_REPO_ROOT/scripts/hpc/run_continuous_input_audit.sh"
fi
af_receipts="$AF_OUTPUT_ROOT/submission"
if [[ -f "$af_receipts/job.id" ]]; then
  printf 'Already submitted: %s\n' "$(cat "$af_receipts/job.id")"
  exit 0
fi
if ! mkdir "$af_receipts"; then
  echo "Submission already attempted; inspect $af_receipts and sacct before retrying." >&2
  exit 2
fi
af_command=(sbatch --parsable --account="$af_account" --partition=cpu
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=01:00:00
  --job-name="$af_job" --export=ALL
  --output="$AF_OUTPUT_ROOT/logs/audit-%j.out"
  --error="$AF_OUTPUT_ROOT/logs/audit-%j.err")
if [[ "$AF_REFERENCE_MODE" == refine ]]; then
  af_command+=(--array=0-13%2)
fi
af_command+=("$AF_REPO_ROOT/scripts/hpc/run_continuous_input_audit.sh")
printf '%q ' "${af_command[@]}" > "$af_receipts/command.txt"
printf '\n' >> "$af_receipts/command.txt"
af_status=0
"${af_command[@]}" > "$af_receipts/stdout.txt" 2> "$af_receipts/stderr.txt" || af_status=$?
printf '%s\n' "$af_status" > "$af_receipts/returncode.txt"
af_reply=$(cat "$af_receipts/stdout.txt")
if [[ "$af_status" == 0 && "$af_reply" =~ ^[0-9]+(\;[A-Za-z0-9_-]+)?$ ]]; then
  printf '%s\n' "${af_reply%%;*}" > "$af_receipts/job.id"
  printf 'Submitted CPU job: %s\n' "$(cat "$af_receipts/job.id")"
else
  echo "Scheduler reply unconfirmed. Inspect $af_receipts; do not resubmit blindly." >&2
  cat "$af_receipts/stderr.txt" >&2
  exit 2
fi
