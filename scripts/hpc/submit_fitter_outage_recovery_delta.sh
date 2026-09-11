#!/bin/bash
# The user submits one campaign; a native CPU smoke gates its recovery array.
set -euo pipefail
campaign="${1:?Usage: $0 stopping|piecewise}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
export AF_BOOTSTRAP_PYTHON="${AF_BOOTSTRAP_PYTHON:-/usr/bin/python3}"
export AF_NUMERICAL_PYTHON="${AF_NUMERICAL_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
dependencies="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
case "${campaign}" in
  stopping)
    source_root="${AF_ORIGINAL_OUTPUT:-/work/hdd/bibo/yxiao2/phase_b/fitter-stopping-v1}"
    legacy_repo="${AF_LEGACY_REPO:-/projects/bibo/yxiao2/repos/autoformalism-fitter-stopping-v1}"
    ;;
  piecewise)
    source_root="${AF_ORIGINAL_OUTPUT:-/work/hdd/bibo/yxiao2/phase_b/piecewise-fitter-v1}"
    legacy_repo="${AF_LEGACY_REPO:-/projects/bibo/yxiao2/repos/autoformalism-piecewise-fitter-v1}"
    ;;
  *) echo 'Expected stopping or piecewise.' >&2; exit 2 ;;
esac
export AF_RECOVERY_OUTPUT="${AF_RECOVERY_OUTPUT:-/work/hdd/bibo/yxiao2/phase_b/${campaign}-outage-recovery-v1}"
export AF_RECOVERY_DRIVER="${AF_RECOVERY_OUTPUT}/recovery-driver.py"
concurrency="${AF_ARRAY_CONCURRENCY:-1}"
[[ "${concurrency}" =~ ^[12]$ ]] || { echo 'Concurrency must be 1 or 2.' >&2; exit 2; }
[[ -x "${AF_BOOTSTRAP_PYTHON}" && -z "$(git -C "${repo}" status --porcelain)" ]] || {
  echo 'Use a clean recovery checkout and a working local bootstrap Python.' >&2; exit 2;
}
mkdir -p "${AF_RECOVERY_OUTPUT}/logs"
if [[ -f "${AF_RECOVERY_DRIVER}" ]]; then
  cmp "${repo}/scripts/recover_fitter_outage.py" "${AF_RECOVERY_DRIVER}"
else
  cp "${repo}/scripts/recover_fitter_outage.py" "${AF_RECOVERY_DRIVER}"
fi
"${AF_BOOTSTRAP_PYTHON}" "${AF_RECOVERY_DRIVER}" prepare \
  --campaign "${campaign}" --source "${source_root}" --legacy-repo "${legacy_repo}" \
  --python "${AF_NUMERICAL_PYTHON}" --dependencies "${dependencies}" \
  --output "${AF_RECOVERY_OUTPUT}"
if [[ -f "${AF_RECOVERY_OUTPUT}/submission.json" ]]; then
  "${AF_BOOTSTRAP_PYTHON}" - "${AF_RECOVERY_OUTPUT}/submission.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print(json.dumps(record, indent=2))
if not record.get("submission_complete"):
    raise SystemExit("Partial submission: reconcile recorded job IDs before continuing.")
PY
  exit 0
fi
mkdir "${AF_RECOVERY_OUTPUT}/submission.intent" 2>/dev/null || {
  echo 'Submission already attempted; reconcile the queue before retrying.' >&2; exit 3;
}
record_job() {
  "${AF_BOOTSTRAP_PYTHON}" - "${AF_RECOVERY_OUTPUT}" "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1]); path = root / "submission.json"
record = json.load(path.open()) if path.exists() else {}
record.update({sys.argv[2]: sys.argv[3], "cpus_per_task": 1, "gpus": 0,
               "submission_complete": sys.argv[2] in ("summary_job_id", "no_work")})
temp = root / "submission.tmp"
temp.write_text(json.dumps(record, indent=2) + "\n")
os.replace(str(temp), str(path))
PY
}
indices="$("${AF_BOOTSTRAP_PYTHON}" "${AF_RECOVERY_DRIVER}" indices --output "${AF_RECOVERY_OUTPUT}")"
if [[ -z "${indices}" ]]; then
  "${AF_BOOTSTRAP_PYTHON}" "${AF_RECOVERY_DRIVER}" summarize --output "${AF_RECOVERY_OUTPUT}"
  record_job no_work true
  exit 0
fi
launcher="${repo}/scripts/hpc/fitter_outage_recovery_delta.slurm"
smoke="$(sbatch --parsable --time=00:10:00 --output="${AF_RECOVERY_OUTPUT}/logs/smoke-%j.out" --error="${AF_RECOVERY_OUTPUT}/logs/smoke-%j.err" --export=ALL "${launcher}" smoke)"
smoke="${smoke%%;*}"
record_job smoke_job_id "${smoke}"
fit="$(sbatch --parsable --array="${indices}%${concurrency}" --dependency="afterok:${smoke}" --kill-on-invalid-dep=yes --output="${AF_RECOVERY_OUTPUT}/logs/fit-%A_%a.out" --error="${AF_RECOVERY_OUTPUT}/logs/fit-%A_%a.err" --export=ALL "${launcher}" run)"
fit="${fit%%;*}"
record_job fit_job_id "${fit}"
summary="$(sbatch --parsable --time=00:20:00 --dependency="afterany:${fit}" --output="${AF_RECOVERY_OUTPUT}/logs/summary-%j.out" --error="${AF_RECOVERY_OUTPUT}/logs/summary-%j.err" --export=ALL "${launcher}" summarize)"
summary="${summary%%;*}"
record_job summary_job_id "${summary}"
cat "${AF_RECOVERY_OUTPUT}/submission.json"
