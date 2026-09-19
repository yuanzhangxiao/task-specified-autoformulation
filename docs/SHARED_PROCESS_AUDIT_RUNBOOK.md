# Run Milestone 1 on the saved round-15 models

This is a short CPU-only equation audit, not a fitting job. No GPU, LLM endpoint,
benchmark trajectory directory or test dataset is needed. It leaves source
campaigns untouched. The first command exports existing selected model equations,
parameters, context and scalar development scores; the new audit uses only the
equations/context/provenance. It does not recompute any scores.

Use the commit provided with this milestone as `AF_AUDIT_COMMIT`. Fetch through an
existing authenticated checkout. The commands use a small source archive under
group scratch to avoid another Git checkout and its inode overhead. Do not change
the revision of a checkout that a running job uses.

On ACES, paste the following in a subshell after setting the exact commit:

```bash
(
  set -euo pipefail
  : "${AF_AUDIT_COMMIT:?Set to the milestone commit first}"
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  AF_CODE="$AF_GROUP/shared-process-audit-code-$AF_AUDIT_COMMIT"
  AF_INPUT="$AF_GROUP/shared-process-audit-inputs-v1"
  AF_OUTPUT="$AF_GROUP/shared-process-audit-v1"
  AF_SOURCE="$AF_GROUP/review-continuation-v5-round15"
  AF_PYTHON="$AF_BASE/.venv/bin/python"

  test -f "$AF_SOURCE/plan.json"
  test -x "$AF_PYTHON"
  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_AUDIT_COMMIT^{commit}"
  mkdir -p "$AF_CODE" "$AF_INPUT"
  git -C "$AF_BASE" archive "$AF_AUDIT_COMMIT" \
    src scripts/mechanism_audit.py scripts/audit_shared_processes.py |
    tar -x -C "$AF_CODE"

  module load GCCcore/13.2.0 Python/3.11.5
  export PYTHONPATH="$AF_CODE/src"
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

  "$AF_PYTHON" "$AF_CODE/scripts/mechanism_audit.py" export \
    --review-root "$AF_SOURCE" --round 15 --arms full brief_only \
    --output "$AF_INPUT/round15-models.json"

  "$AF_PYTHON" "$AF_CODE/scripts/audit_shared_processes.py" \
    --bundle "$AF_INPUT/round15-models.json" --output "$AF_OUTPUT"

  cat "$AF_OUTPUT/SUMMARY.md"
)
```

There is no `sbatch` step: this walks the 24 saved model definitions and checks
their syntax/closure. If a site's login-node policy prohibits even a short
read-only audit, run the same two Python commands in an ordinary one-CPU session.
It does not need a new fitting allocation. Failure to find the source plan should
be fixed by locating the completed round-15 campaign, not creating an empty root.

Rerunning the same commands verifies/reuses the sealed artifacts. If the source
selection or audit implementation changes, choose fresh input/output directory
names rather than overwriting the frozen artifacts. Unavailable selections remain
explicit rows, so the 24-row expectation is not a claim that 24 valid models exist.

Return `SUMMARY.md` and `summary.json` from `shared-process-audit-v1`. The JSON has
the expressions and consumers behind each count. These let us decide which reuse
patterns deserve construction guidance in Milestone 2. **Do not use the count of
similar terms as a mechanism-compliance score or automatically merge their fitted
parameters.** The private reference inventory is a separate diagnostic document
and is not read by this command.
