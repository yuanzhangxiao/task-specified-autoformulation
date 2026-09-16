# Recovering the two unsubmitted v2 visits

The September 16 campaign at scientific commit
`7eff174d6819a13c740240f751c7613d5db9ed51` completed visit zero. Its dispatcher
`2135060` then failed with `Job dependency problem`. Preparation job `2135055`
was complete in accounting but no longer available in Slurm's active controller.
No visit-one jobs were accepted. Reusing that old preparation ID as a new
`afterok` dependency prevented submission.

The 88 missing rows are two future visits of 44 lineages, not 88 numerical
failures. Visit zero has 30 finite fitting results, two fitting failures, and
12 refit controls sharing their full-method parent. All 12 full-method fits
are finite; their validation NMSE spans 0.1344 to 43.1119. The revised feedback
contract has not been exercised until later visits run. Do not discard these
results or pool them with v1.

## Resume this campaign

Use the **original checkout for all scientific work**. A separate checkout
supplies only `scripts/recover_review_deadline_v2.py`. Do not run the ordinary
launcher from the new checkout against the old plan: the plan binds its source,
runtime, public assets and scientific launchers.

On ACES, substitute the recovery commit from the implementation message:

```bash
(
  set -euo pipefail
  module load GCCcore/13.2.0 Python/3.11.5
  RECOVERY_REV=REPLACE_WITH_RECOVERY_COMMIT
  BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  git -C "$BASE" fetch origin codex/prefit-aces-v1
  RECOVERY_REPO="/scratch/user/u.yx126462/repos/autoformalism-review-recovery-${RECOVERY_REV:0:7}"
  if [ ! -e "$RECOVERY_REPO/.git" ]; then
    git -C "$BASE" worktree add --detach "$RECOVERY_REPO" "$RECOVERY_REV"
  fi
  [ "$(git -C "$RECOVERY_REPO" rev-parse HEAD)" = "$RECOVERY_REV" ]
  export AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-review-v2-7eff174
  export AF_PYTHON="$BASE/.venv/bin/python"
  export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-v2
  "$AF_PYTHON" "$RECOVERY_REPO/scripts/recover_review_deadline_v2.py" --round 1
)
```

This queues proposal, fitting, reporting and a small dispatcher for visit one.
The dispatcher queues visit two after visit one finishes. Only one visit's
fitting array is queued at a time. GPU, CPU, memory, call and fitting budgets
match the original submission. It neither repeats visit zero nor grants any
extra revision or optimization attempts. All original failures remain recorded.

The driver checks the original source identity using the original Python
environment and verifier. It checks successful preparation/prior reporting in
`sacct`, confirms the failed original dispatcher, and checks both accounting and
the active queue for possibly accepted next-visit jobs. It refuses recovery if
that visit already has job IDs or scientific artifacts. Missing accounting is a
reason to stop and inspect, not a reason to assume success.

The driver records its hash and prerequisite evidence under `scheduler-recovery`.
Its own later dispatcher is bound to the same hash. It preserves the original
failed intent and logs. Successful submission updates the ordinary manifests,
so existing status and analysis commands continue working. Repeating a fully
submitted visit only returns its saved manifest. A partial or uncertain
submission stops and retains its known IDs for inspection.

The normal v2 launcher is also corrected for **future fresh campaigns**. The
existing campaign uses the separate recovery driver to preserve its original
scientific identity.

## Dispatcher-only repair after the ACES wrapper rejection

The first recovery driver used `sbatch --wrap` for its final dispatcher.
ACES's `/sw/local/bin/sbatch` forwards its arguments without preserving quoting,
so it splits the command string. In the observed attempt, proposal `2135300`,
fitting array `2135301`, and reporting `2135302` were accepted; only submission
of the visit-two dispatcher failed. Do not repeat those three jobs or remove
their intent directory.

The corrected driver submits a real shell-script file through the same site
wrapper. Its explicit `--complete-dispatcher-only` mode verifies and adopts the
three recorded job IDs, checks that no dispatcher was already accepted, and
submits only the missing dispatcher. If the report already completed, it uses
that verified completion instead of another dependency on an expired ID.
It requires expanded accounting records for every array task; a successful
parent record cannot hide a failed or missing child. If accounting is still
catching up, it stops before creating repair intent and may be retried later.
An unexpected or failed accounting record requires inspection rather than
resubmission. A saved reply indicating acceptance or an uncertain outcome also
blocks retry even if queue/accounting snapshots are temporarily empty.

On ACES, use the new repair commit supplied in the implementation message:

```bash
(
  set -euo pipefail
  module load GCCcore/13.2.0 Python/3.11.5
  REV=REPLACE_WITH_DISPATCHER_REPAIR_COMMIT
  BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  git -C "$BASE" fetch origin codex/prefit-aces-v1
  REPAIR_REPO="/scratch/user/u.yx126462/repos/autoformalism-review-dispatch-${REV:0:7}"
  if [ ! -e "$REPAIR_REPO/.git" ]; then
    git -C "$BASE" worktree add --detach "$REPAIR_REPO" "$REV"
  fi
  [ "$(git -C "$REPAIR_REPO" rev-parse HEAD)" = "$REV" ]
  export AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-review-v2-7eff174
  export AF_PYTHON="$BASE/.venv/bin/python"
  export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-v2
  "$AF_PYTHON" "$REPAIR_REPO/scripts/recover_review_deadline_v2.py" \
    --round 1 --complete-dispatcher-only
)
```

The original evidence and accepted job IDs remain untouched. New evidence,
the dispatcher script, and raw scheduler stdout/stderr/exit-code receipts are
saved under `scheduler-recovery/attempt-1/dispatcher-repair`. A repeated uncertain
repair remains blocked for inspection; a completed submission returns its saved
manifest. The newly queued dispatcher uses this corrected driver for visit two.

## Inspect the resumed results

```bash
ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-v2
cat "$ROOT/submission_manifest.json"
sacct -j "$(jq -r '[.jobs[]] | unique | join(",")' "$ROOT/submission_manifest.json")" \
  --format=JobID,JobName%28,State,ExitCode,Elapsed
cat "$ROOT/SUMMARY.md"
```

`submitted_through_round: 2` means all visits were submitted, not that all work
completed. The summary counts should distinguish complete, failed and still
missing results. The controlled intervention demo remains separate from the
six-cell search results. Use `docs/REVIEW_DEADLINE_ANALYSIS_HANDOFF.md` to make
tables and graphs only after checking those status counts.
