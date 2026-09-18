# Recover round 14 after the ACES file quota failure

The original v5 campaign is pinned to
`dfc6f81a613e186ddffdd0b2406feca583a58944`. Its round-14 proposer completed,
but eight fitting workers could not publish results. The logs explicitly report
`Errno 122: Disk quota exceeded`. The finish and dispatch jobs also failed, so
round 15 has not run. This is an execution interruption, not eight demonstrations
of scientific or numerical failure.

This recovery is separate from the frozen fitter. Three standalone files run
with the original checkout and Python; no checkout changes, new clone, fitter
changes, collocation reruns, or extra optimization allocations are needed.

## What is retained

| Array index | Task | Saved point to replay |
| ---: | --- | --- |
| 26 | cell03_seed1_brief_only | polling best; cost 206.98658791355572 |
| 33 | cell04_seed1_full | primal-screen best; cost 363.91799617691277 |
| 34 | cell04_seed1_no_latent | sensitivity best; cost 384.4932775822431 |
| 36 | cell05_seed0_brief_only | None; retain round-13 incumbent |
| 38 | cell05_seed0_no_spec | None; retain round-13 incumbent |
| 39 | cell05_seed0_refit_only | primal-screen best; cost 4407.241097080257 |
| 41 | cell05_seed1_brief_only | None; retain round-13 incumbent |
| 43 | cell05_seed1_refit_only | sensitivity best; cost 4486.001377894568 |

These are the supplied diagnostic values, not new measurements. The script
discovers all supported `best_evaluated.json` files and freezes the one with the
lowest saved full-training cost for each task before replay. It verifies the
original request, data, parent, parameter names and domains. An initializer-only
point is insufficient: its collocation objective is not a rollout training cost.

Each selected vector gets one production Radau replay on training, then validation
if the training cost agrees (`rel_tol=1e-5`, `abs_tol=1e-8`). These checks use the
existing integration tolerances and training normalization. Each split has a
300-second cap; an outer process cap is 720 seconds. They optimize neither
parameters nor initial conditions. The same training predictions provide the
residual feedback packet. Validation never selects a different saved checkpoint.
The original validation/complexity rule chooses between the recovered trial and
the incumbent.

Successful replay establishes finite production scores, not optimizer convergence
or scientific correctness. Native stopping status, original residual-call count,
and original budget exhaustion are unavailable and remain null. Feedback marks
the numerical outcome unresolved. A missing point, interrupted replay, failed
integration or cost disagreement retains the incumbent and an explicit
infrastructure-interrupted visit. No second replay allocation is granted on resume.

The recovery copies public development assets, imports, proposal call caches, and
result/checkpoint files to group scratch. It hashes every copied source file,
preserves completed visits byte for byte, and leaves the original campaign alone.
It does not copy test data or evaluation artifacts. Proposal token/call accounting
survives the copy; recovery replay seconds and zero optimizer calls are recorded
separately. The eight interrupted attempts remain part of the experiment history.

## Submit the recovery on ACES

Replace `REPLACE_WITH_REPORTED_COMMIT` with the implementation commit. Use the
existing authenticated clone to fetch just the three new scripts. This does not
change the original pinned checkout. All new result/log/cache files below go to
group scratch; the old Python, container, and downloaded model remain in place.

```bash
(
  set -euo pipefail
  AF_RECOVERY_REV=REPLACE_WITH_REPORTED_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-review-v5-dfc6f81
  export AF_PYTHON="$AF_BASE/.venv/bin/python"
  export AF_SOURCE_ROOT=/scratch/user/u.yx126462/phase_b/review-continuation-v5
  export AF_RECOVERY_ROOT="$AF_GROUP/review-v5-quota-recovery-v1"
  export AF_RECOVERY_CODE="$AF_GROUP/quota-code-${AF_RECOVERY_REV:0:7}"
  mkdir -p "$AF_RECOVERY_CODE"
  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  for source in scripts/recover_review_quota.py \
    scripts/hpc/run_review_quota_recovery_aces.sh \
    scripts/hpc/submit_review_quota_recovery_aces.sh; do
    dest="$AF_RECOVERY_CODE/${source##*/}"
    git -C "$AF_BASE" show "$AF_RECOVERY_REV:$source" > "$dest.tmp"
    if [ -e "$dest" ]; then
      cmp "$dest" "$dest.tmp"
      rm "$dest.tmp"
    else
      mv "$dest.tmp" "$dest"
    fi
  done
  bash "$AF_RECOVERY_CODE/submit_review_quota_recovery_aces.sh"
)
```

This checks that the four old round-14 job IDs have no active jobs, snapshots the
campaign, and submits an eight-task CPU array plus a finalizer. Five tasks perform
replay; three record that no saved point is available. No GPU is requested.
Submission receipts live in `quota_submission`. An uncertain or repeated
submission stops for inspection instead of duplicating jobs. Keep those receipts.

After the jobs finish:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/review-v5-quota-recovery-v1
cat "$ROOT/quota_recovery_summary.json"
jq '{status_counts, proposal_status_counts}' "$ROOT/summary.json"
```

The snapshot's summary will still have 44 missing **round-15** rows: that round
will live in the separate continuation below. A missing recovery summary means
finalization did not finish; inspect `quota_logs` before continuing. If a replay
worker itself was killed, rerunning that same worker consumes the saved start
marker and records interruption, rather than granting another replay budget.

## Run the already-planned round 15

Run only after `quota_recovery_summary.json` exists. This imports the completed
round-14 selections and authorizes exactly one remaining visit for all 44 tasks.
It neither reruns rounds 13–14 nor extends the experiment beyond round 15.

```bash
(
  set -euo pipefail
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-review-v5-dfc6f81
  export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
  export AF_SOURCE_ROOT="$AF_GROUP/review-v5-quota-recovery-v1"
  test -f "$AF_SOURCE_ROOT/quota_recovery_summary.json"
  export AF_OUTPUT_ROOT="$AF_GROUP/review-continuation-v5-round15"
  export AF_SOURCE_ROUND=14 AF_ADDITIONAL_VISITS=1
  export AF_CONTINUATION_PROTOCOL=review-deadline-5
  export AF_VLLM_IMAGE=/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif
  export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
  export AF_COMPUTE_CACHE_ROOT="$AF_GROUP/review-cache"
  export AF_IPC_TMP_ROOT=/scratch/group/p.nairr260351.000/af-ipc-yx
  export PYTHONDONTWRITEBYTECODE=1
  export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
  export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  bash "$AF_REPO_ROOT/scripts/hpc/submit_review_continuation_aces.sh"
)
```

Read the final round from the **new** root:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/review-continuation-v5-round15
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SUMMARY.md"
jq '{status_counts, proposal_status_counts}' "$ROOT/summary.json"
```

This phase has 88 rows: 44 imported round-14 selections plus 44 round-15 visits.
The import ledger links to the recovery snapshot, which retains all earlier
results, costs, interrupted checkpoints and recovery receipts. Do not add these
import rows again when aggregating experiment counts.

## Verification and limits

`tests/test_review_quota_recovery.py` checks real fixed-vector replay, lineage and
domain validation, saved-cost agreement before validation, best-point selection,
source immutability, no-point retention, consumed replay attempts, publication
idempotence, and compatibility with the ordinary continuation importer.
`python -m scripts.smoke_review_quota_recovery` additionally runs the real worker
subprocess end to end using isolated synthetic data.

This deliberately supports the supplied failure mode: a started sibling fit
without its final result/backend publication. A source with an existing final fit,
a changed runtime, different missing tasks, started later rounds, or sealed test
evaluation fails closed. Incomplete recovery cannot unlock continuation. This
does not reconstruct optimizer calls or parameters lost before checkpointing.
