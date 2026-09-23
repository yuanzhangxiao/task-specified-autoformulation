# Run the pending sign recheck on Delta

This is an execution alternative for the five pending corrected judge reviews,
not another pipeline milestone or a new scientific protocol. Export the already
prepared ACES `judge-sign-recheck-v1/plan.json`. It contains the symbolic requests,
public context, original findings and provenance needed for the recheck. No
trajectory files, fitting outputs, model weights or full source campaign need to
be transferred.

The original source receipts were verified when ACES prepared this plan. Delta
verifies the transported plan's self-seal, request hashes, exact judge protocol,
model revision, affected-evidence classification and placement linkage. It keeps
the origin plan unchanged beside its own separately sealed plan. It does not
pretend to reopen the original ACES paths or reverify unavailable ancestor files.
Self-seals check artifact consistency, not external authentication.

## Avoid duplicate work

At the time of this handoff, ACES prepare job 2157181 has completed, review job
2157182 is pending, and report job 2157183 is waiting. The plan is a prepared input,
not a live status snapshot. This handoff does not import completed corrected
reviews or coordinate schedulers across clusters. Hold the ACES review while
trying Delta; do not launch this handoff if that review has already started.

On ACES:

```bash
(
  set -euo pipefail
  af_state=$(squeue -h -j 2157182 -o '%T')
  [[ "$af_state" == PENDING ]] || {
    echo "ACES review is no longer pending; inspect it before switching."
    exit 1
  }
  scontrol hold 2157182
  [[ "$(squeue -h -j 2157182 -o '%T')" == PENDING ]] || {
    echo "ACES started during the handoff; do not submit a duplicate Delta review."
    exit 1
  }
  squeue -j 2157182,2157183 -o '%.18i %.25j %.10T %R'
  cp /scratch/group/p.nairr260351.000/u.yx126462/judge-sign-recheck-v1/plan.json \
     /scratch/group/p.nairr260351.000/u.yx126462/judge-sign-aces-plan.json
)
```

Download `judge-sign-aces-plan.json` using the ACES file portal, then upload it
using the Delta file portal to `/work/hdd/bibo/yxiao2/phase_b/`. This avoids the
previous jump-host authentication problem. Keep this file separate from output.
The ACES report remains pending behind the held review. To return to ACES, first
ensure the Delta review is cancelled or finished, then `scontrol release 2157182`.
When Delta finishes successfully, the held ACES review/report may be cancelled
without deleting their prepared plan or historical source campaign.

## Submit on Delta

Use an immutable archive of the supplied commit under
`/projects/bibo/yxiao2/repos/sign-delta-<commit-prefix>` and set:

```bash
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_VLLM_IMAGE=/projects/bibo/yxiao2/containers/vllm-openai-v0.27.1.sif
export AF_HF_HOME=/projects/bibo/yxiao2/huggingface-cache
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/judge-sign-recheck-delta-v1
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_judge_sign_delta.py" \
  --source-plan /work/hdd/bibo/yxiao2/phase_b/judge-sign-aces-plan.json \
  --root "$AF_OUTPUT_ROOT"
```

The submitter schedules:

1. CPU preparation: four CPUs, 16 GB, one hour; offline tests/smoke, source pin,
   local runtime and SIF hash, vLLM 0.27.1 check, cache the exact model revision.
2. One `gpuA40x4` job: four A40 GPUs, 16 CPUs, 128 GB, three hours. It runs the
   affected unique pairs only, preserving both stages/orientations, prompts,
   seeds, request limits and interruption semantics. No GPU job is queued for
   an unaffected-only plan.
3. A CPU report after the review terminates. A pending report usually reflects
   its dependency, not a separate resource shortage.

Accounts default to `bibo-delta-cpu` and `bibo-delta-gpu`; overrides are
`AF_CPU_ACCOUNT` and `AF_GPU_ACCOUNT`. No ACES modules are loaded on Delta.
There is no unpinned container download. CPU preparation uses the existing SIF,
checks its vLLM version, and records its hash; the review refuses changed bytes.
The original ACES image digest remains recorded separately. Different SIF bytes
are allowed for this explicit platform handoff and exposed as `same_image_bytes`.
Four A40 GPUs use tensor parallelism 4, rather than the ACES two-H100 configuration.
Runtime/image/hardware differences mean bitwise-identical model answers are not
assumed. This is not a speed comparison or renewed scientific calibration.

Submission is idempotent. Repeating the submit command returns existing job IDs;
it does not resubmit failures. Partial/uncertain scheduler replies retain intents
and stop for inspection. Completed reviews are reused by the run CLI, while an
interrupted started review keeps partial logs and cannot obtain a fresh budget
through resume. There is no automatic follow-up, fitting, model/selection change
or test access. GPU availability and actual serving must still be checked live.

## Inspect on Delta

```bash
ROOT=/work/hdd/bibo/yxiao2/phase_b/judge-sign-recheck-delta-v1
cat "$ROOT/submission_manifest.json"
af_jobs=$(jq -r '[.jobs[]] | join(",")' "$ROOT/submission_manifest.json")
squeue -j "$af_jobs" -o '%.18i %.25j %.10T %R'
sacct -j "$af_jobs" --format=JobID,JobName%25,State,ExitCode,Elapsed
```

After preparation the report exists but can still show pending reviews. After
the report job finishes:

```bash
cat "$ROOT/SUMMARY.md"
jq '{unique_reviews, affected_unique_reviews, status_counts, cost, execution,
     model_changes, selection_changes}' "$ROOT/summary.json"
```

The expected input remains 12 unique reviews: five affected and seven unchanged.
The five affected statuses should no longer be `pending` after a successful run;
inspect `unavailable`, `interrupted` or `indeterminate` outcomes separately.
For failures, inspect `logs/prepare-*.err`, `logs/review-*.err` and
`runtime/server-*.log`. Do not remove markers or submission intents to force reruns.

Local checks cover portable input validation, exact request reuse, completed and
interrupted resume, partial cost accounting, scheduler idempotency and the no-GPU
case. Run `scripts/smoke_judge_sign_delta.py` for the offline mocked workflow.

Verification on 2026-09-23: 40 focused sign-correction/recheck/Delta tests passed,
as did the offline smoke, changed-file Ruff, Python 3.11 syntax parsing and shell
syntax validation. Repository-wide `ruff check .` reports 37 existing findings in
unrelated `analysis/claude` files. No cluster session was opened during this work.
