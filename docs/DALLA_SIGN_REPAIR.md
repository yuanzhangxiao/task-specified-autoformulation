# Public-context sign repair of the Dalla rescue models

## Current follow-up: directional review and advisory citations (v2)

`configs/dalla_sign_repair_v2.json` selects only `full_perturbed_r4` from the
**original rescue** packet. Use a new output root, `dalla-sign-repair-v2-r4`.
Do not use the v1 sign-repair `models.json` as input. Historical v1 results and
their pinned implementation retain the semantics documented below.

This follow-up responds to two separate v1 failures: an exact-quotation rule
discarded all R4 attempts, while exact but irrelevant quotations accompanied
questionable decisions elsewhere. Causal dependence alone does not determine a
positive coefficient. A literal quotation match cannot establish scientific
support. Nor does a correct outer sign prove that its inner law has the desired
effect throughout the state domain.

V2 asks for the mechanism role (source, sink, transfer, feedback or unknown),
directional rationale, and explicit donor/recipient when claiming a transfer.
Mechanical checks enforce the consistency of this claimed interpretation and
the permitted patch scope; they do not establish that the interpretation is true.
A separate cached call to the same GPT-OSS-20B model assesses each proposal using
the full public description and symbolic candidate. It returns supported,
contradicted, or insufficient, with an explanation. This is a fallible semantic
assessment, not the integrated scientific critic or independent certification.

Supported slots are retained while only unresolved slots are retried. An
unrestricted choice is accepted when direction is underdetermined; exhausted
slots preserve their original expressions and declarations. A supported partial
patch can proceed while explicitly recording unresolved slots. No usable
decision means no additional numerical fit. Every call, failure and unknown
usage event remains cached and charged across resume.

**Citations are advisory metadata.** Missing/null quotes become empty strings;
inaccurate or irrelevant quotations do not invalidate otherwise supported sign
decisions. `quote_exact` is only a string-presence check. `citation_credited`
requires both an exact public quotation and an explicit semantic assessment of
its relevance. It remains the assessor's judgment, not verified scientific
evidence. Neither a quote match nor `citation_credited` admits a patch by itself.
The runtime does not invent or replace quotations to obtain acceptance.

The public input contains no reference equation, fitted value, NMSE, trajectory
array, validation or intervention outcome. Named and anonymous channel meanings
remain as disclosed. No Dalla-specific coefficient direction is hard-coded.
Existing sign enforcement, matched rescue allocations and pruning are reused.

The pilot permits at most three proposal/assessment pairs (six physical LLM
requests in total), one H100 review allocation, and one CPU array task. That task
runs the repaired and unchanged-control arms sequentially, with at most six
rescue-profile fits total, plus starting-vector replays. This is demonstration
development, not an unbiased estimate of the general sign-review success rate.
Intervention performance remains to be evaluated after numerical endpoints are
frozen; the review itself cannot establish that a useful model will result.

### Launch v2 on ACES

Upload the supplied `dalla-sign-repair-COMMIT.tar.gz` archive to group scratch and
replace `COMMIT` below with its short commit identifier. These commands submit
jobs; no benchmark fit runs on the login node.

```bash
bash <<'BASH'
set -euo pipefail
AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
AF_ARCHIVE=dalla-sign-repair-COMMIT
mkdir -p "$AF_GROUP/repos"
tar -xzf "$AF_GROUP/$AF_ARCHIVE.tar.gz" -C "$AF_GROUP/repos"
export AF_REPO_ROOT="$AF_GROUP/repos/$AF_ARCHIVE"
export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
export AF_OUTPUT_ROOT="$AF_GROUP/dalla-sign-repair-v2-r4"
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
module load GCCcore/13.2.0 Python/3.11.5
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_dalla_sign_repair.py" \
  --inputs "$AF_GROUP/dalla-demonstration-rescue-v1/models.json" \
  --config "$AF_REPO_ROOT/configs/dalla_sign_repair_v2.json" \
  --root "$AF_OUTPUT_ROOT"
BASH
```

Inspect the review (once the GPU job finishes), then the fitted endpoints:

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-sign-repair-v2-r4
AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" -X --format=JobID,JobName%30,State,ExitCode,Elapsed
jq '{status,physical_requests,reply,attempts,direction_history}' \
  "$AF_ROOT/results/full_perturbed_r4/review.json"
jq '{status,expected,rows}' "$AF_ROOT/summary.json"
```

Download `models.json` after completion. `rows[].citation_audit` and
`rows[].unresolved_slots` explain the review limitations alongside the fitted
outcomes. Repeating the submission command reuses confirmed scheduler receipts;
an ambiguous scheduler reply still requires inspection/adoption, not deletion.

### Recover the rejected review dependency

The observed ACES preparation timeout still submitted job `2162020`, which
completed successfully. Recovery adopted it but the original launcher then
submitted `afterok:2162020`; ACES rejected that review with `Job dependency
problem`. A completed job can remain in accounting after leaving the live
dependency records. The earlier review `2161544` predates this preparation and
does not establish that the new review was submitted.

`scripts/recover_dalla_sign_submission.py` is a scheduler-only helper for this
case. Copy it **outside the pinned scientific checkout** and point `AF_REPO_ROOT`
to the original `dalla-sign-repair-dbb51ed` archive. It verifies the original plan,
source/runtime/launcher identities, owned preparation command, successful exit,
and explicit review rejection. It also checks for any matching original review
before issuing a replacement. It preserves every original receipt and writes
new attempt receipts in `submission-recovery-1/`.

The helper submits review without a dependency after confirming preparation
completed. Fit still requires successful review; report waits for fit termination.
Verified completed stages are omitted from later dependencies. An accepted job
whose reply timed out is recovered by matching the full command and owner in
accounting, including array jobs. If accounting has not caught up, the helper
stops and the same command can be rerun; it never resends an unconfirmed attempt.
An explicit dependency rejection can get a new attempt only after the verified
dependency condition changes. Old rejected receipts remain intact.

After uploading the helper to group scratch (use the actual uploaded filename):

```bash
bash <<'BASH'
set -euo pipefail
AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
export AF_REPO_ROOT="$AF_GROUP/repos/dalla-sign-repair-dbb51ed"
export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
module load GCCcore/13.2.0 Python/3.11.5
"$AF_PYTHON" "$AF_GROUP/recover_dalla_sign_submission.py" \
  --root "$AF_GROUP/dalla-sign-repair-v2-r4" --prepare-job 2162020
BASH
```

Successful recovery creates the normal `submission_manifest.json`, so the status
and result commands above still apply. No new scientific checkout, refreeze,
preparation run, extra model budget or changes to candidate/fitting code are needed.

Recovery verification on 2026-09-25: 61 focused recovery/sign tests passed, including
11 new scheduler recovery cases. A mock-scheduler smoke against the unchanged
`dbb51ed` upload archive confirms plan/receipt preservation and duplicate-free
resume. No live scheduler call, LLM request or benchmark fit ran locally. The
helper and tests pass Ruff; repository Ruff still has 37 unrelated analysis findings.
Accounting delays may still require rerunning the identical helper command; no
unconfirmed scheduler response is treated as permission to resubmit.

### Local v2 verification

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_directional_sign_review.py tests/test_sign_review.py \
  tests/test_dalla_sign_repair.py tests/test_dalla_rescue.py \
  tests/test_topology_owned_sign.py tests/test_process_pruning_campaign.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv/bin/python scripts/smoke_dalla_sign_repair.py --protocol dalla-sign-repair-2
.venv/bin/ruff check .
```

V2 verification on 2026-09-24: 117 focused tests passed, including absent/incorrect
citations, malformed assessments, supported-slot retention, interrupted calls,
budget accounting, paired fits and deterministic resume. The synthetic numerical
smoke passed with two mocked requests and no live LLM or benchmark data. All six
eligible gains of the real R4 input pass the freeze/verify/resume import path
without fitting. Changed Python files pass Ruff; repository-wide Ruff still
reports 37 unrelated findings in `analysis/claude/`. The full pytest suite was
not run for this follow-up.

## Historical v1 protocol and launch

Protocol: `dalla-sign-repair-1`. This is a separate development campaign, starting
from the retained endpoints in the original rescue `models.json`. It does not
overwrite the rescue campaign or change benchmark data, finalized prompts,
historical fit profiles, or current pipeline defaults.

The importer verifies each source result using the rescue writer's sealed-artifact
JSON digest (`staged_topology.content_hash`). Public-fit identities use a different
compact JSON serialization and cannot verify those seals. The original `7f517e2`
launcher mistakenly used the compact digest and stopped with `source result digest
differs` before creating a plan or submitting any job. The corrected importer
retains the original seals and rejects changed payloads; do not rewrite source
checksums or bypass verification. Use the corrected source archive and repeat the
submission command with the same input and output paths. Existing submission
receipts, if any, must still be preserved and checked.

The original constructions behind R4/R13 and R9 left their direct gains real.
The proposer chose unrestricted topology signs, so the fitter could reverse
source/sink interpretations while respecting every numerical bound. This
campaign asks the proposer to review those choices explicitly, then enforces
the repaired contract during fitting and pruning.

## Repair scope and evidence

Only isolated, single-use, unbounded real outer gains in state-equation terms
are eligible. Shared gains, bounded declarations, parameters used in initial
conditions/constraints, internal thresholds, denominators, process definitions
and existing nonnegative/positive parameters are preserved. Restricted AST
parsing identifies the eligible gains; no proposed expression is executed.

The proposer sees the original public scientific description, public channel
roles, symbolic candidate and eligible terms. It sees no fitted parameter vector,
NMSE, trajectory arrays, private reference or intervention outcome. In particular,
anonymous channels keep their anonymous meanings. The review is a sign-only
patch, not a model-generation stage.

Each eligible term requires exactly one decision:

- `positive`: assemble `+a*f`, with a nonnegative fitted magnitude a.
- `negative`: assemble `-a*f`, with a nonnegative fitted magnitude a.
- `unrestricted`: preserve the original expression and real domain exactly,
  with a reason such as a signed deviation, net flux or insufficient public
  information. The runtime does not turn uncertainty into a fixed sign.

The normalized term removes only literal outer minus factors; internal
subtractions and nonlinear expressions remain intact. Decisions include a basis
and explanation. A public-task citation must match an exact excerpt of the
displayed scientific context. Structural explanations remain proposer claims,
not certified scientific findings. Fixed outer signs do not prove global
monotonicity, state positivity or a particular dynamic intervention response.

The pinned proposer is GPT-OSS-20B on one H100, using the existing local vLLM
transport. Each model has at most three physical attempts; every attempt,
failure, uncertainty, token charge and reply is cached. Malformed decisions
receive deterministic feedback within that allocation. Interrupted inflight
requests consume their attempt; resume never silently resends them. The review
checks complete slot coverage, exact public quotes, compilation and public
admission before committing a patch.

For the uploaded six endpoints, four models have eligible gains (6, 6, 5 and 5
respectively). The two hard endpoints have none and are reported without new LLM
or fitting calls. A review that keeps every gain unrestricted is reported as
`unchanged`; exhausted attempts remain failures, with the original model retained
in the report. Neither is labeled a successful repair.

## Fitting and comparison

Each changed model receives two separate arms:

1. The repaired request, with newly constrained gains initialized using the
   fitter's ordinary role-based starts. Their old signed values and old guesses
   are not converted with absolute value. Compatible unrelated parameters and
   all initializer declarations/values are retained.
2. An unchanged-model control starting from the original rescue endpoint.

Both use the existing `collocation-rescue-v1` allocation and the existing rescue
selection/pruning controller. Each gets one starting-vector replay, one refit,
and at most two equal-allocation pruning/control fits. Numerical estimation and
pruning ranking use training data. Existing validation-based pruning selection
is retained. The unrestricted control never replaces the repaired arm and is
never relabeled as sign compliant. This is equal additional fitting allocation,
not identical warm starts or a proof that any observed improvement is caused
only by sign repair.

The selected repaired model is checked for exact surviving outer operators,
unchanged inner laws, nonnegative gain declarations and fitted bound compliance.
Removed gains are explicitly reported as pruned. Numerical failures remain
visible. Original, pre-pruning, repaired, pruned and unchanged-control endpoints
are available for subsequent frozen-parameter intervention inspection.

Maximum work for the uploaded packet: 12 LLM requests, eight starting-vector
replays and 24 rescue-profile fits. One GPU job reviews all models, followed by
a CPU array of six source rows (four potentially active, three concurrent).
Each active CPU row performs its two arms sequentially with one CPU/16 GB and
a 4 h 30 min scheduler cap. The one-H100 review cap is 1 h 15 min. No remote job
is submitted by the local coding agent.

## Launch on ACES

Use the supplied source archive to avoid Git authentication and personal-scratch
quota problems. Upload it to group scratch, extract into `repos`, and set the
directory below to the actual archive directory name.

```bash
(
  set -euo pipefail
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/dalla-sign-repair-COMMIT"
  export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
  export AF_OUTPUT_ROOT="$AF_GROUP/dalla-sign-repair-v1"
  export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  module load GCCcore/13.2.0 Python/3.11.5
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_dalla_sign_repair.py" \
    --inputs "$AF_GROUP/dalla-demonstration-rescue-v1/models.json" \
    --root "$AF_OUTPUT_ROOT"
)
```

The launcher selects an existing group/personal Hugging Face cache containing
the pinned GPT-OSS-20B revision. `AF_HF_HOME` can explicitly select another cache.
The default serving image is the existing
`/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif`; `AF_VLLM_IMAGE` can
override its path, but the frozen checksum must match. Compilation caches go
to group scratch, with short per-job IPC paths under `/tmp`.

Preparation runs the targeted tests and a synthetic end-to-end smoke before
the GPU allocation. It verifies the image checksum and locally cached model
revision. It does not download a model or run a Dalla fit on the login node.
Every Slurm stage has durable intent/reply/ID receipts. Repeating the exact
submission command reuses confirmed jobs. An ambiguous reply requires scheduler
inspection and `--adopt STAGE=JOB_ID`; adoption verifies the job against the saved
command. Never delete submission receipts to force another allocation.

## Inspect results

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-sign-repair-v1
AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" -X --format=JobID,JobName%30,State,ExitCode,Elapsed
jq '{status,expected,rows}' "$AF_ROOT/summary.json"

jq '{task,status,physical_requests,observed_tokens,reply,attempts}' \
  "$AF_ROOT"/results/*/review.json
```

The GPU worker writes a partial report before CPU fitting; the final CPU report
updates it after all fitting tasks terminate. `complete` means all rows are
terminal; individual rows can still be unchanged, exhausted or numerically
failed. Missing reviews/fits remain pending. If necessary, regenerate a report
with the pinned checkout/environment:

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/dalla_sign_repair.py" report --root "$AF_ROOT"
```

Download `models.json` for fitted equation and intervention review. Logs are in
`logs/` and `runtime/`; LLM records are in `results/TASK/calls/`; paired fit and
pruning checkpoints are in `fitting/TASK/results/{repaired,unchanged_control}/`.
Reports retain failed proposals and original endpoints. They do not claim a
successful physiological demonstration until the frozen intervention curves
have actually been examined.

## Local verification

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_sign_review.py tests/test_dalla_sign_repair.py \
  tests/test_dalla_rescue.py tests/test_topology_owned_sign.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv/bin/python scripts/smoke_dalla_sign_repair.py
.venv/bin/ruff check .
```

The smoke uses a prescribed synthetic reply and toy trajectories, performs both
numerical arms, checks retained sign integrity, and verifies exact resume with no
extra request or fit. No benchmark refitting is performed on the user's laptop.

Verification on 2026-09-24: 94 focused sign/rescue tests passed, including cached
request interruption, exact scope, quote/coverage failures, matched allocations,
pruning integrity and scheduler adoption. The real synthetic smoke achieved
training NMSE 5.42e-17 and validation NMSE 6.98e-18 with exact resume. Both sign
choices compiled and produced compatible bounded starts for all 22 eligible
gains in the uploaded real models; this adapter check selected no scientific
sign and performed no benchmark fit. Changed Python files pass Ruff. Repository
Ruff retains 37 pre-existing findings under `analysis/claude/`. A broader pytest
run was stopped after 423 passes and five dependency skips; the full suite was
not completed.

Digest-fix verification: 81 sign/rescue/topology tests passed, including producer-
sealed input, exact resume, modified fitted parameters, corrupt seals and rejection
of the wrong compact digest. All six downloaded rescue endpoints pass the real
freeze/verify/resume path without source changes, LLM calls or benchmark fitting.
The synthetic paired-fit smoke also passes. Changed files pass Ruff; repository
Ruff retains the same 37 unrelated findings under `analysis/claude/`.
