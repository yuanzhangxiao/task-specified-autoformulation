# Public-context sign repair of the Dalla rescue models

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
