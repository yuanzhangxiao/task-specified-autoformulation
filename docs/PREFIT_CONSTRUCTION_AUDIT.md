# Fresh construction audit: Milestone 1 after local feedback experiments

`prefit-construction-audit-1` constructs complete models and stops before fitting.
Its configuration is `configs/prefit_construction_audit_v1.json`. This is the
first remaining integration milestone after the local repair experiments, rather
than a renumbering of the original milestones in `PREFIT_MILESTONES.md`.

The completed campaign constructed and audited all twelve models. The
[saved-model review](PREFIT_MODEL_REVIEW_2026-09-14.md) records what that pass
establishes, the remaining scientific concerns and the next diagnosis milestone.

## Question and frozen matrix

Does the current pre-fitting controller construct internally consistent complete
models, retain already-valid function slots during local repair, and emit causal
initialization policies that can be handed to a fitter later?

The matrix has twelve fresh constructions: two public cells, three proposer
seeds, and `brief_only` / `training_evidence` arms. It retains the earlier pilot's
GPT-OSS-20B revision, sampling, limits, budgets, causal initialization, equation
batching and bounded atomic repair. Each task has at most 128 physical requests,
524,288 charged tokens and three attempts per step, shared across stages and
resume. No previously generated models or provider replies are imported. The
older frozen public directory supplies public release assets only.

The latest local feedback experiment repaired all 24 cases in both arms. Structured
feedback needed 25 requests versus 27 for error text, and repaired 23 versus 21
cases on the first response. Valid controls were preserved with zero calls. The
remaining retries concentrated on a single nonlinear function case; those repeated
seeds are not independent scientific tasks. These results motivate examining fresh
complete models rather than claiming a general feedback advantage.

This new experiment compares training evidence presentations under the **current**
controller. It is not a causal before/after controller comparison. Historical
11/12 construction completion is descriptive context, not a matched control.
There is no fitting, scientific judge, post-fit repair or automatic winner.

## Data and numerical boundary

The new protocol requires `fit: null` and rejects both the worker `fit` command
and direct `fit_task` entry point before creating any numerical marker. The
legacy `prefit-matched-construction-1` protocol retains its existing fitting path.

Freeze copies only `manifest.json`, `proposer_prompt.txt` and `train.csv`. The
manifest retains the release's split fingerprints, but validation and test tables
are neither opened nor copied. `BenchmarkLoader.load_training` verifies and
loads the training split alone. Registry-only contexts and frozen public contracts
remain the source of proposal constraints. The training packet is identical to
the existing descriptive packet; only the evidence arm receives it.

No benchmark data, finalized prompts, fitting algorithms or scientific judge
contracts are changed. Constructor parameters and initializer coefficients remain
unfitted. Numeric values inside the initializer artifact are optimizer guesses,
not learned physical boundaries.

## Audit and artifacts

After construction, the CPU audit:

1. Reconstructs topology from the saved variable inventory and equation types.
2. Recomputes each slot's frozen sources, outer sign and nonlinear obligation.
3. Revalidates the original batch function and rebinds its accepted local reply
   using the restricted production compiler. A valid batch slot must have the
   same accepted reply after any certified deterministic role normalization and
   must not be marked as sent for atomic repair.
4. Checks the rebound canonical functions and functional draft against the saved
   ones, finalizes the base candidate, and recompiles the causal initializer plan.
5. Checks target coverage, initializer coverage, exact model identity and the
   existing deterministic completeness certificate.

The audit reuses the production validators; it is an independent reconstruction
of the saved artifact chain, not an independent implementation of the expression
language or a scientific judge. It records original and final RHS expressions,
local parameter-role changes (including same-name changes), certified role
repairs, canonical effective roles/domains and unresolved scientific review facts.
For example, a quadratic term can satisfy a syntactic nonlinearity obligation
without establishing saturation, scientific adequacy or numerical stability.
Identity mappings are exposed for inspection and are not automatically rejected.

Each task directory under `results/TASK_ID/` contains:

- `construction.json`: sealed terminal construction and existing stage traces.
- `calls/`: cached provider responses and charged request records.
- `audit.json`: sealed audit, input/cache digests, checks, per-slot repair facts,
  and a `handoff` object for a passing audit. That object contains the canonical
  candidate, context and complete causal initialization artifact.
- `model.md`: derived readable equations, declarations, initializers, parameter
  roles and audit details. It can be regenerated after interruption.

The root contains `summary.json` and a `review.md` index linking every task.
Summary denominators include all twelve planned tasks, with separate construction
completion, audit passes, audit failures, unconstructed models and pending audits.
Partial provider costs are retained. `status: complete` means every task has a
terminal audit disposition, not that every model passed. No NMSE is reported.
Local preservation counts concern function slots in completed reconstructed models;
they are not overall repair-success rates. Failed/partial construction retains
its original stage traces for diagnosis.

Frozen plans bind source, launchers, dependency versions, public assets, settings
and evidence. Cached calls remain charged across resume. An audit binds the sealed
construction and the entire cached-call ledger; missing or modified records are
rejected. Sealed completed artifacts cannot be overwritten. `report` can inspect
an older runtime read-only, whereas `verify`, `audit` and `summary` require the
frozen runtime. An interrupted deterministic audit may be recomputed; it performs
no provider call and consumes no fitting budget.

## ACES execution

Use a clean checkout from the pushed commit. Each command below is one physical
line. The user-facing completion message supplies the exact commit to pin; these
documentation commands select the current branch tip when the worktree is created.

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1 FETCH_HEAD
AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1 AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python AF_CONFIG=/scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/configs/prefit_construction_audit_v1.json AF_PUBLIC_ROOT=/scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1-e7ffd12/public AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/scripts/hpc/submit_prefit_construction_aces.sh
cat /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1/submission_manifest.json
squeue -u u.yx126462 -o '%.18i %.18j %.10T %.10M %.30R'
```

The launcher schedules CPU preparation (one hour), one H100 construction job
(four-hour allocation / three-hour worker), and CPU auditing (30-minute allocation).
There is **no fit job** in the submission manifest. Preparation checks the pinned
serving image, ensures the model is cached, runs the fitting-free regression set
and the new smoke test. Test output and temporary files use scratch; pytest's
cache is disabled to avoid the home-directory quota issue. Failed preflight output
is printed into the job error log. The audit depends on `afterany` construction,
so it can inspect finished models even when the GPU allocation was interrupted.

After the jobs finish:

```bash
jq '{protocol,status,construction_complete,arms,limitation}' /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1/summary.json
less /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1/review.md
jq -r '.rows[] | [.task_id,.construction_status,.audit_status,.model_report] | @tsv' /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1/summary.json
```

Open a returned `model_report` path with `less`; its sibling `audit.json` has the
machine-readable checks and canonical handoff. To rebuild the summary without
calling a proposer or fitter:

```bash
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/scripts/prefit_construction_campaign.py summary --root /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1
```

If all prior jobs have stopped and the summary is partial, resume using the same
checkout and output root:

```bash
AF_RESUME=1 AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1 AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python AF_CONFIG=/scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/configs/prefit_construction_audit_v1.json AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/scripts/hpc/submit_prefit_construction_aces.sh
```

Resume refuses active/uncertain jobs, skips construction if all tasks are terminal,
and otherwise continues pending stages with their original caches and budgets.
Repeated initial submission prints the existing manifest instead of submitting
again. An ambiguous `sbatch` outcome retains its submission intent and requires
inspection rather than risking duplicate allocations.

## Local verification and remaining milestones

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff check .
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_construction_audit.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_construction.py
bash -n scripts/hpc/submit_prefit_construction_aces.sh scripts/hpc/run_prefit_construction_aces.sh scripts/hpc/run_staged_topology_server.sh
bash scripts/hpc/run_staged_topology_server.sh --check-config configs/prefit_construction_audit_v1.json
```

The synthetic control constructs both evidence arms, recompiles both models,
preserves valid slots and resumes without further requests. It blocks fitter,
validation and test entry points and does not create held-out observation files.
Failure tests cover bounded construction failure, deferred cross-stage resume,
nonlinear and malformed-expression repair, role-change reporting, invalid saved
artifacts, changed/missing call records and scheduler idempotence. The old synthetic
fit smoke remains a regression check for the legacy protocol, not part of the
new ACES experiment.

Verification on 2026-09-14: the full repository suite passed 1,626 tests with
three optional Torch skips in 420.76 seconds. After the final atomic report-write
and deterministic rendering changes, all 56 focused construction/audit/scheduler
tests passed, as did Ruff and the repeated construction-only smoke. The legacy
fit smoke, shell syntax checks and new server-protocol dispatch check also passed.
These local runs use synthetic provider responses, not live proposer results.

After the ACES results, inspect the complete models and failure traces. Retain
the existing fitting settings under the agreed fitter freeze, validate the
interface against these sealed canonical artifacts, and run a small full
fit–diagnose–repair pilot before expanding the full pipeline. Scientific adequacy
and empirical improvement remain unestablished by this construction-only milestone.

## Recovery from the first ACES preflight failure

Preparation job 2131028 failed with four scheduler-test assertions; construction
job 2131029 was cancelled without running. Audit job 2131030 completed its report
step, which does not establish that any construction took place.

The cause was inherited test configuration. The live job exports `AF_CONFIG`
pointing to the construction-only audit configuration. The offline scheduler
fixture copied that environment into its subprocess, so tests of the legacy
default workflow selected `audit` while correctly asserting the expected legacy
`fit` stage. These are mocked scheduler calls; the failure did not involve fitting
or a live proposer.

Setting the same `AF_CONFIG` locally reproduced all four failures. The fixture
now removes inherited `AF_*` variables before applying its own settings and the
test's explicit overrides. Every scheduler case runs with both an absent config
selector and an exported audit selector. Assertions for both the legacy fit chain
and the construction-only audit chain remain intact. The exact ACES preflight
group passed all 82 cases under the exported audit configuration after this fix.
Only tests and documentation change; runtime code, launchers, scientific settings,
fitting settings, prompts and data stay fixed.

The full repository suite after the fix passed 1,638 tests with three optional
Torch skips in 443.36 seconds. Ruff and the construction-only smoke also passed;
the smoke was run with the audit configuration exported, as in ACES preparation.

Update the existing experiment checkout to the fix commit supplied in the
completion message. Then submit a fresh output root, retaining the failed run's
logs and reusing its frozen public inputs. This is a fresh submission because the regular `AF_RESUME=1` path requires
successful preparation and will correctly refuse job 2131028.

```bash
AF_RESUME=0 AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1 AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python AF_CONFIG=/scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/configs/prefit_construction_audit_v1.json AF_PUBLIC_ROOT=/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1/public AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-construction-audit-v1/scripts/hpc/submit_prefit_construction_aces.sh
cat /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1/submission_manifest.json
```

After the new audit job completes with exit code `0:0`:

```bash
jq '{protocol,status,construction_complete,arms,limitation}' /scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1/summary.json
```
