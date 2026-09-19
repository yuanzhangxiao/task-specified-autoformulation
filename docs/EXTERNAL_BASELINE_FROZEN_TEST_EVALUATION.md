# External-baseline frozen test evaluation

Contract for opening sealed Phase-B test data on the four existing external
baselines: SINDy, PySR, GPT-5.6 Sol, and native Phase-B D3.

**Status: `frozen_before_test_or_private_evaluation`** (finalized 2026-09-19
after advisor review). Steps 1-3 still open nothing. Step 4 opens sealed test
trajectories and the private reference, and now runs, because the plan is
finalized. Autoformalism is not modified, no candidate is generated, no
parameter is refit, and no model is selected using a test result. The frozen
plan is `configs/external_baseline_frozen_test_evaluation_v1.json`.

Nothing in the chain can open test data without passing the prepare stage's
`--authorize-execution` freeze, its re-hash, and the replay stage's
execution-record preflight.

## 1. Scope and roster states

| Method | Source kind | Planned | Evaluator |
|---|---|---:|---|
| SINDy | `sindy` | 120 | ready |
| PySR | `pysr` | 120 | ready |
| GPT-5.6 Sol | `raw_data_agent` | 120 | ready, reuses the existing freeze |
| Native Phase-B D3 | `d3` | 120 | **unsupported**, see gap G1 |

40 registered cells x 3 repetitions x 4 methods = 480 planned identities.
Persistence is excluded: it is a causal previous-observation predictive
reference, not an autonomous mechanistic model.

Every planned identity produces exactly one roster row recording three
independent facts:

- **source availability** — `available` or `missing`;
- **evaluator readiness** — `ready` or `evaluator_unsupported`;
- **adaptation outcome** — `adapted`, `failed`, `missing`, or
  `evaluator_unsupported`.

`evaluator_unsupported` is *not* a claim of scientific non-applicability. It
says only that no frozen evaluator can score the artifact under its declared
execution semantics. Applicability is decided per endpoint against that
endpoint's definition; an absent required mechanism, for example, is an
evaluated structural failure rather than grounds for exclusion. None of
`missing`, `failed`, or `evaluator_unsupported` is a zero score, and none is
evidence of compliance.

A row that is both missing and unsupported reports `missing` as its outcome:
availability is checked first. The manifest's `evaluator_unsupported_count`
therefore counts rows by evaluator readiness and overlaps the missing count;
`adapter_request_count` plus `withheld_outcome_count` is the figure that sums
to the planned total.

## 2. Saved-model provenance

### SINDy and PySR

Source: the **development readiness freeze**, one selected
`BaselineDevelopmentResult` per task at
`<development-root>/common-readiness-freeze/tasks/task_NNN.json`
(schema `phase-b-baseline-development-result-1`), with identity resolved
through `<...>/inputs/task_plan.jsonl`. Coefficients are fitted on training
data only; validation selects the SINDy threshold and the PySR expression.
Coefficients are embedded in the equation strings, so the adapter reports
`parameterization: not_required`. Neither method reads the proposer prompt.

### GPT-5.6 Sol

120 provider-fitted models composed by
`configs/phase_b_raw_agent_deterministic_evaluation_v1.json`: 90
unchanged-prompt runs from `raw-data-agent-fitted-v1` and 30 prompt-v3 refresh
runs from `raw-data-agent-fitted-prompt-v3-refresh-v1`. Parameters are exactly
as returned; `parameter_refit_applied` is `false`. The `structure_only` variant
is a separate labeled ablation and is excluded.

Sol is imported through its own freeze rather than re-resolved here. The import
verifies the manifest's no-test/no-refit flags, re-hashes its
`source_adapter_requests.jsonl` against the recorded digest, and requires its
roster to cover the exact planned cross-product.

### Native Phase-B D3

The campaign writes, per task index from `<campaign-root>/plan.json`:

- `results/<index>/native-selection.json` — sealed, holds the selected
  `BaselineDevelopmentResult` under `"selection"`;
- `results/<index>/result.json` — sealed evaluation and accounting wrapper,
  **not** a development result and carrying no `schema_version`;
- `results/<index>/d3_checkpoint.json` — written conditionally, never assumed.

Both sealed files are required for availability. Historical Phase-A D3 models
are a separate cohort on different cells and cannot populate a Phase-B cell.

### Prompt provenance

All methods evaluate against the prompt-v3 overlay root
(`phase_b/inputs/public-prompt-v3`). Sol's composition is hash-bound to it.
SINDy and PySR are prompt-blind, so a prompt revision cannot change their
models; their prompt hash is provenance only. Native D3 is prompt-sensitive and
its campaign `plan.json` freezes the hashes it used — verify those match the
overlay before adapting.

## 3. Execution semantics

| Method | Test protocol | Update rule |
|---|---|---|
| SINDy, PySR, Sol | unseen-condition free rollout (`solve_ivp`) | continuous derivative |
| Native D3 | discrete recursive rollout | `x_next = x + f`, **no dt multiplier** |

Replaying native D3 through the ODE evaluator reinterprets its increment as a
rate. That is an ODE reinterpretation diagnostic, not native D3 performance,
and must never be reported as a D3 test score. Routing is decided by declared
execution semantics, not by a method-name prefix.

Common ODE settings: target standard deviations fitted once on the public
training split, frozen parameters and initializers, no trajectory-specific
initial-state fitting, no measured-target resets after the initial sample,
300 s per-trajectory wall-clock limit. A partial rollout is a failure.

The classical chain's separate **predictive** endpoint uses causal one-step
observed-state resetting. Those NMSE columns are not interchangeable with
free-rollout NMSE and must never be pooled.

## 4. Bypassing the train+validation refit

`scripts/finalize_phase_b_public_baseline_models.py` refits the selected SINDy
threshold on train plus validation
(`src/autoformalism/rebuttal/baseline_postfreeze.py:188`, protocol
`selected_threshold_refit_on_train_plus_validation`) and writes adapter
requests pointing at those refit models. This contract bypasses that chain and
sources the development freeze instead.

`reject_refit_derived_source` fails closed on any
`phase-b-frozen-baseline-model-1` artifact or any source declaring the refit
protocol, so the bypass cannot be undone by pointing the manifest at the wrong
directory.

Scope note: fitting on train plus validation does **not** invalidate an
independent test estimate in general, and this guard is not a judgement on that
workflow. It is scoped to this campaign, where three specific problems apply —
the refit model's validation score is in-sample, its inherited
`development_validation_normalized_mse` describes the train-only equations
rather than the refit ones, and including it would break the chosen train-only
equal-data contract. The refit finalization remains valid as a separately
labeled analysis with matched data access.

## 5. Missing sources

Missing sources stay in the denominator as explicit outcomes; they are never
retried to success or silently dropped, and they reach final assembly through
`withheld_source_outcomes.jsonl`.

Development-stage counts from the 2026-09-16 saved-baseline validation replay
(`artifacts/baseline-validation-review-2026-09-16/`) are indicative only — they
come from a validation replay, not an inventory of the frozen development
results:

- SINDy 102 / 120 complete, 18 unavailable;
- PySR 102 / 120 complete, 18 unavailable;
- Sol 119 / 120, with one rollout failure on
  `phase_b_anonymous_system_task_canonical_opaque_easy` repetition 0;
- native D3: no Phase-B matrix completion is **recorded locally**. Whether the
  campaign ran is unverified from this checkout; cluster receipts settle it.

The preparation command produces the authoritative inventory.

## 6. Metric definitions

Endpoints are reported separately. **No weighted overall score is defined.**

1. **Runtime validity** — recomputed by the evaluator; a method cannot supply
   its own flag.
2. **Public mechanism compliance** — conjunctive over applicable public graph
   predicates. This is graph compliance, not a scientific rubric. Report failed
   predicates, uncertainty, and non-applicability as distinct states.
3. **Sealed target NMSE** — aggregate and per-target, protocol per section 3,
   reported only when every requested trajectory integrates.
4. **Mechanism recovery** and 5. **conditional hidden NMSE** — response-subspace
   contract, training-only alignment, scored once on test, withheld unless the
   candidate is structurally recovered.
6. **Intervention behavior** — per sealed test trajectory.
7. **Complexity and reliability** — deterministic counts and failure totals.

Endpoint 8 (qualitative LLM assessment) is **not requested**; no LLM is called.

**Reporting.** The shared assembler computes
`source_failed_count = len(outcomes) - adapted_count`, which reads a missing
artifact and an unsupported evaluator as method failures, and its metrics
contain only adapted subjects. This campaign therefore reports through
`scripts/summarize_external_baseline_evaluation.py`, which joins the master
roster by explicit identity and keeps `evaluated`, `adaptation_failed`,
`missing` and `evaluator_unsupported` distinct, with per-method and
per-method-and-cell counts against the full expected totals. NMSE summaries are
labeled conditional on success. No mean is pooled across methods. The finalize
stage prints that report as the primary result; the shared assembler summary is
retained but labeled legacy, because it pools methods and counts missing and
unsupported rows as source failures.

## 7. Aggregation rules

1. Freeze the method/cell/repetition roster before test access. Missing,
   invalid, unsupported, and failing rows stay in the denominator.
2. Report per-cell **median over successful repetitions**, labeled as
   conditional on success, with completed/expected alongside. Full-roster
   availability and failure rates are reported separately. No zero-imputation,
   no silent omission of identities.
3. **Retain all 40 conditions as the primary reporting roster for every
   method.** Named/obfuscated and functional/opaque cells share numeric data
   (`paired_numeric_data_across_semantic_variants: true`), so paired cells are
   **dependent, not independent evidence** — that is a caveat on inference, not
   a reason to drop conditions. Prompt-blind methods (SINDy, PySR) cannot
   differ across a semantic pair; prompt-sensitive methods (Sol, D3) can, and
   the semantic-variant contrast is itself a result worth keeping.
4. If 20-design aggregates are also reported, group semantic variants with
   **identical weighting for every method**. Never compare a 20-design
   weighting for classical methods against a 40-cell weighting for the others.
5. Never pool free-rollout NMSE with the causal one-step predictive endpoint.
6. Any "common completed subset" comparison is conditional and must be labeled,
   with excluded cells named.
7. No test or intervention score may choose which cached source to include.

**Deferred diagnostic.** Whether paired-variant SINDy/PySR models are literally
identical is an artifact-level question; matching medians at six significant
figures do not establish it, and PySR is stochastic. The check (compare
equations or their SHA-256 across paired cells in the development freeze) is
cheap and offline but needs a benchmark pairing map built from
`phase_b_public_spec`, which requires a data root. It is deferred so it cannot
delay the evaluation path.

## 8. Commands

Steps 1-2 are offline audits. Step 3 builds the inventory and opens nothing.
Step 4 runs the whole sealed chain and requires a finalized plan.

```bash
repo=/projects/bibo/$USER/repos/autoformalism-v21
python_bin=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
```

**Step 1 — Sol tool-budget audit (no API call).** The first root is the
structure-only pilot; the last two are the fitted-model primary.

```bash
for r in raw-data-agent-pilot-v1 raw-data-agent-fitted-v1 \
         raw-data-agent-fitted-prompt-v3-refresh-v1; do
  echo "== $r"
  "$python_bin" "$repo/scripts/audit_raw_data_agent_budget.py" \
    --root "/work/hdd/bibo/$USER/phase_b/$r"
done
```

**Step 2 — locate any 12-vs-24 comparison and check its contract (no API call).**

```bash
b24=/work/hdd/bibo/$USER/phase_b/raw-data-agent-budget24-v1
if [ -d "$b24" ]; then
  "$python_bin" "$repo/scripts/audit_raw_data_agent_budget.py" --root "$b24"
  # A run launched through the pilot slurm defaults to structure_only.
  # run_raw_data_agent.py writes each run directly under its output root.
  jq -r '.output_contract // "unrecorded"' "$b24"/*/run_config.json | sort | uniq -c
else
  echo "no 24-call sensitivity run exists at $b24"
fi
```

**Step 3 — inventory only (opens nothing, works on a draft plan).**

```bash
"$python_bin" "$repo/scripts/prepare_external_baseline_frozen_test_evaluation.py" \
  --config "$repo/configs/external_baseline_frozen_test_evaluation_v1.json" \
  --symbolic-development-freeze \
    /work/hdd/bibo/$USER/phase_b/public-baselines-full-v1/common-readiness-freeze \
  --d3-campaign-root /work/hdd/bibo/$USER/phase_b/d3-native-full-v1 \
  --hidden-audit \
    /work/hdd/bibo/$USER/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json \
  --raw-agent-freeze-manifest \
    /work/hdd/bibo/$USER/phase_b/raw-agent-deterministic-evaluation-v1/frozen/raw_agent_freeze_manifest.json \
  --output-root /work/hdd/bibo/$USER/phase_b/external-baseline-inventory-v1/frozen

jq '{expected_source_count, available_source_count, missing_source_count,
     evaluator_unsupported_count, adapter_request_count, counts_by_method}' \
  /work/hdd/bibo/$USER/phase_b/external-baseline-inventory-v1/frozen/external_baseline_freeze.json
```

A missing D3 campaign root is reported as 120 missing rows rather than an
error, and does not block the other three methods. Omitting
`--raw-agent-freeze-manifest` keeps Sol in the roster as 120 missing rows — it
never drops the method — but execution then refuses, because an authorized
freeze requires the upstream Sol freeze. Prepare it first with
`scripts/hpc/phase_b_raw_agent_eval_prepare.slurm`.

**Step 4 — the sealed chain (opens test and private data).** The plan is
finalized, so this runs. Do not start it until step 3's inventory has been
reviewed.

```bash
cd "$repo"
export AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/external-baseline-evaluation-v1
bash scripts/hpc/submit_external_baseline_evaluation_delta.sh
```

Stages, each `afterok` on the last: prepare (freeze, re-hash, adapt, seal the
execution record) -> 24-shard sealed replay -> merge -> 24-shard hidden
subspace -> finalize (merge, combine outcomes, assemble, roster report).

Every stage exports `PYTHONPATH=<checkout>/src` so imports resolve from the
pinned checkout, then re-computes
`scripts/hpc/external_baseline_inputs_digest.sh` and refuses to start if it
changed after submission. That digest is a stable code identity covering this
chain's launchers, Python entry points, the rebuttal modules they call
directly, and the frozen plan. It is deliberately neither a transitive closure
of every imported module nor a repository-wide scan. HEAD equality alone would
miss staged, untracked, or post-submission edits; unrelated local artifacts do
not block a dedicated evaluation checkout.

The prepare stage seals an `execution_record.json` binding the authorized
freeze digests, the adapted subject and outcome hashes, the public-data root
and identity, the evaluator commit, the chain-input digest, and the trajectory
wall-time limit. Public-data identity reuses the loader's own train/validation
split fingerprints and proposer-prompt hash for each planned cell
(`baseline_validation.load_public`), so the numerical trajectories are covered
rather than only JSON and text; test data is never loaded.

The replay stage re-verifies every one of those bound settings through
`--verify-execution-record` before opening test data, so a changed public root,
time limit, authorization manifest, adaptation output, or code identity fails
the preflight rather than silently running a different contract. Verification
is done in Python against resolved paths: a companion digest checked with
`sha256sum -c` from another working directory resolves the recorded basename
against that directory instead and fails spuriously.

Resume without discarding completed CPU work:

```bash
AF_RESUME_FROM=postfreeze AF_POSTFREEZE_ARRAY=3,7,11 \
  bash scripts/hpc/submit_external_baseline_evaluation_delta.sh
```

Each submission is appended to `submission_ledger.jsonl` as it happens, so an
interrupted chain stays recoverable. A fresh full submission into a root that
already holds a receipt is refused; resuming is not. A resumed stage must match
the chain digest **sealed in the original execution record**, not one
recomputed at resume time — the same commit can hold different local edits.

## 9. Implementation gaps

**G1 — native D3 frozen-test integration. Blocking for D3 only.** Native
validation rollout is implemented; the frozen test path is not. Two parts:

- *Campaign reader.* `_adapt_d3`
  (`final_evaluation_adapters.py:271-281`) expects the historical layout —
  `result.json` validating as a development result with a sibling
  `d3_checkpoint.json`. The campaign writes the selection into sealed
  `native-selection.json`. A reader keyed on the plan row index, verifying the
  seal and plan association, is required.
- *Discrete test protocol.* `evaluate_phase_b_postfreeze.py` performs
  `solve_ivp` free rollout only, and
  `TargetPredictionEndpoint.evaluation_protocol` admits only
  `legacy_unspecified` and `unseen_condition_free_rollout`.
  `d3_rollout.predict(..., teacher_forced=False)` already implements the native
  recursion, but `evaluate_validation` restricts splits to TRAIN/VALIDATION and
  that restriction is deliberate and should stay. A separate frozen-test entry
  point is required, preserving the selected model and parameters,
  training-derived normalization, native increments, the public
  auxiliary-input contract, and no observed-target resets after
  initialization. Any one-step diagnostic stays separately labeled.

**G2 — private endpoints for native D3.** `phase_b_hidden_subspace.py:635`
raises on any protocol other than free rollout and `:247` pins `solve_ivp`.
Even with G1 done, hidden recovery, conditional hidden NMSE and intervention
behavior must return `evaluator_unsupported` with a reason rather than raising.
This does not block native target test NMSE for available D3 models.

**G3 — adapter-level refit guard.** `_adapt_symbolic_baseline` still accepts
`phase-b-frozen-baseline-model-1`. The freeze blocks it, but a hand-written
request file passed straight to `export_phase_b_frozen_subjects.py` would not
be caught. Recommend the same guard inside the adapter or an explicit export
policy flag.

**G4 — D3 campaign completion unverified locally.** Only pilot roots appear in
this checkout. Step 3 reports 120 missing D3 rows if the full campaign has not
run. No new discovery is requested.

**G5 — Sol is budget-conditioned.** See section 10.

**G6 — paired-variant equality check deferred.** See section 7.

**G7 — shared assembler summary remains coarse.** `_write_outputs` in
`assemble_phase_b_final_evaluation.py` still reports one `source_failed_count`.
This campaign reads its results from the external roster report instead. A
shared-layer fix would benefit every method but needs its own authorization.

## 10. Sol budget audit

The primary full matrix and the prompt-v3 refresh both use
`max_tool_calls: 12`, with 30,000 output tokens, a 20-minute request limit and
two provider attempts. Every Sol result here is therefore **budget-conditioned
at 12 tool calls** and must be described that way. Twelve is the historical
primary operating point; its existence is not an argument that it is
sufficient. Report actual calls, termination reasons, and available token
accounting, distinguishing per-request limits from total experiment usage
across retries. Unknown accounting stays unknown.

`scripts/audit_raw_data_agent_budget.py` reads cached Responses API objects
offline. Terminal statuses are `completed`, `incomplete` and `failed`;
`in_progress` and `interpreting` are nonterminal and counted separately. The
run-level `tool_call_count` in the earliest six pilot artifacts includes one
retained `interpreting` record; the offline audit is authoritative.

**12-versus-24: no completed comparison is recorded locally.** What exists is
the frozen plan `configs/raw_data_agent_budget_sensitivity_v1.json`
(`status: frozen_before_calls`; two cells, three repetitions;
`max_tool_calls` 12 -> 24 as the single changed field) and a documented
submission to `/work/hdd/bibo/$USER/phase_b/raw-data-agent-budget24-v1`
(`docs/ACCESS_NAIRR_OPERATIONAL_SETUP.md:1711`). No summary or artifact
references that root in this repository.

**Contract caveat.** `scripts/run_raw_data_agent.py:48-52` defaults
`--output-contract` to `structure_only`, and
`scripts/hpc/phase_b_raw_data_agent_pilot.slurm` — the launcher in the
documented budget24 command — does not pass the flag. The fitted-model
launchers pass `--output-contract fitted_model` explicitly. A budget24 run
submitted as documented is therefore most likely structure-only: a valid
matched comparison against the **structure-only** pilot, but not a sensitivity
of the primary fitted-model baseline. Check each saved run's recorded contract
and refit provenance before comparing anything.

No further paid budget work is requested at this stage. Any later sensitivity
study is a separate decision taken on the audit's evidence; if one is
commissioned after test results are seen, disclose that timing and report it
separately rather than replacing the frozen primary comparison.

## 11. Authorization boundary

Steps 1-3 open no test or private data. Step 4 opens sealed test trajectories
and the private reference, and runs only from a finalized plan. Once test data
are opened for a frozen subject, no parameter, threshold, source selection, or
aggregation rule in this document may be changed for that subject. A later D3
increment gets its own frozen receipt and identified report composition; it
must not overwrite an earlier receipt or change its evaluated subjects.
