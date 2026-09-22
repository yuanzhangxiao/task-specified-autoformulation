# Milestone 2: general pipeline integration

Protocol: `shared-process-integration-1`. This is a small integration pilot,
not a shared-process efficacy comparison or a final pipeline freeze.

The general route uses the existing variable → topology → interaction constructor,
whole-model revision compiler, training residual evidence, public predicate checks,
and frozen `collocation-single-target-v2` fitter. It does not import the basin
repair controller, reference equations, basin threshold/balance probes, or saved
basin replies. The basin experiments remain separate development diagnostics.

## What changes

1. Select variables once. Preserve the inventory and public memory assignments.
2. Ask for optional shared processes: one named law, drivers, consumers, declared
   signs, and any known conversions. Empty or invalid optional suggestions do not
   stop ordinary construction. A failed admitted process route can fall back using
   the same pre-process inventory and the same construction call/token budget.
3. Reuse existing topology checks and identified interaction-function delivery.
   Compile each law once and reference it in every declared consumer. Existing
   sign normalization, explicit dependency/conversion decisions, and atomic
   repairs remain active.
4. Assemble gains from declarations, before looking at fit scores. With complete
   explicit conversions, transfers use one common nonnegative amplitude;
   influences use separate amplitudes. If any conversion is unknown, use effective
   independent consumer gains for that process. This does not certify conservation
   and does not compare or select gain policies by validation performance.
5. Fit with the existing fixed budget. Give the proposer the full current model,
   equation-derived public findings, training residual evidence, and numerical
   status. Validation values are not included in this revision prompt.
6. Apply one bounded whole-model revision through the established compiler. An
   edit to a named law propagates to all references. Coordinated consumer, state,
   and initializer changes are allowed; complete causal boundaries are required.
   Every retry sees the actual retained model, and the compiled trial is saved.
   There is no additional LLM confirmation gate.
7. Refit using compatible parent parameter values and select the incumbent using
   the existing validation/complexity rule. Failed or no-change revisions reuse
   the visit's existing fit allowance; they do not add a new allowance.

`full` and `brief_only` retain their existing initial-evidence difference. Both
receive fitted training residual feedback during revision. The optional process
phase is proposer construction, not the later calibrated scientific critic.

Runtime compilation enforces declared sharing and checks grammar, closure, causal
initialization, algebraic acyclicity, and public requirements. It does not infer
new scientific laws or unit conversions. Opposite signs and shared names alone
are not physical certification. Basin-specific `same_as` repair actions are not
added to the general reply schema: the existing named laws and shared parameter
identities remain the common mechanism.

## Bounded development matrix

| Family | Public cell | Variants | Seeds |
| --- | --- | --- | --- |
| Dalla Man | canonical obfuscated | full, brief_only | 0 |
| CSTR | canonical named mechanism | full, brief_only | 0 |
| Alien device | canonical functional mechanism | full, brief_only | 0 |

Six fresh constructions (round 0), followed by one revision per lineage (round 1):
at most twelve campaign fits. GPT-OSS-20B low reasoning and the existing serving,
per-call, construction, revision and fitting budgets are unchanged. No automatic
round 1 submission or further expansion. No critic, pruning, new optimization
method, test evaluation, or independent BDF/Radau comparison is included.

The first gate is deterministic integration: law identity and propagation,
initialization, payload separation, sealed resume, and old-protocol compatibility.
The live gate is that each planned visit is accounted for, failures remain visible,
and empty/invalid optional processes do not create an additional mandatory gate.
Model fits and accepted revisions may still fail; excellent NMSE is not required
to establish integration. Inspect causes before deciding whether Milestone 2 can
close. A scientific benefit claim needs a later controlled comparison.

## Run on ACES

Use a git archive of the reported commit in group scratch, with `SOURCE_COMMIT`
containing its full hash. This avoids creating another complete Git checkout.
The final task response supplies the pinned archive command.

```bash
export AF_REPO_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/repos/shared-integration-COMMIT
export AF_PUBLIC_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/shared-process-integration-v1
bash "$AF_REPO_ROOT/scripts/hpc/submit_shared_process_integration_aces.sh" 0
```

The launcher loads the existing Python module, checks the public manifest/prompt/
train/validation files, freezes them, and submits prepare → proposer → six CPU
fit tasks → report. Prepare runs compatibility tests and the offline real-fit
smoke. The proposer uses one H100; each fit uses one CPU. Test CSVs are not copied.
Do not modify an existing frozen root or delete scheduler intents to resubmit.
Completed submissions return their existing receipts. Interrupted physical calls
resume through existing caches, and consumed fit attempts do not silently rerun.

After round 0 has finished, inspect results and submit the one revision:

```bash
cat "$AF_OUTPUT_ROOT/INTEGRATION.md"
bash "$AF_REPO_ROOT/scripts/hpc/submit_shared_process_integration_aces.sh" 1
```

Round 1 refuses submission until every round-0 predecessor has a sealed result.
The launcher defaults to the paths above if only `AF_REPO_ROOT` is set.

```bash
cat "$AF_OUTPUT_ROOT/INTEGRATION.md"
jq '{status_counts, planned_round_rows, rows: [.rows[] |
  {task_id, round, status, proposal_status, construction_fallback,
   trial_validation_nmse, retained_validation_nmse,
   all_graph_requirements_certified, retained_complexity,
   cumulative_requests, cumulative_tokens}]}' "$AF_OUTPUT_ROOT/integration_summary.json"
```

`EQUATIONS.md` separates trial and retained equations. `integration_summary.json`
also records actual law consumers, transitive affected equations, revision
propagation, and both models' public predicate findings. Full bounded LLM usage,
raw attempts, initialization, numerical fit diagnostics, and training residual
packets remain in the ordinary per-visit artifacts. Missing values stay missing.

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_shared_process_integration.py tests/test_process_gain_comparison.py tests/test_shared_process_pilot.py tests/test_shared_process_pilot_submission.py tests/test_review_revision_v6.py
PYTHONPATH=src .venv/bin/python scripts/smoke_shared_process_integration.py
bash scripts/run_tests.sh full
.venv/bin/ruff check .
```

The smoke uses a prescribed transport and isolated synthetic fixture with real
fitting; it makes no live provider calls. No physical-call/model-recovery claim is
made from mocked responses. Historical parents have different total budgets.
