# Whole-model requirement feedback: the next pre-fitting milestone

`prefit-requirement-feedback-1` tests whether runtime feedback repairs the missing
nonlinear feedback identified in the twelve saved constructions. It runs before
fitting and does not change the defaults of existing construction or fitting
protocols. The completed source campaign remains immutable.

The local repair/audit milestone established 12/12 construction completion and
preservation of valid sibling slots. The subsequent
[model review](PREFIT_MODEL_REVIEW_2026-09-14.md) found an entirely linear model
despite an explicit public nonlinear-feedback requirement. Its local certificate
passed because no individual term carried a nonlinearity obligation. This new
milestone binds that public obligation to the assembled model, independently of
how the proposer describes each term.

## Scope and scientific interpretation

The opt-in configuration contains reviewed bindings: cell, requirement ID,
`nonlinear_feedback` kind and the exact public requirement text. Freeze rejects
missing IDs or changed text. There is no general keyword-to-hard-rule inference
and no new requirement inferred from private reference equations.

The runtime checks whether nonlinear source dependence occurs on a feedback
path connected to the required input and output. Feedback can include a self
interaction, a cycle through several states or a generated algebraic process.
Input-only nonlinearity and disconnected nonlinear processes do not satisfy the
check. Obvious zero, identity-power and identical-expression cancellation tricks
are removed before examining syntax. General symbolic equivalence is not solved.

A successful revision establishes necessary syntax on a relevant path. It does
not certify the correct feedback mechanism, absence of all cancellations,
stability, global numerical feasibility, identifiability or predictive accuracy.
Unknown fitted parameter values can still make a contribution ineffective.

Other reviewed problems remain visible but advisory: repeated contributions,
saved relaxation/sign facts, identity terms, dimensional consistency and
state-versus-algebraic interpretations. This milestone does not automatically
rewrite glucose balances, require every rate to be algebraic, or require meal
storage when the public task does not ask for it.

## Repair contract and routing

For a missing bound feature, the runtime enumerates existing interactions on a
relevant feedback path. The proposer sees the exact public brief, complete
canonical model, diagnosis, eligible local replies and protected-slot list.
It returns exactly one interaction ID, an RHS expression, and parameter
names/roles. Equation type and LHS belong to the frozen topology. The proposer
does not write assignments such as `x = ...` in the expression field.

Every attempted revision:

1. Selects one eligible interaction; an unrelated ID or extra model-edit field
   is rejected.
2. Passes the existing restricted parser, source set, outer sign, parameter-role
   and local-obligation checks. Certified outer-gain normalization remains
   available and is recorded separately from the proposer reply.
3. Rebinds all functions, comparing every unselected canonical function exactly
   with its saved value. The public topology, states and observation mappings
   remain fixed.
4. Reuses the original causal initialization plan and guesses, recompiles it
   against the revised model, and checks unchanged initial-condition expressions.
   The new initializer artifact identifies the original initialization decision
   and the new base model; no initializer LLM call or parameter fit occurs.
5. Rechecks the whole-model requirement. A locally valid but still linear reply
   is rejected with actionable feedback. Invalid attempts leave the parent
   candidate unchanged. The first accepted revision becomes the episode result.

When the existing topology offers no eligible interaction, the result is
`topology_revision_required`, without an LLM call. A topology editor is outside
this bounded one-RHS experiment. Multiple missing requirements that cannot all
be addressed by one RHS will not be silently declared fixed.

Already-covered models and models without a bound requirement are retained
without a proposer call. Preservation is a runtime property; it is not evidence
that a proposer learned to avoid unnecessary changes, nor that all other
scientific requirements are met.

### Parameter-role normalization

The eleven same-name changes in the source review all changed `coefficient` to
`nonnegative_coefficient`, with a recorded direct-outer-gain certificate. For
example, in a fixed subtractive slot containing `k*x`, the assembled contribution
is `-k*x`. Allowing a negative `k` would reverse the selected outer sign. The
runtime restricts that outer gain to the nonnegative domain while keeping the
parameter name and RHS text. This is an intentional domain correction under the
sign contract, not a new functional form or fitted numerical value. Arbitrary
internal parameters are not normalized this way.

## Frozen experiment

Configuration: `configs/prefit_requirement_feedback_v1.json`.

- Source root:
  `/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1`.
- Source plan:
  `c722b0a2de2736150b2f82db93973126a63a480b4c8b82f15dbd76e33b334ab2`.
- Twelve source models; each must have a sealed passing audit. Preparation
  rechecks the cached request ledger and reconstructs the canonical model and
  initialization using the current production compiler. Any difference fails
  preparation visibly. No training CSV, validation/test data or fit artifact
  is opened. Runtime source files are read to compute the code fingerprint.
- One reviewed nonlinear-feedback binding for the anonymous-system task. Six
  anonymous models have that binding; six Dalla Man models do not. The latter
  remain controls only for preservation, not for scientific correctness.
- Three new repair seeds, 0/1/2, for every source model, in two arms. Arm order is
  counterbalanced. This gives 72 episodes, with source generation seed and arm
  retained as provenance rather than treated as new scientific tasks.
- `local_only`: retain the already locally valid model. Global gaps remain
  visible in reporting; this arm makes no repair requests.
- `requirement_feedback`: repair only a diagnosed gap, preserving controls and
  unrelated slots. There are at most three physical requests and 262,144 charged
  tokens per episode, including uncertain calls and failures.
- Same GPT-OSS-20B revision, low reasoning, temperature 0.2 and 8,192 output-token
  limit as the source campaign. No scientific judge is called.

Inspection of the supplied complete equations identifies one source gap:
anonymous-system, training-evidence generation arm, source seed 1. Thus the
expected live work is three repair episodes, at most nine physical LLM requests.
ACES preparation verifies this using the original individual slots; its frozen
diagnostics are authoritative for the actual case count. Five anonymous controls
already contain relevant nonlinear syntax; six other controls have no bound rule.

This compares local-only acceptance with requirement-aware repair on the same
saved inputs. It is not another brief-only versus training-evidence ablation, a
matched comparison with an old full controller, or a calibrated scientific judge
evaluation. The single natural missing-feature case limits generalization.

The user's hypothesis that training evidence primarily helps function forms
requires a separate experiment with a shared frozen topology, initializer policy
and budgets. The original construction campaign supplied the packet at multiple
stages, so its different term counts cannot isolate a function-form effect. The
current feedback experiment does not replay the training packet to the proposer.

## ACES commands

Use the exact commit supplied in the completion message to make a new clean
checkout. The commands below are single physical lines; they include explicit
paths and never rely on a `python` command in the login shell or its current
working directory. The documentation checkout example uses the fetched branch
tip; the completion message supplies a literal commit instead of `FETCH_HEAD`.

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1 FETCH_HEAD
AF_RESUME=0 AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1 AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python AF_CONFIG=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1/configs/prefit_requirement_feedback_v1.json AF_SOURCE_ROOT=/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-requirements-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1/scripts/hpc/submit_prefit_requirements_aces.sh
cat /scratch/user/u.yx126462/phase_b/prefit-requirements-v1/submission_manifest.json
```

The launcher submits CPU reconstruction/preflight, then one H100 repair job
dependent on successful preparation. CPU preparation has one hour; GPU allocation
has two hours with a one-hour worker budget plus startup/drain allowance. Failed
preflight tests print the error tail and full log path. Temporary files and pytest
work are placed in scratch; pytest's home cache is disabled.

After preparation, inspect the initial diagnoses:

```bash
jq '[.[] | {source_task, diagnosis}]' /scratch/user/u.yx126462/phase_b/prefit-requirements-v1/runtime/selected-cases.json
```

After completion, rebuild/print the summary without another LLM call:

```bash
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1/scripts/prefit_requirement_campaign.py summary --root /scratch/user/u.yx126462/phase_b/prefit-requirements-v1
```

Inspect the actual attempted RHS changes and final remaining gaps:

```bash
jq 'select(.attempts | length > 0) | {episode: input_filename, stop_reason, attempts, final: {selected_interaction: .final.selected_interaction, diagnosis: .final.diagnosis, candidate: .final.candidate, protected_slots_preserved: .final.protected_slots_preserved}}' /scratch/user/u.yx126462/phase_b/prefit-requirements-v1/results/*/state.json
```

Expected reporting distinguishes `requirement_gap_remaining` from local validity,
`no_bound_requirement_gap_final` from scientific certification, and bound versus
unbound source models. Controls should show zero changed models and zero calls.
Successful feedback episodes should report one changed model, the preserved
sibling count and a named selected interaction. Failures retain the original
model and remain in the denominator; exhausted episodes are terminal.

For interrupted, nonterminal work, after prior jobs have stopped:

```bash
AF_RESUME=1 AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1 AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python AF_CONFIG=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1/configs/prefit_requirement_feedback_v1.json AF_SOURCE_ROOT=/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-requirements-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v1/scripts/hpc/submit_prefit_requirements_aces.sh
```

Resume checks the pinned code/config, successful preparation and prior scheduler
state. Ambiguous submissions leave an intent directory and cannot silently create
a duplicate job. Each worker locks its episode; requests, budgets, uncertain
responses and accepted/rejected attempts survive interruption. The copied frozen
models are sufficient for worker resume without reopening the historical source.

## Local verification and artifacts

The source module `search/requirement_feedback.py` owns diagnosis and atomic model
repair. `rebuttal/prefit_requirements.py` owns frozen-source reconstruction,
episode state, accounting and reports. The CLI, two ACES scripts, configuration,
synthetic smoke and three regression files complete the experiment. The only
shared execution change is an extra protocol route in the existing server script.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_requirement_feedback.py tests/test_prefit_requirements.py tests/test_prefit_requirements_submission.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_requirements.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_construction_audit.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Focused verification passed 41 tests. They cover local-keyword omission, affine
failure, relevant versus disconnected/input-only nonlinearity, generated algebraic
feedback, obvious fake nonlinear terms, source and schema violations, preserved
initializers, no-op exhaustion, rollback, deferred/uncertain calls, tampered cache,
source provenance, no dataset reads, scheduler idempotence and direct CLI use.
Both the new requirement smoke and the legacy construction-audit smoke passed.
The full suite passed 1,695 tests in 793.06 seconds; three optional Torch tests
were skipped because Torch is not installed locally. `ruff check .` and shell
syntax checks for the launchers passed.

The output root contains a sealed `plan.json` with copied model evidence,
`runtime/selected-cases.json`, `summary.json`, and per-episode `state.json`/`calls/`.
Accepted results include the revised canonical candidate, a recompiled reusable
initializer artifact, preservation facts and the remaining requirement diagnosis.
No generated experiment outputs or supplied model artifacts are committed.
