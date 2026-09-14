# Pre-fitting milestones

This prospective plan starts from repair commit `9013e84`. Existing ACES
comparison-3 runs and Astra's separate fitter work retain their pinned code.
No benchmark data or finalized public prompts change.

## Current modeling contract

Public task requirements, allowed channels, and executable expression rules are
authoritative. Confirmed deterministic failures block execution. Unresolved
static domains remain risks, not proof of reachable failure. Scientific concerns
are advisory. Numerical evidence describes actual evaluated points and budgets;
finite replay and local convergence do not establish scientific recovery or
global optimality. Failure to find a fit does not prove structural infeasibility.

For the open-loop repair pilot, observed states use the permitted initial
measurement and then propagate. Generated variables may drive other equations;
later measured targets may not. Validation/test prediction receives initial
public observations and permitted inputs/auxiliaries, not later measured targets.
Benchmark-specific availability rules still govern every channel.

Unknown latent boundaries may be a shared training-fitted value or a function of
permitted initial observations, initial input values and fixed covariates. Map
parameters are fitted once on training and frozen; the evaluated initial value
can differ across trajectories. Trajectory IDs and future target values are
never initializer features. Future inputs may drive dynamics when publicly
supplied, but are not silently interpreted as preparation information. Identical
available initial information gives the same deterministic latent initialization.
Independent fitted training-trajectory initials would be a separate diagnostic,
not a deployable validation initialization rule.

The current functional model admits fitted internal shape/scale parameters. It
is not restricted to a fixed basis with only fitted linear outer coefficients.

## 1. Evidence strength and repair priorities

Add an opt-in `repair-evidence-priority-1` policy. Its ordered support levels are
unresolved, observed, corroborated, and certified. These are an explicit routing
hypothesis, not calibrated probabilities or a new scientific score.

- Confirmed runtime contract failures take precedence.
- Scientific absolute concerns have corroborated support only when the exact
  criterion, subject, evidence, request and strict paired consensus match.
  Identical cached requests in a self-pair count as one observed review, not
  additional corroboration.
  Orientation agreement is not independent scientific validation. Comparative
  preference has narrower scope and receives observed support.
- Finite selected-point production training replay has observed support. Verified
  local optimizer convergence at the same selected parameters adds corroboration.
- State-only success with augmented integration failure supports numerical
  difficulty at evaluated points. Timeout or budget exhaustion alone remains
  unresolved model evidence. Search opportunity is separately reported, including
  missing values. Collocation status does not establish latent-boundary coverage.
- Equal support retains joint focus. Stale-candidate findings remain in the audit
  but cannot determine current-model priorities. Validation metrics do not set
  these evidence levels; the existing validation selection remains unchanged.

The existing `legacy_category` control stays available. A new freeze using
`--routing-policy evidence_strength` is comparison-4. Its plan and report bind the
new policy and provider cache identity; changed roots cannot resume old plans.
The judge's scientific prompts, model, response schema, scoring, and retries are
unchanged. No fitter implementation is imported from Astra's private diagnostics.

Gate: routing regressions, full pytest, Ruff, real fitter/resume smoke, and
inspection of matched public live traces before claiming a search benefit.

Implementation status: complete; live benefit remains untested. Local verification
ran the full suite: 1,439 passed, 39 failures from absent benchmark fixtures, two
source-provenance guards that detected edits during the run, and three optional
Torch skips. Both provenance cases passed after the source was held fixed. Final
priority/feedback checks passed 39 tests, including the self-pair and new-policy
live-request/resume regressions; the earlier combined priority/comparison/feedback
run passed 91. Ruff, shell syntax, diff checks and the real CPU fitter/resume smoke
passed. No benchmark fixture was copied or changed to mask the missing files.

From this commit's clean checkout, using the project's Python environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_repair_priority.py tests/test_repair_comparison.py tests/test_repair_feedback_evidence.py tests/test_repair_drafts.py tests/test_repair_action_replay.py tests/test_atomic_occurrence_judge.py
PYTHONPATH=src python scripts/smoke_repair_feedback_comparison.py
PYTHONPATH=src python -m pytest -q
ruff check .
```

For a later matched ACES routing comparison, the existing launcher accepts
`AF_ROUTING_POLICY=evidence_strength` with a new `AF_OUTPUT_ROOT`; omit it for the
legacy control. Pin the same source candidates and fitter for both policies.
This implementation has not submitted a new job or changed the active v3 runs.

## 2. Trajectory-dependent causal initialization

Implementation: `CAUSAL_INITIALIZER_CONSTRUCTION.md` describes the new opt-in
constructor, canonical fitting handoff, checkpoints and verification commands.

Make existing causal maps an explicit pre-fitting construction choice, preserving
the runtime ownership of numeric guesses and parameter fitting. Test distinct
initials and held-out initial observations, no future-target/ID access, matching
training/validation boundary semantics, and deterministic resume. Use a controlled
nonconstant initializer so a shared constant cannot accidentally pass the test.
No arbitrary validation/test initial-state fitting is permitted.

## 3. Training evidence before construction

Implementation: `TRAINING_EVIDENCE_CONSTRUCTION.md` records the descriptive
packet, constructor integration, provenance and verification commands.

Build a deterministic training-only packet covering observed initial variability,
input changes and output trends/turning points, with exact trajectory/window
provenance and uncertainty. Do not infer hidden labels or make oscillation, lag,
polarity or decay claims stronger than the sampling and experiment support.
Allow the same packet into variable, topology and function construction.
Test train-only access, deterministic serialization, unavailable evidence,
constant/noisy/short trajectories and mixed input schedules.

## 4. Matched end-to-end experiment

Implementation: `PREFIT_MATCHED_CONSTRUCTION.md` documents the frozen twelve-task
initial-construction pilot (six matched pairs), separate construction/CPU-fit
workers, cost and failure accounting, resume, and runnable commands. Its sole
candidate makes first and best fit identical; post-fit repair remains a separate
follow-up. The current sensitivity transfer supports only `v01`, so both packet
arms use the same existing general bounded-rollout fitter for this two-cell pilot.

Compare initial construction with/without the packet, holding initializer policy,
proposer settings, budgets, scientific judge setting and fitter fixed. Freeze
source data, code, protocol and seeds before calls. Report completion, first-fit
and best predictive quality, structural changes, initialization choices, total
cost and all failures. Preserve checkpoint/resume and task-independent completion.
Do not combine a fitter change or priority-policy change with the packet ablation.
The initial six-pair public matrix is an engineering pilot, not a general winner.

Gate: offline mocked orchestration and real numerical smoke, then a user-run
bounded campaign; interpret its results before choosing subsequent architecture.
