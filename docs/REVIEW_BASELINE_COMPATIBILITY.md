# Baseline compatibility for the corrected review campaign

This is a public-code and configuration audit, not a report of new baseline
scores. Remote completion counts still require inspection of the corresponding
development artifacts. No test data or private reference was opened for this
audit. Use this document alongside `REVIEW_DEADLINE_ANALYSIS_HANDOFF.md`.

The validation-only inventory, frozen replay and CPU submission workflow is now
implemented in `BASELINE_VALIDATION_REPLAY.md`. It also reuses historical D3 on
its original cells, with an explicit separate cohort and original training
checkpoint parameters. This does not establish remote artifact availability.

## Findings

1. All six review cells appear in the full classical Phase-B configuration.
   The full GPT-5.6 Sol protocol also plans the Phase-B matrix. Planning a cell
   is not evidence that its result completed.
2. The named canonical and perturbed Dalla Man T1 cells require the GPT prompt-v3
   refresh results. Their pinned refreshed prompt hashes match the classical
   prompt-v3 hashes. The other four cells use the unchanged-prompt source after
   its exact prompt hash is verified.
3. Classical development scores and the separate classical predictive-test
   reports use causal one-step observed-state resetting. Current review scores
   use open trajectory rollouts. Existing NMSE columns are not interchangeable.
4. The common frozen-subject evaluator already supports free rollout with no
   observed-target resets, frozen parameters/initialization, training-derived
   normalization, and common integration settings.
5. Historical SINDy finalization refits its selected threshold on train plus
   validation. That artifact must not be used to report held-out validation
   accuracy or silently compared with a train-only model under a claim of
   identical fitting-data access.
6. The symbolic adapter now accepts `BaselineDevelopmentResult` directly.
   It preserves the selected SINDy/PySR equations and hashes their original
   source, without a new fit or any copied test metric. This closes the
   train-only artifact handoff gap; it does not certify remote provenance.

## Six-cell source plan

| Review cell | Classical planned source | GPT-5.6 Sol required source |
| --- | --- | --- |
| Named canonical Dalla Man T1 easy | Full prompt-v3 classical matrix | Prompt-v3 refresh |
| Obfuscated canonical Dalla Man T1 easy | Full prompt-v3 classical matrix | Original full matrix, hash-checked unchanged prompt |
| Named perturbed Dalla Man T1 easy | Full prompt-v3 classical matrix | Prompt-v3 refresh |
| Obfuscated perturbed Dalla Man T1 easy | Full prompt-v3 classical matrix | Original full matrix, hash-checked unchanged prompt |
| Named CSTR easy | Full prompt-v3 classical matrix | Original full matrix, hash-checked unchanged prompt |
| Functional alien device easy | Full prompt-v3 classical matrix | Original full matrix, hash-checked unchanged prompt |

Authoritative plan/configuration files:

- `configs/review_deadline_v1.json`
- `configs/phase_b_public_baseline_full_delta_cpu_v1.json`
- `configs/raw_data_agent_fitted_model_full_v1.json`
- `configs/raw_data_agent_fitted_model_prompt_v3_refresh_v1.json`

The first two named cells both use public prompt hash
`b8ec7ca58d32da32652894397ca708b8058d533826d8be6dc994096c5fda92d4`.
Different dynamics still require separate dataset/cell identities; matching
prompt hashes alone does not establish matching data.

## What to reuse and what to reevaluate

| Method | Reusable scientific artifact | Required common evaluation | Important qualification |
| --- | --- | --- | --- |
| SINDy | Original `development_complete` equations and selected threshold | Open rollout of those exact equations | Native selection used one-step validation; report that policy |
| PySR | Original `development_complete` equations | Open rollout of those exact equations | Preserve the native selected expression, not a new test-selected expression |
| GPT-5.6 Sol | Candidate, returned parameter vector, causal initialization, and run identity | Open rollout with exact returned values | Primary result has no Autoformalism refit; refresh changed prompts |
| D3 | Selected checkpoint candidate and full parameter vector for matching Phase-B cells | Common ODE rollout, labeled as such, alongside native semantics if reported | Historical Phase-A cells are not substitutes; native fitting/selection differs |
| Persistence | Original simple predictive reference | Separate clearly labeled endpoint | Previous-sample prediction has ongoing target access; it is not an open-rollout ODE |
| Current Autoformalism | Frozen selected candidate and all learned initializer parameters | Same common ODE rollout | Conditional on supplied auxiliary paths where the prompt permits them |

Do not use test or intervention scores to choose which cached baseline model to
include. Freeze the cell/repetition roster before evaluation and keep missing,
invalid, or numerically failing runs in the denominator. A new baseline call is
needed only when the required matching scientific artifact is absent or has the
wrong public information contract. Re-evaluating compatible saved equations
does not require rerunning symbolic search or making another hosted-agent call.

For the deadline, prioritize GPT-5.6 Sol, SINDy, and PySR artifacts on the six
cells. Add D3 only when matching Phase-B artifacts and its checkpoint provenance
are confirmed. Old D3 hard-tier tables remain historical context. Do not block
all matched baselines on a fresh GPU discovery campaign.

## Comparison contract

Before using a source, verify and record:

- exact cell, tier, repetition, public prompt hash, and train/validation data
  fingerprints;
- target, auxiliary, forcing, and initial-observation permissions;
- model source hash, selected equations, fitted constants, and all causal
  initializer coefficients;
- whether parameters used training only or train plus validation;
- native selection criterion, number of proposals/tool calls, CPU/GPU/provider
  usage, and any prior held-out evaluation;
- independently frozen evaluation release, solver/tolerances, normalization,
  sample inclusion, failure policy, and target-reset policy.

Use the same public inputs and auxiliaries for the common rollout. Initial
observed states come from the permitted initial observations. Hidden boundaries
use the model's frozen shared value or causal map; do not fit them on validation
or test trajectories. Internal states may differ across methods; the primary
behavioral comparison scores the public output trajectories.

Primary deadline comparison: retain each method's train-fitted, validation-selected
model and perform no additional fitting during common evaluation. The original
SINDy development result is the right source for that comparison. The historical
train-plus-validation SINDy finalization remains available as a separately
labeled analysis, with compatible data access for any methods compared against it.

Common evaluation does not imply equal training/search algorithms, identical
compute, or equal information processing. GPT-5.6 Sol receives raw development
files and uses a hosted tool budget; the current proposer receives bounded
summaries and uses a local model. Report these differences. Do not manufacture
a compute-matched claim or treat hosted repetition IDs as matching stochastic
seeds with open-weight runs.

## Implementation paths

- `src/autoformalism/rebuttal/final_evaluation_adapters.py` converts compatible
  saved artifacts into content-addressed `FrozenEvaluationSubject` objects.
  SINDy/PySR now accept the original development-result schema as well as the
  historical schemas. Test-contaminated or incomplete development records are
  rejected by the typed schema.
- `scripts/export_phase_b_frozen_subjects.py` performs requested source exports;
  every expected cell/repetition should appear in the prespecified request
  ledger, including missing sources.
- `src/autoformalism/rebuttal/postfreeze_evaluation.py` performs common replay
  with `reset_observed_states=False`, frozen parameters, and training-only
  normalization. It opens the test split only after a selection is frozen.
- `src/autoformalism/rebuttal/review_deadline_reporting.py` binds the review
  campaign's subject hashes and evaluator settings before using that endpoint.
- `scripts/assemble_phase_b_final_evaluation.py` reports runtime validity,
  public graph compliance, behavioral endpoints, complexity, and failures
  separately.

Do not pass a raw development-result JSON through the historical classical
finalizer merely to make its schema acceptable: that would refit SINDy. Use
the newly supported direct adapter instead.

## Remote audit still required

The launchers document these Delta roots, which must be checked rather than
assumed to contain completed results:

```text
/work/hdd/bibo/yxiao2/phase_b/public-baselines-full-v1
/work/hdd/bibo/yxiao2/phase_b/raw-data-agent-fitted-v1
/work/hdd/bibo/yxiao2/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1
```

Inspect development status and source ledgers first. Record `available`,
`missing`, `failed`, or `incompatible` for every planned source. A full matrix
configuration or a submission record is not a completed experiment.
