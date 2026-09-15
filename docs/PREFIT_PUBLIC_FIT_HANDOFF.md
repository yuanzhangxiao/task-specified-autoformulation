# Saved requirement repair to public fitting

This milestone joins an accepted pre-fitting repair to the
[frozen public fitter interface](PUBLIC_FITTING_HANDOFF.md). It does not change
numerical algorithms or perform a post-fit proposer revision.

The declared source in `configs/prefit_public_fit_handoff_v1.json` is requirement
repair seed 0 of the previously identified anonymous-system gap, from requirement
plan `6f426e6893035be751a8f00b2f01dd491b7099947c99e506f154737879f5b6cd`.
It was selected before fitting. All three repair seeds supplied the same quadratic
RHS; this milestone fits only the declared seed. It is not another repair-rate
comparison or a search over candidates.

## What is checked

The exporter verifies the pinned construction and requirement plan seals, original
construction audit, complete provider-call ledger, raw visible responses, ordered
attempts and accepted final candidate. It reconstructs the accepted repair under
the current restricted compiler without executing or resuming the historical
campaign. A changed acceptance, candidate, initializer, public asset or source
identity prevents the handoff.

Training assets and the public manifest/prompt must match the original construction
hashes. Validation is loaded through the development-only benchmark loader and
verified against that manifest. No test CSV, derivative labels or private equations
are loaded. The frozen original brief remains authoritative; current local prompt
files do not replace the historical task specification.

The request contains the base candidate, original context and causal initialization
plan. Initializer guesses come from that plan. After preparation, the lowered
candidate must equal the accepted repaired candidate exactly. Initializer
coefficients remain training-fitted and frozen for validation.

## Failure feedback and provenance

New requirement campaigns can explicitly set `feedback_policy: stage-aware-1`.
Their retry feedback distinguishes provider truncation/unavailable final JSON
from an evaluated function-contract violation. It requests concise complete JSON
within the existing budget; it does not enlarge token or request limits. Legacy
configurations retain their historical behavior.

New stage-aware results place earlier construction flags under
`selected_function.construction_provenance`. The separate
`requirement_repair_provenance` compares the repaired RHS against its immediate
parent. Old recorded artifacts are never rewritten: the exporter derives this
stage-specific report and retains historical feedback alongside the corrected
attempt diagnosis. Parameter declaration comparisons are exact structure, including
rebound names; they are not claims that scientific parameter roles changed.

## One bounded CPU job

The default ACES worker requests one CPU, 16 GB and a 30-minute scheduler limit.
It runs focused regressions, checks CasADi and numerical library versions, runs a
synthetic handoff smoke, then prepares and executes the selected public fit. It
uses `collocation-feasible-v1`: the existing 120-second initializer and 180-second
refinement profile, unchanged evaluation/probe limits and numerical settings.
Setup and scoring add time outside those cooperative optimizer deadlines.

There are no GPU or live LLM calls. ACES runtime versions are recorded; wall times
are not a controlled comparison with Delta. The Dalla Man targets `Gp`, `I`, `U`
remain outside this anonymous-only pilot and need a separate C+S adapter milestone.

Use a clean checkout pinned to the published milestone commit. Set `AF_REPO_ROOT`
and `AF_OUTPUT_ROOT`, then run `scripts/hpc/submit_prefit_fit_handoff_aces.sh`.
Defaults refer to the existing ACES requirement-v2, construction-audit-v1-fix1,
public release and Python paths. Override `AF_SOURCE_ROOT`, `AF_CONSTRUCTION_ROOT`,
`AF_PUBLIC_ROOT` or `AF_PYTHON` only if their locations differ. The script prints a
submission manifest; invoking it again with identical inputs returns that manifest
without another submission. An uncertain submission leaves its intent for inspection.
The worker checks the submitted commit, clean checkout and selection-file bytes
before preflight and again before preparation. Identity is published before
`sbatch`, so a fast-starting job need not wait for the returned job-ID manifest.

The standalone CLI is `scripts/prefit_fit_handoff.py`:

```bash
python scripts/prefit_fit_handoff.py prepare --source REQUIREMENTS_ROOT --construction-root CONSTRUCTION_ROOT --public-root PUBLIC_ROOT --selection configs/prefit_public_fit_handoff_v1.json --output NEW_OUTPUT_ROOT
python scripts/prefit_fit_handoff.py run --output NEW_OUTPUT_ROOT
python scripts/prefit_fit_handoff.py summary --output NEW_OUTPUT_ROOT
```

Use the actual absolute environment Python on ACES; run fitting only in an allocated
CPU job. The last command reads the terminal result without starting an optimizer.
`handoff.json` records the source, equations, repair evidence and public-file hashes;
`fit/freeze.json` seals the fitting inputs; `fit/result.json` and
`fit/backend_result.json` retain numerical evidence; `summary.json` is produced by
`run`. Preflight errors are printed in the job log and saved under `runtime/`.

Identical terminal runs return the same fit. An attempt interrupted after its
start marker is terminal as `interrupted` and gets no fresh budget. Old campaign
roots remain untouched. Inspect logs before manually recovering a failed preparation.

## Interpretation and next boundary

`complete` means the fitter returned finite train/validation rollouts. Review the
NMSE, budget and convergence evidence separately. Capability failures route to the
fitter interface; interruption routes to execution recovery. Numerical failure or
poor fit does not prove instability, identify the faulty term, or establish
structural infeasibility. The nonlinear syntax can also become ineffective at
particular fitted coefficients; a syntax pass does not certify a mechanism.

The report contains candidate-bound numerical feedback for the next controller
milestone. Its scientific verdict remains unset. Connecting this evidence to an
actual proposer revision and re-fit remains the next step; this job does not
claim a completed iterative pipeline or select a final model for test evaluation.

## Verification

The full shared-checkout suite passed 1,797 tests with three optional Torch skips.
The focused interface/exporter/requirement suite passed 82 tests. After the final
launcher-only identity guard, all 19 affected exporter and scheduler tests passed,
including queued commit/working-tree/selection drift and immediate job startup.
Full-repository Ruff and shell syntax checks passed. The real synthetic handoff
smoke preserved the accepted candidate and terminal result exactly and produced
finite train/validation NMSEs below 3e-16; this is an integration control, not a
benchmark accuracy result.
