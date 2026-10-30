# Receiver review: 23 scaled-readout construction candidates

29 October 2026. Orion has independently checked the delivery for fitting
intake. No candidate fitting, trajectory integration, validation read, test read,
LLM call, or remote job submission was performed in this review. Numerical
qualification remains pending. The separate detention continuation is unchanged.

## Accepted inputs and verification

Source: [Astra's handoff](../../docs/PHASE_C_SCALED_READOUT_HANDOFF.md) and
`output/construction-scaled-readout-delivery-20261029/`.

- Delivery source commit: `3eb449bc2554a5cd1b81853dde774d1b46805612`.
- Archive SHA-256:
  `ca85ea734c9121da1ea1cd813c8b07cb46ac637f8f6af70858c1095d5596fbe4`.
- All 165 manifest-listed files match both archive bytes and extracted files;
  the separate checksum list also verifies. Archive paths and entry types were
  checked without executing supplied packaging scripts.
- All 23 roster packets pass typed parsing, restricted compilation, initializer
  lowering, parameter-domain checks at their declared guesses, and the existing
  prefix/future-forcing interface for all 16 training trajectories.
- All 368 model/trajectory initial points have finite initialized states,
  vector fields, readouts and local analytic partial derivatives. This is a
  pointwise check, not a successful integration or derivative-accuracy check.
- Public training content SHA-256:
  `169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795`.

The machine-readable receiver evidence is local generated output:
`output/scaled-intake-receiver-audit-20261029/intake-audit.json`.
Do not commit that artifact or the supplied model/data packets.

Two existing digest conventions must be distinguished in the importer. Source
`problem_sha256` uses `staged_topology.content_hash` (sorted JSON with spaces),
whereas fitting identities use `public_fitting.content_sha256` (compact sorted
JSON). Both are recorded in the receiver audit. Their different strings are not
evidence of changed equations; authenticate each with its declared convention.

This population comprises 23 complete-model/domain identities for one T2-easy
parent, not 23 benchmarks or necessarily 23 different scientific mechanisms.
No numerical score was used to select the roster.

## Numerical structure

Fifteen models are jointly affine in states and supplied forcing, so the existing
exact affine propagation certificate applies. Their dependence on unknown
parameters can still be nonlinear; fitting is not thereby convex.

Eight require the general ODE path: **01, 02, 03, 04, 10, 13, 14, 17**.
These include rational/saturating responses and, for candidate-17, the genuine
multi-argument interaction `Gp*S_insulin`.

Every model has one or two independent hidden initial roots per trajectory.
Sixteen training trajectories therefore contribute 16 or 32 root coordinates.
Including 17–21 shared parameters gives 35–53 joint coordinates before reduction.
Certified gain products remove 5–8 coordinates per model. Existing structural
certification additionally removes ten unused coordinates across six models,
leaving 27–45 coordinates. These are exact coordinate reductions, not pruning
based on small fitted coefficients or a claim to identify the original factors.

| Candidate | Propagation | States | Roots per row | Original coordinates | After gain and structural reductions |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 | Affine | 5 | 2 | 52 | 44 |
| 01 | ODE | 4 | 1 | 35 | 27 |
| 02 | ODE | 4 | 1 | 36 | 29 |
| 03 | ODE | 4 | 1 | 36 | 28 |
| 04 | ODE | 4 | 1 | 37 | 31 |
| 05 | Affine | 5 | 2 | 51 | 43 |
| 06 | Affine | 5 | 2 | 53 | 45 |
| 07 | Affine | 5 | 2 | 52 | 44 |
| 08 | Affine | 5 | 2 | 53 | 45 |
| 09 | Affine | 5 | 2 | 53 | 45 |
| 10 | ODE | 4 | 1 | 36 | 29 |
| 11 | Affine | 5 | 2 | 52 | 44 |
| 12 | Affine | 5 | 2 | 52 | 44 |
| 13 | ODE | 4 | 1 | 37 | 29 |
| 14 | ODE | 4 | 1 | 36 | 29 |
| 15 | Affine | 5 | 2 | 50 | 42 |
| 16 | Affine | 5 | 2 | 53 | 45 |
| 17 | ODE | 5 | 2 | 53 | 45 |
| 18 | Affine | 5 | 2 | 53 | 45 |
| 19 | Affine | 5 | 2 | 49 | 43 |
| 20 | Affine | 5 | 2 | 52 | 44 |
| 21 | Affine | 5 | 2 | 49 | 42 |
| 22 | Affine | 5 | 2 | 53 | 45 |

The generic training oracle lacks the `.training` metadata attribute used by the
existing inactive-initializer certificate. It was attached only in memory for
this static audit; a production adapter is still needed. No implementation file
was changed to produce these counts.

## Preserve the initialization and information contract

Keep the frozen eight-unit prefix: nine observed samples per row, followed by
292 suffix samples. Fit shared dynamics/map parameters and declared training
roots jointly on training. For a prefix-conditioned forecast, freeze the shared
parameters and estimate only the declared independent roots from those nine
samples; derive dependent initials through the graph.

Report these separately:

1. Full-training reconstruction with jointly fitted training roots.
2. Training suffix forecasts after fresh prefix-only root inference.
3. Validation suffix forecasts using the same prefix-only inference rule, after
   the previously frozen matching validation split is separately authenticated.

Using full-trajectory-fitted roots for item 2 would give training forecasts more
information than validation forecasts. Validation is absent from this delivery;
do not create a replacement split. No held-out metric goes to the proposer.

Candidate-01 requires ongoing supplied `Uii(t)`. The other 22 use only the two
external-input schedules for future forcing. The frozen policy authorizes this
auxiliary dependency; retaining only `Uii(0)` would change candidate-01. The
zero-valued base-candidate initializer placeholders are not the fitting
initializer; the graph is authoritative.

## Issues to retain explicitly

- Several meal-to-glucose terms have identical functional dependence, and some
  insulin excitation/decay terms overlap. Current reductions do not remove every
  such redundancy. Further reductions require an algebraic and feasible-domain
  certificate, not a small numerical sensitivity column.
- Candidates 05, 06, 07 and 17 merit hidden-state scale-symmetry checks. These are
  potential ambiguities, not yet certified reductions in this intake review.
- Candidate-01 has `Uii/(1 + alpha*Gp)` with real `alpha`. Other rational models
  also need denominator/domain diagnostics during integration and optimization.
  A finite initial point does not rule out later singularities or entry into a
  protected-division region. Preserve the original executable semantics.
- Candidates 10 and 14 have the same Hill-form expressions but different
  parameter domains. Keep both. Candidate-10's real `Km`, appearing as `Km**2`,
  has an exact sign ambiguity; accurate predictions cannot identify that sign.
- A nonnegative outer gain does not fix the physical effect's sign when its
  inner coefficient is unrestricted. Process names such as endogenous production
  also do not establish a baseline source when the expression is proportional to
  meal input. Retain these scientific questions separately from fit convergence.

None of these notes modifies a model, adds a physiological bound, or establishes
that the best attainable fit is poor. Astra's scaled-readout contradiction proof
also concerns exact reproduction, not an approximate-error lower bound.

## Recommended next implementation milestone

Build a separate CPU qualification campaign for the complete 23-model roster.
The historical returned-eight launcher and its validation requirement cannot
accept this delivery unchanged.

1. Authenticate the manifest, preserve every original start/domain/initializer,
   and freeze a common numerical policy and per-stage nonrenewable allowances.
2. Route certified affine systems to exact propagation and all others to the
   general ODE oracle. Apply only certified coordinate reductions, preserving a
   lift to the original parameters and original-equation verification.
3. Bridge the general oracle to the current lazy search driver: residual-only
   trial evaluation, Jacobians when needed, durable useful-point retention,
   training-only continuation and explicit domain/derivative failure outcomes.
4. First qualify initial rollouts and derivative reliability, then bounded search
   on all 23 models. Retain failed starts as outcomes; never silently delete them
   or classify them as equation failures. Benchmark the numerical overhead before
   adding a large restart population.
5. Independently replay retained parameters, report convergence/remaining descent
   separately from integration accuracy and parameter ambiguity, and produce
   full reconstruction plus prefix-reinferred training forecasts. Bind validation
   only after its provenance is established.

The old Model-6 multi-output profiler requires distinct identity-state outputs.
None of these packets meets that implementation's criterion because `U` is a
composite readout. Composite-output profiling may be possible mathematically,
but extending it is a separate milestone. Likewise, detention-specific parameter
charts must not be transplanted unchanged.

Only qualified training evidence should support subsequent equation feedback.
High error with unresolved optimization calls for numerical recovery, not an
assertion that a new nonlinearity or state is required. Preserve all 23 outcomes
and report median/unscaled MAD with failure counts; do not present this intake
as a benchmark-wide comparison.

## Verification of this review

- Independent manifest, packet, prefix/domain and 368 initial-point checks passed.
- 81 focused prefix/schema/gain/recovery/structural-reduction tests passed.
- `ruff check .` reports the same 37 findings in unrelated untracked
  `analysis/claude/`; no implementation files changed in this review.
- `git diff --check` passed. Full repository pytest was not rerun for this
  read-only implementation review; the focused run is not a full-suite claim.
- The runtime checks above are distinct from the synthetic rollouts performed
  inside the existing unit tests. No supplied candidate trajectory was integrated.

This document is the receiver's review and implementation recommendation, not a
claim that a launcher has been implemented or that the 23 fits have started.
