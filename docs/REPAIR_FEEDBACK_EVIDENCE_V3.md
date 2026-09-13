# Feedback evidence v3: actual capabilities, effects and scientific references

This opt-in milestone follows ACES comparison-2 (`c6f929a`). The no-judge arm
attempted redundant initializer edits. The judge arm changed one sign while its
explanation claimed the opposite direction; one other review exhausted atomic
response validation. These are development observations, not scientific labels.

## Changes and boundaries

- `repair-feedback-comparison-3` pins `repair-feedback-evidence-3`. It requires
  a new output root. The action schema remains `RepairActionV2`; its request/cache
  identity now also binds the feedback protocol. Old checkpoints are not resumed
  as though they answered these revised requests.
- A report lowers the same initializer plan as the fitter, without reading data
  or optimizing. It names latent states, shared-value/map policy, optimizer
  guesses, and available selected training-fit parameter values. Missing values
  stay unknown. Observed initial channels remain distinct from latent boundaries.
  Collocation-stage status explicitly does not assert missing latent initializers.
  A zero guess is not a prescribed zero physical initial value. Validation/test
  parameters are not estimated or exposed through any new path.
- Each validated action records actual before/after equation, mapping and
  initializer effects. The audit's changed lists contain real changes, not all
  attempted targets. `causal_map:null` is unchanged when that same shared-value
  rule already exists, but can be a real change from a causal map. These effects
  enter subsequent history/report requests and compact summaries.
- No-change reasons distinguish `explicit_decline`, `empty_edit`, and
  `equivalent_model_and_initialization`. The existing identity test still owns
  commit/deduplication; AST equality is not a general symbolic equivalence proof.
  Repeated no-ops are not counted as repaired models or successful scientific
  decisions. The two-consecutive-no-change stopping policy is unchanged.
- Only after judging, the adapter reconstructs both exact atomic occurrence
  plans and maps referenced current-candidate IDs to equation/process, component,
  unsigned expression and certified outer polarity. Parent-only and unknown IDs
  stay unresolved. Original evidence, verdicts and scores are retained. The
  proposer is told that a scientific expected sign is advisory, while an outer
  sign is syntax—not the sign of the state or evaluated term. Neither is promoted
  to an explicit public requirement without public evidence.
- Review status, prior terminal validation failures and request/token costs are
  shown independently of scientific concerns. `indeterminate` does not mean pass
  or no defect. Missing occurrence sets and duplicate occurrence IDs have named
  diagnostics. No missing scientific answers are imputed, no duplicates are
  silently removed, and no additional judge retries are introduced.

The 20B proposer, 120B judge, scientific prompts, atomic sign blinding, symmetry,
scoring, fitting algorithm/budgets, source candidates, public data, beam/round
limits and production defaults are unchanged. This is a feedback-only successor,
not a scientific recalibration or a claim that judge validation failures are
solved. A different atomic output interface would require a separate experiment.

## Local checks and ACES experiment

Before GPUs, run `tests/test_repair_feedback_evidence.py`, the existing repair
comparison/draft/replay tests, atomic-judge tests, and the real CPU fitter smoke
`scripts/smoke_repair_feedback_comparison.py`. The ACES CPU preparation job runs
all these checks and blocks the GPU dependency on failure. The smoke also checks
that learned initial parameters appear in feedback and resume is identical.

The new regressions reproduce timeout-versus-boundary confusion, redundant
initializers, explicit decline versus attempted no-op, a real causal-map change,
sign flips whose explanations disagree with equations, both judge orientations,
unresolved/parent-only references, omitted atomic units and duplicate atomic IDs.
Mocked response tests measure plumbing, not scientific accuracy.

Submit the two existing split arms independently from a clean pinned checkout
and a new shared root, using `scripts/hpc/submit_repair_comparison_aces.sh`:

- `AF_ARM=redesigned_runtime`: one H100, no judge calls.
- `AF_ARM=redesigned_prefit_judge`: two H100s, same frozen comparison plan.

Both use the existing SIF path/hash and model revisions. Do not rebuild an image,
change its frozen SHA, or overwrite earlier results. There are two seeds and at
most four repair rounds per seed per arm. The same early-stop rule remains.

Report baseline and best NMSE, actual committed/no-op/declined actions, reasons,
named diagnostics, review availability and costs separately. A judge-arm task
with no usable scientific review is not successful exposure to that intervention.
Keep such failures in the denominator and do not select away costly failures.
There is no automatic winner, no test access, and no private-reference access.

### Verification on the development checkout

- Focused feedback, action/replay, campaign and atomic-judge tests: 101 passed.
- Final full `pytest` run: 1,422 passed, 39 failed because benchmark fixtures are
  absent from this isolated checkout, and three optional Torch tests skipped.
  No benchmark files were copied, generated or changed to mask those failures.
- Real CPU fitter/resume smoke: pass; training NMSE `3.116437e-16`, validation
  NMSE `2.922292e-16`, with fitted latent initial value `2.00000000169` correctly
  distinguished from its zero optimizer guess. Resume was identical.
- `ruff check .`, launcher Bash syntax and `git diff --check`: pass.

## Remaining limitations

Better evidence does not guarantee the proposer interprets it correctly. No
automatic check proves that hypothesis prose matches an edit or that a chosen
sign is scientifically justified. Judge response validation and fitter timeouts
can still fail; this milestone makes those outcomes distinguishable and auditable.
