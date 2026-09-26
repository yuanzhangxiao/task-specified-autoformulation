# Model inspection: findings, completed fixes and remaining priorities

Status: consolidated on 2026-09-26 from the reports linked below. This is an
index and handoff, not a new fitting experiment or an assertion that remote
campaigns have finished. The companion [CSTR note](2026-09-26-cstr-reference-defect-summary.md)
documents the confirmed reference-data defect separately.

Subsequent work: [the reference-integrity fix and audit](2026-09-26-reference-integrity-fix-and-audit.md)
corrects the generators and reports the remaining release/input-contract work.

## Main conclusions

We found useful demonstration candidates, especially reduced T1-hard models
and the assisted canonical R9 repair. We did **not** find one model that clearly
wins on every intervention or a clean matched Full/no-specification/no-latent
triple with equally good training fits and failures attributable to omission of
an explicitly required mechanism.

The recurring lesson is that low NMSE, an executable graph and legal parameter
values are different from a correct fitted mechanism. Missing dependencies,
wrong effective directions, almost-zero mechanism gains, negative latent
initial values and compensating transients can all coexist with good curves.
Conversely, a plausible skeleton can still be badly fitted. Both equations and
frozen-parameter intervention responses need inspection.

## What we inspected

The earlier inspection covered all 217 exported historical Full/Brief-only
fitted records, not only the lowest-NMSE endpoints. The newer
`final-components-v1` export contained **1,334 records: 688 fitted and 646
unfitted**, across T1/T2 easy/hard. We expanded algebraic processes and
substituted fitted parameters throughout that inventory. Detailed manual review
covered leading candidates, sign/coverage alternatives regardless of NMSE, and
all 37 unmatched unfitted candidates. These are records, including repeated
incumbents, not independent runs or 1,334 scientific certifications.

Six new T1-hard candidates underwent 162 frozen-parameter rollouts: training,
validation and seven previously defined probes. All were finite; the 12 saved
train/validation scores reproduced within 2.45e-15. The newer archive lacked
original split fingerprints; the replay recorded local same-cell data hashes.
Metric agreement is a consistency check, not recovery of missing provenance.

The export was incomplete: 747 of 1,200 planned visits were complete, 12 had
construction failures, 21 were interrupted and 420 were missing. Most
critic-enabled lineages stopped at R5/R6 while critic-disabled lineages reached
R14. **No exported record came from a pruning endpoint.** This snapshot cannot
support a fair critic comparison or a claim that pruning finished.

All round numbers below are zero-based. Record the campaign, task, candidate
hash, parameter vector and selection status: “R4” or “R9” alone is ambiguous.
In the newer task names, `c/v/s` mean critic/verifier/shared processes; `full`
means the training-evidence arm, not necessarily all switches enabled.

## Findings that changed our interpretation

### 1. Reference and input semantics must be checked first

CSTR's reference solver missed a feed-temperature pulse. Full's flat prediction
therefore looked better than No latent's dip for the wrong reason. The
[CSTR summary](2026-09-26-cstr-reference-defect-summary.md) gives the confirmed
4.36 K physical dip and the correction still needed.

A separate Dalla issue affects interpretation of time-zero meals. The physical
generator applies meal mass as a stomach-state jump. Common continuous replay
linearly interpolates the sampled `meal_event_g` channel. On the one-minute
grid, an isolated 90 g sample at time zero has interpolated area 45, whereas an
interior sample has area 90. The generator applies 90 g in both cases. Nonzero
fitted initial states can partly compensate, confounding comparisons to models
with zero meal-state initial values. This was diagnosed, not corrected and
rerun. Later probes place meals after time zero, but a continuous pulse remains
an approximation to the physical jump.

### 2. The traced wrong signs were not bound violations by the fitter

The original named Full R4/R13 ancestor and anonymous Brief-only R9 ancestor
declared all six outer signs **unrestricted**, with real coefficient domains.
The runtime did not discard fixed signs in those two construction handoffs.
For example, `-k_Uii*Uii` becomes a source if an unrestricted `k_Uii` fits
negative. That is legal under its declaration but conflicts with a utilization
interpretation.

The six rescue endpoints had zero compiled-bound violations among 67 retained
parameters. The newer inventory likewise had no positive/nonnegative-domain
violation among its 688 fitted records. These are scoped audits, not proof that
every historical sign path is correct. Not all inspected signs were wrong.

Inspect the **whole fitted contribution**: outer sign, gain domain, inner law,
state values and initialization. A positive gain multiplying a negative latent
state can still reverse an effect. A correct-looking graph with a nearly zero
gain does not establish an active mechanism. Likewise, a huge latent coordinate
may have a moderate contribution after multiplication by a tiny gain; rescale
before judging its size.

### 3. Fitting helped some models; it could not repair every skeleton

Refitting the historical perturbed R4 moved train/validation NMSE from roughly
`36.23/37.87` to `0.08545/0.06756`. The selected result in that rescue was the
unchanged pruning control: the improvement was from fitting, not pruning.
Many starting values were still 0.1, but that alone does not establish the exact
reason the original optimizer failed. Worker crashes also occurred on ACES;
they were not demonstrated to be a laptop-only problem.

For canonical R2, correcting the tissue-return sign and refitting improved
train/validation from `0.21856/0.39694` to `0.04332/0.05486`. A wrong fixed
negative outer sign had pushed its nonnegative gain toward zero. This is a
useful repair example, but not two equally good training fits diverging only
under intervention.

`status=complete` often meant finite completed rollouts, not verified optimizer
convergence. Initializer limits/failures and retained finite fits can coexist;
native convergence was unknown for several rescue results. A budget stop is
not evidence of structural infeasibility.

### 4. Missing dependencies can explain failure exactly, with limited claims

The historical no-specification canonical-obfuscated T1 seed-0 model has no
physical auxiliary dependence in either its equations or initializer. With
meal history and initial plasma glucose fixed, changing tissue glucose `Gt`
therefore produces **exactly no response**. Assisted R9 retains a positive
`Gt` contribution and responds. Their training NMSEs are similar:
`0.043576` versus `0.041554`.

This is our clearest omission/response illustration. However, T1 does not
explicitly mandate `Gt`; auxiliary use is optional. The omission is not a
literal T1 specification violation. The assisted model and historical ablation
also do not form a controlled, same-seed comparison of specification guidance.

T2 explicitly requires delayed insulin action on disposal, but does not
prescribe a meal/glucose-to-insulin secretion equation. Missing insulin responses
should be inferred from observed trajectories or assessed scientifically, not
converted into a fabricated public requirement. In the newer T2-easy inventory,
140/154 fitted records lacked a syntactic meal-to-I or Gp-to-I path. The better
T2-hard skeleton still had an approximately 10.4-million-minute memory time
constant and a large negative initial memory. Neither cell supplied a ready
whole-model physiological demonstration.

### 5. Pruning and intervention metrics need precise interpretations

Removing an excretion term when its training channel is identically zero does
not prove excretion is unnecessary outside training. Removing one input term
from a latent equation does not necessarily remove that latent state. Paired
unchanged/pruned fits are needed to separate pruning from extra fitting budget;
the implemented selection uses validation, despite an earlier descriptive
metadata label that incorrectly said training-only selection.

Absolute trajectory NMSE and intervention-response error answer different
questions. A model can predict the baseline glucose level well but respond
incorrectly to a small intervention. For a model and reference, form responses
relative to their respective matched controls, with the same fitted parameters:

```
delta_model = prediction(intervention) - prediction(control)
delta_ref   = reference(intervention)  - reference(control)
response_RSE = sum((delta_model - delta_ref)^2) / sum(delta_ref^2)
```

Handle a zero reference-response denominator explicitly. Report both absolute
curves and response curves. A tissue-initialization probe changes initial `Gt`
while holding initial `Gp` and other physical states fixed; it is not a
mass-conserving redistribution. These T1 evaluations are conditional on the
permitted reference-generated auxiliary channels.

## Completed engineering fixes, and their scope

These changes were implemented incrementally in versioned protocols. Their
presence in current source does not retroactively change an old frozen run.

| Issue | Implemented correction | Evidence/runbook |
|---|---|---|
| Single-output assumptions in revisions | Explicit target-addressed mappings, joint-output fitting and per-target feedback; omitted mappings remain unchanged. | [Multi-target continuation](2026-09-22-codex-dalla-multi-continuation.md) |
| Construction failures closed useful drafts | Expose the existing algebraic-U contract before construction, return failures for bounded repair, retain failed drafts for later visits. This was not a new benchmark requirement. | [Continuation policy](../../docs/DALLA_MULTI_TARGET_CONTINUATION.md) |
| Failed revisions erased/stalled usable models | Preserve fitted incumbents; invalid/worse fits cannot replace them; bounded unchanged-model fallback fits where the protocol permits. | [Continuation corrections](2026-09-22-codex-dalla-multi-continuation.md) |
| Oversized numerical evidence caused delivery failure | Compact per-target errors and response examples, tokenizer preflight/output reserve, full residuals kept separately; provider failure preserves incumbent and stops later automatic dispatch. | [Response feedback](2026-09-22-response-feedback-implementation.md) |
| Existing-parameter collisions or role changes | Exact references inherit declarations without fixing values; conflicting explicit roles reject with focused feedback; fresh meanings need fresh names. | [Parameter/selection fixes](2026-09-23-codex-parameter-and-selection-fixes.md) |
| Tiny score changes rewarded extra terms | Fixed validation comparison band; fewer additive terms win within it, otherwise equal-complexity ties keep incumbent. | [Integrity v8](../../docs/REVIEW_INTEGRITY_V8.md) |
| Scientific directions remained freely signed | Separate opt-in sign-review campaigns assign justified outer signs and nonnegative magnitudes, protect inner laws, refit repaired/control arms and audit retained signs/bounds. | [Sign repair](2026-09-24-dalla-sign-repair-implementation.md) |
| Citation mistakes blocked usable decisions | Record exact-match/citation issues without treating them as verified support or automatically rejecting otherwise supported decisions; semantic assessment remains fallible. | [Directional review](2026-09-24-dalla-directional-sign-review.md) |
| Pilots used older construction paths | Fresh T1-hard/T2 integration combined shared processes, multiple outputs, compact feedback, protected parameters and final paired pruning. That fresh campaign had its critic off; later component campaigns have separate flags. | [Fresh integration](2026-09-23-fresh-shared-multi-pruning.md) |

Assisted sign choices and refits are separate diagnostic experiments, not an
automatic correction applied to every historical model. CSTR reference repair,
Dalla meal-event semantics, universal state positivity and complete physiological
mechanism certification are **not** established by these fixes.

## Candidates worth retaining

Use the linked reports for exact candidate hashes, coefficients, initializers,
probe definitions and plotting files. Values below are approximate.

| Candidate and provenance | Train / validation NMSE | Why keep it; main limitation |
|---|---:|---|
| **New T1-hard Brief R4**, `cell06_seed1_brief_only_c1v1s1`, retained through R5 | .05182 / .07377 | Simple positive meal-memory state and plausible tissue return; Gt -20% response RSE .0457. Not an exported final pruning result. |
| **New T1-hard Full R4**, `cell06_seed1_full_c1v1s0`, fitted trial | .04676 / .06701 | Similar clean reduced skeleton; Gt -20% response RSE .0245. Shared processes off and not the selected final endpoint. |
| **New T1-hard Brief R10**, `cell06_seed1_brief_only_c0v1s1`, retained through R14 | .01484 / .01546 | Strong meal trajectories; split-meal NMSE .02077. Negative latent starts and poor tissue-response sensitivity; critic off. |
| **New T1-hard Brief R14**, same lineage, unselected trial | .01216 / .02220 | Split-meal NMSE .01274. Negative absorption initial state; cannot replace R10 as the final endpoint after viewing probes. |
| **Assisted canonical-obfuscated R9**, historical Brief-only lineage, repaired arm | .04155 / .05846 | Best missing-Gt-dependency contrast to historical no-specification. Assisted/exploratory; Gt is optional under T1. |
| **Assisted canonical-named R2**, historical Full lineage, repaired arm | .04332 / .05486 | Clear rescue from a wrong tissue-return sign. Original model already fitted poorly, so it does not demonstrate hidden failure after equally good training. |

The two **new** R4 models are different from the old **perturbed assisted R4**.
After an exact change of latent coordinates, their equations are:

```
New Full R4:
  B'  = 0.0949650 u - 0.00802076 B;  B(0) = 1.509686
  Gp' = B - 0.0709602 Gp + 0.0904661 Gt

New Brief R4:
  B'  = 0.0864003 u - 0.0134820 B;    B(0) = 1.133964
  Gp' = B - 0.0459638 Gp + 0.0594458 Gt
```

These have interpretable positive meal memory, decay, tissue return and plasma
clearance. They approximate other physiological processes rather than recover
the full reference equation. Their meal responses start too early. Their
promise is a defensible reduced-model illustration, not universal superiority.

The old perturbed assisted R4 is a lower-priority mechanism example: its
train/validation NMSE is `.10715/.08113`, worse than the matched unchanged
control `.07078/.06811`; it has near-zero production/utilization gains and
effectively instantaneous plasma dynamics. It has some meal-spacing gains but
poor tissue-initialization responses. Preserve it as a useful negative finding.

External comparisons were mixed. Canonical assisted R9 improves fasting
tissue-response error over Sol repetitions 0 and 1, but Sol repetition 2 is
essentially exact there; all three outperform it on the split-meal probe.
Absolute and response rankings can disagree. Report all repetitions and both
metrics, not only a favorable baseline or intervention.

## Most promising next work

1. **Resolve data/input integrity before another broad fitting sweep.** Fix
   the CSTR release separately; define and test Dalla event mass, time-zero
   events and initialization consistently across methods. Keep historical
   artifacts unchanged and identify corrected versions explicitly.
2. **Inspect the clean reduced T1-hard candidates first.** Retrieve completed
   critic-enabled rounds and pruning outputs for the named R4 lineages and fully
   enabled Full lineages. Compare the actual retained endpoints with matched
   refit/pruning controls. Do not infer completion from a successful job alone.
3. **Strengthen fitted-mechanism diagnostics.** Report effective contributions,
   relevant directional derivatives, initial states, fitted time constants,
   active gains and excitation of each mechanism. Enforce nonnegativity only
   when the declared variable semantics support it; signed deviations and
   coordinate changes must remain possible. A graph path is necessary but not
   sufficient evidence of an active delayed mechanism.
4. **Improve fitting diagnoses and evidence-directed repair.** Separate poor
   optimization from absent dependencies and weak identifiability. Examine
   per-target errors/sensitivities and unusually large or near-zero scales.
   Continue fitting when justified by optimization evidence; propose structural
   changes when data support them. Do not infer a unique faulty term from high
   aggregate NMSE or a budget stop.
5. **Test a genuine required-memory example prospectively.** T2's delayed
   insulin action is a stronger explicit requirement than T1's optional Gt
   dependence. A bounded delayed-versus-instantaneous comparison should first
   achieve adequate training fits, freeze parameters, then evaluate fixed insulin
   pulse/spacing schedules and all outputs. A manual instantaneous ablation
   tests memory's value; it is not a no-specification proposer experiment.
6. **Keep exploratory demonstrations distinct from benchmark evidence.** The
   probes inspected repeatedly are no longer untouched holdouts for further
   model selection. Freeze any next repaired candidate before a new declared
   evaluation family, regenerate all permitted auxiliaries consistently, and
   report all schedules/failures. A more plausible skeleton does not guarantee
   a favorable intervention result without correct parameters and excitation.

## Existing detailed summaries and reproducible artifacts

| Subject | Existing report |
|---|---|
| Complete newer inventory, six-model equations and replay | [New Dalla model inventory](2026-09-25-new-dalla-model-inventory-review.md) |
| Historical refitting and pruning outcomes | [Rescue results](2026-09-24-dalla-rescue-results.md) |
| Original topology declarations versus fitted signs | [Topology sign origin](2026-09-24-dalla-topology-sign-origin.md) |
| Perturbed assisted R4 and matched control | [Sign-diagnostic results](2026-09-25-dalla-sign-diagnostic-results.md) |
| Canonical assisted R9/R2, no-specification and external models | [Canonical rescue results](2026-09-25-canonical-r9-r2-rescue-results.md) |
| Meal-event semantics and broader meal grid | [R4 equations and intervention grid](2026-09-25-r4-equations-and-intervention-grid.md) |
| Why the strongest illustration is not a literal requirement violation | [Required-mechanism search](2026-09-25-required-mechanism-demonstration-search.md) |
| Limits of the matched Full/ablation story | [Full versus ablation audit](2026-09-25-full-versus-ablation-demonstration-audit.md) |

Useful local artifact directories (not committed, transfer separately):

- `artifacts/dalla-all-models-review-2026-09-25/`: `screen.json`, exact
  `shortlisted-models.json`, `shortlist-curves.csv`, `shortlist-probes.csv`,
  all-training/all-validation figures and sealed replay records.
- `artifacts/dalla-canonical-rescue-inspection-2026-09-25/`: canonical assisted
  repairs, unchanged controls, matching external models and no-specification.
  Transfer package: `transfers/dalla-canonical-rescue-curves-20260925.tar.gz`.
- `artifacts/dalla-sign-diagnostic-inspection-2026-09-25/`: perturbed R4
  assisted/control curves and sign checks.
- `artifacts/dalla-sign-audit-2026-09-24/origin-audit.json`: original topology
  and compiled-domain provenance.

Do not merge canonical and perturbed dynamics, named and obfuscated information
conditions, assisted and autonomous candidates, or selected endpoints and
unselected trials in a single unlabeled comparison. This handoff adds no new
benchmark fitting, LLM calls, remote jobs or benchmark-test access.

## Verification of this documentation update

All 25 relative links in the two summaries resolve locally. The relevant CSTR,
T1 replay, rescue and sign-repair regression tests passed: **74 passed**. The
synthetic rescue smoke passed three toy fits and performed no additional fits
on resume; it used no benchmark data or LLM calls. No implementation or dataset
was changed, and the full repository test suite was not rerun for these notes.
`ruff check .` still reports 37 pre-existing findings under `analysis/claude/`.
