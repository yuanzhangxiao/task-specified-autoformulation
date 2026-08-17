# Current judge-rubric audit

## Current rubric

Every benchmark and tier uses the same six category weights:

| Category | Weight | Deterministic overlap | Residual qualitative role |
|---|---:|---|---|
| Task/output coverage | 0.20 | High: output mappings and declared targets are validated | Whether all prose objectives are substantively addressed |
| Mechanism/state adequacy | 0.25 | Partial: state dynamics and declared roles are validated | Coherence and plausibility of the proposed mechanistic decomposition |
| Mathematical completeness | 0.15 | Very high: closure, symbols, equations, mappings, parameters, and initials are validated | Little residual role beyond model-level conceptual well-posedness |
| Data/causal consistency | 0.15 | Very high: channel availability and target leakage are validated | Detecting conceptual shortcuts not captured by the grammar |
| Constraint compliance | 0.15 | Partial: implemented explicit constraints are checked | Qualitative plausibility or prompt constraints without deterministic checks |
| Parsimony/interpretability | 0.10 | Partial: term, state, process, and parameter counts are deterministic | Whether remaining complexity has an understandable scientific role |

The aggregate score is therefore not a pure mechanistic-plausibility score. It
mixes deterministic properties, qualitative scientific judgment, and
parsimony. It also overlaps with the explicit complexity term proposed for the
selector.

## Current safeguards

- The judge receives the proposer prompt, a public checklist, and candidate
  structure, but no fit metric or trajectory data.
- Candidate content is treated as untrusted text.
- The runtime schema fixes all six categories and their ranges.
- Judge red flags are advisory. They cannot override deterministic validity.
- The production controller ranks validation NMSE first and uses aggregate
  judge score only as a tie-breaker.

## Interpretation of the frozen score

Existing cached judge outputs remain valid evidence for the historical method,
but their aggregate score should be described as **public task-compliance and
qualitative-mechanism assessment**, not structural recovery. It cannot verify
agreement with a hidden simulator and should not be used as a substitute for
private structural-validity or hidden-trajectory evaluation.

For the zero-call frozen-pool study, the historical aggregate score is retained
unchanged. Reweighting its cached categories is allowed as a sensitivity
analysis, but changing category definitions would require new judge calls.

## Implemented prospective v2 rubric

New judge calls use schema version 2. Historical version-1 prompt text and
checkpoint payloads remain supported for retrospective experiments, while a
runtime amendment explicitly supersedes the old categories for prospective
calls. The v2 implementation:

1. moves all implemented syntax, closure, availability, causality, mapping, and
   explicit constraint checks exclusively into the deterministic feasibility
   layer;
2. exposes those check results to the judge as certified facts, not questions;
3. asks the judge only about mechanistic coherence, source/sink balance
   semantics, dynamic plausibility, mechanism coupling and task sufficiency,
   nonredundancy/accounting, and latent-state complexity justification;
4. returns category scores separately and computes the weighted aggregate in
   deterministic runtime code, so the LLM cannot choose or manipulate it;
5. keeps private reference mechanisms, hidden trajectories, fit metrics, and
   test information outside the prompt; and
6. keeps all scientific red flags advisory.

The six v2 weights are 0.20, 0.20, 0.20, 0.20, 0.10, and 0.10 in the order
listed above. The resulting score still requires calibration against blinded
expert ratings and adversarial pairs, and should not be combined with an
explicit complexity penalty without checking for double counting.

The prospective judge must be evaluated as a measurement instrument: category
reliability, family sensitivity, adversarial preference accuracy, expert
agreement, and association with private structural recovery should be reported
before its weight in selection is increased.

## Empirical calibration result

The frozen 336-call adversarial analysis is documented in
`docs/JUDGE_CALIBRATION_ANALYSIS.md`. On the 24 genuine dynamics mutations per
family, the historical aggregate distinguishes the valid member of a matched
pair in 91.7% of Gemini comparisons and 100% of GPT comparisons. Repeat
ICC(1,1) is 0.975 and 0.941, respectively.

Leave-one-benchmark-out category calibration improves Brier calibration but
does not improve pairwise accuracy, and it reduces Gemini AUROC. Consequently,
the fitted coefficients are not adopted as production selector weights.
Mechanism/state adequacy remains the most useful qualitative category;
constraint compliance is discriminative but overlaps deterministic checks.
The future rubric should narrow the LLM's role rather than merely reweight all
six historical categories.
