# Selection and evaluation contract

## Purpose

This contract prevents private benchmark knowledge and test outcomes from
entering proposal generation, fitting, pruning, or model selection. It also
separates deterministic validity, qualitative LLM assessment, predictive fit,
and post-selection scientific evaluation instead of calling all four
properties "model quality."

## Permitted information by stage

| Signal | Proposal | Fit/prune | Selection | Final evaluation |
|---|---:|---:|---:|---:|
| Public task prompt and manifest | Yes | Yes | Yes | Yes |
| Training trajectories and metrics | Summary only | Yes | Yes | Yes |
| Validation trajectories and metrics | Metric feedback only | Yes | Yes | Yes |
| Deterministic runtime validity | Yes | Yes | Hard gate | Report |
| LLM judge categories and aggregate | Yes | No | Yes | Report |
| Post-pruning additive term count | Yes | Yes | Yes | Report |
| Test trajectories or metrics | No | No | No | Once, after freeze |
| Private mechanism reference | No | No | No | Yes |
| Hidden reference trajectories | No | No | No | Yes |
| Private intervention outcomes | No | No | No | Yes |

All frozen-pool selector analyses must consume `CandidateArtifact`, which does
not contain test metrics or private mechanism references. Hyperparameters must
be frozen before joining selections to test results.

## Meanings of validity and quality

### Deterministic runtime validity

A mandatory feasibility gate computed from public information. It covers
schema validity, restricted-expression parsing, equation closure, defined
symbols, legal channel use, causal availability, observation mappings, safe
domains, declared parameters and initial conditions, and explicit public
constraints that have deterministic implementations. Failure makes a
candidate ineligible; superior fit cannot compensate for it.

### Public structural compliance

Deterministic task predicates derived only from public requirements may become
selection-time gates or scores after their implementation has been frozen.
Examples include the presence of a required output mapping or a dynamic path
from a declared input to a target. A requirement must not encode the hidden
simulator structure more specifically than the public prompt.

The current frozen candidate pool records runtime-valid candidates, but it does
not yet contain a separate public structural-compliance score. Therefore the
Phase A6 retrospective comparison treats deterministic validity as a common
hard gate and does not invent such a score after the fact.

### LLM mechanistic assessment

An advisory assessment of qualitative properties that are not completely
captured by deterministic public rules: mechanistic coherence, plausibility of
latent-state roles, unsupported shortcuts, and whether complexity is
scientifically justified. The LLM judge receives no trajectory-fit metric.

### Private structural recovery

Post-selection agreement with the hidden reference mechanism: required edges,
signs, dynamic memory, mechanism identities, hidden trajectories, and private
intervention responses. This is a scientific evaluation endpoint and must
never influence search or model selection.

## Selection objectives under study

All objective components are development-only. Lower values are preferred.

1. **Validation only:** causal validation NMSE, judge score as a deterministic
   tie-break only. This is the production behavior.
2. **Normalized weighted sum:** robustly standardized log validation NMSE,
   negative-log judge penalty, and log post-pruning term count.
3. **Pareto compromise:** discard candidates dominated simultaneously in
   validation NMSE, judge penalty, and term count, then choose a normalized
   compromise on the frontier.
4. **Epsilon constrained:** retain candidates within `delta` of the best
   validation NMSE, then prefer judge score, term count, and validation NMSE in
   that order.

The normalized weighted objective is

```text
z(log(validation_nmse))
  + lambda_judge * z(-log(judge_score + 0.05))
  + lambda_sparse * z(log(1 + additive_term_count)).
```

Here `z` uses the within-run median and interquartile range, with the observed
range and then one as deterministic zero-spread fallbacks. This removes units
without using information from another split.

Pruning and sparsity-aware selection remain distinct. Pruning removes weak
terms within a candidate using fixed development-only rules. The sparsity term
compares the post-pruning complexities of different candidates.

## Hyperparameter discipline

- Use one prespecified grid across benchmarks and tiers.
- Do not optimize weights on test MSE, hidden MSE, or private structural
  recovery.
- Report the full sensitivity surface rather than only a favorable point.
- Prefer a stable region that preserves validation quality over a sharp optimum.
- Freeze any chosen selector and weights before deterministic refitting and one
  test evaluation.
- Treat a selector that changes beam membership or proposer feedback as a new
  end-to-end algorithm; retrospective final selection alone does not establish
  its end-to-end performance.

## Final reporting

No scalar test score defines a universally better scientific model. Report the
vector of observed test error, private structural validity, hidden-trajectory
error where defined, post-pruning term count, and completion/failure status.
Use "lower test MSE" rather than "better model" when only prediction improved,
and report Pareto dominance only when one model is no worse on every applicable
endpoint and strictly better on at least one.
