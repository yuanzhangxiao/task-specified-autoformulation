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

Final evaluation recomputes this endpoint from the frozen candidate and the
serialized public validation context. It does not trust a validity flag emitted
by the discovery method. Runtime-invalid candidates retain their diagnostics and
cannot carry test, hidden-mechanism, intervention, or public-mechanism scores.

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

A proposed-method component that assesses qualitative properties not completely
captured by deterministic public rules: mechanistic coherence, plausibility of
latent-state roles, unsupported shortcuts, and whether complexity is
scientifically justified. Its frozen output may guide proposal refinement and
selection, but it receives no trajectory-fit metric and is not treated as the
source of private benchmark truth. Final reports retain its influence, call
budget, response coverage, and ablations separately from deterministic and
private-reference endpoints.

### Private structural recovery

Post-selection agreement with the hidden reference mechanism: required edges,
signs, dynamic memory, mechanism identities, hidden trajectories, and private
intervention responses. This is a scientific evaluation endpoint and must
never influence search or model selection.

## Selection objectives

All objective components are development-only. Lower values are preferred.

1. **Validation only:** causal validation NMSE, with judge score as a
   deterministic tie-break. This remains the default for historical
   reproducibility.
2. **Normalized weighted sum:** robustly standardized log validation NMSE,
   plus a weighted negative-log judge penalty. This is available to the online
   beam and final selector as `normalized_weighted_sum`.
3. **Incumbent-relative hybrid (development pilot):** one challenger per round
   is compared with the incumbent frozen before that round. The first valid
   candidate seeds the incumbent; later pairwise values are never globally
   sorted because they may use different references. This mode is restricted to
   `beam_size=1` and cannot open test data.
4. **Pareto compromise:** discard candidates dominated simultaneously in
   validation NMSE, judge penalty, and term count, then choose a normalized
   compromise on the frontier.
5. **Epsilon constrained:** retain candidates within `delta` of the best
   validation NMSE, then prefer judge score, term count, and validation NMSE in
   that order.

The normalized weighted objective is

```text
z(log(validation_nmse))
  + lambda_judge * z(-log(judge_score + epsilon)).
```

Here `z` uses the within-run median and interquartile range, with the observed
range and then one as deterministic zero-spread fallbacks. This removes units
without using information from another split.

The runtime records the policy, raw judge score, normalized components, and
final scalar objective in the frozen selection. `validation_only` records the
same diagnostics but ranks on raw validation NMSE. Weighted selection requires
the judge to be enabled. Pruning remains distinct and no additional complexity
penalty is included in this first prospective online objective, avoiding double
counting with the judge's complexity-justification category.

The incumbent-relative pilot uses only one new tradeoff parameter, `alpha`:

```text
fit_delta = (nmse_incumbent - nmse_challenger)
            / (nmse_incumbent + nmse_challenger)
science_delta = bounded challenger preference from the frozen hybrid decision
combined_delta = (1 - alpha) * fit_delta + alpha * science_delta
relative_score = (1 + combined_delta) / 2
```

The hybrid judge's frozen tie interval maps to `science_delta = 0`; values
outside it are linearly scaled to `[-1, 1]`. The challenger replaces the
incumbent only when `combined_delta > 0`. A tie, indeterminate scientific
decision, or terminal paired-response failure retains the incumbent. Both
orientations use the same seed. If either orientation fails, the entire
incomplete pair is discarded and both orientations are retried once at the next
distinct seed. `configs/hybrid_search_objective_pilot_v1.json` freezes this
development contract before any search calls.

Version 2 changes only the deterministic treatment of comparative
`indeterminate` answers after A/B question consensus. Each of the three required
comparative questions contributes a signed vote: challenger preference `+1`,
incumbent preference `-1`, and tie or indeterminate `0` after identities are
normalized. The mean always divides by three. This prevents the remaining
determined question from inheriting the weight of questions withheld by
consensus. `configs/hybrid_search_objective_pilot_v2.json` freezes this scoring
boundary; the version-1 exclusion rule remains available for exact replay of old
artifacts.

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

The prospective typed implementation and conjunctive public-mechanism metric are
specified in `docs/PHASE_B_FINAL_EVALUATION.md`. The final record deliberately
defines no weighted overall score.
