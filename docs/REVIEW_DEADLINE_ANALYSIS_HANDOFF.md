# Internal-deadline analysis handoff

This document is a handoff for producing tables, figures, and concise result
summaries from the twenty-four-hour review campaign. It distinguishes current
campaign evidence from historical experiments and records decisions that should
guide the next pipeline revision.

The campaign implementation is pinned at commit
`a590b7a27edf145b23de6f7cbdfb4438fcaa8600`. The matched refit control was added
at `cf6d32a`. The frozen plan hash is
`995f6304bb0b0fe6e9ce03d185100b2493c7f1476a095e64a05bd0595ac27535`.
The ACES output root is:

```text
/scratch/user/u.yx126462/phase_b/review-deadline-v1
```

The result snapshot below was made after the first two visits and before the
newly submitted final fitting/finish jobs had been incorporated. Refresh every
count and endpoint from the final `summary.json`; never turn a missing row into a
failure or a numerical value.

## Decisions supported by the campaign

### Do not make repeated refitting a routine search branch

The `refit_only` arm shares the full arm's visit-zero model, then applies the
same frozen fitting protocol again without changing structure. Eleven finite
paired lineages showed negligible changes in validation NMSE. This is evidence
that another identical fit call is not a useful default branch under the tested
budgets and implementation.

The operational policy should therefore be simple:

1. perform the frozen fit and its bounded numerical qualification;
2. retain the best finite, independently replayable parameters;
3. if the result is poor after that budget, return qualified residual evidence
   to the proposer for scientific revision.

This conclusion is limited to the tested repeated-refit protocol. It does not
prove that more compute, a different optimizer, or a continuation preserving
all internal optimizer state can never improve a candidate. Those alternatives
are outside the frozen fitting protocol.

### Route scientifically plausible revisions instead of rejecting them

Several saved revision responses proposed scientifically meaningful changes,
such as delayed meal absorption, glucose-dependent absorption, a direct input
path, or an additional CSTR coupling. They were submitted through a contract
that allowed only a new function over an existing interaction's exact source
set. The runtime consequently classified the replies as invalid rather than
routing them to source or topology revision.

The next proposer contract should permit exactly one typed scientific action:

| Action | Meaning | Runtime response |
| --- | --- | --- |
| `keep` | Retain the fitted model | Close the revision visit |
| `revise_function` | Change one interaction while retaining its sources | Validate with the restricted expression grammar and refit |
| `revise_sources` | Add, remove, or replace a source for one interaction | Return to interaction construction, regenerate its function, validate, and refit |
| `add_pathway` | Add one scientifically justified interaction or process | Return to the bounded topology stage, complete affected functions, validate, and refit |
| `add_state` | Add one scientifically justified persistent state and its required paths | Return to bounded topology construction and complete all affected contracts before refitting |

The action schema should require a scientific rationale tied to the supplied
brief and residual evidence. Deterministic checks must still block unsafe
expressions, undefined symbols, unavailable channels, target leakage, and
incomplete equations. A well-formed request for a new source or pathway should
be routed to the appropriate earlier stage rather than forced into a
function-only reply and rejected. This retains proposer flexibility while
keeping each visit bounded and auditable.

### The 98,304 limit is an experiment limit, not a GPT-OSS limit

The served model has a 32,768-token context and the campaign requests at most
8,192 output tokens. Later visits override the cumulative client budget to
98,304, which was chosen as three times 32,768 for three allowed attempts. It is
imposed in `review_deadline_pipeline.py`; it is not a hard GPT-OSS output limit.

There is also an accounting mismatch. Before a call, the client conservatively
reserves:

```text
serialized request bytes + maximum output tokens
```

After a successful call, it replaces that charge with provider-reported total
tokens. The pre-call and post-call quantities therefore use different units.
Large feedback packets can consume enough byte-based reservation that the
98,304 budget admits only one attempt, even though the protocol declares three
attempts. The observed `provider_budget_exhausted` rows must be described as a
campaign budget/accounting limitation, not evidence that GPT-OSS exhausted its
context or could not repair the model.

For future runs, use one unit throughout. Prefer provider/tokenizer prompt-token
counts plus the output allowance. Preflight the largest request and reject a
configuration whose cumulative budget cannot fund the declared attempts. Do not
retroactively change the pinned campaign.

## Current campaign snapshot

The plan contains 44 lineages, three visits each, for 132 planned rows. Before
the final fitting stage, the summary reported:

| Status | Rows | Interpretation |
| --- | ---: | --- |
| `complete` | 48 | A visit produced a retained result |
| `shared_full_round_zero` | 12 | Refit control reuses the corresponding full visit zero |
| `no_change` | 3 | Proposer explicitly retained the parent |
| `provider_budget_exhausted` | 18 | No further provider request fit the client budget |
| `fit_failed` | 3 | The frozen numerical stage did not yield a retained fit |
| `closed_lineage` | 4 | A prior terminal result prevented another visit |
| `missing` | 44 | The final stage had not yet been incorporated |

Use the last nonmissing retained endpoint through visit one for the following
interim descriptive values:

| Arm | Finite lineages | Median validation NMSE | Mean validation NMSE | Range |
| --- | ---: | ---: | ---: | ---: |
| Full | 11 | 0.5817 | 0.5986 | 0.1344–0.9941 |
| Brief only | 10 | 0.6591 | 9645.58 | 0.0532–96430.40 |
| No latent | 4 | 0.6816 | 0.6901 | 0.6020–0.7954 |
| No specification | 4 | 1.0077 | 0.9969 | 0.5685–1.4036 |

The brief-only mean is dominated by one extreme failure and should not be used
without the median, individual points, and sample size. Among nine complete
full/brief matched pairs, full was better in four and brief-only in five. This
small, variable comparison is inconclusive.

The specification ablation has a clearer mechanistic result. Across the four
matched anonymous-system and alien-device lineages, the full arm produced four
of four graph-certified models, while `no_spec` produced one of four. Report
this together with output NMSE: graph certification and predictive accuracy
measure different outcomes.

## Controlled demonstration

The controlled system is:

```text
z' = -a z + b u_j
y' = z - y
```

Only `y` is observed. Training and validation make `u1` and `u2` identical, so
both driver choices fit observational data equally well. A held intervention
decouples the drivers after both models are frozen.

| Chosen driver | Train NMSE | Validation NMSE | Held-intervention NMSE |
| --- | ---: | ---: | ---: |
| Correct `u1` | 1.0222e-16 | 8.9670e-17 | 1.2163e-18 |
| Wrong `u2` | 1.0222e-16 | 8.9670e-17 | 2.9922 |

Safe claim: a public scientific requirement can distinguish observationally
indistinguishable models and select the interventionally correct mechanism in
this controlled example. This does not show that the LLM autonomously discovers
the correct mechanism on every benchmark.

## Baseline inventory

The current twenty-four-hour campaign did not rerun external baselines. Earlier
experiments and execution paths exist, but their numbers must not be mixed with
this campaign until benchmark cells, data release, prediction semantics,
selection policy, and metric definition have been matched.

| Method | Existing evidence | Status for the current six-cell campaign | Primary source |
| --- | --- | --- | --- |
| SINDy | Historical hard-tier and intervention summaries; full Phase-B runner exists | Not rerun | `docs/PROPOSER_AND_BASELINE_EXECUTION.md`; `artifacts/rebuttal/analysis/secondary_tables/baseline_ablation_hard.csv` |
| PySR | Pilot artifacts and a full Phase-B runner exist | Not rerun; the local checkout does not establish completion of the full matrix | `docs/PROPOSER_AND_BASELINE_EXECUTION.md`; `artifacts/baselines/pysr/` |
| D3 | Historical hard-tier and held-intervention evaluations exist | Not rerun | `docs/INTERVENTION_RESULTS.md`; `artifacts/rebuttal/interventions/phase_a3_synthesis_v1/method_rollups.csv` |
| LLM-feature-SINDy | Historical hard-tier and held-intervention evaluations exist | Not rerun | `docs/INTERVENTION_RESULTS.md`; `artifacts/rebuttal/interventions/phase_a3_synthesis_v1/method_rollups.csv` |
| GPT-5.6 Sol raw-data agent | A two-cell, three-repetition fitted-model pilot and full-matrix protocol are documented | Not rerun; verify the exact completed remote outputs before quoting a full-matrix count | `docs/RAW_DATA_AGENT_BASELINE.md`; `docs/PHASE_B_FINAL_EVALUATION.md` |
| Autoformalism ablations | Historical no-judge/no-latent intervention results plus the current brief/no-latent/no-spec/refit controls | Current ablations are part of this campaign | `docs/INTERVENTION_RESULTS.md`; final campaign `summary.json` |

The older hard-tier table is already rendered in
`artifacts/rebuttal/analysis/secondary_tables/baseline_ablation_hard.md`. It
includes SINDy, D3, LLM-feature-SINDy, no-judge, and the older full method. PySR
is absent from that consolidated table. Treat it as historical evidence, not as
the result table for the new six-cell campaign.

The common final-evaluation adapter is documented in
`docs/PHASE_B_FINAL_EVALUATION.md`. Use it when a matched comparison is needed;
do not copy values from experiments with different prediction semantics into a
single unqualified table.

## Files for table and figure generation

After the finish job completes, use these campaign files:

```text
/scratch/user/u.yx126462/phase_b/review-deadline-v1/summary.json
/scratch/user/u.yx126462/phase_b/review-deadline-v1/rounds.csv
/scratch/user/u.yx126462/phase_b/review-deadline-v1/SUMMARY.md
```

Use the controlled-demo files under the same output root; locate the exact file
from the frozen summary rather than guessing its name. The final summary is the
authority for paths and hashes.

Historical table-ready sources are:

```text
artifacts/rebuttal/analysis/secondary_tables/baseline_ablation_hard.csv
artifacts/rebuttal/interventions/phase_a3_synthesis_v1/method_rollups.csv
artifacts/rebuttal/interventions/phase_a3_synthesis_v1/paired_comparisons.csv
```

## Requested tables

1. **Campaign coverage.** Rows are arms; columns are planned lineages,
   completed finite endpoints, fit failures, budget exhaustion, closed
   lineages, and missing endpoints. Keep failure categories separate.
2. **Six-cell endpoint table.** One row per cell, seed, and arm. Show the last
   retained visit, train NMSE, validation NMSE, graph certification, and terminal
   status. Include all points rather than only successful examples.
3. **Few-visit progress.** For each full lineage, show retained train and
   validation NMSE at visits zero, one, and two. A retained parent after a
   rejected/no-change proposal is a flat endpoint, not a new fit.
4. **Paired ablations.** Compare full with brief-only by cell and seed; compare
   full with no-latent and no-spec only where the corresponding ablation was
   planned. Report pair counts and direction of change.
5. **Refit control.** Show visit-zero and repeated-refit NMSE and their absolute
   and relative changes. This table supports removing repeated identical refits
   from the default controller.
6. **Controlled demonstration.** Use the three metrics in the controlled-demo
   table above and state the observational non-identifiability explicitly.
7. **Historical baseline context.** If included, label it historical and retain
   the original suite, metric, and sample-count columns. Do not combine it with
   current-campaign NMSE as if it were matched.

## Requested figures

1. **Visit trajectories.** Small multiples by cell; x-axis is visit, y-axis is
   validation NMSE on a log scale, color is arm, and individual seeds remain
   visible. Use markers for no-change, budget exhaustion, fit failure, and
   missing status.
2. **Full versus brief-only paired plot.** One point per matched cell/seed with
   log-scaled axes and an identity line. Annotate the extreme brief-only point
   rather than clipping it silently.
3. **Specification and certification.** For the four matched full/no-spec
   lineages, show validation NMSE beside graph-certification outcome. Do not
   collapse certification into the error metric.
4. **Refit delta plot.** Plot repeated-refit validation NMSE minus the shared
   full visit-zero NMSE for every finite pair, with a zero reference line.
5. **Controlled intervention demonstration.** Grouped bars for observational
   validation NMSE and held-intervention NMSE for the two driver choices. A
   log scale or a broken axis is necessary because the observational errors are
   near machine precision.

## Analysis rules

- Pair only identical cell and seed values.
- Use the retained validation score for search outcomes. Training residuals may
  guide revision but are not validation results.
- Never use test data for proposal generation or model selection.
- Report medians, individual points, ranges or interquartile ranges, and sample
  counts. Means alone are misleading in this small heavy-tailed sample.
- Do not impute missing or failed endpoints.
- Treat `provider_budget_exhausted`, `fit_failed`, `closed_lineage`, and
  `missing` as distinct outcomes.
- Use log axes only for positive NMSE values and show how exact zeros are
  handled.
- Do not run significance tests on these two-seed descriptive comparisons.
- Finite numerical replay means the prediction is reproducible at the retained
  parameters; it does not mean the prediction fits well or is scientifically
  correct.
- A graph-certified result satisfies the implemented public mechanism checks;
  it is not proof that the latent mechanism is unique or true.

## Safe current conclusions

- The pipeline executes end to end on multiple public families and preserves
  train/validation separation.
- Another identical refit did not materially improve the finite paired models
  in this campaign.
- Function-only repair is too narrow for residual evidence that points to a
  missing source, pathway, or state; typed routing should preserve those
  scientific proposals.
- The current revision-attempt failure rate is confounded by an imposed and
  inconsistently accounted cumulative budget.
- The controlled example demonstrates why public scientific requirements can
  matter when observational trajectories alone do not identify the driver.
- The small, incomplete campaign does not establish broad superiority over
  SINDy, PySR, D3, or GPT-5.6 Sol.

## Claims to avoid

- Do not call 98,304 a GPT-OSS context or output limit.
- Do not interpret `provider_budget_exhausted` as scientific inability.
- Do not claim that an unchanged second fit proves the model structure is
  wrong; it shows that the tested repeat did not help.
- Do not call historical baseline values a matched comparison with the new
  campaign.
- Do not claim convergence from three visits.
- Do not describe held-intervention results as having been available to the
  proposer or selector.
