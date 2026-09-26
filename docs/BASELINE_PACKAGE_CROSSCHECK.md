# Independent external-baseline table audit

Run from the project root:

```bash
python scripts/audit_experiments_results_package.py \
  --package /path/to/results-package-v1.tar.gz \
  --output output/results-package-v1-crosscheck
```

The command produces `REPORT.md`, `audit.json`, and a complete LaTeX table
fragment, `BASELINE_TABLE.tex` (requires `booktabs`). It reads only the supplied
archive and the local roster/configuration. It performs no inference, fitting,
equation execution, or trajectory replay. The source archive is unchanged.

The auditor parses `NINE_CASES` from `build_experiments_table_results.py` as
literal data, checks equality with `final_component_campaign_v1.json`, and
computes the statistics independently. It does not import the existing table
builder. It requires the complete planned roster of three repetitions for each
external method/case, including explicit rows for missing or failed runs.

For NMSE, it takes the median over all planned repetitions within each case,
then the median and unscaled MAD over the nine case medians. Confirmed failed
predictions rank as `+infinity` **before either median**. The evidence must be
`target_status=failed` or a missing score with `terminal_status` equal to
`failed`/`timed_out`; an unknown evaluation is not relabeled as a failure.
Unknown predictions leave the aggregate unavailable pending resolution.
The internal infinity sentinel is written as the string `"+inf"` in valid JSON.
An infinite median is displayed as **Failed** and its MAD is undefined.
Graph and complexity use their recorded evidence, with availability reported
separately. This implements the revised rule in `EXPERIMENTS_DRAFT_NOTES.md`.
Large finite NMSEs remain observations; they are not clipped as outliers.

Prediction and graph-assessment availability are independent. An exported NMSE
can exist even when the runtime contract blocks the graph endpoint. The auditor
retains that NMSE and reports the graph score as unavailable for that run. It
does not infer that an unavailable graph assessment is either a pass or a fail.
The graph column uses `mechanism_compliance`, never `mechanism_coverage`, which
is based on annotations. A macro median of 100% does not mean every case or run
satisfies all obligations.

The supplied v1 export does not contain per-obligation unresolved counts or
token usage. Accordingly, its graph percentage cannot fill the draft table's
entire **P / U** field, and the token column remains unavailable. The generated
table labels the available graph endpoint explicitly and adds separate graph
coverage. It does not substitute it for the independently implemented fitted
equation/activity assessment. The existing `evaluate_frozen_subject` entry point
calls the graph checker on the candidate rather than substituting all fitted
parameters first.

CSV/JSONL fields, request identities, planned repetitions, and manifest totals
are checked. Payload SHA-256s are computed for reproducibility. The supplied
manifest lists hashes of upstream evaluation files that are absent from the
archive; these are provenance references, not independently verified payload
checksums. Its `test_data_opened=false` describes the packaging operation, which
reads previously written final-evaluation metrics.

D3's recorded native discrete-increment scores are retained. The older supplied
archive lacks per-subject execution semantics, fitted parameter values, and
trajectory arrays, so it does not support an independent numerical replay.
The current packager exports fuller model records, complete saved public graph
evidence, and runtime rejection reasons. It also hashes its two payloads. This
recovers evidence already on disk; it does not run the stronger assessment.

For the supplied archive, the corrected NMSE macro median [MAD] values are
SINDy `2.4197498357479286 [0.28373868439420535]`, PySR
`2.3313604665455183 [0.829431917937252]`, D3
`2.302477485484474 [1.4624632802723798]`, and GPT-5.6 Sol
`0.6665992675462672 [0.5241191790458177]`. SINDy and PySR each have three
confirmed unsuccessful predictions among 27 planned runs. PySR's T2-hard case
has a failed median (two budget timeouts); SINDy's CSTR case has three failures.
The other three headline medians/MADs happen to remain unchanged.

## What 100% graph median means

D3 has graph scores of 1 in 10/17 assessed runs, 0.5 in three, and 0 in four;
10/27 runs were unassessed because the runtime contract blocked that endpoint.
Five of nine **available-score case medians** equal 1, producing a macro median
of 1 and MAD 0. GPT-5.6 Sol has 23/27 scores of 1 and four of 0; eight of nine
case medians equal 1. These are not universal passes. Zero scores can include
unresolved requirements; the old flattened archive cannot separate that from
failed predicates.

The graph builder reads syntactic dependencies in state/process/output
expressions. `evaluate_frozen_subject` does not substitute fitted parameter
values before this check. Consequently, a syntactic path can remain even with
a zero fitted coefficient or cancelling terms. General signs, conservation,
nonlinear feedback, and correct response behavior are not certified by this
graph fraction. Complexity of an expression is not evidence of a mechanism.
For example, the legacy alien spec lists `nonlinear_feedback` as a tag alias;
that label does not invoke a nonlinear-feedback predicate. The named T1
easy/hard graph specs each require a meal-input-to-`Gp` path. T2 additionally
checks for a latent-state witness on the insulin pathway, but does not measure
the fitted pathway's delay, direction or strength.

The existing stronger deterministic tool is documented in
`DETERMINISTIC_MECHANISM_ASSESSMENT.md`. Report **fitted-equation requirement
compliance** as pass/(pass+fail+unresolved), alongside the unresolved fraction,
and keep **fitted pathway activity** and **response behavior** separate. The
current reviewed configuration binds the original six easy cases; three
additional T1/T2 cases require public-prompt bindings before a nine-case
comparison. Its nonlinear-feedback predicate is available, but is not an
obligation in the six easy-case bindings. Do not invent new requirements from
hidden ground truth or because a baseline scored 100%. Its numerical response
assessment currently uses development trajectories, not test trajectories.

## Export the missing evidence on Delta

Use the updated `scripts/package_phase_b_results.py` from this checkout. This
stdlib-only command reads the two evaluation roots named in the supplied
manifest. It performs no optimization, LLM calls, rollouts, or trajectory reads:

```bash
python3 scripts/package_phase_b_results.py \
  --evaluation sept=/work/hdd/bibo/yxiao2/phase_b/external-baseline-evaluation-v1 \
  --evaluation d3=/work/hdd/bibo/yxiao2/phase_b/external-baseline-d3-test-v2 \
  --supersede sept=d3_native_no_tools \
  --out /work/hdd/bibo/yxiao2/phase_b/results-package-v2

tar -czf /work/hdd/bibo/yxiao2/phase_b/results-package-v2.tar.gz \
  -C /work/hdd/bibo/yxiao2/phase_b results-package-v2
```

The new `models.jsonl` includes `model.parameterization`, initializers and
observation mappings in `model.candidate`, `model.validation_context`,
`execution_semantics`, `public_mechanism`, and `runtime_diagnostics`. Missing
saved evidence remains null. These enable per-requirement inspection and
preparation of a stronger assessment; no new mechanism scores are claimed by
the export itself. D3 must keep its native discrete-increment semantics.

Checks:

```bash
pytest -q tests/test_audit_experiments_results_package.py \
  tests/test_experiments_table_results.py tests/test_package_phase_b_results.py
ruff check scripts/audit_experiments_results_package.py \
  tests/test_audit_experiments_results_package.py
```

Generated results stay outside version control. The audit makes no changes to
the experiment section, Orion's case study, or any campaign's selection policy.
