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

For every endpoint separately, it takes the median over available repetitions
within each case, then the median and unscaled MAD over the nine case medians.
A missing whole case makes that endpoint's macro statistic **unavailable**.
Partial repetition coverage remains visible and does not silently become zero.
This reproduces the rule in `EXPERIMENTS_DRAFT_NOTES.md`. Large finite NMSEs remain
observations; they are not clipped or discarded as outliers.

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
The current packager can export fuller model records, but a calculation audit
does not manufacture those absent fields or reinterpret D3 as an ODE.

Checks:

```bash
pytest -q tests/test_audit_experiments_results_package.py \
  tests/test_experiments_table_results.py tests/test_package_phase_b_results.py
ruff check scripts/audit_experiments_results_package.py \
  tests/test_audit_experiments_results_package.py
```

Generated results stay outside version control. The audit makes no changes to
the experiment section, Orion's case study, or any campaign's selection policy.
