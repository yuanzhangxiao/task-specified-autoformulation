# Failure recovery and status-integrity audit

## Scope

This audit reconciles the 486 prespecified core method cells against the
authoritative consolidated results, historical failure records, and numerical
failure sentinels. Family-study and experimental ablation cells are outside
this count. Test NMSE is used only to identify the documented `1e12` failure
sentinel; it is never used to select a model or recovery configuration.

## Reconciled inventory

| Item | Count | Interpretation |
|---|---:|---|
| Prespecified core cells | 486 | Seven methods with their predefined seed counts |
| Rows reported complete | 467 | Authoritative consolidation before integrity checks |
| Valid completed rows | 452 | Excludes 15 rows carrying the failure sentinel |
| Unresolved expected cells | 19 | 15 PySR, 3 SINDy, and 1 full-method cell |
| Invalid “complete” rows | 15 | LLM-feature-SINDy on obfuscated-original, all tiers/seeds |
| Historical failed records | 37 | Includes superseded attempts |
| Superseded historical failures | 19 | All repaired D3 cells |

## Root causes and decisions

### D3

The 19 historical D3 failures include structured-metadata ingestion,
structured-expression validation, truncated structured output, and no-valid-fit
failures. All 19 cells have authoritative successful V18 replacements. They
are retained as provenance but require no replay.

### SINDy and PySR

SINDy has three and PySR has fifteen explicit safe-rollout failures, all on
`obfuscated_original_case01`. These occurred after the structured-metadata
loader fix: every fixed support/expression considered by the prespecified
algorithm failed causal safe rollout. Repeating the identical configuration is
not a recovery. These are valid algorithm failures and must contribute to
completion-rate reporting rather than being silently dropped or assigned an
MSE sentinel.

### LLM-feature-SINDy

Fifteen rows were serialized as complete although their train, validation, and
test NMSEs are all `1e12`. The proposed feature call succeeded, but the final
equation failed rollout. The shared baseline result builder previously ignored
`failed_trajectories` from final test evaluation. It now raises an explicit
failure, so future runs cannot convert the numerical penalty into a successful
result.

These historical cells should be reclassified as failures in statistical
tables. Replaying their cached feature proposals can verify the corrected
status without an LLM call, but changing SINDy's library or threshold grid
would define a new baseline variant rather than repair the original result.

### Full method

`benchmark5/hard/seed1` is the only missing full-method cell. Four attempts
produced 31 unique cached structures. Across the 32 full-method round
checkpoints, 27 fits reached the wall-time limit before an optimizer evaluation
and five terminated after one evaluation with numerical failures. This is a
continuous-optimization failure, not absence of proposals.

The next recovery step is a development-only refit of these frozen structures:

1. use train and validation only;
2. compare scale-aware bounds and positive-parameter log transforms;
3. use fixed RK4 screening with bounded time per structure;
4. select a structure and numerical configuration by validation NMSE and fit
   success only;
5. freeze the choice before one final train-plus-validation refit and test
   evaluation.

No new proposer or judge calls are needed.

## Benchmark5 hard seed-1 recovery result

The zero-call recovery was completed after this audit:

1. deduplicate the failed attempts into 31 frozen structures;
2. screen each with both the historical declared-midpoint initialization and a
   scale-aware near-zero initialization, using one fixed-RK4 optimizer
   evaluation;
3. refine the three lowest-validation-NMSE structures for three evaluations;
4. freeze the lowest-validation-NMSE structure;
5. perform one train-plus-validation adaptive refit and one test evaluation.

| Endpoint | Recovered value |
|---|---:|
| Screening fits successful | 53/62 |
| Refined finalists successful | 3/3 |
| Selected validation NMSE | 0.003468 |
| Test NMSE | 0.007494 |
| Structural validity | 1.000 |
| Hidden affine-aligned NMSE | 0.6504 |
| Additive ODE terms | 8 |

The selected model contains separate `u01`-driven transformation and
`u03`-driven exchange latent states plus a direct `u02` term. Its test NMSE is
inside the range of the other four full-method Benchmark5-hard seeds
(0.006719–0.007632). This supports a numerical-initialization diagnosis rather
than an absent-structure diagnosis.

The result remains labeled as a **numerical recovery** in provenance. The
candidate had failed before historical judge evaluation, so it has no cached
judge score. The judge is advisory and validation NMSE determined this
selection, but the recovered cell should not be presented as an unqualified
historical end-to-end completion without this disclosure.

Recovery artifacts are stored under:

- `artifacts/rebuttal/benchmark5_hard_seed1_refit_v1`;
- `artifacts/rebuttal/benchmark5_hard_seed1_recovery_test_v1`.

## Status policy

- A final rollout with any failed trajectory is `failed`, never `complete`.
- The `1e12` numerical penalty is an internal failure sentinel, not a reportable
  observed MSE.
- Superseded attempts remain in provenance but do not count as current gaps.
- An algorithm that returns no safely rollable expression counts as an explicit
  method failure and remains in completion-rate denominators.
- Recovery hyperparameters are frozen using development splits only.

## Reproduction

```bash
python scripts/audit_failure_recovery.py \
  --runs artifacts/rebuttal/analysis/authoritative_runs.csv \
  --artifact-root artifacts/rebuttal/consolidated_inputs \
  --output-root artifacts/rebuttal/failure_audit_v1
```

Outputs include historical failures, unresolved cells, invalid complete rows,
a prioritized recovery manifest, a machine-readable summary, and a Markdown
report.

The recovery itself is reproducible with:

```bash
python scripts/refit_failed_candidate_pool.py \
  --data-root data_raw \
  --artifact-root artifacts/rebuttal/consolidated_inputs \
  --output-root artifacts/rebuttal/benchmark5_hard_seed1_refit_v1 \
  --screen-max-nfev 1 --screen-timeout-seconds 30 \
  --refine-top-k 3 --refine-max-nfev 3 --refine-timeout-seconds 180

python scripts/evaluate_failed_candidate_selection.py \
  --data-root data_raw \
  --frozen-manifest \
    artifacts/rebuttal/benchmark5_hard_seed1_refit_v1/frozen_development_selection.json \
  --output-root artifacts/rebuttal/benchmark5_hard_seed1_recovery_test_v1 \
  --max-nfev 1 --timeout-seconds 900
```
