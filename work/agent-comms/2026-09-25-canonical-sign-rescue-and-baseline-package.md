# Canonical R9/R2 assisted rescue and baseline package audit

The requested experiment is implemented as `dalla-canonical-sign-rescue-1`.
See `docs/DALLA_CANONICAL_RESCUE.md` for decisions, budgets, provenance and launch
commands. The ordinary pipeline and previous R4 experiment retain their scope.

## Frozen candidates

- Brief-only, canonical obfuscated T1-easy, seed 1, global R9, original inventory
  model `4d630f90e574895a0046686a7e0f1380bf99cb1dc154e399bc9ff33f360d660d`.
- Full, canonical named T1-easy, seed 0, global R2, original inventory model
  `076b95891e38358230e28131f2aad822bab81e74c18cdbf925b59d495e27ee98`.

Each original request/vector branches to an unchanged control and a sign-repaired
arm. R9's six direct gains receive +,+,−,−,+,−. R2's constrained tissue-return
term changes from − to +. R2's remaining direct signs are audited and preserved.
The initializers, inner laws and unselected terms remain unchanged. Each changed
gain receives the ordinary role start, even when its declaration was already
nonnegative and only the outer operator changed. This avoids retaining R2's
effectively zero tissue gain solely because its declaration was unchanged.

Anonymous channel decoding and canonical tissue-return interpretation are
reference-informed analyst hypotheses. This is **not autonomous scientific
discovery**. No reference coefficient values, replacement gastric equations,
baseline metrics, intervention observations, or test trajectories enter the
numerical run. Prior exploratory results motivated the hypothesis; the study
is not a blinded confirmation.

Four CPU elements reuse the rescue/refit/paired-pruning allocation; at most two
run concurrently. Fitting and pre-pruning retention use training, while pruning
acceptance uses validation. Both final arms must be kept and replayed under the
same frozen interventions. Never choose a new fit by its intervention result.

## Attached package audit

Inputs: `Downloads/manifest.json`, `Downloads/models.jsonl`, and
`Downloads/models_and_metrics.csv`, received on September 25.
The CSV and JSONL agree on every identity and equation. All 480 identities are
unique: 40 cells, four methods, three repetitions.

| Method | Planned | Adapted model entries | Missing |
|---|---:|---:|---:|
| SINDy | 120 | 102 | 18 |
| PySR | 120 | 102 | 18 |
| GPT-5.6 Sol | 120 | 120 | 0 |
| D3 | 120 | 119 | 1 |

PySR's 18 missing entries are recorded timeouts. SINDy's missing entries comprise
12 timeouts and six failed runs. D3 is missing canonical obfuscated T2-easy,
repetition 0. Neither missing nor failed entries should be silently removed from
the planned denominator.

All 12 method/repetition entries are present in each of the two canonical
T1-easy cells and the perturbed named T1-easy cell used for R4.
Coverage was checked without using target scores to select a model.

The attached export contains only state-equation summaries. It omits full
candidate processes, observation mappings, initializers, parameter vectors,
validation context and execution semantics. It is therefore insufficient for
replaying all methods. In particular, D3 equations must retain their declared
discrete-increment interpretation, rather than being passed through an ODE
solver. The complete adapted subjects exist at the two roots in the manifest:

```
/work/hdd/bibo/yxiao2/phase_b/external-baseline-evaluation-v1/adapted/frozen_evaluation_subjects.jsonl
/work/hdd/bibo/yxiao2/phase_b/external-baseline-d3-test-v2/adapted/frozen_evaluation_subjects.jsonl
```

Use SINDy/PySR/Sol from the first and D3 from the second, following the manifest's
explicit D3 supersession. These subjects contain the fitted parameterization
and full candidate. Downloading those files avoids any new baseline training.
The package's `test_data_opened=false` describes the packaging operation; it does
not mean its already-computed score fields are validation scores.

The local coverage audit is saved (not committed) under
`artifacts/baseline-package-coverage-2026-09-25/coverage.json`. Source hashes and
missing identities are recorded there. Generated model packets and transfer
archives remain outside git. Concurrent changes to the general results packager
are owned by the other task and were not edited as part of this implementation.
