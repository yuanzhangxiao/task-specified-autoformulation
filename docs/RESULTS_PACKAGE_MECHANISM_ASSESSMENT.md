# Fitted equation assessment of the v2 baseline package

This handoff reuses the existing deterministic mechanism checker. It does not
produce a complete scientific-compliance score. The old graph column and this
fitted necessary-feature assessment must remain separate from numerical activity,
physical-sign/balance evidence, and predictive accuracy.

The CLI verifies the archive payload hashes, all CSV/JSONL fields, and the full
nine-case, four-method, three-repetition roster. It substitutes complete saved
parameter vectors before simplification and checking. D3 keeps native increments;
supplied auxiliary equations are excluded from causal evidence and restored only
for the native replay adapter. Missing legacy continuous execution fields are
resolved only for recognized source adapters, with that inference recorded.

The original six reviewed predicate bindings are unchanged. Canonical named
T1-hard adds the identical public meal-path requirement at its own pinned prompt
hash. **T2-easy and T2-hard remain explicitly unbound:** the delayed insulin-action
requirement needs an operational assessment that respects generated plasma insulin
and, for T2-hard, the unobserved disposal mechanism. Neither direct connectivity
nor output integration alone establishes that mechanism. No nine-case stronger
compliance score is emitted.

For the subsequent nine-case functional rubric, including delayed-action tests,
see [Stronger deterministic public-mechanism assessment](FUNCTIONAL_MECHANISM_ASSESSMENT.md).
That protocol is a separate endpoint; it does not retroactively rename the
necessary-equation scores below.

## Results on the supplied v2 archive

Archive SHA-256:
`f5145810dcff3bf7e386ab155e5e1a950c7de11c19b478d361422d88dcddeac1`.
Two payload hashes and 13,440 CSV/JSONL fields agree. Of the 108 planned runs,
103 have fitted models. The seven supported cases contain 81 assessable models,
three missing SINDy models, and 24 separate T2 rows without predicate bindings.

| Method | Confirmed necessary predicates, median [unscaled MAD] | All predicates pass / 21 planned runs |
| --- | ---: | ---: |
| SINDy | 0% [0 pp] | 0/21 |
| PySR | 0% [0 pp] | 0/21 |
| D3 | 100% [0 pp] | 19/21 |
| GPT-5.6 Sol agent | 100% [0 pp] | 21/21 |

Compute the fraction of confirmed predicates for each run, take its case median
over three repetitions, then the macro median and raw MAD across seven cases.
Unresolved and unassessed slots stay in the denominator as unconfirmed evidence;
they are not asserted to be false. SINDy's absent CSTR models are not omitted.
For interpretation, use the saved pass/fail/unresolved/unassessed counts as well.
These 100% medians do not certify correct direction, units, physical balances,
activity at fitted parameters, or the public mechanisms in their entirety.

Parameter-declaration violations are saved separately. They must not be used as
a common scientific score: declaration domains differ across methods, and D3's
native optimizer did not enforce all of its model declarations.

## Reproduce locally or on Delta

Use the project Python environment and the pinned checkout supplied with this
change. All script paths are absolute to avoid invoking them from a login home
directory by accident:

```bash
export PYTHONPATH="$AF_REPO_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_PACKAGE=/work/hdd/bibo/yxiao2/phase_b/results-package-v2.tar.gz
export AF_EVIDENCE=/work/hdd/bibo/yxiao2/phase_b/package-fitted-equations-v1

"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_results_package_mechanisms.py" \
  --package "$AF_PACKAGE" --output "$AF_EVIDENCE"
cat "$AF_EVIDENCE/REPORT.md"
```

Outputs are the sealed `models.json` replay bundle, `summary.json`, a readable
`REPORT.md`, and `assessment_config.json`. Repeating the same inputs reuses
identical sealed output. Use a new output root for changed inputs.

## Optional existing CPU activity/response assessment

The archive has no trajectory arrays. On Delta, read the public development-data
root from the existing evaluation receipt, then prepare the existing numerical
assessment. This opens train/validation only; it does not open held-out data or
feed evidence back into construction or selection. This still covers seven cases.

```bash
export AF_ASSESSMENT=/work/hdd/bibo/yxiao2/phase_b/package-mechanism-activity-v1
export AF_PUBLIC_ROOT="$("$AF_PYTHON" - <<'PY'
import json
from pathlib import Path
record = Path('/work/hdd/bibo/yxiao2/phase_b/external-baseline-evaluation-v1/execution_record.json')
print(json.loads(record.read_text())['public_data_root'])
PY
)"

"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_mechanisms.py" prepare \
  --bundle "$AF_EVIDENCE/models.json" \
  --config "$AF_EVIDENCE/assessment_config.json" \
  --public-root "$AF_PUBLIC_ROOT" --root "$AF_ASSESSMENT"

jq '.rows[] | select(.status != "ready") | {index,method,benchmark_id,error}' \
  "$AF_ASSESSMENT/plan.json"

AF_COUNT=$(jq '.rows | length' "$AF_ASSESSMENT/plan.json")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$AF_ASSESSMENT/logs"
sbatch --account=bibo-delta-cpu --partition=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=03:00:00 \
  --array="0-$((AF_COUNT-1))%8" --job-name=mechanism-activity \
  --output="$AF_ASSESSMENT/logs/%A_%a.out" \
  --error="$AF_ASSESSMENT/logs/%A_%a.err" --export=ALL \
  --wrap='"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_mechanisms.py" run --root "$AF_ASSESSMENT" --index "$SLURM_ARRAY_TASK_ID"'
```

After completion:

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/assess_mechanisms.py" report \
  --root "$AF_ASSESSMENT"
cat "$AF_ASSESSMENT/SUMMARY.md"
```

Use these existing probes as conditional activity evidence, not whole-system
physical counterfactuals. See `DETERMINISTIC_MECHANISM_ASSESSMENT.md` for exact
probe sizes, solver checks, descriptive response thresholds and resume policy.
Source prompt hashes absent from earlier runtime-invalid records are explicitly
marked unverified; preparation checks the configured public prompt and channel
roles rather than pretending the missing historical hash was recovered.

Verification: package CLI, native/continuous/zero-parameter failure fixtures,
checkpoint smoke, `pytest`, and scoped Ruff. Repository-wide Ruff currently
reports unrelated pre-existing findings under `analysis/claude`.
