# Paired canonical T1 sign rescue

`dalla-canonical-sign-rescue-1` is a separate, assisted development diagnostic
on two preselected historical models. It does not change the ordinary proposer,
fitter, pruning rule, benchmarks, or the earlier R4 and automatic-review runs.

| Starting model | Cell | Seed | Intervention before fitting |
|---|---|---:|---|
| Brief-only R9 | T1 canonical obfuscated easy | 1 | Constrain the six direct target terms to meal +, production +, utilization −, excretion −, tissue return +, target self-feedback − |
| Full R2 | T1 canonical named easy | 0 | Correct the tissue-return assembly term from negative to positive; retain the other five existing target signs |

These are the original inventory checkpoints. R9 is not imported from its
previous all-positive automatic repair or from its subsequently pruned endpoint.
R2 was not in the original six-model rescue. The whole request, starting vector,
public context, and development data are frozen in a portable source packet.

## Scientific scope

The interpretation of the anonymous channels and the canonical tissue-return
direction is **reference-informed analyst input**. This experiment must be
reported as an assisted diagnostic, not autonomous mechanism recovery. No
reference coefficient values or replacement nonlinear laws are inserted.
The decisions and assumptions are visible in
`configs/dalla_canonical_rescue_v1.json` and in every published endpoint.

Only isolated, single-use outer numerator gains can be changed. Explicit bounds,
initialization dependencies, shared parameters and inner functions are protected.
An exactly identified already-nonnegative gain is eligible for this explicit
repair, which is what permits R2's wrong assembly sign to be corrected.
The automatic public-context sign reviewer retains its original scope.

Changed signs or parameter roles reset the affected magnitudes to the fitter's
ordinary role-based starts (0.1 for these gains). Fitted absolute values are not
used to invent positive starting magnitudes. Other parameters, including signed
latent initial values, are retained. A positive outer gain does not establish
nonnegative latent states or a monotone complete intervention response.

Four independent CPU tasks run: R9 repaired/control and R2 repaired/control.
Each arm receives the same `collocation-rescue-v1` allocation: initial-vector
replay, one rescue refit, then the existing paired one-deletion pruning check.
Parameters, seed retention and deletion ranking use training. Pruning acceptance
uses validation. No intervention outcomes, baseline scores or test data enter
the fitting or selection interface. Both endpoints are kept for later frozen
intervention replay; a successful fit is not a scientific certification.

## Submission

The supplied `dalla-canonical-rescue-COMMIT.tar.gz` contains pinned code and
`inputs/canonical-sources.json`. The generated data packet and archive are not
committed. Upload it to the ACES group directory, then run:

```bash
bash <<'BASH'
set -euo pipefail
AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
AF_PACKAGE=dalla-canonical-rescue-COMMIT
mkdir -p "$AF_GROUP/repos"
tar -xzf "$AF_GROUP/$AF_PACKAGE.tar.gz" -C "$AF_GROUP/repos"
export AF_OUTPUT_ROOT="$AF_GROUP/dalla-canonical-rescue-v1"
bash "$AF_GROUP/repos/$AF_PACKAGE/scripts/hpc/launch_dalla_canonical_rescue.sh" aces
BASH
```

There is no GPU or separate preparation dependency. Each array element requests
one CPU, 16 GB and 2h15; at most two run together. This is a scheduler ceiling,
not an expected duration. The report runs after the array terminates and can
report partial failures. Preparation validates the packet without numerical fits.

The same archive supports Delta instead: extract under
`/projects/bibo/yxiao2/repos`, set
`AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/dalla-canonical-rescue-v1`, and use
`delta` as the launcher argument. Run on one site. Existing Python/account
defaults match the R4 CPU diagnostic and can be overridden with `AF_PYTHON`,
`AF_ACCOUNT` (ACES), or `AF_CPU_ACCOUNT` (Delta).

On a submission timeout, preserve receipts, inspect the scheduler, and rerun the
same launcher with `--adopt fit=JOB_ID` or `--adopt report=JOB_ID` only for a
matching job. Scheduler command identity is verified. No blind resubmission is
performed. Numerical resume retains completed stages; an interrupted started
fit consumes its attempt rather than silently resetting its budget.

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-canonical-rescue-v1
AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" -X --format=JobID%22,JobName%26,State%20,ExitCode,Elapsed
jq '{status,decision_source,expected,recorded,sign_integrity,rows}' "$AF_ROOT/summary.json"
```

Download top-level `models.json` and `summary.json`. The reporting CLI also works
while fits run: `scripts/dalla_sign_diagnostic.py report --root "$AF_ROOT"` under
the pinned checkout's Python/PYTHONPATH. It detects the frozen protocol and does
not start a fit. Missing reports before the report job runs are expected.

## Verification and source reconstruction

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_dalla_canonical_rescue.py tests/test_dalla_sign_diagnostic.py \
  tests/test_dalla_rescue.py tests/test_sign_review.py
PYTHONPATH=src:. .venv/bin/python scripts/smoke_dalla_canonical_rescue.py
```

The smoke uses synthetic trajectories and real paired fits with exact resume.
The portable source packet is reconstructed by
`scripts/build_dalla_canonical_sources.py --inventory INVENTORY --public-plan PLAN
--output OUTPUT`: it selects the two fixed inventory IDs, verifies lowered models
and complete parameter vectors, checks public brief/target-contract identity,
and copies only training/validation cells. It neither fits nor imports scores.
The source-row hashes in the committed decision configuration prevent substituting
a different checkpoint or vector.
