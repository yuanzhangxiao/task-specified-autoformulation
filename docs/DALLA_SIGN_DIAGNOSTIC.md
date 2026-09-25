# Assisted R4 sign diagnostic, with no GPU dependency

The user authorized this CPU-only diagnostic while the automatic GPT-OSS sign
review remains queued. Its protocol is `dalla-sign-diagnostic-1`, and every
export identifies `assistant_specified_under_user_authorization`. It does not
fabricate an assessor response, a provider-call receipt or an autonomous success.
This is an exploratory, post-hoc diagnostic of a selected historical model.

## Frozen decisions

`configs/dalla_sign_diagnostic_r4_v1.json` binds the original rescued
`full_perturbed_r4` result, exact request and public context by SHA256. The
decisions concern only isolated outer gains, retaining all inner expressions,
states and initialization rules. The already saved data and starting vectors
are reused; no new benchmark observations or reference model are imported.

| Existing target contribution | Decision | Basis |
|---|---|---|
| Meal filter `Y` | Positive outer gain | Explicit interpretation of the existing positive-gain meal filter as meal appearance |
| `EGP` | Positive outer gain | Publicly disclosed glucose production |
| `Uii` | Negative outer gain | Publicly disclosed glucose utilization |
| `E` | Negative outer gain | Publicly disclosed renal excretion |
| `Gt` | Unrestricted, unchanged | An observed tissue mass does not specify a flow direction |
| `Gp` self-term | Negative outer gain | Explicit dissipative self-clearance hypothesis |

The meal and self-clearance choices are modeling assumptions, not logical
consequences uniquely forced by the task text. In particular, a nonnegative
meal gain does not certify that the learned latent initial state is nonnegative
or that a complete intervention response is monotone. Existing signed latent
initialization policies remain unchanged. The public source/sink descriptions
support the other fixed directions; this runtime checks decision consistency,
not scientific truth. Citation text is advisory.

The runtime reuses the existing sign adapter and compatible-seed policy. Fixed
outer signs use nonnegative magnitude parameters. Changed declarations receive
the same role-based starts used by the automatic sign-repair path; the saved
wrong-signed coefficients are not silently converted to absolute values. The
initialization plan and compatible parameters are preserved. A separate unchanged
arm retains the original request and full parameter vector. `warm_start_audit`
records every reset and reuse.

## Numerical work and timing

One CPU array has two independent elements:

- index 0: repaired model;
- index 1: unchanged control.

Each gets one CPU, 16 GB and a 2h15 allocation, and runs the same seed replay,
rescue refit and paired pruning experiment used by the automatic path. Each
arm has at most three fits; total numerical budgets are unchanged. Both tasks
may run concurrently, subject to scheduler resources. There is no GPU job,
provider request, container download, separate preparation dependency or new
fitter. The report depends on termination of the CPU array.

Parameters, seed-versus-refit retention, and pruning contribution ranking use
training data. The inherited pruning acceptance rule compares validation NMSE
against a paired unpruned baseline with a frozen tolerance. Thus validation is
used for development selection, not solely for untouched evaluation. The
configuration prose and exported limitation in commit `d28c408` incorrectly
described selection as training-only; this is a metadata correction, not a
change to fitting or selection. Existing frozen results remain unchanged and
do not need rerunning. Use their original pinned checkout for resume; the
corrected source/configuration has a different identity.

Freezing the small input packet on the login node performs validation only;
benchmark integration and optimization occur in CPU jobs. Every worker verifies
code, decisions, source packet, runtime and paired plan identities before fitting.
Completed stages are reused. As in the underlying rescue protocol, an interrupted
started fit consumes that attempt; resume does not reset its numerical budget.

The existing automatic sign-review jobs and output roots are untouched. This
diagnostic has a distinct purpose, so it may run alongside the automatic review.
Run this CPU diagnostic on one cluster, not duplicated on both.

## Upload and submit on ACES

The provided `dalla-sign-diagnostic-COMMIT.tar.gz` includes the pinned code and
the original public rescue packet under `inputs/rescue-models.json`. Generated
packets and archives are not committed. Upload it to
`/scratch/group/p.nairr260351.000/u.yx126462`, then replace `COMMIT` below:

```bash
bash <<'BASH'
set -euo pipefail
AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
AF_PACKAGE=dalla-sign-diagnostic-COMMIT
mkdir -p "$AF_GROUP/repos"
tar -xzf "$AF_GROUP/$AF_PACKAGE.tar.gz" -C "$AF_GROUP/repos"
export AF_OUTPUT_ROOT="$AF_GROUP/dalla-sign-diagnostic-r4-v1"
bash "$AF_GROUP/repos/$AF_PACKAGE/scripts/hpc/launch_dalla_sign_diagnostic.sh" aces
BASH
```

The launcher defaults to the known ACES Python environment and account. Overrides
are `AF_PYTHON` and `AF_ACCOUNT`. No Hugging Face cache or SIF is needed. The
submission record contains only `fit` and `report`. On a scheduler timeout,
inspect the saved command and accounting before verified `--adopt fit=JOB_ID` or
`--adopt report=JOB_ID`; never delete receipts to force resubmission.

The same archive can instead run on Delta: extract it under
`/projects/bibo/yxiao2/repos`, export
`AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/dalla-sign-diagnostic-r4-v1`,
and invoke the launcher with `delta`. It defaults to
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python` and `bibo-delta-cpu`
(`AF_CPU_ACCOUNT` override). It uses the `projects&work` filesystem constraint.

## Inspect progress and return results

On ACES:

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-sign-diagnostic-r4-v1
AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" -X --format=JobID%22,JobName%28,State%22,ExitCode,Elapsed
```

After the report job finishes:

```bash
jq '{status,decision_source,expected,recorded,sign_integrity,rows}' "$AF_ROOT/summary.json"
```

To generate a partial report as soon as either arm finishes, without starting or
repeating a fit (set the archive path using the supplied commit):

```bash
AF_REPO=/scratch/group/p.nairr260351.000/u.yx126462/repos/dalla-sign-diagnostic-COMMIT
module load GCCcore/13.2.0 Python/3.11.5
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$AF_REPO/src:$AF_REPO" \
  /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python \
  "$AF_REPO/scripts/dalla_sign_diagnostic.py" report --root "$AF_ROOT"
```

Download **the top-level `models.json` and `summary.json`**. Their decision
provenance accompanies the fitted equations and parameters. We can compare the
saved GPU review separately when it completes, without overwriting either result.
Intervention rollouts follow endpoint freezing, with parameters held fixed; they
are not performed by these jobs or used to select the fitted endpoint.

## Local verification

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_dalla_sign_diagnostic.py tests/test_dalla_rescue.py \
  tests/test_dalla_sign_repair.py tests/test_directional_sign_review.py tests/test_sign_review.py
PYTHONPATH=src:. .venv/bin/python scripts/smoke_dalla_sign_diagnostic.py
```

The smoke uses synthetic data and real fitting, makes no LLM calls, and checks
partial reporting, both arms and exact resume. No benchmark fit runs locally.
