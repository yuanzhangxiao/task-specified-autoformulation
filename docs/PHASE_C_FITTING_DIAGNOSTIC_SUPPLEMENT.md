# M33: portable evidence verification and CSTR diagnostic completion

M33 addresses the two open issues in the
[M32 review](PHASE_C_FITTING_M32_RESULTS_2026-10-13.md). It performs no fitting,
proposal generation or model selection. Equations, fitted parameters, initial
values, training data, perturbation points and diagnostic acceptance thresholds
stay fixed. Production fitting defaults and benchmark data are unchanged.

## Portable verification

Saved artifact hashes, candidate/point identities, stage identities, budgets,
statuses and discrete conclusions remain exact. The verifier reads the saved
residuals and Jacobian and validates only explicitly named numerical summaries
within declared numerical tolerances. It checks singular values against matrix
backward-error tolerances and validates a weakest direction against the relevant
singular subspace; nearly null directions need not be identical between LAPACK
implementations. This does not turn the entire JSON packet into an approximate
comparison or loosen scientific derivative-acceptance thresholds.

The direction actually used to execute an experiment is an input, not a quantity
to replace while reconstructing its report. New diagnostics seal the full
perturbation plan before its first numerical call and record the numerical
environment, including BLAS/LAPACK configuration and thread settings. Archived
M32 diagnostics recover their executed directions from the saved assessment,
check those directions against the Jacobian, and verify exact point hashes.
An incomplete legacy experiment without recoverable directions cannot silently
invent new points during resume.

The M33 campaign audits all 23 M32 outcomes read-only, including the two
unavailable sources. It copies their necessary JSON evidence into an immutable
source snapshot so a downloaded review is self-contained. The original M32
campaign is never modified. Qualification reconstructs the original controls,
output-verification and preserved-accuracy gates; it never treats unavailable
models as successful fits.

## Linked CSTR supplement

CSTR already has accurate independent output replay. Its joint local parameter
assessment is incomplete because the aggregate perturbation allowance expired.
The saved experiment has 62 completed solver calls and one failed attempt;
21 solver calls remain if completed results are reused.

The supplement freezes the same model, fitted vector, data, step sizes, actual
weak direction and tolerances. It adds **1,200 seconds** of incremental evaluation
budget, with at most **120 seconds per call**. The original exhausted attempt and
its charge remain in the ledger. A separate identity authorizes one attempt at
each missing evaluation, including an explicitly new retry of the old failed
point. It reuses all completed evaluations without relabeling their identities.

Each new attempt is durably recorded. Resume reuses completed attempts and
charges interrupted attempts without automatically rerunning them or resetting
the allowance. Any subsequent supplement would need another explicit linked
identity and budget. Original, new and combined costs are reported separately.

The queue reservation is 45 minutes for one CPU and 16 GiB; this includes setup,
verification and filesystem overhead. It is not a 45-minute numerical allowance.
There are no GPU requests, LLM calls or validation/test observations.

The report distinguishes recorded work from derivative qualification. If all
21 calls finish but their numerical checks disagree, the parameter conclusion
still remains conservative. A local sensitivity pass is not global
identifiability, scientific correctness, or a calibrated probability of accuracy.

## Delta commands

The delivery is `phase-c-fitting-m33.tar.gz`. Upload it to
`/work/hdd/bibo/yxiao2/phase_c/`, then run:

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
tar -xzf "$AF_BASE/phase-c-fitting-m33.tar.gz" -C "$AF_BASE"
bash "$AF_BASE/phase-c-fitting-m33/scripts/hpc/submit_phase_c_diagnostic_supplement_delta.sh"
BASH
```

The defaults use the existing M32 source
`/work/hdd/bibo/yxiao2/phase_c/fitting-staged-feedback-v1` and write a **new** root
`/work/hdd/bibo/yxiao2/phase_c/fitting-diagnostic-supplement-v1`.
Set `AF_SUPPLEMENT_SOURCE` or `AF_SUPPLEMENT_ROOT` only if those locations differ.
Use the same extracted code, root and policy for resume. An ambiguous scheduler
reply stops submission for reconciliation instead of duplicating jobs.

After the jobs finish:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m33/scripts/hpc/inspect_phase_c_diagnostic_supplement_delta.sh
```

This validates the saved source and supplement, prints the compact result, and
writes a timestamped review archive containing both old and new evidence. The
printed path is the file to download. No additional work is submitted by the
inspection command.

## Local verification

The 51 new portable-verification, supplemental-accounting and campaign tests pass.
A wider 110-test piecewise/staged regression run passed before the final targeted
roundoff hardening and three additional campaign checks; the final 51-test run
covers those changes. All 21 available archived diagnostic packets validate, and
read-only reconstruction exactly reproduces the original 23-row M32 summary.
The actual CSTR preflight confirms 62 reusable completed evaluations and 21
remaining calls. CLI, Bash syntax, snapshot preparation and incomplete-report
smokes pass without integrating the archived models.

Changed-file Ruff and `git diff --check` pass. Repository-wide Ruff still reports
37 pre-existing findings in `analysis/claude`. Repository-wide pytest was started
and interrupted after 210 passes and three missing-Torch skips; a full-suite pass
is not claimed. The Delta prepare job reruns the 51 focused tests before allowing
the numerical supplement to start.

## Following this milestone

While the remote diagnostic is running, prepare the three items in the
[Vega handoff](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md): explicit measurement and
residual provenance; a bounded noise-aware fitting design; and a small controlled
study separating specification errors, data errors and fitting failures. These
are preparation for subsequent work. They must not alter this frozen diagnostic
or attribute a residual mismatch to a false specification without evidence.
