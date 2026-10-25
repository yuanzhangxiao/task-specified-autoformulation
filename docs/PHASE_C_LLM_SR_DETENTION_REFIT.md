# LLM-SR detention models as fitting controls

Date: 2026-10-25. Protocol: `llm-sr-detention-refit-1`.

## Question and scope

Can our numerical fitter recover the low training errors already demonstrated
by two fixed LLM-SR detention equations, when their coefficients are perturbed
or initialized without copying the supplied values?

The preceding instantaneous-control campaign **replayed** supplied coefficients;
it did not fit them. Independent Radau and DOP853 rollouts gave training NMSEs
of approximately 0.0001283 and 0.0002329 for native tasks 14 and 15. Maximum
prediction differences, divided by training target scale, were below 8e-8.
These results establish useful attainable reference errors for the continuous
equations. They do not certify exact equivalence to the native discrete-time
Python program, or unique correct coefficients.

Both detention models are included because of their predefined numerical-control
role. The two CSTR transcriptions are retained in input provenance but excluded
from this refit roster: their supplied-value continuous rollouts were not
accurate controls. This pilot is separate from constructed-model evaluation and
does not promote LLM-SR structures into Autoformalism results.

There is one observed depth state, initialized from its permitted time-zero
reading for each event. There are no latent initial values to estimate here.
The six training events are fitted jointly with shared coefficients. No later
target observations drive the rollouts. No validation or test data are loaded.

## Fixed experiment

Each model gets five predetermined starts:

1. Supplied coefficients.
2. Two reproducible perturbed starts.
3. Two generic starts based on public training scales and declared domains.

All ten tasks use the same fitting settings and budgets. A perturbed or generic
task never receives the supplied solution as an incumbent, fallback, or
additional optimization start. Reference accuracy is used for reporting only.
The working accuracy threshold is training NMSE <= 0.001; it is not a
coefficient-accuracy or scientific-validity threshold.
This band is looser than the supplied endpoints. Report the actual NMSE and
its ratio to the supplied reference before claiming comparable accuracy.

The equations and their retained branches are fixed. Task 14 has eight adjustable
coefficients; task 15 has nine. Native constants and omitted branches remain
fixed. In particular, a signed coefficient is not changed to a positive-only
coefficient because its sign appears scientifically questionable. Such a change
would be a mechanism repair, outside this optimization control.

Finite numerical working bounds use public training depth and input scales.
They are experimental search domains, not bounds established by the public
scientific specification. Contact with a bound limits what can be concluded
about an optimum over the full equation family.

## Numerical procedure

For each start:

1. Independently integrate its supplied initial coefficient vector with Radau
   and DOP853 and record accuracy and solver agreement. A high error alone does
   not prevent fitting.
2. Fit by bounded direct rollout least squares. Use the existing optimizer with
   fixed physical coordinate units, guarded relative scaling, invalid-trial
   handling and saved improving checkpoints. Qualify local numerical derivatives
   using different local perturbations and tighter integration, rather than
   declaring a derivative reliable by comparing it to itself.
3. Independently verify the best available fitted checkpoint. Retain the better
   independently verified endpoint from this task's initial and fitted vectors.
4. Diagnose that retained endpoint: numerical derivative reliability, active
   bounds, local sensitivities, piecewise-function limitations and known exact
   parameter ambiguities. These diagnostics do not certify a global optimum.

The task reserves 600 seconds for optimization, 120 seconds for initial
verification, 120 seconds for endpoint verification and 120 seconds for retained
endpoint diagnosis. Setup and diagnostic overhead are recorded; these allowances
are not all described as optimizer time. The scheduler allocates one CPU, 8 GB
and 30 minutes per task, with two tasks concurrent. The report has a separate
15-minute CPU allowance. There are no GPU requests or LLM calls.

The initial derivative qualification may use up to 240 seconds **inside** the
600-second fitting allowance. Its first check receives that complete window;
tighter-precision retry uses only the remaining time. A representative supplied
endpoint needed about 115 seconds for a complete check locally, so prematurely
splitting a short check budget would prevent useful fitting. The remote timing
and generic-start performance remain to be measured.

The right-hand sides contain nonlinear depth responses. Their coefficients are
not an exact linear output block under rollout. Consequently this pilot uses
direct rollout fitting and the applicable numerical improvements; it does not
force an inapplicable bounded linear projection onto these equations.

## Exact ambiguity in task 15

The original parameter names should not be mistaken for independently
identifiable quantities. Write `T=dt`, `b=k_gw-k_loss`, `d=k_gw*kappa`, and
`M=h_max+T*k_gw`. On the preserved positive-capacity branch, before its final
dry-state and post-step guards, the derivative can be written as

```
min(T * (r - outlet(h) + b + d*h), M - h + T*d*h).
```

The final guards preserve this dependence. Four declared parameters
`k_gw, k_loss, kappa, h_max` therefore enter through three combinations. A local
null direction, when `k_gw != 0`, is

```
(delta k_gw, delta k_loss, delta kappa, delta h_max)
    = (1, 1, -kappa/k_gw, -T).
```

All nine coefficients cannot be independently identified. The report records
this algebraic limitation even if a numerical SVD spuriously appears full rank.
Useful equivalent coordinates are `T`, `T*C`, `T*Q_max`, `beta`, `eps_s`,
`T*b`, `T*d` and `M`. Additional ambiguity remains possible if some branches are
not excited. In this transcription, `dt` multiplies the vector field; it is
**not** the adaptive numerical solver's integration step.

This first pilot retains the original coordinates and reports the ambiguity.
It does not fix an arbitrary coefficient and silently shrink the feasible
equation family. An exact reduced-coordinate optimizer would additionally need
a certified feasible lifting through the original bounds and branches.

## Interpretation

Report every start, with initial/final error, verification status, termination
reason, cost and parameter limitations. Do not only report the best of five.

- Recovery from generic starts supports optimization robustness on these two
  equations, not robustness on latent-state or other nonlinear models.
- Failure to recover the known attainable error leaves an optimization issue
  under the tested budgets/domains; it does not establish that the equations
  cannot fit the observations.
- Lower loss with different coefficients is not necessarily a mistake. Supplied
  LLM-SR coefficients are references, not ground truth.
- Low error alone does not establish mechanism compliance. These transcriptions
  include gains/losses whose interpretation can conflict with the public
  negligible-unmeasured-gain/loss assumption.
- Neither sampled inactive guards nor local derivative agreement certify global
  smoothness, global monotonicity, or full-family optimality.

## Implementation and reproducibility

- `src/autoformalism/fitting/llm_sr_detention_refit.py`: bounded direct rollout,
  starts, derivative qualification and endpoint diagnostics.
- `src/autoformalism/research/llm_sr_detention_refit_campaign.py`: frozen roster,
  authenticated historical evidence, nonrenewable operation budgets and report.
- `scripts/run_llm_sr_detention_refit.py`: prepare, submit, run, status, report,
  inspection archive and execution identity.
- `scripts/hpc/start_llm_sr_detention_refit_delta.sh` and worker: portable Delta
  entry points using the dedicated `AF_LLMSR_DETENTION_REFIT_ROOT` variable.

The portable input includes the predecessor plan and all four replay receipts,
so checking provenance does not depend on the original remote experiment
directory. Source, data, starts, domains, runtime and launcher identity are
sealed. Re-running submission reuses confirmed scheduler receipts. An ambiguous
submission is stopped for reconciliation instead of submitting duplicates.
Interrupted stages retain checkpoint evidence and consume their reserved
allowance; retrying cannot reset their budget. Reporting and inspection do not
perform fitting.

After extracting the delivered source bundle on Delta:

```bash
export AF_LLMSR_DETENTION_REFIT_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-refit-v1
bash scripts/hpc/start_llm_sr_detention_refit_delta.sh run
```

Use the same bundle and root for subsequent inspection:

```bash
bash scripts/hpc/start_llm_sr_detention_refit_delta.sh status
bash scripts/hpc/start_llm_sr_detention_refit_delta.sh inspect
```

The archive is written beside the campaign root as
`llm-sr-detention-refit-v1-inspection.tar.gz`. Download it from a local terminal;
do not run a workstation-destination `scp` command inside the Delta shell.
