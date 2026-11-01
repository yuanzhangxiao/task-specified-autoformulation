# Construction portfolio: qualified searches and timescale diagnostics

31 October 2026. Opt-in follow-up to
`PHASE_C_CONSTRUCTION_PORTFOLIO_CONTINUATION.md`. This is a numerical experiment,
not a change to benchmark inputs, construction prompts, or production selection.

## Frozen scope

The source is the complete five-model continuation under plan
`e92e3bf12f64431d6db74c812f849396747ed98bcb0a30ac02a60a47b3491ea1`, normally at
`/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-continuation-v1`.
Every source terminal is reconstructed from its operations and nested probes
before exporting a separate snapshot. Historical results and reservations are
never rewritten. Derived timing reconciliation follows the narrowly bounded
reporting repair; scientific fields remain exact.

The next snapshot carries compact, hash-pinned projections after full source
authentication. Shared arrays are stored once; raw perturbation arrays and
duplicated progress payloads stay in the source archive. Each exported operation
records its original receipt/value hashes and projection hash. Production
projections must also match the five reviewed digests compiled into this pilot.

| Model | Work in this milestone | Numerical limit |
| --- | --- | --- |
| 03 | New BDF/Radau replays, one paired Jacobian, four local direction checks | 600 s per prerequisite, 240 s per direction |
| 10/14 | Import complete qualification, verify start, bounded shared/root fitting | 1,800 s and at most 500 search calls |
| 12/15 | Separate endpoint diagnoses; ten timescale/nuisance profiles | 120 s and at most 120 calls per profile |

Each independently verified endpoint has its own additional verification and
diagnostic allowance. Numerical, checkpoint, and inclusive wall costs are
reported separately. The largest inclusive task reservation is 8,820 seconds;
workers request four hours, one CPU and 8 GB, with at most two concurrent tasks.
There are no GPU or LLM calls and no validation/test data access.

## Models 10 and 14: use the qualified objective

The prior run qualified all 29 directions at the frozen supplied start, with
BDF tolerances `rtol=1e-11`, `atol=1e-13`. Import authenticates the exact model,
training rows, chart, settings, baseline/Jacobian arrays and every coordinate
receipt. It reconstructs the full qualification and reuses those arrays only at
that exact point. It does not recharge the inherited numerical work.

The new optimizer uses BDF for both state-only residuals and augmented
sensitivities. A default method argument cannot silently change the objective
to another integrator. Away from the imported point, the existing lazy residual
and Jacobian evaluations, pairing checks, invalid-trial handling, coordinate
scaling and accepted-iterate checkpoints apply.

The start and candidate endpoints are independently replayed with original
equations using Radau. Saved search losses must agree with their replays. Only
a qualified, independently verified training improvement may replace the old
incumbent. An unavailable/worse search preserves the source fit; where no source
fit exists, the freshly verified supplied start remains an explicit fallback.
This establishes neither global optimality nor parameter identifiability.

## Model 03: targeted precision, not an enlarged perturbation

Only indices 8, 10, 19 and 22 are investigated: the selected insulin coefficient,
`k_Vmax`, and insulin-action initial roots for training trajectories 007/010.
Integration tightens to `rtol=1e-12`, `atol=1e-14`; the existing local stencil,
step rules, and derivative acceptance criteria remain unchanged.

Fresh BDF and independent original Radau replays precede one fresh paired
Jacobian. Each state-only perturbation has its own durable receipt inside its
coordinate reservation. Failed prerequisites block dependent work. The summary
always labels these four checks as partial diagnostics: passing all four cannot
combine with 24 checks made at older precision to authorize whole-model search.

## Models 12 and 15: interpret the fast response without changing equations

The previously fitted insulin and insulin-action self coefficients have become
very negative, while equilibrium response gains and predictions scarcely move.
For an eligible balance, write

\[
\dot z=a_z z+\sum_j b_j v_j+f_{\mathrm{fixed}}(t),\quad a_z<0,
\qquad \tau=-1/a_z,\quad g_j=\tau b_j.
\]

Then the same balance is

\[
\dot z={-z+\sum_j g_j v_j\over\tau}+f_{\mathrm{fixed}}(t).
\]

Exact algebraic consumer checks establish the coordinate transformation after
the existing product/unused/additive reductions. Fixed forcing terms and
constant offsets are preserved. Every lifted point must satisfy original
parameter domains; finite-precision cancellation is rejected rather than
silently changing a self coefficient. No arbitrary positive lower bound on
`tau` is added. This chart describes the conditional negative-self-coefficient
branch. Zero/positive self coefficients remain part of the original family.
These are self-relaxation times at fixed drivers, not eigenmode timescales or a
certificate of stability for the coupled system.

For each of two states, fix `tau` at its saved value and at 0.001, 0.1, 1 and 10
native time units. Every profile starts from the same saved model transformed
into the new chart. Record its fixed-timescale, otherwise unchanged control;
then allow all remaining parameters and trajectory initial roots to adjust.
This diagnostic uses bounded derivative-free Powell optimization, avoiding the
cost of a complete Jacobian qualification at each new profile point. It does
not inherit derivative qualification from a different point.

The best checkpoint at each profile is independently checked against original
ODE rollout. These are **finite-budget nuisance-refit upper envelopes**, not
certified minima, likelihood intervals or confidence percentages. A good fit
at a slower timescale can demonstrate compatibility within the tested budget;
a poor fit cannot prove that timescale impossible. A failure to improve with
120 seconds may simply be insufficient nuisance optimization. Profiles never
replace the main incumbent, and their computation costs remain explicit.

## Endpoint diagnostics and interpretation

Diagnose both the retained best point and the optimizer's last accepted point
after separate original-equation verification. Keep the source chart when
interpreting source search history. Accepted history is usable only when its
parameter dictionary equals the diagnosed endpoint exactly. Never attach the
last accepted point's convergence evidence to a different best checkpoint.

New search endpoints are scored only by the original full-training objective.
The report keeps historical prefix forecast metrics under
`source_training_prefix_forecast`; this milestone does not reestimate prefixes.
Any new model intended for forecasting needs a separate causal prefix replay
before making a predictive claim. Existing source prefix errors must not be
presented as errors of a newly fitted endpoint.

## Running on Delta

Use the delivered committed source bundle in a new extracted directory. Do not
change or re-use an earlier campaign's frozen launcher/code location. The
source results stay on Delta; the bundle contains code only.

```bash
export AF_PORTFOLIO_SEARCH_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-continuation-v1
export AF_PORTFOLIO_SEARCH_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-search-v1
bash scripts/hpc/start_construction_portfolio_search_delta.sh run
```

`prepare` authenticates the historical evidence and prints progress before its
potentially slow filesystem work. `run` then submits the five-task array and a
dependent report. Repeating a command reuses confirmed reservations and does
not renew numerical allowances. Ambiguous scheduler replies require checking
saved intents against Slurm; do not erase submission records or change roots
just to bypass an unconfirmed submission.

```bash
bash scripts/hpc/start_construction_portfolio_search_delta.sh status
bash scripts/hpc/start_construction_portfolio_search_delta.sh inspect
```

Inspection writes `$AF_PORTFOLIO_SEARCH_ROOT/inspection.tar.gz`. Reports and
inspection are receipt-only operations, so they can be repeated without fitting.
The report's operational `complete` status means the bounded task has terminal
evidence; `search_status`, precision coverage and verification fields state the
scientific outcome, which may remain unresolved.
