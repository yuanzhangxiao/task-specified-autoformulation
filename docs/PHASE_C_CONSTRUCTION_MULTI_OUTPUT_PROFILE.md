# Construction fitting: multi-output profiling and precision probes

This opt-in pilot follows `construction-precision-reduction-1`. The returned
archive (`39c73fdafd05c1618d89ca943c01172073497d95583325acaf0ae8fee608b279`)
contains eight independently verified retained endpoints. Six searches ran, five
improved slightly, and none established optimization adequacy. Model 6's retained
training NMSE is 1.0426742273380412. Small-step termination and accurate numerical
rollouts have not established that these equations reached their best fit.

## Model 6: solve the affine coordinates together

The frozen model has three measured outputs and one latent memory state. After
certified aggregation of private coefficients, its equations can be written as

```text
G' = a_m * meal + a_E * EGP - U / tau_G
I' = c_q * insulin_input + c_m * meal - c_I * I + c_M * M
U' = c_U * Uii + M - lambda_U * U
M' = k_act * I - lambda_M * M
```

This is a change of optimizer coordinates, not a scientific revision. In
particular, `c_q >= 1` and `c_U >= 1` are retained, positive rates retain their
domains, and the two signed coefficients remain signed because their frozen
declarations permit it. Every new coordinate vector must lift to valid original
parameters. The certificate must check the entire assembled model and
initialization graph; a matching fragment of an equation is insufficient.

Fix the six coordinates `(tau_G, c_I, c_M, lambda_U, k_act, lambda_M)`. The
trajectories of all three outputs are then jointly affine in five shared forcing
coefficients and the 16 trajectory-specific memory initial values. Stacking the
same normalized training residuals gives

\[
r(\eta,\beta)=A(\eta)\beta+b(\eta),\qquad
\beta^*(\eta)\in\arg\min_{\ell\leq\beta\leq u}
\|A(\eta)\beta+b(\eta)\|_2^2.
\]

The inner bounded least-squares solve handles 21 coordinates together. The outer
optimizer searches six coordinates, using the derivative of the profiled
residual when its numerical and active-set checks permit it. The direct control
optimizes the same 27-coordinate family jointly. Both start from the same saved
incumbent and use the same observations, normalization and independent
original-equation verification. No new latent state, nonlinear response law or
scientific sign constraint is introduced.
The basis propagation evolves `G`, `I`, `U` and `M` together with their declared
causal initializers. Beyond the permitted initial observations, measured target
trajectories enter only the residual. Thus this is exact coupled-rollout
profiling, with the declared auxiliary/input schedules supplying the forcing.

The inner optimum is conditional on the six outer coordinates. It is not a
certificate of global optimality, original-parameter identifiability, or a better
scientific model. Rank deficiency or an ambiguous active set must remain visible.
An independently verified inner-only improvement can be retained even if the
profile derivative check blocks the outer search.

## Targeted precision checks

Models 0, 3, 4, 5, 6 and 7 receive separate fixed-endpoint checks. These address
stationarity uncertainty, unresolved rank or derivative comparisons; they do not
optimize parameters or change a model-selection decision.
They retain their parent diagnostic coordinate systems. In particular, model 6's
precision task probes its saved 34-coordinate chart, while its two fitting arms
use the new 27-coordinate chart. Their Jacobian/rank diagnostics are therefore
not interchangeable.

Increasing precision only inside the matrix exponential leaves roundoff in
generator construction, parameter lifting, initialization, propagation and
residual subtraction. The new diagnostic evaluates that complete calculation
with multiprecision arithmetic through a restricted expression interpreter.
It compares local finite differences at two shrinking step sizes and two
precision levels against the float64 analytic derivative at the identical saved
endpoint. Bounds and the local relative-step cap still apply. Large perturbations
are not used as evidence about local derivatives.

Columns are chosen by a deterministic rule from the saved diagnostic, with a
finite column cap and deadline. Where the saved diagnostic contains a shrinking
step pair, the probe reuses those local steps, capped by the current locality
rule; it falls back to the declared initial rule only when no pair is available.
This targets the observed numerical floor rather than testing only larger steps
where truncation error may dominate. Reports preserve the selected columns, steps,
precision levels, discrepancies, incomplete checks and compute costs. Numerical
agreement is diagnostic evidence, not a statistical confidence level. These
probes do not automatically override the existing qualification gate or promote
a stronger parameter-confidence conclusion.

## Experiment scope and budgets

The eight-task roster consists of two model-6 fitting arms and six precision
checks. It is not another continuation of all eight models. All inputs and
historical receipts are authenticated; a new root records new costs separately.
Interrupted operations retain their consumed or reserved allowance, and restarting
the launcher does not renew a fitting budget.

Both fitting arms have a 600-second optimization allowance. The profiled arm
reserves 120 seconds for the initial conditional solve and 480 seconds for its
derivative check and outer search; unused initial allowance does not enlarge the
outer budget. Independent verification and final diagnostics have separate
allowances. The profiled arm's additional inner-endpoint verification is reported
as extra diagnostic overhead, so equal optimization budgets do not imply equal
total runtime.

The pilot uses CPU jobs on Delta, at most two array tasks concurrently. It has
no GPU or LLM calls, no validation or test access, and no changes to production
fitting defaults. Schema preparation avoids rerunning the full repository test
suite on the cluster; local validation and its source hashes accompany the
bundle.

## Representative saved endpoint

Before the outer-optimization experiment, the actual model-6 endpoint was compiled
and checked locally. Certification gave 27 aggregate coordinates, six outer
coordinates and 21 inner coordinates. At the frozen outer values, the bounded
inner solve took about 1.14 seconds and changed training NMSE from
1.0426742273380412 to 1.0426727339918398. Its normalized design had rank 21 and its
normalized KKT residual was approximately `7.01e-14`, below the declared
`1.23e-8` numerical tolerance.

An independent Radau replay of the lifted original equations returned NMSE
1.0426727339918513; the maximum normalized residual discrepancy was `3.37e-10`.
The solve and replay together took about 20 seconds. This supports the
implementation of the conditional solve, not successful completion of the outer
optimization. The improvement is tiny; the Delta comparison is needed to learn
whether profiling helps optimize the remaining dynamics.

A separate fixed-endpoint probe tested model 7's `af_process_gain_1_0` at the
same saved steps (`7.42e-7` and `3.71e-7`). Float64 residual finite differences
disagreed with the analytic derivative by RMS `2.54e-5` and `1.78e-5`; full
multiprecision reduced both discrepancies to about `9.00e-12`. The two
multiprecision levels agreed, and shrinking the step changed the high-precision
derivative by RMS `1.20e-16`. This is strong evidence of a residual-evaluation
precision problem for that column, rather than evidence of a wrong derivative or
a kink. It does not certify other columns or stationarity. The 15-call diagnostic
took 13.79 seconds and performed no fitting. Working precision includes 88
declared guard digits in addition to the 40/70 precision levels, to limit loss
during bounded scaling and squaring.

## Local validation

The final focused suite passed 255 tests, including the profiling certificate,
bounded solve, precision interpreter, campaign resume and submission behavior,
and adjacent numerical components. The full repository suite in a test-only
snapshot reported 9,301 passed, nine skipped and two failures: existing numerical
tests exceeded their wall-clock limits under parallel execution. Both passed
when rerun in isolation. This is not an uninterrupted green full-suite result.
Two small final hardening changes were covered by the final focused suite.

`ruff check .` passed on the tracked source plus the pilot files. The working
checkout also contains 37 lint findings in unrelated untracked `analysis/claude`
files, which were left unchanged. Native preparation, status and inspection
packaging were exercised without contacting Slurm. The delivery includes a
`VALIDATION.json` receipt with source hashes and these limitations.

## Delta commands

The delivery archive contains committed source, authenticated training-only
inputs, a validation receipt and the offline `mpmath` dependency. Extract it into
a new directory under `/projects/bibo/yxiao2/repos`, verify `CONTENTS.sha256`, and
run from that extracted directory:

```bash
export AF_PLATFORM=delta
export AF_PROFILE_PILOT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-profile-pilot-v1
export AF_ACCOUNT=bibo-delta-cpu
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
bash scripts/hpc/start_construction_profile_pilot.sh run
```

The array requests one CPU and 8 GB per task, a one-hour task limit and concurrency
two; the dependent report has a 15-minute limit. The fitting stages have bounded
allowances within that scheduler limit. Status and packaging use the same root:

```bash
bash scripts/hpc/start_construction_profile_pilot.sh status
bash scripts/hpc/start_construction_profile_pilot.sh inspect
```

`inspect` writes `inspection.tar.gz` in the experiment root. Upload that archive
for review. The launcher authenticates source/runtime identities on resume. It
does not silently reuse a different source tree, overwrite a completed result or
resubmit a job after an ambiguous scheduler response. Inspect the recorded
submission intent before resolving such a response.
