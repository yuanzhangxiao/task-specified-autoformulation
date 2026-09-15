# Final targeted joint/alternating comparison

The user explicitly reopened the difficult reference case for **one paired test**
after the nine-arm [closeout](FITTER_CLOSEOUT_2026-09-14.md). This experiment does
not change production fitter defaults or the running pre-fitting comparisons.
There are exactly two new reference fits; no automatic follow-up or sweep.

## Question and controlled comparison

Does solving the conditionally linear weights as a separate bounded least-squares
block help on the difficult 48-parameter model? Both arms reuse the **ordinary
start**, equations, observations and known per-trajectory initial conditions of
`free_shapes/collocation_exact` in `fitter-final-alternatives-v1`. No previous fitted
estimates or generating parameter values initialize either arm.

- **J+S:** jointly optimize state values at collocation nodes and all 48 parameters,
  then refine the physical parameters with full-trajectory forward sensitivities.
- **A+S:** alternate a bounded linear least-squares update of 26 weights/rates with
  a nonlinear update of node values and the 22 shape parameters. Follow this with
  exactly the same screening and sensitivity refinement as J+S.

The nonlinear block uses IPOPT with an exact Hessian. The linear block uses SciPy
BVLS with column scaling; parameter bounds and shared coefficients are preserved.
The restricted symbolic compiler certifies affinity before enabling that block.
No unconstrained normal-equation inverse is used. Rank, conditioning, projected
gradient and objective change are recorded at each linear step. A step is retained
only if finite, within the original domains, sufficiently stationary for that
linear problem, and non-increasing in the common objective.

Both arms minimize this **same penalized objective** during initialization:

\[
L_C=\tfrac12\left[\frac1{N_y}\sum (\widehat v_{01}-v_{01})^2/\sigma_y^2
+\frac{10^4}{N_D}\sum (D_i/s_i)^2\right].
\]

Here D is the two-stage Radau IIA integration defect, s is a frozen state unit,
and sigma_y is the training standard deviation. This differs from the previous
exact-constraint C implementation: soft defects make the conditional weight
update a bounded linear least-squares problem. **Only J versus A isolates the
block-update strategy.** The earlier C results are contextual, not a matched
ablation. A small surrogate objective is not evidence of a good physical rollout.

## Scaling audit and boundary conditions

The common transformation z -> c z, latent outer amplitudes -> c times their
values, and latent tanh scales -> their values/c is a symmetry of the equation
family if hidden initial conditions transform too. Biases, latent linear
couplings, decay rates, input shape and output outer amplitudes stay unchanged.
The audit infers these exponents from the restricted expression tree, including
tied parameters and tanh products, and verifies RHS covariance at c=0.5 and 2.
Independent per-state rescaling is not assumed; shared coefficients can forbid it.

The retained reference preparations include fixed **nonzero hidden initials**.
Those physical boundaries anchor the common scale; moving just the parameters
along this direction need not preserve predictions. The preflight lists the
anchors and measures the output change along a small scale-direction perturbation
with physical boundaries held fixed. An unavailable rollout makes this last
measurement unavailable, not a false symmetry conclusion or a block on starting
collocation. No parameter is fixed as a gauge in either arm.

Both arms use a single saved, hashed set of node guesses. A five-second total
warmup tries the ordinary rollout; observed/constant guesses supply nodes when
it is unavailable. The hidden state unit is max(1, RMS of those hidden guesses),
shared across hidden states; the output unit is the training output standard
deviation. Parameter units are max(1, |ordinary value|). They remain fixed through
optimization. Sensitivity refinement uses the same parameter units as `x_scale`.
This is numerical scaling, not an identification constraint.

The preflight also transforms equations, parameters, state units **and boundary
coordinates together**, checking that dimensionless collocation residuals stay
unchanged. This audit is a units transformation; the actual fits retain the
original physical boundaries and domains.

## Mesh, stages and budgets

The mesh planner targets 24,000 variables (nodes plus parameters), with a minimum
of 120 intervals
per trajectory and preserves sampled-input corners. These are targets, not hard
caps when mandatory input knots require more nodes. Every original observed
sample enters the objective through the same quadratic collocation interpolant.
Sampled input interpolation is unchanged. Initial states are fixed boundary
values in this reference diagnostic, not additional optimization variables.

| Stage | Budget per arm | Outcome retained |
| --- | ---: | --- |
| J or A initializer, including graph setup | 1,200 s | Best finite in-domain surrogate iterate |
| Screen ordinary and saved initializer parameters on full training | 240 s; at most two calls, 60 s per point | Best successful physical training rollout |
| Full-trajectory sensitivity refinement | 3,360 s; at most 118 residual calls, 180 s per point | Best successful physical training rollout |
| Fresh Radau/BDF replay, each train/validation split | 240 s per solver/split | Independent integration results |

Thus each fit has at most 4,800 fitting seconds and 120 training residual calls,
plus replay. Early stage completion does not enlarge a later stage. J has at most
1,000 IPOPT iterations; A has at most 40 cycles of 25 nonlinear iterations and one
linear update. Both also have the same hard initializer time cap. A may stop when
its objective progress is small. Neither native convergence nor exhausting an
iteration limit establishes recovery.

The sensitivity stage starts from the best screened training point and optimizes
only physical parameters; hidden trajectories are obtained by integration. It
does not optimize independent nodes. `ftol` is disabled; `xtol` and `gtol` are
1e-8. A failed initial sensitivity evaluation is reported explicitly; later
failed trial rollouts are rejected without supplying a fabricated Jacobian.
There is no extra random restart. If sensitivity fails, the valid screened
incumbent survives. If screening has no feasible point, refinement is explicitly
unavailable. Both failures remain in the comparison denominator.

Each stage runs in a supervised child process, so an unresponsive native solver
cannot consume later stages' budget. Atomic checkpoints retain initializer
arrays and the best physical parameters. On deterministic resume, completed
stages are reused; an interrupted stage is charged its whole original budget
and its saved incumbent is retained. It is never silently given fresh fitting
time. Code, runtime, source problem, common guesses and checkpoint identities are
checked. Partial scheduler submission is recorded rather than resubmitted.

## Interpretation and scope

Scores are output trajectory **v01 only**, normalized with the training output
variance. No hidden trajectory errors enter fitting, screening or reporting.
Training chooses the final parameters; validation is used only for final scoring.
Fresh BDF/Radau replay must agree within 1e-4 in maximum prediction difference
scaled by the training standard deviation. Agreement establishes numerical
consistency, not accurate prediction or scientific correctness. Report native
solver outcomes, surrogate objective, physical NMSE and replay separately.

Strict/good/practical bands require both train and validation NMSE <= 1e-4/0.01/0.1
and passing replay. These are descriptive, not certification. The result is one
ordinary-start pair on an already opened difficult case, not a general estimate
of an algorithm's success rate. The restricted diagnostic supports expanded
smooth polynomial/tanh equations with identity v01 output and fixed boundaries;
it is not a replacement for the generic proposer-facing fitter.

Known hidden initial conditions are reference information and remain isolated
from proposer/judge feedback. No test data, hidden trajectory labels, LLM calls,
new model selection, or ACES GPU jobs are involved.

## Run and review

Use a clean checkout pinned to the implementation commit. The default source is
`/work/hdd/bibo/yxiao2/phase_b/fitter-final-alternatives-v1`; the default output is
`/work/hdd/bibo/yxiao2/phase_b/fitter-scaled-alternating-v1`.

```bash
bash scripts/hpc/submit_scaled_alternating_delta.sh
```

The submission creates a 16 GB preflight, then two 64 GB CPU jobs (one CPU each,
up to two concurrent, two-hour scheduler limit), followed by a lightweight report.
The preflight runs the scale/affinity audits and a separate small two-state native
smoke test. Fits depend on its successful completion.

After completion, from the same checkout:

```bash
python3 scripts/run_scaled_alternating.py summarize \
  --output /work/hdd/bibo/yxiao2/phase_b/fitter-scaled-alternating-v1
cat /work/hdd/bibo/yxiao2/phase_b/fitter-scaled-alternating-v1/SUMMARY.md
```

The summary needs only Python's standard library. Detailed diagnostic files are
`gate/result.json`, `common.json`, and `results/task_000` (J) /
`results/task_001` (A). Each task includes `initializer/native/checkpoint.json`,
`screen/result.json`, `refinement/result.json`, and `selected.json`, plus stage
logs and immutable replay results. Review those results before any further action.
