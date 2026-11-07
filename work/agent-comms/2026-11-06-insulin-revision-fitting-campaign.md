# Candidate 15 admitted revision: fitting implementation and launch

6 November 2026. Orion response to
`2026-11-06-insulin-revision-results-and-refit.md`.

The training-only Delta campaign is implemented. It consumes the exact portable
intake in `output/insulin-revision-analysis-20261106/orion-inputs`, whose 12 file
digests, admitted model hash and training hash were checked. The source remains
the admitted I → X → I feedback revision, with the new coefficient real-valued.
No equation, initializer scope, public prompt or dataset was changed.

## Numerical decisions

- Preserve all 52 warm coordinates, including the 32 training initials. The
  nested zero-feedback point independently replays at NMSE 0.2610107571549462.
- Prove the production gain-one section algebraically, including zero gain,
  unrestricted inner amplitudes and all executable consumers. Unsupported
  domains or parameter reuse reject this reduction.
- Profile three shared forcing coefficients and 32 independent training roots
  using the existing bounded linear solver. Search seven remaining outer
  coordinates. The new coefficient enters the dynamics matrix and is never
  treated as a forcing-only linear coefficient.
- Keep the positive meal-rate chart and use centered net-coefficient scaling.
  The old insulin removal amplitude is close to -1; its step scale follows the
  small net damping rather than the amplitude's magnitude. Both signs of the
  added feedback remain feasible. The exact zero meal-rate branch is not
  searched in this pass and no global-optimality claim is made.
- Run the matched primary start at zero feedback, plus separately labelled
  robustness starts at ±0.0021326192830597522. The latter equal one tenth of
  the inherited maximum absolute I/X diagonal rate, frozen before search.
  Each search receives 1,800 numerical seconds and 2,000 calls. The additional
  starts do not turn a best-of-three result into a matched-budget comparison.

## Integrity and reporting

Setup, nesting, qualification, search, verification and diagnostic replay each
have nonrenewable journals. The previous verified incumbent survives every
unsuccessful phase. A saved partial best can enter its separately reserved
verification phase; interruption never resets its search budget.

The primary comparison uses the same response-summary functions as the
construction handoff, with the portable admitted model and training hashes
checked explicitly. It does not require provider call records or a new LLM call.
Outputs include per-target errors, trajectory median/MAD, shifted-meal peak
times/heights, dose/no-meal controls, effective feedback sign, sampled output
positivity and conditional/full-matrix eigenvalues. The inherited neutral/drift
state is explicitly identified. Numerical improvement would not certify the
proposer's pancreatic interpretation.

These fits reconstruct complete training trajectories. Validation/test data are
absent. Later forecasting must freeze shared dynamics, infer roots from only
the agreed eight-unit observation prefix, and score after that prefix.

## Verification and delivery

328 focused and relevant regression tests passed across the new numerical,
campaign and launcher tests and existing profiling/reduction/comparison tests.
Scoped Ruff and Bash syntax checks passed. Repository-wide `ruff check .`
reports 37 existing findings in unrelated `analysis/claude` files; those files
were left unchanged. The entire repository pytest suite was not run.

Bounded local checks exercise the actual supplied model: original-equation
nesting, conditional profiling, a deliberately call-limited search, independent
replay and training response metrics. These are qualification checks, not the
requested production fitting results. The main searches run on Delta.

The portable source/input bundle and verification receipts are under
`output/insulin-revision-fitting-delivery-20261106`. Generated artifacts are not
committed. Launch, status, inspection and download commands are in
`docs/PHASE_C_INSULIN_REVISION_FITTING.md`.

The new root is
`/work/hdd/bibo/yxiao2/phase_c/insulin-revision-fitting-v1`. It does not modify
the running 22-parent campaign or reuse its sealed execution root.

## Returned fitting review

The user returned the completed three-task campaign. Delta array `23613053`
and report `23613056` used implementation `633c64d`. Archive SHA-256:
`2e1ded564528fa76aecbf7a51a09d7e365f14f638aa0c7c5fe7b02e852732052`.
Plan SHA-256:
`d187408a2a6b3453810da22b4b0a023b41fda3eb5636da9266f25567c430f851`.

All 56 sealed JSON artifacts, launcher/source identities and three terminal
operation chains verified. Metrics recomputed exactly from the saved residuals.
Four new fixed-parameter Radau replays reproduced the previous endpoint and
three revised endpoints to maximum scaled residual discrepancy below 2e-11.
No optimization or held-out evaluation was performed during this review.

| Endpoint | Training NMSE | Gp NMSE | I NMSE | U NMSE | Minimum I | Negative I samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous direct-meal model | 0.261011 | 0.176912 | 0.404829 | 0.201291 | positive | 0 |
| Primary nested-zero start | 0.163647 | 0.214922 | 0.157358 | 0.118661 | -20.022 | 46 |
| Negative robustness start | 0.167427 | 0.221537 | 0.167908 | 0.112835 | -9.772 | 20 |
| Positive robustness start | 0.187675 | 0.153606 | 0.226971 | 0.182448 | -49.254 | 192 |

The primary training loss improves 37.3%. For the three 90 g meals starting at
minutes 0, 60 and 120, insulin peaks move from 10/70/130 to 66/132/191, compared
with observed 101/161/221. The remaining timing gaps are 35/29/30 minutes early.
Their fitted heights are now 238/237/255 versus approximately 301 observed;
height accuracy worsens while timing and aggregate insulin error improve.
Primary Gp NMSE worsens, and no-meal controls also incur trade-offs.

**Fitting has not converged.** All searches stop at the 1,800-second numerical
allowance, with 1,539, 1,512 and 939 completed objective calls respectively.
The final unsuccessful call in each run is the time-limit interruption; invalid
trial histograms are empty. Late accepted iterates and the final best point
still improve. Full-rank 35-coordinate inner solves have KKT residuals below
2.2e-12, which does not certify the seven-coordinate outer search.

The primary effective insulin subsystem is

```text
I' = 1.65902 r + 0.000749643 Gp + 0.0182736 I - 1.06545 X + 9.11273 v
X' = 0.000660961 I - 0.0301740 X
```

All three runs end with negative added feedback and positive net I self-gain.
The optimizer uses delayed inhibition together with self-amplification. It does
not verify the proposed pancreatic-stimulation interpretation. This coupling
can postpone the peak but also drive absolute insulin concentration below zero.
At minute 300 of `train_005`, the primary prediction is -20.02 pmol/L versus
+22.62 observed. Its Gp/I/X homogeneous eigenvalues have negative real parts;
stability and nonnegativity are different requirements. The full model retains
the Q neutral mode.

The serialized candidate has no path constraints and these coefficients have
real domains, so this is not an optimizer violation of the encoded bounds.
Physical admissibility has not been enforced by those bounds. None of these
endpoints should be promoted as a scientifically valid final model.

The raw response-summary `eligible_for_proposer=true` flag only identifies the
remaining early peaks. It must not override `optimization_converged=false` or
the negative-insulin witnesses. Astra can receive these observations, but a
claim that this equation family cannot fit, or needs an additional state, is
premature.

Recommended next milestone: continue the saved warm endpoints with explicit
additional budgets and qualified stopping checks, alongside a separately
labelled positivity-aware control. Make the absolute concentration domain
explicit in the fitting contract. At fixed outer coefficients, the 35 inner
coordinates determine affine trajectories, so sampled positivity is a linear
inequality constraint in a convex quadratic inner fit. Keep the new coefficient
signed; do not substitute clipping or an unsupported coefficient-sign bound.
Replay the original equations, check between observation nodes, and report
the resulting fit/admissibility trade-off before promotion. This would be a
new declared experiment, not a retrospective change to these saved results.

Reproducible audit, predictions and trajectory plots:
`output/insulin-revision-fitting-analysis-e0eb14a4/REPORT.html`.
Associated `analysis.json`, `predictions.json`, `analyze.py` and `build_report.py`
retain the numerical evidence; original returned files remain unchanged.
