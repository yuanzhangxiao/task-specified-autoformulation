# Phase C fitting: plan and first qualification milestone

Agreed with the user on 2026-09-30. This task owns fitting; Astra owns
construction. Development proceeds one milestone at a time. The completed
28-cell development release is the input boundary. No test data, LLM calls,
benchmark changes or production-default promotion are part of milestone 1.

## Research objectives and five directions

Distinguish accurate output prediction, parameter/latent-state recovery, and
prediction under new interventions. A low observed NMSE cannot certify the latter
two. Separate numerical failure, weak/structural identifiability, inadequate
initialization information, and model-class mismatch before requesting revisions.

1. **Condition the joint optimization problem.** Center/scale parameters,
   collocation state variables, initial-value parameters and dynamical constraints.
   Preserve physical equations, domains, starts and observation weights in the
   scaling comparison. Test scientifically justified constraints separately.
2. **Approach dynamically consistent latent trajectories gradually.** Investigate
   multiple shooting, mesh/window adaptation and continuation. Final scoring must
   use a continuous free rollout with the deployed initialization rule.
3. **Exploit equation structure.** Investigate conditional linear subproblems,
   redundant latent scales and transfer of unchanged fitted blocks across revisions.
   Conditional least squares does not make the complete rollout objective linear.
   Earlier alternating/continuation trials had mixed outcomes; these are hypotheses.
4. **Allocate fitting effort using observed progress.** Retain useful checkpoints,
   distinguish improving/stalled/unavailable calculations, and avoid discarding
   incumbents or resetting consumed budgets after interruptions. Longer fitting
   is justified by progress, not by a generic timeout label.
5. **Return interpretable numerical uncertainty to construction.** Describe
   unexplained response features, weak parameter combinations and disagreement
   among fitted models. A weakly excited mechanism is not necessarily unnecessary.
   Experimental-design suggestions are future work outside the present paper.

Prioritize direction 1, while collecting evidence that will support directions
4 and 5. Existing collocation, sensitivities, multistart, local polling and
checkpoint recovery remain the baseline. Novelty must be established through
comparisons; established numerical techniques may already produce large gains.

The CSTR lead is documented in
[the September 28 findings](PHASE_C_FITTING_FINDINGS_2026-09-28.md): train/validation
NMSE 0.00014821/0.00006713 from known equations, despite substantially different
hidden initial values. That result bundled positivity, coordinate changes and
initializer extrapolation. It did not isolate conditioning or certify latent
recovery. The Phase C CSTR release now has common hidden preparation; this study
does **not** reuse the obsolete affine hidden-initial shift from that old release.

## Milestone 1: fixed-equation qualification

Protocol: `phase-c-fitting-qualification-1`.
Config: `configs/phase_c_fitting_v1.json`.

The first bounded study uses three predetermined development cells: named CSTR
easy, named CSTR hard, and coupled clean detention basins. These cover a nonlinear
thermal/concentration problem and a storage/threshold problem. Alien-device and
Dalla Man are deferred until this comparison is inspected. Dalla Man's accepted
continuous-ODE approximation of private bookkeeping is a separate model-class
question, not an exact-parameter qualification test.

Known equation skeletons are **evaluator assistance**. Original public training
and validation arrays are copied byte-verifiably from `phase-c-development-2`;
no trajectories are regenerated. CSTR reference parameters are verified against
the release's private specification hash. The portable input artifact retains
the public prompts, data identities, equation requests and diagnostic witnesses.
It contains no hidden trajectories or test payload. It must never become a
construction/proposer input.

The supplied CSTR template already uses a temperature-centered Arrhenius law and
its existing concentration/temperature guards. Every arm keeps this same template.
The study measures the additional effect of optimizer coordinates; it does not
compare raw Arrhenius parameterizations or automatically rewrite discovered laws.

| Case / fitted block | Arms | Numerical starts | Fits |
|---|---|---:|---:|
| CSTR easy: dynamics, measured initial auxiliaries | control, scaled | 3 | 6 |
| CSTR hard: dynamics, reference hidden initials fixed | control, scaled | 3 | 6 |
| CSTR hard: shared hidden initials, reference dynamics fixed | control, scaled, domain, scaled+domain | 3 | 12 |
| CSTR hard: dynamics and shared hidden initials jointly | control, scaled, domain, scaled+domain | 3 | 12 |
| Coupled basins: dynamics, public initial gauge | control, scaled | 3 | 6 |
| **Total** | | | **42** |

The known-block controls deliberately supply private reference values for their
fixed block; label this assistance when interpreting them. Generic starts for
the fitted block use fixed broad guesses with reproducible perturbations, not
reference-centered draws. Each seed's physical starting vector is identical
across its arms. Three starts measure numerical sensitivity, not three independent
scientific problems. Report every endpoint; do not select the best seed.

### What scaling changes

For each parameter and state use `physical = center + scale * coordinate`, with
strictly positive finite scales. Parameter centers are their numerical starts;
scales are `max(abs(start), 1)`. Shared initial-value parameters undergo the same
transform. State centers/scales use training channel means/standard deviations
(standard-deviation floor 1 in that channel's units), or candidate initial guesses
when no observed channel exists. The explicit CSTR unit proxies are reactor
temperature for T, feed concentration for C, and jacket-feed temperature for Tj.
These are recorded dimensional choices in this assisted experiment, not hidden
state estimates or automatic unit inference. Basin upstream depth falls back to
the public initial gauge. The policy is deliberately simple and may be suboptimal.

Collocation uses transformed decision variables, transformed parameter bounds,
and state-scale-normalized continuity defects. Observation residual weights,
physical node guesses, forcing interpolation and original equations are unchanged.
IPOPT uses its existing settings; constraint tolerance now refers to scaled
defects. Thus even an invertible transform can change numerical stopping behavior.
Checkpoint feasibility in these coordinates is not a physical accuracy certificate.

Sensitivity refinement applies the chain rule to parameter gradients/Jacobians;
polling uses the same frozen parameter scales. Callbacks, checkpoints, returned
parameters and Jacobian reports stay in physical units. Native optimizer optimality
is labeled as measured in scaled coordinates. Actual ODE integration still uses
physical state units and unchanged solver tolerances. Time is not rescaled.
Initial-map basis rotation, full matrix preconditioning and adaptive latent-scale
inference are deferred.

### What the domain arm changes

Only the **shared initial CSTR concentration** becomes nonnegative, using its
explicit concentration interpretation in the supplied equation skeleton. No
reference numerical bounds are supplied. Dynamic coefficients retain their existing
domains in every arm. Jacket initial temperature is unchanged. Signed latent
variables are not automatically made positive.

This is an initial-domain experiment, not a certificate of nonnegative complete
state trajectories. No affine clipping or extrapolation change is bundled with it.
There are no redundant domain arms where initial concentration is already known.

### Budgets, scoring and provenance

All fits have 120 seconds of collocation initialization, 180 seconds of refinement,
240 refinement residual calls, five seconds of node warm-up, the same feasibility
screening policy, and local polling for guarded equations. Scaled and control arms
retain identical allowances. Startup/compilation and final scoring are recorded
overhead; a 45-minute scheduler allocation is not extra optimizer budget.

Before the array, the supplied full reference vector must pass both development
splits with NMSE <= 1e-6 and Radau/DOP853 disagreement <= 1e-4 training standard
deviations. It is a witness of attainable output error, not uniqueness.

Each fit selects parameters on training rollouts only. Validation uses the same
frozen parameters and causal initialization; no validation-specific initials.
Independent Radau/DOP853 replay has a separate 240-second total allowance.
Accuracy passes only when both split NMSEs <= 0.01 and the solvers agree within
1e-4 training standard deviations. Preserve optimizer convergence, execution
status, budget exhaustion and accuracy as separate fields. Record parameter
absolute errors without asserting identifiability, per-trajectory replay errors,
fit duration, actual residual calls and saved collocation checkpoints.

Plans, input content, Python source and numerical package versions are frozen.
Completed tasks are reused. An interrupted native fit retains its files and is
marked interrupted on reconciliation; it receives no automatic fresh budget.
If the backend completed before interruption, independent scoring can resume.
Floating-point results and wall-clock-limited endpoints need not be bitwise equal
across different machines. Resume guarantees artifact reuse and budget integrity,
not identical scheduler timing or portable native IPOPT state.

Exit criteria: account for all 42 attempts, verify reference attainability and
paired starting-point invariants, inspect within-case coordinate/domain effects
and scope-specific failures. Numerical failures are results, not missing rows.
Promotion to construction requires a subsequent explicit decision after these
results, including regressions and limitations, have been reviewed.

### Local qualification before submission

The three reference witnesses passed independent replay on the exact bundled
Phase C development arrays. CSTR easy/hard train/validation NMSE was
2.01e-17/1.88e-17; coupled basins was 2.42e-17/3.68e-18. Maximum Radau/DOP853
disagreement was 2.78e-6 training standard deviations. These are supplied-reference
checks, **not newly fitted performance**. The 42 fitting outcomes remain pending
Delta execution.

The focused tests exercise transformed bounds and Jacobians, a signed-latent
native collocation/refinement fit, physical checkpoint values, data/source drift,
paired starts, known-block isolation, resume budget preservation and uncertain
scheduler receipts. The signed-latent smoke fit reaches NMSE below 1e-6 on both
splits without fitting validation initials.

Implementation verification (2026-09-30): full `pytest -q` completed with
3,734 passed and eight optional-dependency skips. The standalone source/input
bundle passed its 14 focused tests and all three reference replay checks.
Changed Python files pass Ruff and the three shell scripts pass `bash -n`.
Repository-wide `ruff check .` still reports 37 pre-existing issues in unrelated
`analysis/claude` files; those files were not changed.

## Delta execution

Delta CPUs are sufficient. Use one CPU and 8 GiB per array task with at most two
concurrent fits; no GPU, model server, API key or ACES allocation is involved.
The 42 nominal optimizer allocations total 3.5 CPU-hours, plus reference replay,
independent endpoint replay and overhead. Queue time is not predicted.

Use the supplied `phase-c-fitting-m1.tar.gz` portable bundle. It includes source,
focused tests, configuration, sealed inputs, `SOURCE_COMMIT`, `SHA256SUMS`, and
the convenience script below. Upload the **single archive**, not its loose files.
An existing Delta project Python is reused. If CasADi is installed only in the
existing supplemental fitter dependency folder, the script detects it; no package
installation or environment mutation is performed.

```bash
bash <<'BASH'
set -euo pipefail
AF_REV=$(tar -xOf "$HOME/phase-c-fitting-m1.tar.gz" SOURCE_COMMIT)
AF_CODE="/projects/bibo/yxiao2/repos/phase-c-fitting-${AF_REV:0:7}"
mkdir -p "$AF_CODE"
tar -xzf "$HOME/phase-c-fitting-m1.tar.gz" -C "$AF_CODE"
(cd "$AF_CODE" && sha256sum -c SHA256SUMS >/dev/null)
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_c/fitting-m1
export AF_ACCOUNT=bibo-delta-cpu AF_CONCURRENCY=2
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_delta.sh"
BASH
```

Expected: 42 array tasks, with preparation, fitting and report job IDs written to
`/work/hdd/bibo/yxiao2/phase_c/fitting-m1/submission_manifest.json`.
Repeat submission with the same frozen source only to reuse receipts; a recorded
ambiguous scheduler reply stops for reconciliation instead of duplicating jobs.

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/fitting-m1
AF_IDS=$(jq -er '.jobs | [.prepare,.fit,.report] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" --format=JobID,JobName%26,State,ExitCode,Elapsed
jq '{status,expected,recorded,status_counts,groups}' "$AF_ROOT/summary.json"
```

The summary is created at preparation and refreshed after each fit, so unfinished
rows appear explicitly. For a failed preparation inspect its `.err` log and
`qualification/result.json`; the fit array should not proceed. For manual refresh
or a download archive, use `scripts/hpc/inspect_phase_c_fitting_delta.sh` from the
same bundle. It reports the existing campaign and packages its diagnostics; it
does not submit jobs or refit. Keep the original campaign for full call traces.

## References for later milestones

- Fides: https://doi.org/10.1371/journal.pcbi.1010322
- Multiple shooting/collocation: https://web.casadi.org/docs/
- Generalized profiling: https://www.jstatsoft.org/article/view/v075i02
- MAGI: https://doi.org/10.1073/pnas.2020397118
- Diffusion tempering: https://proceedings.mlr.press/v235/beck24a.html
- Observability/identifiability: https://doi.org/10.1007/s11538-025-01415-3
