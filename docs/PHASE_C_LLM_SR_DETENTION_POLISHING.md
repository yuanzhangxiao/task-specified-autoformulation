# Detention endpoint polishing and local assessment

Date: 2026-10-27. Training-only numerical development; opt-in experiment.

Returned results: both tasks improved, to 4.27636e-5 and 2.13343e-5 training
NMSE, while reliable descent remained. The
[longer continuation with staged endpoint checks](PHASE_C_LLM_SR_DETENTION_STAGED_POLISHING.md)
records the completed review and the next explicitly bounded experiment.

## Evidence motivating this step

The six-task probe-recovery report reconstructed exactly from its saved
operation receipts. Both continuations improved substantially under unchanged
equations and independent Radau/DOP853 rollout verification:

| Start and coordinates | Before recovery | After three search segments |
|---|---:|---:|
| `perturbed_0`, original | 0.0013033657533 | 0.0000626810910 |
| `generic_1`, reduced | 0.0932183407186 | 0.0000674663445 |

These are training NMSEs. Each search segment exhausted its 600-second allowance;
none established convergence. The final loss discrepancies between independent
solvers were approximately 3.46e-14 and 3.90e-14. Large, reliably nonzero gradient
components remain. Their magnitudes cannot be compared directly between different
coordinate charts.

For `generic_1`, the first `b_fraction` check still failed. The first segment
therefore held it fixed while improving the other coordinates. At that new point
it qualified, and subsequent segments released it. This supports retaining a
useful partial search and revisiting held coordinates; it does not establish
superiority over a matched-budget alternative. Final passing diagnostics must
not obscure the initial failure.

Recovery used about 30 minutes of search and 24 minutes of derivative checking
per continuation. Across all six tasks, recorded operation wall time was
8,747 seconds: 5,049 diagnostics, 3,628 search, and 70 verification. Invocation
wall time contains these operations and must not be added to them. Unmeasured
filesystem and numerical subdivisions remain unknown.

Source archive SHA256:
`33ef4a6165a43e7d33b1d239a7887b65ad8cfb6f9fbbc324eedd88290268b543`.
Source report file SHA256:
`ccfdd0ed14cc7d4cad5dfbe5f3c070daaf3f872fe9bc42338f0acbfae52546dc`.

## Bounded experiment

There are two tasks, using source tasks 0 and 1 separately. Each starts at its
own final independently verified parameters and receives one 600-second lazy
rollout search. No coefficients move between starts. The original equations,
training observations, parameter domains, integration settings, and derivative
acceptance thresholds are retained. Neither validation nor test data are opened;
there are no LLM calls.

The initial residual and active Jacobian columns can be reused only after their
saved qualification authenticates the model, data, parameter point, coordinate
chart, bounds, numerical policy, settings, and raw array digests. Reuse applies
to that exact initial point. Changed points require new evaluations. Held
columns remain unavailable, rather than being replaced by zero derivatives.
Reports distinguish imported arrays and reuse hits from new numerical work.

The improving base rollout is checkpointed before its Jacobian is attempted.
An interrupted derivative calculation therefore cannot erase a useful candidate.
The candidate must still pass independent original-equation verification before
it replaces the incumbent. A failed or worse candidate preserves the original
verified solution. Completed receipts are reused on resume; consumed allowances
are never renewed silently.

After search, every coordinate of the retained endpoint receives a stricter
local audit with at most 360 seconds per coordinate. These checks pursue the
existing stationarity precision requirement, rather than stopping as soon as
derivatives become adequate for search. Completed samples may be reused only
under their exact identity. A missing or timed-out check remains unresolved.

## Meaning of the final assessment

The report distinguishes three conclusions:

- **Reliable descent remains:** a checked, feasible direction has a gradient
  larger than its empirical numerical error envelope and the stationarity
  tolerance. More fitting may help.
- **Qualified local stationarity:** complete, sufficiently precise local
  derivative evidence supports the constrained first-order condition at this
  retained endpoint. This is not a global-optimality certificate.
- **Unresolved:** the available local evidence establishes neither conclusion.
  Missing checks, genuine kinks, or inadequate precision are recorded explicitly.

At a bound, only movement into the feasible domain counts. A derivative pointing
out of the feasible domain is not an available improvement direction. Temporary
freezing during search does not remove a coordinate from this final assessment.
Empirical error envelopes are numerical diagnostics, not probability-based
confidence intervals.

For the normalized training residual vector `r` of length `m`, the objective is
`L = rᵀr / m`. With coordinate unit `s_i` and residual derivative column `J_i`,
the reported dimensionless gradient is `g_i = 2 s_i J_iᵀr / m`. A checked
gradient error `e_i` gives an empirical range `[g_i-e_i, g_i+e_i]`. At an interior
point, `abs(g_i)-e_i` exceeding the tolerance supports descent. At a lower bound,
only the increasing direction is available; at an upper bound, only the
decreasing direction is available. The final report retains both the numerical
evidence and the conclusion, so an unresolved result can guide a later bounded
check without being mistaken for convergence.

Model 15 has an exact original-coefficient ambiguity. Neither low NMSE nor local
stationarity removes it, and this experiment does not claim unique coefficient
recovery or parameter confidence. The two continuations are not a controlled
comparison of coordinate methods or a generalization experiment.

## Delta operations

Use the versioned portable bundle and a fresh result root:

```bash
export AF_LLMSR_DETENTION_POLISHING_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-polishing-v1
bash scripts/hpc/start_llm_sr_detention_polishing_delta.sh run
```

The launcher authenticates the compact saved input, prints progress before
lengthy reads, and records confirmed Slurm IDs. Each task requests one CPU,
8 GB, and a two-hour scheduler ceiling; at most two tasks run concurrently.
The ceiling allows diagnostics and I/O overhead and does not enlarge the
600-second search allowance. Reporting has a thirty-minute ceiling.

Keep the extracted checkout available for queued jobs. From the same checkout:

```bash
bash scripts/hpc/start_llm_sr_detention_polishing_delta.sh status
bash scripts/hpc/start_llm_sr_detention_polishing_delta.sh inspect
```

Status reads submission records without replaying historical fitting. Inspection
reconstructs the report from saved operations and packages the evidence. After
an ambiguous submission response, inspect the recorded intent and scheduler
before retrying; never assume silence means that no job was accepted.

## Scope and remaining limitations

This is a bounded polishing pilot, not a new generic-start campaign. Numerical
deadlines are cooperative: blocked I/O or a solver call can overrun a reservation,
and recorded wall time must remain honest about that overrun. A short segment
may still end without reaching local stationarity. The experiment leaves
construction, benchmark data, finalized prompts, and production fitting defaults
unchanged.

## Local verification

The integrated fitting, campaign, launcher, derivative and lazy-search regression
run passed 280 tests. Both actual saved endpoints authenticate and seed all 9/8
active derivative columns with zero new integrations; a changed point correctly
requires new evaluation. Their saved evidence reports remaining reliable descent,
without promoting search qualification to stationarity.

Changed Python files pass Ruff, both shell scripts pass syntax checks, and the
diff passes whitespace checks. Repository-wide Ruff reports 37 pre-existing
issues in `analysis/claude`. Full-suite completion is not established. These local
checks qualify the implementation and portable input, not the outcome of the
new Delta polishing experiments.
