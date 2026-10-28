# Longer detention continuation with staged endpoint checks

Date: 2026-10-28. Opt-in training-only numerical experiment.

## Returned polishing evidence

The two-task polishing report reconstructed exactly from its operation receipts.
Both searches accepted five improving steps before reaching their 600-second
allowance. The last improving rollouts survived interrupted Jacobian calculations
and passed independent Radau/DOP853 verification:

| Start and coordinates | Before polishing | Retained training NMSE | Reduction |
|---|---:|---:|---:|
| `perturbed_0`, original | 6.2681091015e-5 | 4.2763648676e-5 | 31.78% |
| `generic_1`, reduced | 6.7466344544e-5 | 2.1334319661e-5 | 68.38% |

The independent solver loss differences were 2.90e-14 and 1.96e-14. Recomputed
gradients from all 17 saved residual/Jacobian columns matched the reported
gradients to within 1.74e-18. Both final assessments found reliable feasible
descent, rather than stationarity. In particular, the scaled `kappa` gradient
was -0.01127 with empirical error 3.14e-9; the reduced `b_fraction` gradient was
-1.0214 with error 1.26e-5. Magnitudes in different charts are not directly
comparable.

All 17 endpoint audits finished without timeout. Two original-coordinate columns
(`h_max`, `Q_max`) were exactly zero locally and remained inconclusive. Fifteen
columns passed the local agreement check, but only four original-coordinate
columns met strict stationarity precision. None of the reduced-coordinate
columns met that stricter target. This does not invalidate the much stronger
signal of remaining descent.

Initial sample reuse worked: each task reused one base residual and its 9/8
active Jacobian columns. This is an operational check, not a controlled speedup
comparison. New operation wall time was 2,807 seconds: 1,207 search, 1,580
diagnostics, and 21 verification. Each ten-minute search was followed by about
11 or 15 minutes of endpoint diagnostics. Invocation time contains those stages
and must not be added again.

Returned archive SHA256:
`dab67764da1dc5295612c64eddda78af9f0b744d1ac1771f5e833e689acf0cb0`.
Report file SHA256:
`87c7e3d5805ad594ee82a5d863e572b72d37d033d2c9f2ffe96d5101215db535`.

## New bounded continuation

This is a new experiment identity and result root. Previous allowances, receipts
and the original 600-second polishing API remain unchanged.

The two starts remain separate. Each receives exactly one uninterrupted search
with an allowance of at most 1,800 seconds. Its complete initial qualification
authenticates the same equations, training data, parameter point, bounds,
coordinate chart, integration settings and raw arrays. The initial residual and
active Jacobian are reused at that exact point; changed points require fresh
evaluation. Existing scaling and numerical acceptance thresholds are unchanged.

Every improving base rollout is checkpointed before its Jacobian is computed.
Independent original-equation verification has a separate 120-second allowance.
Only verified improvement can replace that start's incumbent. A failed, worse,
or unverified candidate leaves the incumbent intact. There is no cross-start
selection, fitting of initial values, held-out access or LLM call.

## Staged endpoint evidence

First, every coordinate at the retained endpoint receives a search-quality
audit, with at most 360 seconds per coordinate. Audits share exact completed
residual samples and stop when the existing useful-search precision requirement
is met. The complete resulting qualification is retained for future work.

The runtime then makes an explicit evidence-based allocation decision:

1. If at least one checked feasible direction demonstrates reliable descent,
   report that conclusion and skip the full strict audit.
2. If the endpoint is independently verified, every coordinate has complete
   search-qualified evidence, and no coordinate's empirical feasible descent
   lower bound exceeds the existing stationarity tolerance, perform a strict
   audit of all coordinates. This means stationarity is not excluded by the
   available intervals; it does not mean stationarity has been established.
3. Otherwise report unresolved evidence, identifying missing, failed or
   insufficiently qualified directions. An unavailable audit is not convergence.

There is no new multiplier defining a "small" gradient. For normalized residuals
`r` and coordinate units `s_i`, `g_i = 2 s_i J_iᵀr / m`. The empirical range is
`[g_i-e_i, g_i+e_i]`. At an interior point, reliable descent requires
`abs(g_i)-e_i` above the unchanged gradient tolerance. At bounds, only feasible
directions count. The existing search error requirement
`e_i <= max(absolute_error_limit, relative_error_limit * abs(g_i))` remains in
force before strict escalation can be considered. Temporarily held coordinates
are included in the final full-chart checks.

When triggered, strict audits receive their own nonrenewable 360-second
allowance per coordinate. They reuse completed raw samples from the search
audit at the same point, but must recompute the stronger conclusion. Search-only
stopping evidence cannot certify stationarity. The existing full-chart strict
assessment and tolerances are unchanged; missing checks, actual kinks and
inconclusive zero columns remain explicit limitations. Empirical numerical error
envelopes are not statistical confidence intervals or formal interval bounds.

## Accounting, resume and interpretation

The largest possible numerical allowance is 8,400 seconds for nine coordinates:
1,800 search + 120 verification + 3,240 search-quality audits + 3,240 optional
strict audits. The eight-coordinate task has a maximum of 7,680 seconds. Strict
checks that were never triggered are unconsumed potential allowances, not
executed work or charged reservations.

Each operation has a durable reservation and checkpoint. Resume reuses completed
operations and consumes interrupted reservations without renewing their budgets.
The report reconstructs the allocation decision from the saved endpoint and its
matching evidence. It explicitly distinguishes a skipped full strict audit,
completed strict work, incomplete strict work, and unresolved search evidence.
Diagnostic availability is distinct from optimizer termination and retained
prediction accuracy.

Model 15 retains its exact original-coefficient ambiguity. No result here implies
unique coefficients, statistical confidence, scientific correctness, a global
optimum, or held-out accuracy. The two starts are not a matched comparison of
coordinate methods. A longer allowance may still be insufficient to reach local
stationarity.

## Delta runbook

After extracting the supplied versioned bundle into a persistent project code
directory, run:

```bash
export AF_LLMSR_DETENTION_STAGED_POLISHING_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-staged-polishing-v1
bash scripts/hpc/start_llm_sr_detention_staged_polishing_delta.sh run
```

The launcher uses two tasks, at most two concurrently, one CPU and 8 GB each.
The four-hour scheduler ceiling accommodates the optional strict branch and I/O
overhead; it does not enlarge the thirty-minute search allowance. Reporting has
a thirty-minute ceiling. No GPU or model provider is required.

Keep the extracted code available for queued jobs. From that checkout:

```bash
bash scripts/hpc/start_llm_sr_detention_staged_polishing_delta.sh status
bash scripts/hpc/start_llm_sr_detention_staged_polishing_delta.sh inspect
```

Progress is printed before potentially slow reads and each scheduler submission.
Confirmed job IDs are saved. Inspect an ambiguous scheduler reply before retrying;
the launcher does not blindly duplicate jobs. Inspection creates a portable
archive of inputs, reports, operation receipts and sample caches. Cooperative
deadlines cannot interrupt every blocking I/O or solver call; overruns remain
visible in measured accounting.

## Local verification

- 375 focused tests passed across the new kernel/campaign/launcher, earlier
  polishing, continuation, lazy search, derivative recovery and stationarity.
- Both actual retained endpoints passed input authentication. A metadata-only
  smoke check reused their exact residual/Jacobian arrays, required fresh
  evaluation at a changed point, and classified both as reliable remaining
  descent. No real fitting or ODE integration was performed by this smoke check.
- Changed Python files passed Ruff; both Bash wrappers passed syntax checks.
- The repository-wide pytest attempt was stopped after 225 passes and three
  optional Torch skips; it was not a complete suite pass. Repository-wide Ruff
  reported 42 findings in unrelated analysis and construction files. Those files
  were preserved.

The source report and compact input are pinned independently. The portable input
content SHA256 is
`91ee51ad094d53e40fcb6727dbca95ae01cbfa56395ed4b9e1eda13e219ea616`.
The delivery bundle includes the saved-endpoint smoke record and a checksum
manifest. Portable preparation, status and inspection are checked before
delivery; actual numerical continuations remain for Delta.
