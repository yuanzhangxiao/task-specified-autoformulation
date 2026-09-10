# Numerical validity of collocation/sensitivity fits

The ACES multiround run at `a865f8a` reported native `gtol` success after every
training rollout at the ordinary start failed. The symbolic oracle supplied a
constant failure penalty with a zero Jacobian, so SciPy correctly terminated its
numerical surrogate. That is not a successful fit of the proposed equations.

This change separates native optimizer termination from accepted numerical
validity, without changing equations, parameter domains, initializers, or the
optimization objective. The failure penalty remains an internal optimizer device.

## Contract

- `optimizer_native_success`, `native_optimizer_message`,
  `native_optimizer_parameters`, and `native_optimizer_cost` retain the raw result.
  Existing status, gradient, active-mask, and Jacobian diagnostics refer to that
  native returned point, including when another point is retained for replay.
- `valid_residual_evaluations` counts complete finite training evaluations during
  optimization. A finite penalty after an integration failure does not count.
- With no valid training evaluation, return `parameters=null`, `cost=null`,
  `optimizer_success=false`, and `numerical_status=no_feasible_rollout`.
  The adapter returns `fit_failed` and does not open validation for this case.
- Otherwise, check the native returned point with a fresh full training call,
  inside the existing fitting deadline. An earlier valid point alone cannot
  certify an invalid returned point. If necessary retain the best valid training
  evaluation, verify it if time remains, and mark native convergence unaccepted.
- Preserve the existing best-evaluation fallback on optimizer timeout. It remains
  unverified at this layer and does not claim convergence. The adapter can still
  perform its existing independent production replay after fitting.
- `optimizer_residual_calls` and `verification_residual_calls` distinguish the
  additional checks; `actual_residual_calls` includes both. Native `nfev` keeps
  its SciPy meaning. No extra optimization budget is granted.
- The collocation adapter also requires its fresh production **training** replay
  to pass before retaining `optimizer_success=true`. Validation failure makes
  overall status `rollout_failed` but does not redefine training convergence.

`complete` still means a numerically usable replay, not optimizer convergence,
scientific correctness, high trajectory accuracy, or unique latent recovery.

## Failure evidence for the proposer controller

Each symbolic call records its failing trajectory and, for integration failures,
the integration interval, last attempted RHS time, state values, and state RHS.
Nonfinite state/sensitivity RHS indicators distinguish the affected part of the
augmented system. Failed solver returns include the last returned observation
sample when available. Deadline failures preserve `TimeoutError` behavior and
also carry integration context.

The first five failed or timed-out calls are included in the fit report as
`failure_evidence`; all individual calls remain in `sensitivity_calls/*.json`.
Each includes its parameter vector and call number. These are training-only
measurements for the controller to turn into concise feedback. An attempted RHS
state can be rejected by the solver and is explicitly labeled as such. It is not
proof that the accepted trajectory crossed a pole. This patch does not infer a
denominator failure, invent scientific repairs, or change revision routing.

## Integration

This branch starts at Sol's `a865f8a`. Its numerical files and tests can be
cherry-picked into Sol's next protocol revision. Sol owns the static domain audit,
function/topology repair contracts, and feedback presentation. Use a new frozen
campaign output after integration; do not resume old results under changed code.
No new GPU run is needed to test the gate itself: local tests include an actual
finite-time explosive ODE and reproduce native `gtol` success on its failure
penalty while the accepted fit correctly fails.

This fixes misleading success and supplies factual diagnostics. It does not yet
find a feasible start for every candidate, repair a scientifically invalid model,
or provide automatic per-term singularity attribution. Those decisions need the
proposer-side feedback work and subsequent real-candidate experiments.

A further code finding: `matching_probe.py::latent_start` requires a successful
full-horizon `symbolic_rollout` at the nominal parameters before it constructs the
collocation nodes. An unstable ordinary start therefore aborts initialization
before IPOPT optimizes any nodes or parameters, after which refinement retries
that same start. This is an initialization-path failure, not evidence that
collocation optimization failed or that the candidate is structurally infeasible.
A separately tested numerical milestone should allow finite collocation node
guesses without first requiring a successful full-horizon rollout. This patch
does not change that initialization policy or start fitting latent initial values.

## Verification

`pytest -q`: 1,252 passed, three optional PyTorch tests skipped. The 21 focused
numerical tests include an actual explosive ODE, all-failure rejection without
validation scoring, invalid returned-point fallback, deadline preservation,
successful collocation/sensitivity recovery, and production replay failure gates.
`ruff check .` and `git diff --check` passed. No ACES/Delta job was submitted.
