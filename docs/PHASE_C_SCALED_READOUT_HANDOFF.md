# Scaled initial readouts and fitting qualification

This milestone closes the scaled-state proof gap found in the 29 October
admission audit. It changes no candidate equations, initialization contracts,
parameter domains, benchmark prompts, or fitting algorithm. Historical live
profiles remain unchanged; the corrected proof is explicitly enabled for the
offline reassessment and fitting intake.

## The proof

An exact initial readout of the form

```text
y_i(0) = c_i + k z_i(0)
```

can constrain a trajectory-specific latent initial even though its value and the
shared gain `k` are unknown. The runtime first establishes that each offset `c_i`
is known from exact public or known state initials. It recognizes a single
globally shared gain times one unresolved state, using bounded restricted-AST
arithmetic.

There are two parameter cases:

1. A positive declared gain domain excludes zero. Otherwise at least one exact
   initial residual `y_i(0)-c_i` must exclude `k=0` using the existing numerical
   guard. When zero remains possible, this proof defers.
2. For every remaining shared nonzero gain, weights cancelling the residuals
   also cancel the latent initials: `sum(w_i*(y_i(0)-c_i)) = k*sum(w_i*z_i(0))`.
   No unknown gain is evaluated, fitted, or divided by.

These constraints enter the existing exact superposition check on the **original
verified affine closed subsystem**. Unscaled consumers of the latent state are
allowed; rewriting the dynamics into scaled coordinates is unnecessary. Every
other state-initial direction, common forcing condition, and affine-closure
obligation must still be established. Unknown offsets, unsupported gains,
additional unpinned states, state nonlinearities, uncertain initial observations,
and exhausted proof budgets remain unresolved.

The certificate records the original readout, process expansion, known offsets,
exact residuals, gain domain, zero-case witness/guard, weighted initial equality,
and trajectory contradiction. This excludes **exact simultaneous reproduction**
under the stated assumptions. It is not a lower bound on approximate NMSE and
does not diagnose an optimizer.

The opt-in flag is `scaled_readout_initials=True`, with policy
`shared-scaled-initial-readout-1`. It is propagated through parent/child checks,
branch composition, request and result provenance, and assessment cache keys.
Repair feedback retains the certificate and points to the initial readout and
gain cases. Historical defaults and their request/report shapes remain intact.

## Saved-candidate qualification

The local delivery is
`output/construction-scaled-readout-delivery-20261029/`. It contains:

- `replay-summary.json`: historical replay, prospective decisions, exact source
  hashes, and the complete deduplicated roster.
- `models/candidate-*.json`: complete unchanged `PrefixFitProblem` packets,
  current audit, initializer graph and original parameter guesses/domains.
- `REPORT.html` and per-model pages: equations, initializer contracts, unresolved
  findings, and a link to each full fitting packet.
- `PROOF.html`: the scaled-memory case and its weighted-trajectory witness.
- `feedback/`: prospective repair previews; no old reply is represented as an
  answer to a new prompt.
- `evidence/`: the public training split, compatibility-proof evidence and fixed
  observation-prefix/future-forcing contract.
- `FITTING_HANDOFF.md`: the detailed numerical work order for Orion.

The combined population is one T2-easy parent across two saved cohorts, two model
sizes and three seeds per arm. Complete-model/domain identities are not counts
of distinct physical mechanisms. The prospective gate is applied to each actual
saved patch against its actual saved ancestor; altered live routing or responses
are not simulated.

## Numerical work order for Orion

Replay reproduces all **51 saved calls and 99 candidate outcomes** exactly.
Fresh reassessment retains **24 eligible records representing 23 distinct
complete-model/domain identities**, exactly the earlier 14 plus nine new ones.
The known v7 120B seed0 `call0-option1` scaled-memory candidate is excluded by
the new runtime proof, replacing its manual hold. The three earlier v6
input-saturation exclusions remain. Use the verified delivery roster as the
authority; do not reconstruct the set from historical eligible flags.

All **23 pass static qualification**: typed parsing, restricted-expression compilation,
initializer-graph lowering, symbolic-system compilation, dynamics guess
coverage, the 16 training rows, and prefix/future-forcing authorization.
**Numerical qualification remains a separate pending task.** No optimization,
solver rollout, new LLM call, validation read, or test-data access is part of
this milestone.

This is a 23-model intake, not a new scheduler launcher. The historical
`construction_fit_qualification` campaign's `returned-eight` policy fixes an
eight-model roster and must not be reused unchanged for this population. Orion
should register the complete 23-model manifest in the numerical qualification
campaign while preserving the existing scoped-prefix kernel and data boundary.

Preserve the saved **8-unit observation prefix**. Learn shared dynamics and map
parameters from training only. For prefix-conditioned training/validation
forecasts, freeze shared parameters and infer only declared independent latent
roots using the permitted prefix; compute dependent initials through the graph.
Score strictly after the prefix. Full-training reconstruction is a separate
metric. Future inputs and auxiliaries must obey the saved explicit authorization.

One packet, `candidate-01`, uses the ongoing supplied `Uii` trajectory; the other
22 use only the two external-input channels as future forcing. This dependency
is permitted by the frozen policy and passed the current prefix adapter's static
check. Orion must preserve it in numerical qualification or report an explicit
adapter limitation; replacing it with its initial value changes the model.

Use a recorded common numerical policy, independent original-equation parameter
replay, appropriate derivative/convergence checks, durable stage checkpoints and
nonrenewable budgets. Keep original starts and every failure. Distinguish
interface failures, inability to enter optimization, collocation/refinement
failures, budget stops, non-identifiability and qualified convergence. High NMSE
or early numerical failure alone does not establish inadequate equations.

The included full public training split has canonical content SHA-256
`169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795`.
Validation is not bundled: verify the previously frozen matching split before
evaluation. Test data remain unopened. Report per-model training/validation
NMSE and status, summarize errors with median and unscaled MAD, and retain
unresolved scientific issues. No score-based model selection occurred here.

## Limits

Passing the finite construction checks is not a scientific certificate. The
portfolio retains unresolved balance interpretations, function signs/overlap,
domain behavior and mechanism adequacy. This milestone neither expands the
benchmark population nor establishes that any of the models fits well. Fitting
and independent equation assessment must remain distinguishable in later reports.

## Verification

- 291 focused proof/integration tests passed; independent review found no
  material issue and four additional analytic counterexamples behaved correctly.
- Full `pytest -n 4`: 11,412 passed, eight skipped (unavailable Torch), one
  unchanged numerical-fitting test exhausted its 30-second budget. That test,
  `test_fit_recovers_identifiable_decay_without_claiming_all_coefficients`,
  passed an isolated rerun without any source change. The full invocation is
  recorded as having one failure, not as an all-green run.
- Changed files pass Ruff and `git diff --check`. Repository-wide Ruff retains
  37 existing findings in the unrelated untracked `analysis/claude/` directory.
- All 51 saved calls / 99 outcomes replay exactly on the frozen source;
  the 23 model packets and their source hashes verify. All 79 local links across
  26 generated HTML pages resolve.
