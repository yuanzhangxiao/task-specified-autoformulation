# Candidate 15 insulin feedback revision and numerical handoff

To Orion, from Codex, 6 November 2026.

The one-call GPT-OSS-120B revision completed on ACES and passed the current
construction admission checks. It adds the existing insulin-action state to
insulin production. No new state or initializer was proposed. Numerical fitting
of the new coupling is the next step; improved NMSE or delayed peaks have not
yet been demonstrated.

## Returned proposal and equation interpretation

The proposer replaced `insulin_production` with

```text
insulin_production = k_insulin_term_1_0_4e94c3cad2 * Gp
                   + k_insulin_delay * X_insulin_action
```

It declared `k_insulin_delay` as a global `coefficient`, with domain `real`.
The runtime initialized its numerical guess to 1. This is an unfitted default,
not an estimated coefficient. All existing dynamics, initializers, outputs and
data access remain unchanged.

Writing `r` for ingestion rate, `v` for supplied insulin, `X` for
`X_insulin_action`, and `Q` for the generated Uii alias, the assembled family is

```text
Gp' = r - Q - X + a A
 A' = c_A r - a A
 I' = c_I r + b Gp - d I + kappa X + h v
 X' = e I - f X
 Q' = q
 U  = Q + X
```

Here `kappa = af_process_gain_1_0 * k_insulin_delay`. The inherited fitted values
are approximately `a=0.00205851347`, `c_A=24.6376953`, `c_I=3.11416174`,
`b=0.000975926039`, `d=0.00717575139`, `e=0.000513988269`,
`f=0.0213261928`, `h=2.90478483`, and `q=0`. The outer production gain is 1 at
this endpoint. Effective coefficients can change during the new fit.

The new connection is **I → X → I**. The existing required path I → X → U
remains. X is driven by insulin, not directly by glucose, so the added term
reuses insulin-action memory as feedback into secretion. It is not an
independent glucose-driven secretion state. The proposer's explanation that X
is a pancreatic stimulation proxy is a scientific hypothesis, not a consequence
of the public task or a certified physiological interpretation.

At the inherited positive I-to-X coupling, a positive new effective coefficient
closes a positive-feedback loop. It can change response timescales, but its
effect on the fitted peak is unknown. The declaration permits negative values
too. The claim that the early
rise remains unchanged is also unestablished: nonzero X at the initial boundary
already changes the insulin derivative when the new coefficient is nonzero.

## Admission and independent replay

The worker exited 0. The one response finished with `stop` after 8.86 seconds,
using 5,427 prompt tokens and 489 completion tokens. There were no retries,
optimizer calls, or held-out evaluations. The server log contains transport
fallback/JIT warnings and process/resource cleanup warnings at shutdown; the
generation and terminal result completed successfully.

All eight sealed artifacts verified. The source plan, launcher identities,
runtime probe, exact cached call and exported model hash reconciled. Reapplying
the saved response reproduces the admission result, normalized patch, model,
and training-consistency assessment exactly.

Static admission is `eligible`. The meal-to-Gp and delayed I-to-U requirements
retain paths. The simplest delayed witness is I → X → U; the saved runtime
witness includes a longer cyclic walk, which is unnecessary for this model.
The model-data assessment is `deferred`, with no qualified contradiction found.
Several equality checks defer because the training roots of A and X may differ
across trajectories. The bounded initial/function/affine-family checks also
remain deferred. This status does not certify that the model can fit the data
or meets every scientific requirement.

## Verified nested numerical start

Setting `k_insulin_delay=0` exactly recovers the previous fitted family. Symbolic
expansion confirms equality of every RHS and output, and the initialization
graph is unchanged. The revised model remains autonomous and jointly affine
in states and supplied inputs at fixed parameters.

The provided starting vector carries all 51 previous fitted coordinates,
including the **32 trajectory-specific training initials**, and adds the new
coefficient at zero. Using this 52-coordinate vector, one original-equation
Radau replay of the admitted model gives:

| Quantity | Value |
| --- | ---: |
| Previous verified training NMSE | 0.2610107571537155 |
| New family at the nested start | 0.2610107571549462 |
| Absolute discrepancy | 1.23e-12 |
| Trajectory NMSE median [unscaled MAD], n=16 | 0.185688 [0.166996] |

This is a fixed-parameter replay, not a fit of the added coefficient. Do not
initialize trajectory roots from the generic guesses in `problem.json` when
claiming recovery of the previous endpoint; use `warm_start_parameters.json`.
The full `PrefixFitProblem` graph is authoritative. Its base candidate alone
contains placeholder latent initials and is not the fitting problem.

## Conditional stability at the inherited coefficients

With all old coefficients fixed and Q held fixed, the homogeneous Gp/I/X block
has characteristic polynomial

```text
lambda^3 + (d+f) lambda^2 + (d*f-kappa*e) lambda + b*e.
```

At the inherited positive b,d,e,f, that block is asymptotically stable when
`kappa < d*f/e - b/(d+f)`. With the inherited outer gain 1, the threshold is
`k_insulin_delay = 0.26349268345`. At the default guess 1 the eigenvalues are
approximately `0.00816866`, `0.00160437`, and `-0.03827497` per minute. The fastest
growing homogeneous mode doubles in about 84.85 minutes. At zero, all three
real parts are negative.

This is a conditional starting-point diagnostic, not a universal parameter
bound, a full-system stability certificate, or grounds to reject the family.
Refitting d, e, f and the other coefficients changes the threshold. A and Q
are outside this three-state block. Q retains its inherited constant-drift
equation, and output positivity also requires a separate check.

## Numerical work for Orion

1. Verify the package hashes and reproduce the nested start before optimization.
   Retain that verified incumbent throughout the search.
2. Qualify the revised model in the current fitting backend. Affine propagation
   and conditional profiling of initial roots remain structurally applicable.
   The new coefficient enters the **state matrix**, so it must not be treated
   as another forcing-only column like the earlier direct-ingestion gain.
3. Preserve the declared real domain for the primary fit and choose scales and
   additional starts using the nested endpoint. Any nonnegative-only hypothesis
   is a separate declared control, not a silent change to the returned model.
   Report the effective coefficient kappa and its sign.
4. Use the saved matched numerical allowances as the initial comparison:
   1,800 search seconds and 2,000 calls, with separate qualification/replay
   allowances. Log any altered budget explicitly. A failed default start does
   not establish equation inadequacy.
5. Independently replay retained parameters in the original equations and
   compare training NMSE, target NMSEs, trajectory medians/MAD, all three shifted
   peak times/heights, dose responses and no-meal controls. Report positivity
   and relevant stability behavior at the fitted endpoint. Do not conclude
   that the secretion interpretation is established merely from better NMSE.

The baseline insulin peaks remain at minutes 10, 70, 130 versus observations
101, 161, 221. The goal is to evaluate whether the new family closes this
91-minute timing gap while preserving the other targets and controls.
`compare_refit_replay` in `scripts/propose_meal_insulin_revision.py` accepts the
verified residual and checks the admitted model and training identities.

Full-training reconstruction may estimate training initials from each complete
training trajectory. Later validation/test work must freeze shared dynamics,
infer roots only from the existing eight-unit prefix, and score after that
prefix. No held-out data is needed for this work order.

## Files and provenance

- Construction implementation: `34a35bc80d6d1a9babf1a112a91e662b345b37ae`.
- ACES job: `2310039`; remote root:
  `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-insulin-revision-120b-v1`.
- Returned archive SHA-256: `7edce17bf4e8dabce946fb1a7f2f52665ada2a6e66b5fb6188e0a0d65e25405d`.
- Construction plan: `d46deadb4e3a7f38970cc20caf3fe15ccdc75b493126dffe1c733b911584c9f8`.
- Admitted `PrefixFitProblem`: `76a4fadfbd2dd45a5c3ca3aad4b2f07e4d6c2f89f7564cd135ef4fc81a7c77af`.
- Public training hash: `169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795`.
- Local returned records: `output/insulin-revision-analysis-20261106/returned`.
- Visual equation and stability report: `output/insulin-revision-analysis-20261106/REPORT.html`.
- Portable fitting inputs: `output/insulin-revision-analysis-20261106/orion-insulin-revision-inputs.tar.gz`.

The portable package contains `problem.json`, `parent_problem.json`,
`warm_start_parameters.json`, `baseline_endpoint.json`, the 16 public training
rows in `training.json`, `numerical.json`, `fitting_contract.json`, proposal and
admission records, `nested_replay.json`, provenance and per-file SHA-256s.
It contains no validation/test data. `analyze.py` and `export_handoff.py` in the
local analysis directory reproduce this review and export without generation
or optimization. The package is a fitting handoff, not an automatic launcher.
