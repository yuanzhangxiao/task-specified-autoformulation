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
