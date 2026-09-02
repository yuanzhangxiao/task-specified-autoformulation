# Phase-B classical post-freeze evaluation

This stage evaluates the complete public persistence, SINDy, and PySR matrix
only after its 360 public train/validation selections have passed the common
development readiness freeze.

## Frozen final models

The preparation job validates every artifact in the development freeze before
applying method-specific finalization rules:

- persistence retains its causal previous-observation rule;
- SINDy retains the validation-selected sparsity threshold and refits only the
  coefficients on the combined public train and validation trajectories;
- PySR retains the validation-selected equations exactly and performs no new
  symbolic search.

The resulting 360 models, equations, normalization scales, source hashes, and
development fingerprints are written to a new immutable final-model freeze.
No test or private reference is opened during this preparation job.

## Separate evaluation endpoints

The Delta chain reports two non-interchangeable endpoints:

1. A matched predictive endpoint for all 360 trials. Persistence uses the
   previous observation, while SINDy and PySR use the same causal one-step
   observed-state-reset protocol used for public validation.
2. The common model endpoint for the 240 symbolic trials. Frozen SINDy and
   PySR equations are adapted to the method-neutral candidate schema and
   evaluated for deterministic runtime validity, public mechanism compliance,
   complexity, and unseen-condition free rollout.

The metrics remain separate. Persistence is a predictive reference and is not
misrepresented as an autonomous mechanistic ODE. Test data is opened only
through a grant bound to the final-model freeze hash. The chain does not use a
private reference, exact oracle derivatives, or true latent states.

## Delta submission

The submission script reads the completed development readiness job identifier
from the original full-matrix submission manifest and places every new stage
behind an `afterok` dependency:

```bash
bash scripts/hpc/submit_phase_b_public_baseline_postfreeze_delta.sh
```

The final receipt is
`common_evaluation_readiness.json`. Its status is
`ready_for_private_hidden_evaluation`, which means the public and sealed-target
endpoints are complete; it does not claim that private hidden-mechanism scoring
has already run. The receipt also binds a `slurm_accounting.psv` ledger for the
final-model, predictive-test, free-rollout, and summarization jobs.

All task workers are checkpointable by immutable per-task result files. To
resubmit after an infrastructure interruption, use a new output root or submit
only missing array indices after verifying the frozen manifest and code commit.
