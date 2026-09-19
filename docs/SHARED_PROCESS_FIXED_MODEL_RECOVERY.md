# Shared-process pilot: parameter-free execution recovery

The original round-0 pilot reported two failures with
`ValueError: zero-size array to reduction operation maximum which has no identity`:

- `cell00_seed1_full_shared_on`: `v01' = u01 - v01`.
- `cell01_seed0_full_shared_on`: `T' = Cf*Tf - Cf*(T-T) + C + Tj - T`.

Both models have no learnable parameters after lowering their initialization
plans. Passing an empty parameter vector to optimization was an execution bug.
The public fitter now evaluates such models through causal rollouts without
collocation, sensitivity construction or optimization. It uses the profile's
existing integration method, tolerances and total time allowance, and training
normalization for both development splits. Learnable latent initial values still
introduce parameters and use the ordinary fitting route. The change does not
remove terms, introduce gains or change scientific parameter domains.

Results retain `parameters: {}`, zero optimizer/residual-evaluation calls and
`native_optimizer_converged: null`. Failed integrations have unavailable scores,
not numerical penalty scores. Training residual feedback says
`fixed_model_evaluated`; it does not claim optimizer convergence or structural
failure. No independent solver replay is added. In particular, `Cf*(T-T)` is
identically zero, but this execution fix does not repair that modeling choice.

## Recovery contract

`shared-process-fixed-model-recovery-1` creates a **new** pilot root. It requires
all original round-0 terminal results and refuses a source with revision-round
submission intent or work. It imports original results, proposals, public
development files and logged call accounting. Existing successes, unrelated
numerical failures and construction failures keep their original outcomes.
Only the exact empty-vector failure, verified against the lowered model and
sealed fitting handoff, is eligible for rollout evaluation.

The original root is preserved. Equations, seeds, observations, fitting settings,
construction/revision budgets and selection rules stay the same. Historical
construction tokens and physical requests remain in cumulative reporting; this
repair adds neither LLM calls nor optimization attempts. The report separately
records fixed-model evaluations and the source result hashes. Duplicate recovery
commands reuse sealed results. Interrupted numerical attempts retain their
consumed markers, without receiving an automatic fresh budget.

Round 1 remains unsubmitted. Review the recovered round-0 scores and residual
packets before submitting that milestone's already planned revision round. This
is a correction to round-0 execution, not a fresh guidance/control experiment.
Its timings should not be presented as a controlled speed comparison.

## ACES: one CPU recovery job

Set `AF_COMMIT` to the full pushed commit supplied with this fix. Use the existing
git repository to fetch and export the code into group scratch; no new clone,
virtual environment or model copy is required. Run in a subshell so a failure
does not close the login session.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?set the full fix commit first}"
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/shared-fixed-${AF_COMMIT:0:7}"
  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  export AF_SOURCE_ROOT="$AF_GROUP/shared-process-pilot-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/shared-process-pilot-fixed-v1"
  mkdir -p "$AF_OUTPUT_ROOT/logs"
  sbatch --parsable --account=156264627414 --partition=cpu \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=00:30:00 \
    --job-name=shared-fixed-recovery --export=ALL \
    --output="$AF_OUTPUT_ROOT/logs/recover-%j.out" \
    --error="$AF_OUTPUT_ROOT/logs/recover-%j.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_shared_process_fixed_recovery_aces.sh"
)
```

This queues only recovery: no GPU, no collocation rerun and no next-round jobs.
Keep the returned job ID. If the scheduler reply is uncertain, inspect the queue
before submitting again. The worker runs focused regression tests first.

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/shared-process-pilot-fixed-v1
cat "$ROOT/RECOVERY.json"
cat "$ROOT/SHARED_PROCESS_SUMMARY.md"
jq '{status_counts, proposal_status_counts}' "$ROOT/summary.json"
```

Successful rollout evaluation is not a guarantee of good NMSE or mechanism
compliance. Share `RECOVERY.json` and the summary before proceeding.

## Revision round after review

Use the **recovery commit and output root**, not the original pilot source.
The existing round-1 launcher reads the imported parent results and requests
fresh revision responses only for that planned round. It preserves the original
round-0 token costs. Do not issue this command until the recovery is reviewed:

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?use the same full recovery commit}"
  export AF_REPO_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/repos/shared-fixed-${AF_COMMIT:0:7}
  export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/shared-process-pilot-fixed-v1
  export AF_PUBLIC_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1
  bash "$AF_REPO_ROOT/scripts/hpc/submit_shared_process_pilot_aces.sh" 1
)
```

## Local verification

Regression tests cover all three public fit profiles, the two reported equations
against analytic solutions, causal initial maps, learned-initialization routing,
integration failure, time exhaustion, immutable recovery, historical token/call
accounting, deterministic resume and a real proposal handoff to revision round 1.
The original shared-process smoke also exercises ordinary parameterized fitting.
Cluster NMSEs for the two recovered candidates remain to be measured.

Local verification on 2026-09-19: full `pytest` completed with **2,428 passed**
and four optional Torch-dependent tests skipped. The public-fitting and
shared-process end-to-end smokes passed, including real parameterized fitting
and deterministic resume. Changed-file lint, shell syntax and whitespace checks
passed. Repository-wide `ruff check .` reports 37 existing findings in the
unrelated `analysis/claude` scripts.
