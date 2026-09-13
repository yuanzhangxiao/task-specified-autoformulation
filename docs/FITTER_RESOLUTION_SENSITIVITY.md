# Resolution and sparse sensitivity milestone

The previous reference experiment exposed two execution failures: a global mesh
budget could leave a whole constant-input trajectory in one interval, and every
attempted sensitivity refinement timed out before its first Jacobian returned.
That is not evidence of local optimizer convergence to a bad fit.

## Changes

* The opt-in mesh safeguard bounds each trajectory's collocation interval by its
  duration divided by `collocation_minimum_intervals`. The v3 profile uses 120.
  Required input corners and this state-resolution floor are both allocated
  before the remaining global budget. The target is soft, and the report shows
  actual variables, maximum gaps and why the target was exceeded. It never
  inserts observations or discards forcing samples. Original sample gaps larger
  than the requested resolution are reported, not silently certified.
* The opt-in sparse solver Jacobian retains CasADi's structural sparsity when
  supplying the analytic augmented Jacobian to Radau/BDF. Equations, first
  parameter sensitivities, tolerances, observation residuals and input segments
  are unchanged. Small systems may not benefit from sparse factorization.
* Recovery can prioritize the original parameter guess and retains its aliases
  when an identical collocation checkpoint exists. It records the last actual
  evaluation status, including timeout errors, rather than treating every
  unfinished point as evidence of mathematical infeasibility.
* Distinct retry mode excludes already attempted starting vectors when selecting
  a second sensitivity start. Every feasible incumbent remains eligible for the
  final training-cost comparison and for polling; exclusion applies only to
  identical sensitivity retries under the same allowance.
* V3 grants 60 seconds per state screen and 180 per sensitivity evaluation while
  retaining the same 600-second total refinement budget and 480-call cap. The
  300-second/1000-iteration collocation allocation is unchanged. This is an
  allocation correction, not an increase of the total fitting budget.

The old configuration defaults retain their original numerical choices. This
milestone is explicitly selected by `configs/fitter_resolution_v3.json`.
Reference information remains isolated from public proposer and judge feedback.
No benchmark table or prompt changes. No LLM calls and no test observations.

## Small gated Delta campaign

Preparation reads the completed `fitter-attainability-v2` source. It verifies
source hashes, the reference translation and initial/input protocol, generation
audit, and numerical environment. It copies the exact actual and synthetic
observations into a new root, retaining original generation provenance. This is
a newly authorized experiment with changed numerical settings, not outage
recovery or continuation of exhausted v2 fits. V2 outputs are never modified.

The dependency chain is:

1. Sparse/resolution numerical smoke, with a terminal checkpoint.
2. Six fixed-point profiles: generating, original near-truth and ordinary
   parameters, each with dense and sparse Jacobians. The generating and near
   points are explicitly privileged controls. Each profile measures a whole
   training residual (60-second limit), then a whole training Jacobian
   (180-second limit). Two deterministic directional finite differences on the
   first training trajectory check analytic derivatives. The profile has a
   360-second total budget and an outer process-group supervisor.
3. Two fixed-parameter node solves: original full mesh and safeguarded reduced
   mesh. All initial state values and all generating parameters are known for
   these controls; hidden state trajectories are not supplied to the optimizer.
4. A preflight gate requires both node solves to converge with node NMSE <=1e-4
   and constraint maximum <=1e-6. The generating sparse profile must complete
   both directional checks (relative error <=0.005) and have full-training
   synthetic NMSE <=1e-6. Where both formats return a full Jacobian, their
   maximum prediction difference must be <=1e-5 in training-output-SD units and
   Jacobian relative difference <=0.001. Comparing near-zero residuals by their
   relative difference would incorrectly amplify negligible numerical error.
   A timed-out
   dense control is reported but does not veto an independently verified sparse
   profile. These are engineering checks chosen before this campaign, not a
   new scientific holdout.
5. Only after the gate passes: four synthetic fits (ordinary/near-truth crossed
   with full/safeguarded mesh) and two actual-observation fits (ordinary and
   near-truth, safeguarded mesh). These reference fits all use known per-run
   hidden initial conditions. They do not resolve shared-initial limitations.
6. Report every profile, gate and fitting outcome, including blocked fits. Final
   scores use independent production replays; collocation node paths are never
   scored as fitted predictions. Strict recovery still requires both synthetic
   NMSEs <=1e-4; practical scores are shown even when this threshold is missed.

The profile arrays are retained as hashed compressed artifacts for numerical
comparison, not committed. Profiles checkpoint terminal outcomes; interrupted
profiles and fits cannot receive fresh budgets on resume. Native interrupted
optimizer state is not reconstructible, and is reported as interrupted. The
submission script returns existing job IDs instead of submitting duplicates.
The two arrays run sequentially before fitting, so at most two single-CPU tasks
run concurrently. No GPU or login-node numerical computation is needed.

## Commands

From a clean checkout of `codex/fitter-resolution-sensitivity-v3` on Delta:

```bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-attainability-v2
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-resolution-v3
export AF_ARRAY_CONCURRENCY=2
bash scripts/hpc/submit_resolution_delta.sh
```

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-resolution-v3/preflight_summary.md
cat /work/hdd/bibo/yxiao2/phase_b/fitter-resolution-v3/summary.md
```

Reconstruct reports without numerical computation if the summary job fails:

```bash
"$AF_PYTHON" -S scripts/run_resolution_campaign.py summarize --output "$AF_OUTPUT_ROOT"
```

Local verification uses `pytest`, `ruff check .`, shell syntax checks, the sparse
numerical smoke and its deterministic resume, and reference fixed-point checks.
Passing the preflight certifies execution of the tested numerical components;
it does not establish ordinary-start parameter recovery or readiness to freeze
the complete fitter. The six gated fits provide that next evidence.
