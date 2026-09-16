# One numerical-feedback sibling

The opt-in successor is documented in `PREFIT_NUMERICAL_FEEDBACK_V2.md`. It adds
logged citation normalization, precise missing-reference feedback and an explicit
structural-hypothesis task. This document retains the original pilot contract.

The primary response to an improving fit stopped by its budget is further bounded
fitting. Its present error does not establish a bad equation. This separately
authorized pilot explores one alternative while the original model continues
fitting in `PUBLIC_FIT_CONVERGENCE.md`. It does not replace that branch or consume
its evolving results. The final pipeline's continuation limit remains undecided.

The fixed source is the completed **first continuation**, after two fitting
windows: identity `46e9914dd9d2097ba492adcf8edaecb638da6a8bd93404089afa5143d07c91bb`,
backend SHA `d92f087138f921ab7f634c9c94d9c87963b9a160b4930fa508e77045d6b9e986`.
The selected point had training NMSE 0.8273814092 and validation NMSE 0.9285191153.
Only training evidence reaches the proposer. The recorded vector includes all
eight equation parameters and five learned causal-initialization coefficients.

## Stages and decisions

1. Reconstruct the accepted requirement repair from its immutable construction
   audit and cached responses. Check its exact public brief, topology, current
   functions, initialization and bound nonlinear-feedback requirement against
   the historical public fit. No historical fit root is executed under new code.
2. Replay the current and previous parameter points on training trajectories,
   using the original production solver, initialization and normalization.
   Each point has a 300-second replay cap; no optimization occurs. The retained
   score must reproduce the saved score. An unavailable retained replay blocks
   proposer work; the previous comparison may be unavailable separately.
3. Give GPT-OSS-20B the public brief, equations, exact source/sign contracts,
   unchanged initializer plan, explicitly labeled retained parameter estimates,
   and the bounded training residual packet. Allow one episode, seed 0, at most
   three physical requests, low reasoning, temperature 0.2 and 8192 output tokens.
4. Accept `no_change`, `topology_revision_needed`, or `revise_function`.
   Every reply includes a hypothesis and visible evidence IDs. The first two
   decisions finish without editing or fitting. An exact canonical no-op also
   gets no fit. Exhausted invalid attempts preserve the parent.
5. A revision changes exactly one existing interaction RHS. Recheck its exact
   source set, restricted expression grammar, parameter roles, topology-owned
   outer sign, every other canonical function, the exact causal initializer
   plan, and the bound nonlinear feedback cycle. No topology or initializer
   redesign is performed. A failed check rolls back the whole proposal.
6. Fit only a changed accepted child with the unchanged
   `collocation-feasible-v1` profile: 120-second collocation allowance,
   180-second screening/refinement allowance, and 240-call ceiling. Report
   parent and child separately. There is no automatic winner, scientific judge,
   pruning, further revision, or test-data evaluation.

The proposer may reasonably return **insufficient evidence; keep the model**.
Residual patterns motivate hypotheses at the retained parameters. They do not
identify a faulty term, demonstrate instability, or prove structural inadequacy.
No quadratic-specific rejection or mandatory replacement is introduced.

## Measured evidence and fitting start

[Training residual feedback](TRAINING_RESIDUAL_FEEDBACK.md) defines aligned
observed/predicted samples, signed standardized residuals, trajectory and time
window errors, common-tolerance turning counts/ranges, selection rules, omissions,
and numerical uncertainty. Predictions are open-loop and refer to exactly the
parameter vector supplied alongside them. Latent states are never supplied as
observations. The packet carries candidate, vector and training-content digests.

The child fit's full initial vector is a separate sealed adapter input. It is
not encoded by changing `PublicFitRequest` or the scientific initialization plan.
Runtime role starts and initializer guesses supply defaults; learned values
overlay those defaults only for identical surviving parameter declarations.
Changed/new declarations start fresh, removed names are omitted, and every
unchanged initializer coefficient is carried forward. Child domains and complete
parameter coverage are checked before the backend runs. This is an initial
vector for the existing profile, not resumed optimizer state.

The continued-fitting parent and the child receive different additional work.
Their scores form a descriptive branch exploration, **not an equal-compute
comparison**. One case cannot establish a general allocation rule or controller
advantage. Better train or validation error alone is not scientific acceptance.

## Checkpoint and execution contract

The new campaign root contains `plan.json`, `evidence/`, `results/calls/`,
`results/state.json`, optional `child_fit/`, and `summary.json`. Provider bodies
and responses are cached. Canonical request serialization preserves retry identity
across checkpoint reloads; uncertain delivery consumes its physical request.
Provider time sums recorded request latencies; missing timings are counted
separately as `unknown_latency_requests`.
Reports replay cached acceptance and bind any child fit to this exact decision,
parent vector, packet, public data and frozen fitting profile.

A reservation beside the historical experiments under `.numerical-siblings/`
permits one sibling allocation for the selected completed continuation. Changing
the output directory cannot grant another allocation. A started incomplete replay
or child fit gets no new numerical budget. A completed stage resumes unchanged;
pending proposer attempts resume within the original three-call allocation.
Reports never invoke numerical work. A campaign lock prevents concurrent workers
or reports from publishing inconsistent states.

ACES submission saves an intent before `sbatch`; uncertain or partial submission
requires inspecting that intent and the queue instead of automatically retrying.
Repeating a completed submission command returns its existing job IDs. Workers
recheck the pinned clean checkout, environment paths, configuration and historical
file digests before work. All historical experiment directories remain read-only.
The original convergence job is neither a dependency nor an input.

## ACES execution

Use a fresh clean checkout at the published milestone commit. These scripts do not
require `python` on PATH or the shell's current directory. The shared virtualenv
and scratch cache paths match the existing ACES experiments. Each command below
is a single physical shell line; replace `PUBLISHED_COMMIT` with the commit supplied
with the milestone, not a moving branch tip.

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-public-fit-convergence-v1 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-public-fit-convergence-v1 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v1 PUBLISHED_COMMIT
```

```bash
AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v1/scripts/hpc/submit_prefit_numerical_sibling_aces.sh
```

The chain is CPU preflight/replay → one H100 proposer → CPU child-fit/report.
Proposer submission depends on successful preparation. The final CPU stage runs
after any proposer outcome, publishes available results, and does not interpret
scheduler completion as scientific success. If preparation never produced a plan,
the summary reports `preparation_missing` and points to the preparation logs.

```bash
sacct -j "$(jq -r '[.prepare_job,.propose_job,.fit_job]|join(",")' /scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v1/submission_manifest.json)" --format=JobID,JobName,State,ExitCode,Elapsed
```

```bash
jq '{status,replay_status,proposal_outcome,physical_requests,observed_tokens,parent_training:.parent.training.normalized_mse,parent_validation:.parent.validation.normalized_mse,child_training:.child.result.training.normalized_mse,child_validation:.child.result.validation.normalized_mse,decision:.decision.reply,limitation}' /scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v1/summary.json
```

After jobs have stopped, the read-only report can be regenerated:

```bash
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v1/scripts/prefit_numerical_sibling.py report --root /scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v1
```

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_residual_evidence.py tests/test_fit_residual_feedback.py tests/test_numerical_sibling.py tests/test_sibling_fit.py tests/test_prefit_numerical_sibling_integration.py tests/test_prefit_numerical_sibling_submission.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_numerical_sibling.py
```

The smoke uses explicit synthetic timeout histories with production-scored
parameter points, real production training replay and child fitting, and mocked
LLM delivery. It verifies the complete edge and exact resume; it does not predict
fresh 20B proposal quality or the real benchmark's fitting result.
