# Twenty-four-hour review evidence campaign

This is a bounded integration experiment. It does not reopen fitter research or
promise recovery within a fixed deadline. The useful outcome is an auditable
matrix of successes, failures, few-round progress, and a controlled demonstration.
The experiment makes no automatic extension, topology-repair, or private-test
selection decision.

## Matrix and priorities

The configuration is `configs/review_deadline_v1.json`:

| Generating family | Public cell | Seeds | Visits |
| --- | --- | --- | --- |
| Dalla Man | canonical, named, T1 easy | 0, 1 | 3 |
| Dalla Man | canonical, obfuscated, T1 easy | 0, 1 | 3 |
| Dalla Man | perturbed, named, T1 easy | 0, 1 | 3 |
| Dalla Man | perturbed, obfuscated, T1 easy | 0, 1 | 3 |
| CSTR | controlled reactor, canonical, named, easy | 0, 1 | 3 |
| Alien device | unknown device, canonical, functional, easy | 0, 1 | 3 |

These are **three families, six cells**, not six independent systems. The complete
Dalla Man 2×2 contrast separates description changes from changes in dynamics.
Easy single-output cells make the expanded matrix compatible with the frozen
fitter. Auxiliary observations remain available exactly as their public contracts
permit: these are conditional predictions when the model uses supplied auxiliary
paths, not claims to simulate every unobserved subsystem autonomously.

Priorities are: complete public construction and fit; report all round endpoints;
open held-out evaluation only after global freeze; inspect the controlled example;
then interpret ablations. The first visit is also the no-iteration endpoint. Two
further visits test one evidence-citing function revision each, rather than claiming
that a three-point learning curve establishes convergence.

## Ablations that actually run

| Arm | Cells/seeds | Intervention |
| --- | --- | --- |
| full | all 6 × 2 | training shape summary, scientific brief, persistent states, deterministic checks, then residual feedback |
| brief_only | all 6 × 2 | withhold the initial shape summary during topology/functions/initialization; later training residual feedback remains enabled |
| refit_only | all 6 × 2 | reuse full visit zero, then apply the same per-visit fitter budget to unchanged equations and the retained learned parameters; no new proposer calls |
| no_latent | named canonical Dalla and CSTR × 2 | permit differential equations only for directly observed target states; other generated quantities must be algebraic |
| no_spec | obfuscated canonical Dalla and alien × 2 | withhold scientific context, requirements and required dependencies; keep channel roles, data, grammar and runtime safety |

There are **44 lineages and 132 planned round rows**. Round zero has 32 fresh
constructions; 12 refit controls share their full counterpart. The maximum is
120 actual fitting allocations, each 120 seconds for collocation plus 180 seconds
for refinement. Actual counts, provider tokens (including unknown usage), fit
calls, retained parameters and numerical qualifications remain in the artifacts.
Refit controls' shared initial cost must be included when comparing total costs;
reports separately label their incremental counts. No-iteration is a smaller-budget
endpoint, not a compute-matched method. Refit-only repeats the same C+S allocation
from learned parameters; it is not continuation of the optimizer's internal state.

The judge is off in every arm; this is the judge-free operating protocol, not an
estimate of the judge's incremental benefit. The controlled demonstration contains
a fixed-pool verifier-on/off audit. That audit isolates acceptance of incorrect
equations; it is not a complete search-without-verification experiment. No graph
meta-model, judge-on, or new external baseline implementation is claimed here.
Wrong specifications, broader system coverage and those more expensive ablations
are explicitly deferred. Existing frozen baselines may be compared only after
checking target roles, rollout rules, information access and compute budgets.

## Construction, fitting and feedback

Visit zero uses the current staged topology, equation-batch function construction,
outer-sign contract and train-fitted causal latent initialization. Each call is
cached under its complete request, with delivery outcomes and charged usage.
The same model/revision, seed policy and per-construction request/token caps apply
to paired arms. Benchmark prompts and observations are copied, not modified.
Ablated instructions are explicit experimental overlays in the saved brief.

`collocation-single-target-v2` removes the historical `v01` spelling restriction:
it binds the sole declared output (including `Gp` or `T`) without renaming public
variables. Solver settings, budgets, state-boundary rules, collocation and
sensitivity/polling choices remain those of the frozen adapter. Multi-output
collocation is still unsupported. Named-versus-anonymous residual/Jacobian
invariance and a real named-target fit are tested.

Each later visit receives a fixed-parameter **training-only** residual packet,
complete retained coefficients, and numerical-status qualifications. It can change
one existing interaction function while retaining topology, source identities,
other functions and initialization rules. Matching parameter declarations reuse
learned values, including causal initial-map coefficients. Requests for topology
changes are saved as such and close this bounded branch; no covert architectural
rewrite follows. A no-change reply also closes the revision branch.

Fitted coefficients are selected using training only. Parent versus child selection
uses finite validation NMSE, then additive term count, among admissible models.
The incumbent survives a worse or failed child. The proposer does not receive
validation scores or test information. This is a simple bounded controller,
not MCTS and not a globally optimal search.

Equation-derived target/driver/memory predicates are separate from scientific
interpretation. Certified failures exclude a full-arm model from selection.
Requirements whose semantics cannot be certified remain explicitly **ambiguous**;
they are never counted as certified success. In the no-spec arm, withheld science
requirements are evaluated for reporting only and cannot secretly select a model.
The structural checker does not certify arbitrary signs, units, physiological
interpretations or causal uniqueness.

## Controlled identifiability demonstration

Two supplied, equally complex models have

```
z' = -a*z + b*u_j
y' = z-y
z(0) = y(0) = 0
```

Only `y` is observed. Both models use the same ordinary parameter guesses and the
frozen fitter. On train/validation, `u1=u2`, so choosing `j=1` or `j=2` is exactly
observationally equivalent. A true public requirement specifies a persistent
`u1`-driven hidden pathway to `y`; the equation verifier accepts one and rejects
the other **before** held-out inputs are opened. Held-out inputs decouple the two
drivers. Without the driver-specific requirement, both models are reported as
compatible; small numerical score differences cannot justify a confident choice.

This is a controlled identifiability/verifier example, **not evidence of LLM
model discovery**. It adds no independent benchmark family and tests no wrong
specification. The fixed-pool audit also rejects disconnected and instantaneous
substitutes while accepting latent renaming and rescaling.

A local numerical smoke produced equal train NMSE about `1.02e-16`; intervention
NMSE was about `1.22e-18` for the correct driver versus `2.99` for the other model.
These are controlled local results, not results from the six-cell campaign.

## Running on ACES

Orion's parameter-declaration reuse patch is included. These scripts do not touch
Orion's existing experiment directories. One H100 serves each proposer visit;
CPU fits use one CPU/16 GB each, at most sixteen concurrent array tasks. GPU
visits have six hours of worker time plus startup/drain margin. Queue delays are
not included: the complete matrix may exceed 24 hours if allocations wait. Freeze
a partial matrix at the deadline rather than changing the declared denominators.

The known two-cell prefit releases are insufficient. If the complete six-cell
release has not already been staged on ACES, run this **on the Mac**, from this
checkout. It copies only manifest, prompt, train and validation from the documented
Delta prompt-v3 release through the existing `delta` and `aces` SSH aliases:

```bash
cd /Users/yuanzhangxiao/Projects/autoformalism
bash scripts/hpc/stage_review_deadline_public.sh
```

No test or private-reference files are transferred by that helper. It uses
`/work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3` as source and
`/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1` as destination.
Override `AF_DELTA_PUBLIC_ROOT` or `AF_PUBLIC_ROOT` if those releases moved.
The helper checks ACES authentication first and verifies all 24 remote file
checksums before reporting success. It accepts both macOS openrsync and GNU
rsync new-file markers, retains identical files on resume, and refuses to
overwrite differing content. A failed or interrupted transfer can be rerun.
Do not submit until the helper prints `Verified 24 development files on ACES.`
Prompt hashes, manifest identities and loaded channel roles are checked before
submission; a missing/mismatched release stops before GPU use.

On **ACES**, use an existing authenticated clone to fetch the commit reported in
the task's final response. `REV` must be that exact hash:

```bash
set -euo pipefail
REV=REPLACE_WITH_REPORTED_COMMIT
BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
git -C "$BASE" fetch origin
export AF_REPO_ROOT="/scratch/user/u.yx126462/repos/autoformalism-review-${REV:0:7}"
if [ ! -e "$AF_REPO_ROOT/.git" ]; then
  git -C "$BASE" worktree add --detach "$AF_REPO_ROOT" "$REV"
fi
[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" = "$(git -C "$BASE" rev-parse "$REV")" ]
export AF_PYTHON="$BASE/.venv/bin/python"
export AF_PUBLIC_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1
export AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/review-deadline-v1
bash "$AF_REPO_ROOT/scripts/hpc/submit_review_deadline_aces.sh"
```

The submission prints a job manifest. Preparation runs focused tests and a real
fit smoke before downloading/verifying model assets. Construction, CPU fitting
and barrier/report jobs are dependency-linked. The controlled demo is a separate
CPU job. Repeating the submission returns the same job manifest. A partial
submission intent stops rather than silently creating duplicate jobs.

Check progress/results:

```bash
ids=$(jq -r '.jobs|values|join(",")' "$AF_OUTPUT_ROOT/submission_manifest.json")
sacct -j "$ids" --format=JobID,JobName,State,ExitCode,Elapsed
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" report --root "$AF_OUTPUT_ROOT"
cat "$AF_OUTPUT_ROOT/SUMMARY.md"
cat "$AF_OUTPUT_ROOT/controlled-demo/SUMMARY.md"
```

Cache/fit checkpoints resume only under the same frozen source, runtime, assets,
config and hashes. Missing proposals are not numerical failures. An interrupted
CPU worker consumes its recorded allowance; it cannot silently acquire a new
optimization budget. Inspect `results/TASK/round_XX/worker.log` for execution
failures. Resume commands operate on the same root; never delete markers or
change a frozen plan to turn an interrupted run into a fresh experiment.

## Global freeze and held-out evaluation

Once all jobs are terminal, seal all round endpoints:

```bash
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" export --root "$AF_OUTPUT_ROOT"
```

At the deadline, if results remain missing, cancel only this campaign's recorded
pending/running jobs, wait for them to stop, then use `export --allow-partial`.
The missing rows remain in the denominator. Active workers prevent freezing.
After `evaluation_freeze.json` exists, no further proposal or fit is allowed in
that root. No automatic private-test job is submitted.

`subjects.jsonl` is compatible with the existing common evaluation schema.
Its exact source/result/candidate hashes and **all** learned parameters are frozen.
The campaign `evaluate` command uses the existing unseen-condition free-rollout
and intervention endpoint; it never fits parameters or validation/test initials.
It verifies that the evaluation release's public files match the campaign.
Use the complete Delta release for test access. After global freeze, copy the
campaign through the Mac (the destination should be a new experiment directory):

```bash
# Mac, after ACES export has completed:
mkdir -p /tmp/review-deadline-transfer
rsync -a aces:/scratch/user/u.yx126462/phase_b/review-deadline-v1/ /tmp/review-deadline-transfer/
ssh delta 'mkdir -p /work/hdd/bibo/yxiao2/phase_b/review-deadline-v1'
rsync -a /tmp/review-deadline-transfer/ delta:/work/hdd/bibo/yxiao2/phase_b/review-deadline-v1/
```

On Delta, check out the same reported `REV` using an existing authenticated clone,
set `AF_REPO_ROOT` to that clean checkout, set
`AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/review-deadline-v1`, and set
`AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`.
Do not copy an ACES virtual environment to Delta. Evaluation seals the local
numerical package versions separately and requires the original source hash.

```bash
# On Delta, after copying the frozen campaign and checking out REV:
export PYTHONPATH="$AF_REPO_ROOT/src"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" evaluate \
  --root "$AF_OUTPUT_ROOT" \
  --public-root /work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3 \
  --shard 0 --shards 1
```

For an explicit CPU-only Delta array, use the supplied submission script:

```bash
bash "$AF_REPO_ROOT/scripts/hpc/submit_review_deadline_evaluation_delta.sh"
# When it finishes:
cat "$AF_OUTPUT_ROOT/EVALUATION.md"
```

It records its job IDs, runs 44 disjoint shards with at most 16 active CPU workers,
and schedules a summary after all tasks terminate. `EVALUATION.md` includes all
132 planned round rows; missing models and missing evaluations remain distinct.
The array is deliberately a separate user-run step after global freeze.

Use CPU batch jobs, not a login-node evaluation. The command supports disjoint
`--shard`/`--shards` values and per-subject checkpoints. Reports retain rollout
failures and do not describe independent numerical consistency as good prediction.
Hidden perturbation-response endpoints already exist in the common evaluation
code, but this campaign does not claim to have run them until their separate
results are attached. The primary deadline result is observed-output free rollout
under held-out conditions, with task-structure and complexity reported separately.

## Foundation and what remains after the deadline

Keep the manifest, every proposal, cached call, fit result, complete learned
initializers, immutable round selections, and evaluation freeze. The same schema
supports more seeds, more visits, and additional families in **new** experiment
roots. Expand structural routing only after inspecting the recorded requests for
topology changes. Treat reused/development benchmark cases as exploratory.

Do not claim unique latent coordinates, broad cross-domain recovery, a calibrated
Bayesian prior, a general formal specification compiler, or successful conflict
detection from this campaign. Write the paper around what actually completes.
The frozen numerical protocol may fail; that is an experimental outcome, not an
invitation to spend the deadline reopening fitting research.
