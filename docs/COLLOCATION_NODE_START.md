# Optional starting rollout for collocation

The earlier adapter had to simulate every training trajectory at the ordinary
parameter start before IPOPT could begin. A failed starting rollout therefore
prevented collocation from optimizing any state nodes or parameters. This
milestone makes that rollout optional under an explicit new policy. It does not
change the equations, parameter domains, prescribed initial conditions, fitting
objective, or sensitivity stopping rules.

## Implementation

`CollocationSensitivityConfig.collocation_node_start` accepts:

- `rollout_required`: historical control; a failed starting rollout aborts.
- `rollout_or_observed`: try a rollout within a shared warm-up allowance, then
  initialize nodes directly if it fails or the allowance expires.

The optional allowance is `min(node_warmup_seconds, initializer_seconds / 4)`
for **all** training trajectories combined, defaulting to 10 seconds. It is part
of the existing initialization budget. This reserves time for construction and
IPOPT instead of spending the whole budget on starting rollouts. Successful
rollout guesses are retained.

For a fallback, directly observed state nodes use training target/auxiliary
observations. Other state nodes repeat the state's prescribed initial value.
The first node always retains the candidate's actual fixed or causal analytic
initial condition, even if the observed sample differs. Interior Radau nodes use
linear interpolation of these guesses. No hidden labels or validation values
enter node construction. These are guesses, not fixed trajectories: IPOPT still
jointly optimizes state values and parameters under the same two-stage Radau
collocation constraints.

Local RHS, observations, and derivatives must be finite at the initial/end/interior
guesses. This checks expressions without integrating an ODE. It adds no new state
bounds or scientific repairs. The node journal records each guess's source, the
failed rollout's factual diagnostics when available, and whether IPOPT started.
The parent retains that journal even if the native optimizer is killed on timeout.

The adapter's default remains the historical policy. Sol can opt into the new
policy explicitly in the next frozen campaign after this comparison. Multiple
shooting retains its original initialization policy.

## CPU comparison

The exporter takes the existing version-3 multiround campaign. It verifies its
plan, public ledger, terminal identities, committed candidate hashes, and parent
chain. It exports **all original parents and all committed round candidates**,
including failed fits. It does not choose candidates by training or validation
score. Files are restricted to public prompt/manifest/train/validation, candidate
JSON, and provenance records; no test table or LLM cache is copied.

The new experiment imports that bundle into a separate immutable snapshot. It
freezes the exact role-based ordinary starting vector for each candidate before
running either policy. Original parents and accepted revisions are reported
separately. Three additional controls cover an explosive observed-state start,
an explosive latent-state start with an algebraic observation, and a successful
starting rollout. The controls use x'=a*x^2 with finite observed trajectories
generated at a=0.1; the unstable start is a=1.0 over [0,2].

For the reported ACES run, two parents plus two completed revisions and three
controls give **14 fits** (seven cases, two policies). The exporter reports the
actual frozen count if the source differs. Each arm receives 120 seconds for
initialization and 360 for refinement, with 80 maximum optimizer evaluations.
Forward sensitivities, parameter roles, Radau tolerances, ordinary starts, and
fresh production training/validation replays are the same in both arms. A
successful starting rollout is never a gate for the optional arm or the campaign.

There is one CPU and 16 GB per task, at most two concurrent tasks, no GPU or LLM
calls, and a 30-minute Slurm limit. The worker cap includes both production replays
and shuts down its process group. Completed arm fits/results are checkpointed.
A completed fit can rebuild a missing result without refitting; an interrupted
native fit is terminal and never silently receives a new budget. Imported bytes,
runtime packages/source, and launcher hashes are checked on resume. The summary
job also gets 30 minutes; duplicate and partial submissions are guarded.

## Export on ACES and transfer from the Mac

The exporter supports Python 3.6+ using only its standard library and can run
through stdin. The earlier version failed on ACES system Python at its postponed
annotations import; the standalone exporter now avoids that import and newer
typing, pathlib, and string APIs. The fitting runtime still uses its existing
Python 3.12 environment on Delta.

From the Mac, using its existing `aces` and `delta` SSH aliases:

```bash
ssh aces 'python3 - --source-root /scratch/user/u.yx126462/phase_b/staged-multiround-feedback-v3-a865f8a --output /scratch/user/u.yx126462/phase_b/collocation-node-cases-v1' < /private/tmp/autoformalism-astra-collocation-node-start/scripts/export_collocation_node_cases.py &&
scp -3 -r aces:/scratch/user/u.yx126462/phase_b/collocation-node-cases-v1 delta:/work/hdd/bibo/yxiao2/phase_b/
```

The second command runs only if export succeeds and routes the copy through the
Mac. The original campaign remains untouched. Repeated export requires
byte-identical contents; a changed source
requires a newly named bundle directory. SSH/MFA is handled by the user's terminal.
The compatibility update preserves bundle bytes and format. The previously
supplied Delta commands pinned to `f92d07a` remain valid.

## Run on Delta

From a clean checkout of this commit:

```bash
export AF_REPO_ROOT="$PWD"
export AF_CONFIG="$PWD/configs/collocation_node_v1.json"
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/collocation-node-cases-v1
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/collocation-node-v1
bash scripts/hpc/submit_collocation_node_delta.sh
```

The launcher uses the existing Python at
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python` and CasADi dependencies at
`/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps`; it installs nothing.

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/collocation-node-v1/summary.md
```

`results/case_NNN_POLICY/fit.json` retains the initializer journal, native and
accepted optimizer outcomes, valid residual counts, parameters, failure evidence,
and production metrics. Distinguish reaching IPOPT, finding a feasible model,
fitting accurately, and satisfying the public scientific requirements.

## Remaining limits

Neutral latent guesses can be poor and collocation can still fail or find a local
solution. The finite-node checks do not certify safety between nodes; fresh ODE
replay remains essential. This experiment does not fit latent initial values,
change topology/functions, add singularity constraints, or establish scientific
correctness. It addresses a blocked initialization path, separately from the
completed exploration of alternating optimization. Real-candidate results still
require the user-run cluster experiment; local verification uses synthetic data.

## Local verification

- Initial milestone suite: **1,268 passed, 3 skipped** (optional PyTorch unavailable).
- Ruff and shell syntax checks passed.
- Actual explosive-start controls recovered `a=0.1` with validation NMSE below
  `1e-9` using the optional policy, for both direct and algebraic observations;
  the historical policy stopped before collocation optimization.
- Campaign tests covered source hashes, parent chains, paired starts, completed
  checkpoint reuse, interruption/timeout handling, and supervised CLI execution.
- The standard-library exporter also passed a standalone stdin smoke test,
  matching the ACES invocation above.
- The compatibility regression checks Python 3.6 syntax, rejects newer
  pathlib/string API calls, and exercises export and identical resume in an
  isolated subprocess without site packages. Containment tests cover traversal
  and symlink escapes.
  The local machine has Python 3.12, so this does not claim an actual ACES run.
