# Nine-benchmark final component campaign

Protocol `final-component-campaign-1`. This is a **fresh, prospectively defined**
search campaign, not a continuation or a combination of selected historical models.
Training fits parameters and supplies proposer feedback. Validation selects search
incumbents and the final pruning endpoint. Test trajectories are opened only by a
separate command after every planned endpoint has been sealed. Running an ablation
on a test benchmark means doing its search on that benchmark's public development
splits and then scoring its frozen model on the held-out split.

## Frozen matrix

| Cell | Benchmark | Matched blocks |
| --- | --- | --- |
| 00 | Dalla Man, T1 easy, canonical named | 0–3 |
| 01 | Dalla Man, T1 easy, canonical obfuscated | 4–7 |
| 02 | Dalla Man, T1 easy, perturbed named | 8–11 |
| 03 | Dalla Man, T1 easy, perturbed obfuscated | 12–15 |
| 04 | CSTR, easy | 16–19 |
| 05 | Alien device, easy | 20–23 |
| 06 | Dalla Man, T1 hard, canonical named | 24–27 |
| 07 | Dalla Man, T2 easy, canonical named | 28–31 |
| 08 | Dalla Man, T2 hard, canonical named | 32–35 |

Each block is one cell/seed/prompt combination: seed 0 Full, seed 0 Brief-only,
seed 1 Full, seed 1 Brief-only, in that order. Every block contains the four
critic/verifier combinations. Both prompt variants remain separately reported;
neither is chosen using test results. There are **144 primary lineages**.

Only cells 00, 06, 07 and 08 add shared-process-off with both critic and verifier
on: **16 additional lineages**. Sharing-on reuses the corresponding primary arm.
Pruning-off uses the exact same final search checkpoint; a paired unchanged-refit
control is also exported. These are endpoints, not additional 15-round searches.
There is no Cartesian product with sharing/pruning, no-spec, no-latent, optimizer
variants, or beam search. Historical baselines/other ablations remain separate.

Every lineage has 15 visits: fresh construction at 0, then 14 revisions. The
single/multiple-output cases use the existing `collocation-multi-target-v1` profile,
without changing fitting budgets or using the separate rescue fitter. A failed
scientific revision preserves the incumbent and the established fallback rule.
A provider/preflight failure stops that lineage's automatic advance for inspection.
Failures remain in the denominator.

The campaign also restores the already agreed whole-model revision contract for
multi-output replies: there is no six-equation/two-new-variable patch quota.
The serializer's actual capacities are shown in the prompt; expression, causal
boundary, role and provider-budget checks remain. Historical protocols retain
their original revision schemas.

## Treatment definitions

- **Critic on:** corrected paired GPT-OSS-120B over the Jetstream managed API,
  existing symbolic stages and aggregation. A new executable draft is reviewed
  before fitting; retained-model advice joins the next revision prompt. Changed
  selected equations after pruning get a new review; unchanged equations reuse
  their review. No judge score or veto enters model selection. The critic sees
  equations and the same public specification, not fitted NMSE or trajectories.
- **Verifier off:** keep the same public specification and scientific agenda, but
  remove scientific driver/pathway/memory/nonlinearity admission and repair gates,
  representation/composition/mechanism eligibility gates, scientific certificate
  feedback, and public-predicate pruning gates. Keep complete generated targets,
  allowed inputs, causal initializers, restricted expressions, parameter meaning,
  algebraic cycle checks, and consistency of the proposer's own shared declarations.
  This is distinct from removing the task specification. Scientific evidence is
  still computed and retained for independent reporting, not exposed as feedback.
- **Shared-process off:** disable the optional dedicated proposal/assembly stage
  and shared-process guidance. Ordinary named algebraic expressions and ordinary
  reuse remain possible; this does not prohibit every mathematically shared term.
- **Pruning:** reuse the one-deletion training contribution ranking and paired
  refits described in [PROCESS_AWARE_PRUNING.md](PROCESS_AWARE_PRUNING.md). The
  verifier assignment applies to pruning too. The same original parent and
  unchanged-refit control compete with the child. The existing 1% validation
  tolerance is explicit and unchanged.

The managed API must pass the existing complete 14-case, five-seed, two-orientation
known-case revalidation before enabling this campaign. The successful one-pair
compatibility check alone is insufficient. Copy only its `plan.json` and
`summary.json`; the campaign validates their identity and protocol and records
an authorization receipt. Passing does not establish fresh holdout calibration;
managed weight revision and settings forwarding remain unverified. See
[JUDGE_JETSTREAM_CALIBRATION.md](JUDGE_JETSTREAM_CALIBRATION.md).

## First launch on ACES

Use group scratch for source, campaign files and runtime caches. Existing input
files and the existing Python environment can remain in personal scratch. Do not
make another personal-scratch clone. Use the full commit printed with the change:

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set the full supplied commit}"
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  AF_GIT="$AF_GROUP/repos/final-components-source.git"
  export AF_REPO_ROOT="$AF_GROUP/repos/final-components-${AF_COMMIT:0:7}"
  mkdir -p "$AF_GROUP/repos"
  if [[ ! -d "$AF_GIT" ]]; then
    git clone --bare --single-branch --branch codex/prefit-aces-v1 \
      https://github.com/yuanzhangxiao/task-specified-autoformulation.git "$AF_GIT"
  fi
  git --git-dir="$AF_GIT" fetch origin codex/prefit-aces-v1
  git --git-dir="$AF_GIT" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git --git-dir="$AF_GIT" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  export AF_JUDGE_GATE_ROOT="$AF_GROUP/judge-jetstream-calibration-v1"
  bash "$AF_REPO_ROOT/scripts/hpc/submit_final_components_aces.sh"
)
```

First upload the completed Jetstream revalidation's `plan.json` and `summary.json`
to `AF_JUDGE_GATE_ROOT` (or set it to their actual existing directory). The script
asks for the API key without echo. It never writes the key into source, manifests,
arguments or logs; Slurm receives it in the inherited job environment. Do not use
shell tracing. All submissions require the gate, preventing partial treatment
execution under an unvalidated critic.

By default the public-only staging command combines the six-cell export at
`/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1` with the three-cell export
at group scratch `dalla-response-feedback-r14-v1/public`. These are known earlier
campaign paths, not a claim that every file is still present. It checks all 36
allowlisted files before copying anything, refuses differing duplicate exports,
and copies no test files. If your exports are elsewhere, stage them explicitly:

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" stage-public \
  --source /actual/six-cell/public-root \
  --source /actual/three-cell/public-root \
  --root /scratch/group/p.nairr260351.000/u.yx126462/final-component-inputs-v1
```

Then set `AF_PUBLIC_ROOT` to that output and repeat submission. Do not regenerate
benchmark data or amend finalized prompts to satisfy a missing-file check.

The default allocation wave requests eight **independent one-H100 proposer jobs**,
four one-CPU API clients, 32 one-CPU fit workers, and eight pruning workers. No
four-GPU allocation is needed for the 20B proposer. Each service runs up to six
hours within a 6h30 allocation, keeping its server alive across rounds. Independent
services claim ready work using OS locks; CPU fitting and API review overlap with
GPU construction. Queued workers start wherever available. No test jobs are submitted.

A worker exits before starting a unit that exceeds its remaining margin. Existing
per-call, fit, replay and pruning limits remain in force. Interrupted units retain
consumed budgets. Repeat the **same wave** to recover submission bookkeeping;
confirmed job IDs are reused. An uncertain reply stops and preserves the intent;
inspect the scheduler and receipt, never delete it to force another submission.

To add resources after a wave finishes, set `AF_WAVE=wave2` and rerun the same
submission block. New workers resume only remaining work, even if another wave
is still running. `AF_PROPOSERS`, `AF_CRITICS`, `AF_FIT_WORKERS`, and
`AF_PRUNING_WORKERS` change worker counts, not scientific budgets. Do not change
source, public files, treatment configuration or runtime within a campaign.

## Progress and the 24-hour target

```bash
module load GCCcore/13.2.0 Python/3.11.5
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" report --root "$ROOT"
jq '{search_status_counts, endpoint_status_counts, judge_cost}' "$ROOT/component_summary.json"
squeue --me -o '%.18i %.24j %.10T %.10M %R'
```

`summary.json`/`rounds.csv` hold all 2,400 search rows; `component_summary.json`
joins final treatment flags, NMSE, complexity, proposer tokens and independent
public graph/target certificates. Raw judge availability/calls remain separate.
The latter are **scoped checks**, not a single scientific-correctness percentage.
The existing three-layer mechanism assessment remains the appropriate supplemental
scientific assessment; this launcher does not invent new benchmark-specific probes
or claim graph compliance equals complete mechanism compliance.

Target allocation: hours 0–2 for input/gate/setup and first complete blocks; 2–18
for concurrent fresh searches; 18–21 for remaining pruning/reviews and coverage
checks; 21–24 for global freeze and held-out evaluation. This is a resource target,
not a queue-time guarantee. Inspect actual throughput after the first hour. Add
workers to the measured bottleneck; do not silently reduce seeds/visits or select
only successful/fast models to meet the deadline. At 24 hours an incomplete matrix
is reported as incomplete; no automatic partial test freeze is permitted.

## Delta, Koa and Jetstream

Jetstream supplies the production critic through its managed API; these calls
already remove the 120B GPU queue from the critical path. CPU SUs do not determine
this API's throughput. A new CPU VM is not assumed to have the pinned numerical
stack or a mounted ACES filesystem. Do not point a worker at a copied partial
campaign: absolute fit paths and runtime receipts are deliberately checked.

For additional sites, assign **disjoint whole blocks before freezing**. Example:
ACES blocks 0–19, Delta 20–27, Koa 28–35. This keeps every matched scientific arm
of a block on one site's numerical environment. If the sites are not ready, run
all 36 blocks on ACES instead; never duplicate a block on two sites and retain
whichever result looks better. A separate block/site policy is recorded alongside
wall-clock comparisons. Numerical environments must be unchanged within each site;
package versions should match across sites. Hardware remains a reporting factor.

On each site, create the same commit archive and copy the public-only inputs and
the two gate files. Install/reuse the project environment and cache the **same**
20B revision and serving-image SHA in the config. The CLI works without Slurm:

```bash
# Example: prepare Delta's disjoint blocks before any worker starts.
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" prepare \
  --root "$AF_OUTPUT_ROOT" --public-root "$AF_PUBLIC_ROOT" \
  --blocks 20,21,22,23,24,25,26,27 --platform delta-a40x1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" authorize-critic \
  --root "$AF_OUTPUT_ROOT" \
  --calibration-plan "$AF_JUDGE_GATE_ROOT/plan.json" \
  --calibration-summary "$AF_JUDGE_GATE_ROOT/summary.json"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_component_campaign.py" \
  --root "$AF_OUTPUT_ROOT" --wave wave1 --proposers 2 --critics 2 --fits 16 --pruning 4
```

The submission helper chooses Delta `bibo-delta-gpu`/`bibo-delta-cpu`, one A40,
`gpuA40x4`/`cpu`; it does not request four A40s. Existing environment path:
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`. The same image's 20B
startup on A40 must succeed before treating that resource as available. For Koa
use `--platform koa-h100x1` and explicitly set `AF_GPU_ACCOUNT`, `AF_CPU_ACCOUNT`,
`AF_GPU_PARTITION`, `AF_CPU_PARTITION`, `AF_GPU_GRES` for one allocated H100,
plus the site's real `AF_PYTHON`, image/cache/temp paths. Koa's exact accounts and
current allocation access have not been verified. Its H100 driver probe alone
would not establish model-serving compatibility.

If using ACES sharding, set `AF_BLOCKS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
before its first launch. The generic CPU/API commands on an already configured
site are `component_campaign.py work --stage fit|prune|critic --root ...`.
A remote proposer service must be the frozen 20B server; substituting the managed
120B API for the proposer would change the experiment and is not this protocol.
No remote sessions are opened by the implementation agent.

Download each site's `component_summary.json` and combine them:

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" merge \
  --source /path/aces/component_summary.json \
  --source /path/delta/component_summary.json \
  --source /path/koa/component_summary.json --root /path/combined
```

The merge rejects overlapping blocks or differences in code, public assets or
scientific settings. Missing blocks and incomplete endpoints remain explicit.

## Explicit final freeze and test evaluation

Finish **all** planned blocks across every site before opening any test split.
Freeze every site, regenerate its report and verify merged `missing_blocks=[]`,
no missing endpoints, and `all_sites_frozen=true`. For one ACES root, no merge is
needed. The export includes all planned endpoints, including unavailable models;
further search is forbidden once the receipt exists. Export is deterministic.

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" export --root "$ROOT"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" report --root "$ROOT"
```

Only then submit **CPU** evaluation workers. Supply the existing original full
benchmark root whose public files match the frozen public snapshot byte-for-byte:

```bash
# Run within a CPU allocation, not a login-node numerical job.
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" evaluate \
  --root "$ROOT" --public-root /actual/original/benchmark-root --shard 0 --shards 16
# Submit the same command for shards 0..15, keeping --shards fixed.
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" evaluation-report --root "$ROOT"
```

The evaluator reuses the existing held-out free rollout/intervention endpoint;
it does not refit parameters or expose test scores to the proposer, critic or
selection. `component_evaluation.json` retains status and endpoint denominators,
per-target scores through the common endpoint record, and independent mechanism
assessment from the same evaluator for all arms. Frozen subject JSONL can also be
used by the existing richer public mechanism-assessment workflow. Models with
failed or unavailable predictions are not silently removed from tables.
