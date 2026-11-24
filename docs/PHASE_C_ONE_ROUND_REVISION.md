# One round of revision on three fitted models

This experiment separates who proposes edits from who selects a fitted model. It starts with one saved parent each on T2-easy, the coupled basin, and CSTR-easy. Each case has its own public equation structure, editable family and fitting contract. The driver uses the same decision sequence and accounting rules in all three cases. The families were chosen after earlier work on these development benchmarks.

## Generation and selection

For each parent, GPT-OSS-20B makes three independent calls (seeds 100–102). Each call must return three alternatives in a bounded edit grammar. Every call receives the same scientific task, editable family and training evidence. It sees neither the algorithm's shortlist nor the development trajectories. Repetitions, unsuccessful replies and the unchanged parent consume proposal slots.

The CPU generator screens the following edits. Its shortlist contains three alternatives per parent. A separate control retains the three candidate edits specified at experiment preparation. On the basin both procedures use the same shortlist.

| Case | CPU screen | Common fit after generation |
|---|---|---|
| Coupled basin | Profile upstream and downstream discharge powers from training water balances, rank 24 nonparent combinations | Two discharge gains, matched starts for all edits, whole-event training omission for ranking |
| CSTR-easy | Two whole-event training folds compare interval temperature-rate errors for 11 reaction-heat laws | Fit feed, reaction heat and jacket terms with matched starts and budgets, verify the full temperature forecasts |
| T2-easy | Score initial disposal contributions with persistent or fading effects. Combine three possible initial sources with nine time choices. Opposite initial plasma/tissue glucose contrasts allow nine equivalent screens to be skipped | Refit the disposal intercept and ongoing tissue gain in every model, plus one signed gain if an initial effect is added. Enforce the same training initial-order and domain constraints; generate glucose, insulin and disposal trajectories |

The CPU can use the underlying training trajectories. The proposer sees numerical summaries of the same data. The comparison describes the performance and resource cost of these two procedures. The number of source-power combinations, CPU fits, LLM calls and proposed slots measure different resources.

All unique submitted edits and the parent go through the same case-specific fit and eligibility checks. Checkpoint each distinct fit once, attributing its availability to every pool that proposed it. Freeze these outcomes before constructing the ranking prompts. Three 20B calls rank the same eligible union in independently shuffled orders. The identifiers hide the proposal origin. The judge sees the scientific evidence and the candidate equations encoded as edits. Child training scores stay hidden from it. Restrict each returned ranking to each pool to select its best available member. This measures scientific preference within a common menu. An incomplete ranking records a failed selection.

Compare against the numerical selector (lowest eligible training score, preferring the parent on an exact tie) and uniform sampling from the same options. Each pool includes the unchanged, matched refit. Report the three individual three-edit 20B batches as the equal-slot comparisons. The pooled nine-slot arm measures the value of additional draws. Deduplication counts literal edits after numeric normalization; equal training predictions with differently named sources can also occur. Pooled scores across different benchmark units have no role in selection.

Freeze all choices before opening the development trajectories. Report development NMSE per target, verified physics where supported, failures, eligible slots, unique edits, time, fit evaluations and provider tokens. Development events have been inspected in earlier work. Test data stay closed. The experiment measures a single revision within a specified edit family; an unrestricted search for new states or mechanisms is a separate comparison.

## Results on November 24

ACES job 2409477 completed in four minutes on one H100. Generation used 6,598 tokens across nine calls, and the nine complete rankings used 9,746 tokens. All 27 proposed edit slots passed the same training fit checks. After repeats and a returned parent, they represent five distinct nonparent edits on the basin, six on T2 and five on the reactor. The CPU screened 24, 27 and 11 combinations, respectively. Slot counts alone cannot establish relative computational cost or model quality.

| Development NMSE of the numerically selected model | Matched refit parent | CPU screen, three edits | 20B, three edits each (seeds 100 / 101 / 102) | 20B, nine pooled | Prespecified three |
|---|---:|---:|---:|---:|---:|
| Coupled basin | 0.07528 | 0.0000276 | 0.0000276 / 0.0000276 / 0.0000276 | 0.0000276 | 0.0000276 |
| T2-easy | 0.39223 | 0.06049 | 0.06028 / 0.06064 / 0.06064 | 0.06064 | 0.06043 |
| CSTR-easy | 0.40426 | 0.58816 | 0.39336 / 0.39540 / 0.39336 | 0.39336 | 0.08637 |

On the basin, every proposer batch included powers 1.5 for both discharges. Every ranking seed and the numerical selector picked that model. On T2, each batch included a fading effect that the numerical selector retained. Two of the three scientific rankings of the CPU shortlist preferred the unchanged parent. T2's parent here refits just two coefficients under the initial-order constraint; the revisions add one more. An earlier, broader fit of the delayed parent had development NMSE 0.05998 under different constraints and did reverse the initial disposal order. This study shows a way to regain forecast quality while satisfying the constraint. [Earlier fitting comparison](../work/agent-comms/2026-11-23-orion-disposal-coupled-refits.md)

On CSTR, the scientific rankings of the pooled 20B edits selected the parent once and a model with development NMSE 0.66557 twice. The numerical selector chose a concentration power of 0.75 with NMSE 0.39336. The strongest evaluated reactor candidate, a square-root concentration term with temperature-sensitive heat, entered through the prespecified control. The CPU screen had evaluated that form, ranking it sixth of eleven by held-training-event interval-rate error. The ranking of a local rate approximation can discard a strong full-trajectory model. A shortlist that retains different function forms is a concrete next comparison.

In the separate historical topology replay, all 40 algorithm-generated graphs cleared the corrected checks after 52 complete-graph assessments. Repairing 15 initial 20B proposals with up to two edits recovered eight parents and 113 distinct admissible children after 5,888 patch checks. The two searches have different stopping rules and produced disjoint graph pools. On the earlier identical menus, 42/45 scientific picks and 41/45 uniform picks survived corrected admission. They cover 16 and 29 distinct eligible graphs, respectively. The previous downstream fits used six deliberately chosen graphs; these data leave the relative forecasting value of the two topology generators open. [Full results](../output/revision-round-20261124/REPORT.html) · [Topology audit](../output/revision-round-20261124/topology-audit.md)

## Running from the project checkout

Keep the source receipts at the paths specified in the three adapters. The basin and T2 adapters use the registered training inputs archived with the trajectory-function study. The reactor uses the registered corrected public release. Scientific packages are loaded from the project virtual environment. Run CPU commands from the repository root.

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
rr=output/revision-round-20261124
.venv/bin/python -m scripts.revision_round prepare "$rr"
.venv/bin/python -m scripts.revision_round screens "$rr"
```

On ACES, put the two provider files and the frozen generation manifest in one run directory. The existing Phase C ACES paths supply the pinned H100 image, model weights and Harmony tokens. The run uses one H100 and a maximum of nine generation calls and nine ranking calls. The worker records the complete request and response, elapsed time and token usage under the corresponding stage. A reserved request survives interruption and requires reconciliation before analysis. Use the same run directory to resume a stage.

```bash
rr_remote=/scratch/group/p.nairr260351.000/u.yx126462/af-revision-round-20261124-v1
ssh aces "mkdir -p '$rr_remote'"
scp scripts/revision_round_provider.py scripts/hpc/run_revision_round_20b.sh "$rr/generate.json" "aces:$rr_remote/"
ssh aces "cd '$rr_remote' && sbatch --parsable --export=ALL,AF_WAIT_FOR_RANK=1 --output=worker-%j.out run_revision_round_20b.sh '$rr_remote' generate"
```

Wait for all nine generation receipts to have terminal statuses. Download them, run the common CPU fits, and stage the rank manifest atomically. In wait mode the original job looks for this manifest for up to eight minutes after generation. If that job has already ended, submit the same worker with stage `rank`. Confirm an ambiguous scheduler acknowledgement against the queue and accounting before submitting again.

```bash
scp -r "aces:$rr_remote/generate-responses" "$rr/"
.venv/bin/python -m scripts.revision_round shortlist "$rr"
scp "$rr/rank.json" "aces:$rr_remote/rank.ready"
ssh aces "mv '$rr_remote/rank.ready' '$rr_remote/rank.json'"
# Only if the generation job ended before rank.json arrived:
# ssh aces "cd '$rr_remote' && sbatch --parsable --output=worker-%j.out run_revision_round_20b.sh '$rr_remote' rank"
```

Once the ranking receipts are complete, evaluate the frozen selections and render the report.

```bash
scp -r "aces:$rr_remote/rank-responses" "$rr/"
.venv/bin/python -m scripts.revision_round freeze "$rr"
.venv/bin/python -m scripts.revision_round validate "$rr"
.venv/bin/python -m scripts.report_revision_round "$rr"
```

[Topology comparison on the saved inventories](PHASE_C_PREPROPOSAL_PILOT.md) · [Earlier function-construction comparison](PHASE_C_TRAJECTORY_GUIDED_FUNCTIONS.md)
