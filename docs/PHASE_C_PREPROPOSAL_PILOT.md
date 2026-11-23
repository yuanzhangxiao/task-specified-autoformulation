# Scientific selection before topology construction

This opt-in Phase C pilot takes a completed variable inventory and a measured differential target. The runtime forms a menu of information that could allow the target trajectories to diverge. GPT-OSS-20B ranks a few choices for scientific plausibility and proposes a complete topology for each. The runtime then rechecks the actual graphs. The experiment ends before function construction and fitting.

## Handoff for another benchmark

Use `autoformalism.research.preproposal_cases.from_variable_exit` with an accepted, completed variables event, the public brief, authenticated complete training evidence, the differential target, permitted ongoing histories, and permitted baseline readings including the target's own observed initial. Supply `required_histories` if fixed by the public task. A generated auxiliary uses its modeled value thereafter. Call `screen_case` and `selection_case` on that handoff. The historical ACES launcher uses a separately sealed experiment roster.

The screen assumes shared deterministic dynamics with unique forward solutions, common measured inputs under their specified interpolation, a reset-free interval, and the observed initial of the selected target. Compare publicly measured trajectories only on aligned times. For each pair, take the first target discrepancy beyond the stated uncertainty and additional target gap. Ask which baseline readings or forcing histories had already differed in time. For a held input, a jump exactly at the comparison time cannot already change a continuous state. Missing bounds, interpolation or comparable clocks can leave a subset unresolved.

The exact enumeration, capped at 65,536 source subsets by default, keeps inclusion-minimal sets covering the witnesses. A larger surviving set can contain a smaller explanation. A source is allowed to reach the target directly, through other generated states, or through a declared initial map as appropriate. This necessary-condition test cannot establish causality, functional dependence, attainable NMSE or mechanism correctness. A zero-size support imposes no additional data obligation. See [the formulation](PHASE_C_PREPROPOSAL_TOPOLOGY_FORMULATION.md).

## Frozen 20B experiment

Both insulin arms use the same 16 training trajectories, time-zero forecast contract, three public targets and seed-102 inventory. Seed 102 originally declared variables and topology together. Only its names and types reach the screen. An additional offline replay uses an authentic T2-easy variable-stage exit. The coupled-basin control also uses a genuine variable-stage exit.

* The exact arm retains two minimal singleton histories, EGP and Gt.
* The sensitivity arm counts insulin observation gaps exceeding 1 pmol/L beyond measurement uncertainty. It retains those two histories and four combinations of the administered insulin and meal histories with one permitted baseline reading. That reading can be EGP, Gp, Gt or U. This is an experimental threshold in insulin units. Other target comparisons keep their original bounds.
* The basin control has six distinct initial downstream levels. The data screen imposes no further choice and the launcher skips this target. Its public transfer and storage requirements continue to apply.

Five draws per insulin arm rank one to three distinct options with explanations. Each choice receives one independent complete-graph request. The maximum is 10 ranking calls plus 25 topology calls on one H100. The runtime checks the named variable inventory, algebraic cycles, public paths, initializer dependencies, the path actually realized for the selected information, and conditional training conflicts in observed differential targets. The sensitivity arm also records the exact comparison. An algebraic target still undergoes the public graph checks. This pilot does not evaluate its algebraic value against all observations. A successful graph is a candidate for subsequent function construction.

Analyze seed-level yield and branch-level yield separately. Count seeds with at least one surviving topology, choices made at each rank, distinct model structures, realized routes, residual conflicts and inference failures. Read the rationales against the target, source ownership and public mechanisms. The earlier five open-edit replies tested a different interface.

## Reproduce or launch

From the project checkout run the following commands to regenerate the training audit and review.

```bash
af_out=output/preproposal-public-cases-gap-20261122
.venv/bin/python -m autoformalism.research.preproposal_cases --out "$af_out"
.venv/bin/python -m scripts.audit_preproposal_transfer --out "$af_out/transfer-audit.json"
.venv/bin/python -m scripts.review_preproposal_pilot --cases "$af_out" --transfer "$af_out/transfer-audit.json" --out output/preproposal-pilot-review-20261122
.venv/bin/python -m scripts.package_preproposal_selection_20b --case "$af_out/selection-intake.json" --out output/preproposal-selection-delivery-20261122 --harmony output/initialization-provider-recovery-delivery/assets/harmony
```

Upload the generated archive to group scratch. On ACES, run the following commands.

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462
tar -xzf phase-c-preproposal-selection-20b-v1.tar.gz && cd phase-c-preproposal-selection-20b-v1
sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_preproposal_selection_aces.sh preflight && bash scripts/hpc/start_preproposal_selection_aces.sh run
```

Use `status` on the same launcher to see scheduler and trial status, `resume` for an interrupted submission, and `inspect` for the inspection archive. Checkpoints store the exact requests, choices, replies and graph assessments. The default result root is `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-preproposal-selection-20b-results-v1`.

## Matched selection and graph-first repair

This experiment uses the same public training cases and five seeds per insulin menu. For each measured target, each displayed “at least one” requirement lists alternative sources that could distinguish a pair of training trajectories in time. A choice must satisfy every such requirement. The menu contains the smallest combinations passing this test. The public mechanism requirements and the full graph still need checking.

| Arm | What it does |
| --- | --- |
| Scientific selection | GPT-OSS-20B chooses one of the two exact-screen options, or three of the six options at 1 pmol/L. For each choice it writes a complete graph. |
| Uniform selection | The runtime samples the same number of options without replacement. For the same case, seed and option both arms use the same graph request and share its response. |
| Graph first | GPT-OSS-20B writes a graph before choosing any options. The runtime then generates and checks bounded edits. Report the original graph and any surviving edits separately. |

The scientific and uniform arms compare how well their selections yield eligible graphs. Their graph-generation budget is equal and the scientific arm also pays for selection. The graph-first arm evaluates whether a proposed graph can pass the checks or be repaired by the runtime. The maximum is 60 physical generation requests on one H100 for one hour. This measures graph yield on the training evidence. Function construction, fitting and forecast evaluation come later.

For a failed graph-first reply with a valid schema, the runtime tries adding an RHS source from the existing inventory or permitted supplied quantities, adding a permitted observed initial value to a latent initializer, and pairs of these edits. It checks each edited graph against the public graph rules and all measured differential targets. It retains the passing edits for which no smaller patch passed. A common hidden preparation excludes changes to initializer arguments. The cap is 1,024 checked patches per parent, with an explicit flag if enumeration stops early. Counts of independently proposed graphs and their algorithm-generated descendants are reported separately.

The offline seed-102 replay reads the original topology and formula receipts. The formula reply chose a trajectory-specific initial value for `Uid`. To compare wiring inside this time-zero pilot, the replay also evaluates the same topology with a shared initial value for `Uid`, recording both outcomes.

```bash
.venv/bin/python -m scripts.replay_preproposal_graph_repair --source output/preproposal-public-cases-gap-20261122/frozen-cases.json --out output/preproposal-matched-review-20261122/seed102-repairs.json
```

From the Mac checkout, build or upload the delivery as follows. Packaging reads tracked source from the current commit.

```bash
af_case=output/preproposal-public-cases-gap-20261122/selection-intake.json
af_out=output/preproposal-matched-delivery-20261122
.venv/bin/python -m scripts.package_preproposal_selection_20b --matched --case "$af_case" --out "$af_out" --harmony output/initialization-provider-recovery-delivery/assets/harmony
scp "$af_out/phase-c-preproposal-matched-20b-v2.tar.gz" aces:/scratch/group/p.nairr260351.000/u.yx126462/
```

In the ACES login shell, verify the bundle, preflight every prompt and submit the job.

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462
tar -xzf phase-c-preproposal-matched-20b-v2.tar.gz && cd phase-c-preproposal-matched-20b-v2
sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_preproposal_matched_aces.sh preflight && bash scripts/hpc/start_preproposal_matched_aces.sh run
```

Run `status` on the same launcher to check Slurm and the experiment. If Slurm times out while acknowledging submission, repeating `run` reconciles the original intent. Use `resume` after an interrupted job. Once the job has finished, package and download the results.

```bash
# On ACES, from the extracted directory
bash scripts/hpc/start_preproposal_matched_aces.sh inspect
# On the Mac
scp aces:/scratch/group/p.nairr260351.000/u.yx126462/phase-c-preproposal-matched-20b-results-v2/inspection.tar.gz ~/Downloads/preproposal-matched-20b-inspection.tar.gz
```

To regenerate the prompt review locally:

```bash
af_review=output/preproposal-matched-review-20261122
.venv/bin/python -m scripts.preproposal_selection_20b prepare --matched --source "$af_case" --root "$af_review/plan"
.venv/bin/python -m scripts.review_preproposal_pilot --matched-root "$af_review/plan" --out "$af_review"
```

## Selecting complete graphs

The new `--complete-graphs` experiment uses four T2-easy variable inventories with the same 16 public training trajectories, time-zero initialization contract and 1 pmol/L insulin screening margin. Three inventories come from completed 20B variable-stage declarations (seeds 0, 1 and 3). The fourth comes from the previous seed-102 joint declaration. A fifth case uses the coupled-basin inventory and its own public units and initialization contract.

The runtime builds and checks a pool once per inventory. It tries direct and mediated pathways, additional RHS terms, and shared constants or initial-value maps for latent states. The frozen limits are 64 distinct graph checks, up to 256 draws, six sources per equation, two observed arguments per generated initial map and eight admitted graphs per inventory. Sampling can stop before all feasible graphs have been found.

| Approach | What the proposer does | What the algorithm does |
|---|---|---|
| Scientific source selection | Selects source IDs, then writes the complete graph | Screens source sets and checks the resulting graph |
| Uniform source selection | Writes the graph for each drawn source set | Draws from the same source menu and checks the resulting graph |
| Graph first | Proposes a complete graph | Checks the graph and searches bounded one- and two-edit repairs |
| Complete graph selection | Ranks three whole graphs by ID when at least three are available | Generates and admits the pool first, then also samples the same number of graphs uniformly from the identical pool |

The two source selectors share a graph response when they select the same option on the same seed. A singleton empty source set requires no ranking call. Proposer-written graphs can include a new latent with an explicit name, type and RHS. Each new latent differential state also needs an allowed initial value. The runtime retains the equations associated with a selected graph ID. Each graph must pass the same implemented public-path and observed-variable checks. The recorded differential memory assignments from the variable stage constrain their public pathways.

The observations can rule out a graph when it forces equal values where the training data differ. The gate compares measured generated differential and algebraic variables, including model-owned measured auxiliaries when present. For algebraic equations it uses repeated complete arguments at time zero and later samples. For differential equations it considers the initialization and upstream forcing. Physical sign, function form, fitted accuracy, unobserved state recovery and mechanisms beyond those formalized in the public graph checks require additional evidence. The 1 pmol/L cutoff applies only to insulin. The report also retains the exact insulin check.

Three paired proposer seeds (100–102) run on one H100 with a two-hour wall limit. The report records the CPU graph checks, reasons for rejection, distinct admitted structures, selections and physical requests. Compare admission before fitting, then compare the fitted scientific and uniform selections under equal fitting budgets. The source-written and graph-first arms can introduce additional variables, so differences in yield include differences in their search spaces.

From the Mac checkout:

```bash
af_out=output/topology-admission-replay-20261123
.venv/bin/python -m scripts.package_preproposal_selection_20b --complete-graphs --case output/preproposal-public-cases-gap-20261122/selection-intake.json --out "$af_out" --harmony output/initialization-provider-recovery-delivery/assets/harmony
scp "$af_out/phase-c-complete-graphs-20b-v2.tar.gz" aces:/scratch/group/p.nairr260351.000/u.yx126462/
```

On ACES:

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462
tar -xzf phase-c-complete-graphs-20b-v2.tar.gz && cd phase-c-complete-graphs-20b-v2
sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_complete_graphs_aces.sh preflight && bash scripts/hpc/start_complete_graphs_aces.sh run
```

Use `status` on the same launcher to check progress, `run` to reconcile an ambiguous submission, `resume` after interruption and `inspect` to package the results. The results directory is `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-complete-graphs-20b-results-v2`. To view the exact messages for a prepared experiment locally, run `.venv/bin/python -m scripts.review_preproposal_pilot --portfolio-root <prepared-root> --out <review-directory>`.

The inspection archive includes `admitted-graphs.json`. Each exported graph has its variable types, complete RHS source lists, initial-value rules and the recorded selection or repair origins. The pool in `intake.json` retains all admitted alternatives, including those the selectors did not choose. An inspection made before completion records its partial status. The export precedes interaction functions, signs and fitting.

### Replaying the first complete-graph run

Job 2403807 used the v1 bundle. The following local command rechecks that run's saved replies, selected graphs and CPU pools using the corrected code. It reads only the archived training evidence and needs no GPU.

```bash
.venv/bin/python -m scripts.replay_topology_admission
```

The checker merges a second declaration of an inventory variable when its type and source set agree. It shows both definitions when they conflict. Additive repair tries missing dependencies and pathways on an otherwise usable graph. The generator connects a driver through the assigned memory to the target on a single path. Surveyed covariates can enter the RHS. In the reviewed basin inventory the physical upstream-depth state starts at its measured event depth, which the runtime includes in the exported graph.

Of 40 graphs in the original CPU pools, 36 still pass. Of the 34 originally selected, 30 still pass. The four exclusions all used a shared constant for the measured upstream depth. Replaying graph-first proposals and their allowed repairs yields 113 distinct T2 graphs from eight parents. These results concern structural admission. The next design question is [how training trajectories should inform function construction](PHASE_C_TRAJECTORY_GUIDED_FUNCTIONS.md).
