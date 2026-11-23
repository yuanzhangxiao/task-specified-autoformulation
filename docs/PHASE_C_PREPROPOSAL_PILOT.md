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
