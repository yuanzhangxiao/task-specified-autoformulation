# Repairing a topology from matched training trajectories

A differential target with a closed set of upstream states must evolve identically in two runs when those states receive equal initial information and equal forcing histories, the laws and parameters are shared, and solutions are unique. The repair generator tests whether a local dependency edit breaks that equality argument. All inputs to this check come from the public training split and the declared forecast contract.

## Reuse on another benchmark

Supply a construction `Draft`, sealed `TrainingCompatibilityEvidence`, a differential target, two training trajectory IDs, one shared sample time, and an explicit list of sources allowed at prediction time. The core entry points are `generate_repairs` and `assess_edit` in `autoformalism.search.topology_repair_generator`. The selected observation intervals must be disjoint, and the unedited topology must imply equal target values for the pair. Missing or different hidden initial conditions can invalidate an equality argument.

For each permitted source the generator tries adding an argument to the target's law and routing that source through a new differential memory with one shared initial constant. A source can itself be an existing state or process. Each patch preserves the parent's existing edges. The checker accounts for all upstream initializer inputs and forcing histories. A sampled difference in a generated trajectory has scientific interest, but the corresponding model quantity must be generated during prediction. Input interpolation governs how early a forcing difference can matter.

Each assessment stores the added edges, any new state, the roots that can carry a distinction, and any missing conditioning information. It also checks the first disjoint target observation on each training pair. Once the histories or initial information differ, equality of the complete input prefix cannot be reestablished by a later matching sample. The four outcomes are:

- `distinguishes` means an allowed path can receive different initial information or forcing by the selected time. Function choice and effect size remain open.
- `still_forced_equal` means the same equality proof survives this edit.
- `unresolved` means missing information, such as a hidden initializer, prevents a decision.
- `invalid` covers duplicate direct arguments, disallowed sources, and edits outside the contract.

The scope is single-source edits to a differential target. Initializer changes, formula construction, arbitrary changes upstream, and optimal parameter fitting are separate experiments. The general checker returns dependency patches. The named-target campaign additionally materializes each patch and reruns the original public graph checks for ownership, cycles and required pathways.

The generic CLI accepts a JSON object with the same `draft` and `evidence` fields as the existing offline topology checker. For example, given a reservoir case with trajectories named dry and wet, run:

```bash
python -m scripts.generate_topology_repairs --input reservoir.json --target storage \
  --left dry --right wet --time 2 --allowed-source rain --allowed-source storage \
  --output reservoir-repairs.json
```

A synthetic reservoir test exercises successful repairs. On the archived coupled-basin seed 0, all six training trajectories had distinct measured initial downstream depths. The generator declined to require a repair for downstream depth on all 15 pairs tested at t = 4.

## Seed 102 and the 20B comparison

The experiment authenticates the topology accepted at call 16 of the fresh 20B campaign, including its provider receipt. At minute 95 the observed insulin values on train_005 and train_011 differ by 559.44 pmol/L despite equal observed initial insulin and zero insulin infusion. The target's only declared sources were insulin and infusion.

Ten of 20 enumerated edits break this selected equality. Both the direct and memory versions using generated Gp, supplied EGP, or supplied Gt leave zero certified conflicting pairs in this insulin check. The two meal-rate edits leave ten and the two E edits leave 91. U and Uid remain unresolved because U is generated and Uid's initial value is unknown. These are graph-level outcomes for a fixed parent. Scientific mechanism, effect magnitude, and held-out performance require their own checks.

To reproduce the offline report from the original inspection on the project computer:

```bash
export PYTHONPATH="$PWD/src:$PWD"
AF_DIR=output/topology-repair-20261120
.venv/bin/python -m autoformalism.research.topology_repair_case \
  --returned output/t2-fresh-20b-results-20261119/returned --out "$AF_DIR/case.json"
.venv/bin/python -m scripts.audit_topology_repairs --case "$AF_DIR/case.json" --out "$AF_DIR"
```

The public request contains the saved parent, task, source permissions, initial conditions, interpolation and compact pair observations. Each draw sees the same evidence. Five independent 20B seeds each choose one edit from the same grammar as the generator. Every draw is scored using the same full training records as enumeration. Report single-draw yield and distinct edits separately from the generator's exhaustive count. The 20B request contains no enumeration results.

To rebuild the upload bundle using an existing verified Harmony vocabulary:

```bash
.venv/bin/python -m scripts.package_topology_repair_20b --case "$AF_DIR/case.json" \
  --harmony output/initializer-choice-delivery/phase-c-initializer-choice-48c2581/assets/harmony \
  --out "$AF_DIR"
```

Upload `phase-c-topology-repair-20b-v1.tar.gz` to the group scratch directory. On ACES:

```bash
cd /scratch/group/p.nairr260351.000/u.yx126462 && tar -xzf phase-c-topology-repair-20b-v1.tar.gz && cd phase-c-topology-repair-20b-v1 && sha256sum --quiet -c SHA256SUMS && bash scripts/hpc/start_topology_repair_20b_aces.sh run
```

Use `status` on the same launcher to see job and per-seed outcomes, `resume` after an interrupted allocation, and `inspect` to export the results. The downloadable archive is `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-topology-repair-20b-results-v1/inspection.tar.gz`. One H100 serves all five calls. The job requests one hour and performs zero fits.
