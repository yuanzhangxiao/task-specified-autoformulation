# Paired state and specification searches

The controlled campaign reuses the component campaign's 20B staged proposer, whole-model revision and repair, 120B advisory critic, multivariate public fitter, validation selection and process pruning. The default is 15 visits and three seeds (0, 1 and 2). These are prospective repetitions. The original campaign used two seeds per prompt. The new study uses the Full construction prompt for every treatment. Each generated calculation and critic request uses the assigned version of the public scientific specification. An endpoint carries all four public source-file hashes for independent evaluation.

The two hard anchor cells in this package use the versioned Phase C development contract. `acquisition_00` is clean T2-hard with continuous meal ingestion. `sensor_00` is corrected CSTR-hard with a common hidden preparation and no supplied auxiliaries. These conditions differ from the submitted glucose event encoding and named CSTR-easy data. The entry point also accepts an original Phase B public directory containing the registered train/validation CSVs, manifest and prompt, so the exact submitted T2-hard export can be run as a separate cohort. Supply the original public data and prompt hashes for that comparison. Existing judged/retained test scores cannot start a search.

## Treatments

| Method | Information and enforcement |
| --- | --- |
| `full` | The original public scientific brief, causal generation, advisory critic and revisable state representation. |
| `fixed_gamma` | Reuses Full's exact executable initial proposal, fitted coefficients, fit/validation result and critic receipt. Subsequent search has the same per-visit proposal, repair and fitting budgets. State declarations, roles and directly observed coordinate identities are fixed. Equations, allowed initializer coefficients and nonidentity algebraic readouts may change. Empty or failed shared starts are reported for both arms. |
| `no_latent` | All differential states must be directly observed target coordinates, with no undeclared time history. The active task and verifier remain enabled. A required unobserved mediator can yield no admissible model. |
| `observed_only` | The same observed-state restriction, with scientific rejection relaxed. The task is still visible and the critic remains advisory. This measures restricted prediction capacity under changed admission. |
| `no_spec` | Keeps channel names, meanings, units, preparation, public domain, global parameters, causal initialization and the analytic grammar. The active mechanism agenda is empty in construction, revision, critique, verification and pruning. |
| `underspecified` | Removes the insulin-action obligation or the jacket-exchange obligation from the active task, including duplicate wording in the introduction. |
| `incorrect` | T2 requires a delayed meal-driven disposal term and excludes a plasma-insulin path to the modeled glucose balance. CSTR excludes a jacket-feed-temperature path to reactor temperature. These are explicit experimental mutations. The checks operate on equation dependencies. |
| `overrestrictive` | Adds an exactly-one-unobserved-state requirement to the original task. The equation checker counts actual states after compilation. |

Fixed-representation edits are checked before retention. A rejected attempt receives the failed treatment check and can use the existing bounded repair allowance. Failed revisions keep the incumbent under the original fallback and validation-selection rules. Algebraic processes cannot store history in this grammar. The role check includes state descriptions and the staged inventory when present; pruning's lowered requests preserve the authoritative state definitions. It measures the value of allowing state revision conditional on the initial state space. The optional `--max-parameters N` freezes one common fitted-coefficient cap, including initializer coefficients, across all arms for a separately labeled capacity sensitivity run. Set the cap before opening the independent tests. `observed_only` uses the relaxed scientific verifier throughout all construction steps. Development failure is an observation about the allotted search and fit budget. No hypothesis-falsification detector is claimed.

## Prepare on the execution host

Transfer the generated `stress/public/` directory to group scratch, or regenerate identical public inputs. Obtain the project commit identified by the package and activate the same Python environment for preparation and every worker. Plans bind the source and runtime. Re-prepare on the cluster in a fresh directory when its runtime differs from the packaged prelaunch snapshot.

~~~bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export AF_PUBLIC_ROOT=/path/to/extracted/stress/public
export AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
export AF_SEARCH_ROOT="$AF_GROUP/iclr-controlled-hard-1"
"$AF_PYTHON" -m scripts.iclr_controlled_search prepare --root "$AF_SEARCH_ROOT" \
    --public "$AF_PUBLIC_ROOT" --cells acquisition_00 sensor_00
~~~

The hard-anchor matrix is six cell/seed blocks and 48 endpoint rows. On the two named canonical/perturbed T1-easy cells, the optional transfer comparison uses `--arms full fixed_gamma no_spec`. The incorrect and incomplete variants are defined for the two hard anchor tasks.

To run the budgeted data-quality and procedural-scaling comparison, initialize a separate campaign with its 19 public conditions and the two arms `--arms full fixed_gamma`. Its standard launch has 57 blocks and 114 endpoints. Read the exact cell roster from the stress plan rather than globbing every public directory, which also contains the corrected CSTR-easy audit cell. Prepare and bind each campaign independently. Scaled simulator dimensions, coupling and private validation checks are excluded from the search prompts.

## Launch and resume on ACES

The critic uses the original completed known-case calibration. Copy or point to its real `plan.json` and `summary.json`. The worker starts the pinned proposer on loopback in each H100 allocation, checks the configured serving-image digest and closes the server at job exit. The default paths for the vLLM SIF and pretrained model cache match the historical ACES campaign. Set `AF_VLLM_IMAGE` and `AF_HF_HOME` if the assets moved. The API key is inherited from the submitting shell. It is not passed to vLLM or written into run artifacts.

~~~bash
export AF_JUDGE_GATE_ROOT="$AF_GROUP/judge-jetstream-calibration-v1"
"$AF_PYTHON" -m scripts.iclr_controlled_search bind-critic --root "$AF_SEARCH_ROOT" \
    --calibration-plan "$AF_JUDGE_GATE_ROOT/plan.json" \
    --calibration-summary "$AF_JUDGE_GATE_ROOT/summary.json"
read -rsp 'Jetstream critic key: ' AF_JETSTREAM_API_KEY; echo
export AF_JETSTREAM_API_KEY
# Set AF_ACCOUNT to your ACES GPU allocation/account.
sbatch --account="$AF_ACCOUNT" --partition=gpu --array=0-5%2 --export=ALL \
    scripts/hpc/run_iclr_controlled_search_aces.sh
# For the separately prepared 57-block stress campaign, use --array=0-56%2.
~~~

The scripts use one node and one H100 per active block. Multiple waves may be needed. `AF_BLOCK_SECONDS` bounds new work starts in each allocation and defaults to 36,000 seconds, leaving two hours for service start and in-flight fitting/critique within the 12-hour allocation. After the previous array has stopped, resubmit the unfinished block indices against the same directory. Completed visits, logged calls and fits are reused. Jobs trying to own the same block concurrently fail a block lock. The primary configurations use the critic. `--without-critic` and `--without-pruning` create explicitly different engineering studies. They must not be pooled with the primary results.

~~~bash
"$AF_PYTHON" -m scripts.iclr_controlled_search report --root "$AF_SEARCH_ROOT"
"$AF_PYTHON" -m scripts.iclr_controlled_search export --root "$AF_SEARCH_ROOT"
~~~

Report and export obtain an exclusive lock, so run them between waves. Export refuses a missing visit or pruning endpoint. It includes terminal failed rows, seals the selection and disables further search. Feed the sealed `frozen_models.json` to the fidelity and stress evaluators after that step. The public validation NMSE is a selection statistic. Independent tests and private state trajectories are never read during search.

The plan includes `ablation_switch_manifest.json` and `spec_variants.json`. The reporter writes `paired_ablation_results.csv`, `spec_sensitivity.csv` and `search_visits.csv`, including fitted complexity, observed calls and tokens, recorded critic findings, rejection reasons and training/validation results. Full is charged for the shared initial construction. Fixed Gamma reports that cost separately alongside incremental work. Counts of critic HTTP attempts referenced by each lineage can overlap if an identical cached symbolic review is reused by other lineages. Wall time measures the full fitting/feedback stage and records missing timing receipts explicitly. Slurm's `sacct` gives allocation-level GPU and CPU resource consumption; inference and numerical fitting share each reserved node.

Offline verification uses schema-valid deterministic synthetic replies, the submitted multi-target fitting profile, pruning, sealed export and deterministic resume. It makes zero provider calls.

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$AF_PYTHON" -m scripts.smoke_iclr_controlled_search
"$AF_PYTHON" -m pytest -q tests/test_iclr_controlled_search.py
~~~

After the 48-row anchor export, render both seed-0 Full models and each paired treatment’s equation and fitted-coefficient changes. Terminal failures remain visible.

~~~bash
"$AF_PYTHON" -m scripts.iclr_equation_examples --search "$AF_SEARCH_ROOT" \
    --output "$AF_SEARCH_ROOT/paired-equations-seed0.md"
~~~
