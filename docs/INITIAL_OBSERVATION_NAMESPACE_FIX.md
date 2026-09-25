# Initial observations and generated process aliases

The September 25 component logs show two T2-easy lineages failing in
`SymbolicODE` before the numerical backend starts: `cell07_seed0_full_c1v1s1`
(`KeyError: Uid`) and `cell07_seed0_brief_only_c0v1s1` (`KeyError: I_mem`).
These are local symbolic compilation errors, independent of the critic API.

An initializer has a different namespace from a differential equation. In
`m(0) = a + b*U(0)`, `U` denotes the initial public measurement, including when
the model defines an algebraic process named `U`. The sensitivity adapter
previously substituted that process's definition into the initializer. This
could introduce unavailable hidden states, causing a crash, or silently change
the boundary if the substituted definition used only parameters.

The correction leaves validated initial observation expressions unexpanded while
retaining their domain and smoothness checks. Differential equations and output
mappings still expand process aliases. Production rollout initialization already
used the correct observation semantics. No prompts, benchmark data, optimization
budgets or model-selection rules change.

## Check saved fits before recovery

From the corrected archive, run the following with the ACES Python environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AF_REPO_ROOT/src"
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/audit_fitted_initialization.py" \
  --fit "$ROOT/results/cell07_seed0_full_c1v1s1/round_00/fit" \
  --fit "$ROOT/results/cell07_seed0_brief_only_c0v1s1/round_00/fit"
```

This reads sealed public training/validation fit bundles and prints JSON. It
makes no LLM calls, optimizer calls, solver rollouts, or writes. It checks:

- the original bundle digest and all non-source handoff fields;
- public observation names that also name generated processes;
- current backend capability;
- agreement between symbolic and production boundaries at saved guesses, when
  every parameter has a saved guess;
- existence of the public fit's start and result markers.

`source_matches: false` is expected for a historical archive. Only this read-only
audit permits that difference; ordinary execution still rejects source drift.
`boundary_comparison.status: agree` demonstrates boundary agreement at those
guesses, not fitting success or predictive accuracy. Missing guesses are reported
without inventing numerical values. A round's outer `worker_started.json` is not
the public backend's `fit/started.json`.

Historical campaign results and budgets remain unchanged. Do not overwrite the
frozen archive, delete terminal results, reseal an old plan, or retry it using the
new source. The next recovery step needs an explicit handoff preserving saved
proposals and prior cost records. Before final evaluation, the scope audit also
needs to include completed fits with the same name collision: absence of a crash
alone does not establish that an initializer was unaffected.

The supplied queue snapshot has only a proposer worker running. Finishing the
unaffected campaign also requires active critic, fitting and pruning workers;
that worker-capacity issue is separate from the initialization correction.

## Recover the two unstarted round-zero fits

The returned ACES audit confirms a supported initializer/process name collision
in both saved requests, with neither `fit/started.json` nor `fit/result.json`.
The `missing_saved_guesses` diagnostic only means that the optional boundary
comparison could not run. Public fitting normally fills unspecified equation
parameters with its existing role-based starts; no new guesses are needed.

`scripts/recover_initialization_namespace.py` provides a separate recovery
directory under protocol `unstarted-initialization-recovery-1`. It admits only
explicitly selected component-campaign round-zero constructions whose supervisor
terminated before numerical work began. It verifies the original plan, public
assets, proposal, interrupted draft, cost record, supervisor marker and fit freeze.
Any numerical start, result, backend directory or unexpected fit artifact blocks
recovery. Existing historical locks prevent recovery reads during an active
original worker without serializing unrelated lineages.

The new plan snapshots this evidence and binds the corrected source/runtime to
the unchanged requests, data, seeds, guesses and fit settings. Each selected model
receives at most one fit attempt in the separate directory. Completed recovery
results resume immutably; an interrupted recovery never receives a fresh budget.
The CPU launcher retains the component protocol's 1800-second outer allowance.
There are no new LLM calls, new equation proposals, automatic model promotion,
later-round replay, or test access. Historical proposal costs are reported as
historical costs, not counted as new LLM usage.

With `AF_REPO_ROOT` pointing to the new immutable archive, `AF_COMMIT` equal to
its `SOURCE_COMMIT`, and `AF_PYTHON` set to the existing ACES virtual environment:

```bash
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AF_REPO_ROOT/src"
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/component-init-recovery-v1

"$AF_PYTHON" "$AF_REPO_ROOT/scripts/recover_initialization_namespace.py" prepare \
  --source /scratch/group/p.nairr260351.000/u.yx126462/final-components-v1 \
  --root "$AF_OUTPUT_ROOT" \
  --task cell07_seed0_full_c1v1s1 \
  --task cell07_seed0_brief_only_c0v1s1

mkdir -p "$AF_OUTPUT_ROOT/logs"
sbatch --parsable --account=156264627414 --partition=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=00:35:00 \
  --array=0-1%2 --export=ALL --job-name=init-recovery \
  --output="$AF_OUTPUT_ROOT/logs/fit-%A_%a.out" \
  --error="$AF_OUTPUT_ROOT/logs/fit-%A_%a.err" \
  "$AF_REPO_ROOT/scripts/hpc/run_initialization_recovery_aces.sh"
```

After the CPU jobs end, use the same archive/Python environment:

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/recover_initialization_namespace.py" report \
  --root "$AF_OUTPUT_ROOT"
```

The report is also written to `summary.json` in the recovery directory. A
`started_without_result` row is an incomplete attempt, not a successful fit;
reporting does not seal it or start numerical work. `complete` means finite
train/validation rollouts, not accurate or scientifically correct recovery.

This step does not repair the historical search trajectories. Later proposals
were generated from their actual unfitted predecessors and must not be attached
retroactively to the recovered parameters. A subsequent campaign handoff must
preserve those contexts and consumed budgets. Completed fits with overlapping
initializer/process names still need a scope audit before final test evaluation.
