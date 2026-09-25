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
