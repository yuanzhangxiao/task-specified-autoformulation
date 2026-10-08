# Per-turn variable checklist and bounded completion repair

`minimal-variable-checklist-1` extends the opt-in `minimal-clarity-1` prompts.
Every call displays a checklist computed from accepted declarations. A rejected
reply cannot change that checklist. During variables, the draft view omits
premature missing-equation warnings and hypothetical balance formulas.

## Checks and stage boundary

The checklist distinguishes `satisfied`, `missing`, `inconsistent`, and `deferred`.
It checks the following declarations before allowing variable-stage completion:

- Every public target has an explicit generated-variable declaration.
- Explicit public target-type restrictions, if present, match the declaration.
  Physical names, descriptions and textbook conventions do not establish types.
- Public inputs, covariates and time remain supplied sources. Existing transaction
  validation rejects conflicting generated declarations atomically.
- Every required memory assignment names declared differential states. Unknown
  requirement IDs, undeclared states and algebraic states produce specific feedback.
  The existing mandatory-mediator contract excludes its driver/target endpoints;
  optional memory assignments retain their existing endpoint freedom.
- When a reviewed target-feedback obligation applies to an algebraic target,
  its proposer-selected differential coordinates are declared. A differential
  target is its own coordinate and requires no additional binding.

Partial inventories and forward binding references may be retained. Missing or
inconsistent declaration items block only a request to finish the variable stage.
Such requests consume the existing bounded local repair allowance. Exhaustion is
reported, not treated as successful stage completion. There is no added self-review
call or automatic selection of a scientific state.

Each required mechanism also appears as deferred topology work, with its public
endpoints and any chosen memory states. Its representation may involve ordinary
contributions, shared processes, or both. An assignment is not a pathway witness.
The existing assembled-topology checks still verify driver/memory/target paths,
cycles, readouts and other reviewed predicates. They do not run on a partial graph.
No new mandatory one-process-per-mechanism annotation is introduced. Missing public
endpoints remain unresolved rather than being supplied by the runtime.

Declaration checks are recomputed after later variable/binding edits and included
in final structural eligibility. Functional adequacy, initialization, actual
mechanism behavior and fitting remain outside this milestone. No variable name or
free-text scientific explanation is parsed as a correctness certificate.

## Live diagnostic

`variable_checklist_confirmation` freezes eight fresh minimal constructions: all
eight public Phase C cases, including both basins, Full context, seed 0. It uses
the same proposer model, settings, total call/token budgets, three global repair
requests, and variables -> shared processes -> ordinary topology schedule as the
earlier minimal refinement. There are no old draft imports, interaction calls,
fitting jobs, scientific critic calls or test-data access. This is a bounded
confirmation, not a matched strategy ranking or an estimate of scientific recovery.

One H100 worker has a three-hour work window within a 3.5-hour allocation. The
dependent CPU report preserves partial results if the worker stops early. Calls,
transactions, checklists and stage outcomes are cached and identity-bound. Use the
original pinned source to resume an existing root; do not replace an older study's
plan. Repeating a submission uses its saved receipt rather than resubmitting jobs.

The portable bundle includes the previously sealed public topology source plan;
no historical raw replies or trajectory tables are included. Upload it to the
ACES group scratch directory, extract it, and run its `RUN_ACES.sh run` wrapper.
The equivalent repository entry point is:

```bash
bash scripts/hpc/start_phase_c_variable_checklist.sh run
# After the jobs finish:
bash scripts/hpc/start_phase_c_variable_checklist.sh inspect
```

The default output is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-variable-checklist-v1`.
`inspect` refreshes the reports without LLM calls and packages `inspection.tar.gz`.

- `VARIABLES.md` / `VARIABLES.json`: first retained declarations, last variable
  snapshot, whether that stage actually completed, and declarations after topology
  and global repair. Every turn links to its request hash and records rejection
  status separately from the retained checklist.
- `SUMMARY.md` / `summary.json`: topology outcomes, delivery and cost.
- `TOPOLOGY.html`, per-task traces and cached calls: exact prompts, responses,
  equations and repair evidence.

The last variable snapshot may be an exhausted or interrupted stage, not a
successful exit. Pending cases have no assessment; they are not scored as zero.
Declaration readiness does not mean all scientifically necessary variables were
found, all scientific meanings were correct, or every mechanism was recovered.

## Verification

Read-only application to the original eight minimal runs finds four declaration-
ready first snapshots and four ready variable exits. One first snapshot lacks a
public target; four exits lack the mandatory memory assignment. All eight final
drafts pass these declaration checks after their historical topology/global
repairs. This is counterfactual earlier feedback, not eight newly correct models,
and no historical acceptance decisions or files were changed.

Focused tests cover explicit type contradictions, optional and mandatory memory,
undeclared references, algebraic readout coordinates, accepted partial edits,
atomic rejection, bounded premature completion, later type changes, graph checks,
interrupted cached resume and the eight-case submission/report path. Public-plan
prepare/verify/report and the existing construction smokes require no LLM calls.
Run details are recorded in the delivery validation notes after verification.
