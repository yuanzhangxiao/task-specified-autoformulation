# Dalla demonstration rescue on ACES

Protocol `dalla-demonstration-rescue-1` imports six explicitly selected starting
vectors in a separate development experiment. It does not restart the interrupted
fresh campaign, modify historical results, call an LLM, or open intervention/test
data. It uses the current fitter and process-aware pruning implementation.

## Frozen tasks

| Task ID | Model / start | Intended comparison |
| --- | --- | --- |
| `brief_canonical_r9` | Brief-only seed 1 R9, canonical obfuscated T1-easy | Same cell as the existing Sol/no-specification intervention comparison. |
| `full_perturbed_r4` | Full seed 1 R4, perturbed named T1-easy, original vector | Simple two-stage meal chain; original fit had default-like parameters. |
| `full_perturbed_r4_from_r13` | Same R4 structure, all 13 compatible parameters from R13 | No new equation or parameter; this is a starting vector, not an inherited R13 score. |
| `full_perturbed_r13` | Full seed 1 R13, perturbed named T1-easy | Retain additional balance terms and refit. |
| `hard_parent_r14` | T1-hard Full seed 0 parent of the saved R15 patch | Unchanged comparison for the corrected delay proposal. |
| `hard_delay_corrected` | Same parent plus the saved R15 meal-delay patch | Explicitly rename the conflicting new `par_004` declaration to `tau_meal_rescue`. |

The correction operates only on the submitted equations and new declaration.
Existing `par_004` retains its nonnegative-coefficient role and fitted value. The
new time constant receives the fitter's ordinary role-based initial guess (30),
and M receives the original proposal's train-fitted shared initializer (guess 0).
These are optimizer guesses, not fixed reference parameters. All seven existing
parameters survive unchanged in the starting vector. The saved evidence citations
are not certified because the original displayed catalog is absent. No other
equation repair is silently introduced; the parent's existing I equation remains.

`scripts/build_dalla_rescue_inputs.py` reproduces the packet from the previously
exported shortlist, public T1 plan and v7 final archive. It checks exact model
hashes, the R15 proposal's historical parent, and compatible declarations. The
input packet shipped with this milestone has SHA256 *content identity*
`817598e2f0b8616618f0d09cf71dad56e05e88a105c678531446140b8ff0159f`.
Its public cells contain only original training/validation arrays, briefs,
contexts and requirement contracts. Dataset bytes and finalized prompts are not
edited. Canonical/perturbed and easy/hard rows remain separate.

## Numerical and selection policy

1. Freeze the input content, code, runtime, start vector, public checks and policy.
   Preparation compiles models without benchmark fitting.
2. Replay the complete starting vector on original training and validation,
   without estimating any parameter or latent initial from validation. A separate
   600-second allowance bounds both splits together. A failed replay is explicit;
   it does not stop the new fit from attempting recovery.
3. Run one compatible warm-start fit using `collocation-rescue-v1`. This opt-in
   profile retains the existing algorithm, tolerances, constraints, feasibility
   strategy and training-only estimation. Only five numerical limits change:

   | Limit | Historical single/multi-target profiles | Rescue |
   | --- | ---: | ---: |
   | Collocation | 120 s | 300 s |
   | Refinement | 180 s | 900 s |
   | Maximum residual evaluations | 240 | 1200 |
   | Node warm-up | 5 s | 30 s |
   | Recovery start screening | 10 s | 60 s |

   These are caps, not convergence guarantees. Profiles and defaults for old
   campaigns are unchanged. Collocation crashes, screening failures and exhausted
   budgets remain in the report. No worker is run on the user's laptop apart from
   small synthetic verification.
4. Retain the lower training-NMSE complete result of starting-vector replay and
   refit, with the starting vector winning exact ties. Both require available
   finite development rollouts; validation magnitudes do not rank this choice.
5. Run Astra's existing one-deletion pruning experiment from that retained model.
   Removal is ranked using training contributions, checked against public graph
   and target obligations, and frozen before child validation scoring. The child
   and unchanged control receive identical rescue-profile allocations and the
   same compatible starting values. Unresolved obligations skip deletion.
6. Use the existing pruning decision: complexity must decrease and validation
   NMSE may increase by at most `max(1e-6, 0.01 * best_parent_or_control_NMSE)`.
   The original parent/control survives rejected pruning. This is one pruning
   step per starting vector, not an exhaustive state-removal search.

Maximum work is six seed replays, six rescue fits and twelve paired pruning fits.
The array uses one CPU/16 GB per task, three concurrently, and a 2 h 15 min
scheduler cap per task. There are no GPU requests. Parent/refit/pruned/control
models remain individually available. A workflow marked complete can still have
failed fits or no usable model; inspect row status and numerical evidence.

Every replay, fit and pruning allocation is checkpointed. A started but incomplete
allocation is consumed on resume rather than silently receiving another budget.
An interrupted pruning stage can resume its remaining allocated work. Source,
runtime or input drift is an error. Scheduler recovery reuses confirmed job IDs
and requires verified adoption after an ambiguous submission reply.

## Submit using the supplied archive

Upload the source/data archive into group scratch and extract it. Substitute its
actual directory name for `dalla-rescue-COMMIT` below. No Git operation is needed.

```bash
(
  set -euo pipefail
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/dalla-rescue-COMMIT"
  export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
  export AF_OUTPUT_ROOT="$AF_GROUP/dalla-demonstration-rescue-v1"
  export AF_INPUTS="$AF_REPO_ROOT/rescue-inputs.json"
  export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
  module load GCCcore/13.2.0 Python/3.11.5
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_dalla_rescue.py" \
    --inputs "$AF_INPUTS" --root "$AF_OUTPUT_ROOT"
)
```

The command prints prepare, fit-array and report job IDs. The CPU prepare job runs
targeted tests and the synthetic rescue smoke. The same submission command resumes
confirmed submissions without duplicating them. For an unconfirmed reply, inspect
`submission-intent/STAGE.reply.json` and the scheduler, then append
`--adopt STAGE=CONFIRMED_JOB_ID` to the same command. Never remove its receipts to
force resubmission. No ACES session is opened by the local agent.

## Read results

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-demonstration-rescue-v1
AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" -X --format=JobID,JobName%30,State,ExitCode,Elapsed
jq '{status,expected,recorded,rows}' "$AF_ROOT/summary.json"
```

The report job writes `summary.json` and `models.json`. If preparation failed,
`summary.json` explicitly says `not_prepared`. A manual report can also be run:

```bash
module load GCCcore/13.2.0 Python/3.11.5
PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" \
  /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python \
  "$AF_REPO_ROOT/scripts/dalla_rescue.py" report --root "$AF_ROOT"
```

Download `models.json` for the equation/intervention review: it contains original
public arrays, all completed model decisions, requests and fitted vectors. For
numerical debugging, also retain `results/*/refit/` and `results/*/pruning/`.
Per-task seed and fit records survive even if the final report is incomplete.

## What would count as the missing positive example

After this campaign, inspect fitted source/sink signs and the activity of the
required mechanism, not just symbolic graph membership. Keep both pre-pruning and
selected curves. Freeze parameters before running every case in the already
defined intervention family. Report training/validation trajectories, absolute
intervention trajectories and effects relative to each model's matched control.
No intervention outcome enters the rescue fitter, removal ranking or selection.

Brief R9 can join the existing canonical obfuscated T1-easy comparison directly.
R4/R13 need perturbed reference interventions; the T1-hard pair has a different
auxiliary-information contract. They cannot simply be overlaid as equivalent
endpoints. T1-easy auxiliaries must be regenerated consistently under physical
interventions, as in the earlier diagnostic.

The Sol model without additional latent states still has glucose-state memory and
supplied auxiliaries. Its known fast tissue-glucose response is not proof of
instantaneous insulin action. The three-latent Sol model did better on tissue
preparations and worse on meal spacing; state count alone does not explain either
result. These are model comparisons, not controlled causal ablations of state count.

This is a post-selection demonstration rescue, not a fresh benchmark-wide test of
the latest complete pipeline. A good outcome is not guaranteed. Public checks do
not establish fitted scientific correctness; no successful intervention example
is claimed until those curves have been evaluated and inspected.

## Verification for this implementation

- 65 targeted tests passed for the rescue adapter, public fitter, sibling starts,
  and paired pruning. They cover changed budgets with unchanged legacy profiles,
  preserved seed values, training-based retention, consumed interrupted attempts,
  malformed inputs, exact resume, and verified scheduler adoption.
- The synthetic rescue smoke completed three real numerical fits and pruning,
  then resumed with identical artifacts and no additional fitting.
- All six real inputs passed preparation and public contract checks without
  benchmark fitting on the laptop.
- The full pytest run was stopped after 677 passes and five optional-dependency
  skips (245.52 seconds); the entire suite was not completed. It reported no
  failure before interruption. Whole-tree Ruff reports 37 existing issues in
  unrelated `analysis/claude/` files; changed Python files pass Ruff.
- No remote session, benchmark numerical fit, live LLM call, or intervention/test
  evaluation was performed while implementing this milestone.
