# Milestone 3: general process-aware pruning

This opt-in pilot starts from the six retained round-1 models in
`shared-process-integration-v1`. It introduces no domain-specific probes, new
proposer/critic calls, or fitter changes. Historical campaigns remain unchanged.
The production controller does not automatically invoke it.

The integration result closes Milestone 2's construction/revision engineering gate:
12/12 round rows completed, all six revisions committed, four new trials retained.
It does **not** establish a benefit from shared processes: the reported models did
not contain a named process used directly by multiple governing equations. The
synthetic smoke therefore separately exercises an actual two-consumer law.

Both CSTR parents remain eligible for development selection, but
`controlled_balance/graph_inference` is ambiguous because its public specification
lacks driver/target anchors. This is neither a failed mechanism nor a certified
pass. This pilot preserves that evidence and skips automatic deletion for these
parents. It does not alter the benchmark prompts or invent missing anchors.

## Frozen pilot policy

1. Read the completed public development lineage, selected fit and backend receipts.
   Copy its public inputs and parent models into a separate sealed plan. No test
   trajectories or reference equations are accessed.
2. Enumerate deletion units from the canonical expressions. Ordinary units are
   complete signed additive terms. A process used by multiple governing equations
   is removed across **all** its consumers together. Nonlinear, denominator or
   overlapping uses that cannot be removed unambiguously are skipped. A subterm of
   a shared law can be removed once, affecting the same function at every consumer.
3. Preserve observation mappings; remove unreachable states, processes, parameters
   and latent initialization rules. Reject cleanup that would silently remove only
   one shared consumer. Compile and rerun the existing target/mechanism predicates.
   No pruning is attempted when a public obligation is unresolved. Declarations of
   conservation, custom or monotonic constraints without an executable verifier
   also require review; this pilot does not silently discard them.
4. On the **parent's training open rollouts**, measure RMS of each term divided by
   RMS of its defining RHS (floor `1e-12`), using the recorded observation times.
   Rank groups by their largest consumer ratio, then stable unit ID. Freeze the
   weakest legal candidate before either new fit or any new validation score.
   This is a conditional heuristic, not a proof of redundancy. There is no raw
   coefficient cutoff or adaptive search through validation-scored candidates.
5. Run at most one pruned fit and one unchanged-parent fit with identical frozen
   `collocation-single-target-v2` caps: 120 seconds initialization, 180 seconds
   refinement, 240 residual calls. Both inherit compatible fitted parameters and
   causal initializers from the same parent. They do not inherit each other's fits.
   Array-index parity alternates control/child order; candidate choice is already
   frozen. Contribution replay has its own 300-second cap.
6. Use the better validation score of historical parent and new unchanged control
   as the baseline. Prefer the pruned model only if every executable complexity
   count is nonincreasing, at least one decreases, public checks pass, and validation
   NMSE is no more than `max(1e-6, 0.01 * baseline_NMSE)` above that baseline.
   These tolerance values are an explicit pilot policy, not scientific thresholds.
   Selection is development-only; fitting and removal ranking use training only.

The complexity vector contains states, named processes, fitted parameters including
causal initialization parameters, additive terms and expression AST nodes. These
are syntactic counts, not counts of independent physical mechanisms. A failed or
interrupted control is reported as an incomplete paired comparison; a valid child
may still be compared with its historical parent, whose total budget differs.

There are six array tasks, at most twelve benchmark refits in general, and at most
ten with the two reported CSTR ambiguities. A skipped deletion still permits its
unchanged control. The preparation job also runs two small synthetic smoke fits.
No benchmark fitting is rerun during preparation. No model is automatically
promoted into the original campaign and no further round is submitted.

## Simple mathematical description

At the retained parameters, write a defining equation as $F_i=\sum_j T_{ij}$.
Over all recorded **training** rollout times, compute

$$
r_{ij}=\frac{\operatorname{RMS}(T_{ij})}
{\max(\operatorname{RMS}(F_i),10^{-12})}.
$$

For a deletion unit $U$ affecting several consumers, use
$r(U)=\max_{(i,j)\in U}r_{ij}$. Choose the legal unit with the smallest $r(U)$;
then remove it and refit. There is no threshold on coefficient magnitude.
This is one greedy deletion, not an exhaustive or iterative pruning search.

Let $E_b=\min(E_{\mathrm{parent}},E_{\mathrm{refit\ control}})$ be the best
available unpruned validation NMSE. Accept the pruned model only if its public
checks pass, its complexity vector is strictly smaller without increasing any
component, and

$$
E_{\mathrm{pruned}} \le E_b+\max(10^{-6},0.01E_b).
$$

The first live pilot produced six completed decisions and ten completed fits:
three pruned selections, one rejected deletion, and two skips for unresolved CSTR
requirements. Each accepted deletion removed one parameter and one term. None of
the accepted changes removed a state or named process. Two retained essentially
the same NMSE; the alien-device brief-only deletion increased validation NMSE
from 0.99737244 to 1.00646147 (about 0.91%), within the declared 1% tolerance.
This establishes working simplification, not improved scientific recovery.

## Checkpoints and outputs

`plan.json` seals source lineage, public train/validation arrays, policy, package,
launcher and runtime identities. Per-task `choice.json` freezes the candidate,
rejected alternatives, contributions, complexity change and predicate evidence.
`control/` and `pruned/` use the existing compatible-sibling fitting checkpoints.
A started but incomplete ranking or fitting attempt consumes its budget; resume
never silently grants another allocation. Scheduler intents and replies prevent
uncertain submissions from being duplicated.

`summary.json` includes historical parent metrics, both new fits, all parameters,
fit status, budget exhaustion, residual calls, selected origin, complexity and
requirement evidence. `SUMMARY.md` is the short comparison. Workflow `complete`
means a decision was published; inspect `fit_status_counts` separately. Missing,
skipped, ambiguous and failed results are not zeros. Tokens and LLM calls for this
pilot are zero; historical construction costs are not attributed to pruning.

There is no independent BDF/Radau replay in this milestone. Passing graph predicates
is scoped structural evidence, not scientific validity or physical conservation.
The present parents are often poorly fitted: a small measured contribution at one
retained parameter vector can become important at other parameters or inputs.
One deletion per parent is an engineering pilot, not exhaustive sparsity search.

## ACES submission

Use the full commit accompanying this runbook in `AF_COMMIT` below. Run in an
existing ACES shell; no new environment or full clone is needed.

```bash
(
  set -euo pipefail
  export AF_COMMIT=REPLACE_WITH_FULL_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/process-pruning-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/shared-process-integration-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/process-pruning-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  module load GCCcore/13.2.0 Python/3.11.5
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_process_pruning.py" \
    --source "$AF_SOURCE_ROOT" --root "$AF_OUTPUT_ROOT"
)
```

This submits preparation, six CPU tasks (one CPU, 16 GB each, four concurrent), and
an after-any report job. The 45-minute scheduler limit covers scoring and startup
as well as the two bounded numerical fits. No GPU or model server is required.
If submission is uncertain, retain receipts and inspect jobs; do not delete the
intent directory and blindly submit again.

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/process-pruning-v1
cat "$ROOT/SUMMARY.md"
jq '{status_counts, pruning_choice_counts, fit_status_counts, selection_counts}' \
  "$ROOT/summary.json"
jq '[.rows[] | {task, choice: .result.choice.status,
  unit: .result.choice.unit, before: .result.choice.audit.before,
  after: .result.choice.audit.after,
  paired_control_available: .result.selection.paired_control_available}]' \
  "$ROOT/summary.json"
```

Review this pilot before extending pruning or proceeding to the calibrated critic.
