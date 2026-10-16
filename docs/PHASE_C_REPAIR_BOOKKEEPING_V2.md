# Saved topology repair: bookkeeping qualification v2

This opt-in milestone repairs defects observed in the 15-start paired experiment.
It changes neither public benchmark prompts nor fitting, and does not increase
the repair allowance. Historical policies and the original frozen bundle remain
reproducible. `saved-actionable-topology-repair-2` applies only to saved repair;
initial variable/shared/topology construction prompts are unchanged.

## Changes

- A rejected mechanism record names the exact memory states missing from its
  component list and shows a conditional corrected record. The runtime does not
  select scientific components, remove states, or apply the example automatically.
- An unchanged reply cannot be repaired by declaring completion while obligations
  remain. Feedback explains that distinction instead of offering contradictory
  “edit or finish” advice.
- Repetition compares the retained scientific model, outstanding obligations and
  rejected scientific alternative. Completion toggles, prose and redundant
  accepted/no-op restatements do not reset it. Different rejected scientific
  alternatives remain distinct. Three consecutive repeated attempts still stop;
  the ten-call and 262,144 charged-token caps remain unchanged.
- Qualified closed-subsystem conflicts explain their invariant using the actual
  matched-initialization/complete-forcing witness. Self-only decay or a shared
  constant changes the function but introduces no between-trajectory distinction.
  The proposer must choose the scientific repair; no particular missing edge is
  supplied by the runtime.

## Required memory and cycles

Ordinary reachability can credit `I -> X -> I -> target` even when the target
uses only current `I`. This does not identify a history effect beyond that current
driver. V2 separately qualifies explicitly required dynamic-memory mechanisms:
at least one selected differential memory must reach each required target without
passing through the mechanism's current driver quantities as intermediate nodes.
Existing driver-to-memory checks and their quantifiers remain in force.

The target endpoint is allowed: genuine delayed self-feedback `I -> X -> I`
remains legal. Other cycles, parallel instantaneous pathways, multiple memory
coordinates and feedback into the driver are not banned. Optional memory claims
are not subject to this additional required-memory gate. It is a structural
qualification of an explicit public history requirement, not proof of a physical
mechanism, function signs, conservation, or predictive adequacy.

On the saved 30 final drafts, the audit flags T2-hard seed 2 in both arms. The
control was already structurally blocked; only the actionable final draft changes
from eligible to blocked under this qualification. Saved assessments are retained
and the new result is separate.

## Arity and state coupling

The independent audit counts each ordinary contribution function and each named
process definition once, regardless of the number of consumers. It distinguishes
explicit argument count from dependencies after expanding algebraic/process
definitions. An ODE state is a leaf in this expansion; its derivative is not
substituted into the right-hand side.

| Cohort | Single-argument laws / all laws |
| --- | ---: |
| Preceding full 40 constructions | 140 / 172 (81.4%) |
| Selected 15 starting drafts | 65 / 75 (86.7%) |
| Historical control finals | 61 / 72 (84.7%) |
| Historical actionable finals | 102 / 110 (92.7%) |

These are declared topology functions, not fitted functions. No resolvable
multi-state model in these cohorts has zero off-diagonal state dependencies.
The full 40 contain 20 single-ODE models, 15 weakly connected one-way multi-state
models, three mutually coupled models, and two models with disconnected state
blocks. The actionable 15 contain one single-ODE model, eleven one-way models,
two mutually coupled models, and one unresolved graph with an undeclared symbol.
Single-state models are reported separately rather than called evidence of
successful coupling. Two disconnected blocks are not the same as every ODE being
independent.

The prompts already allow a joint function via `sources=[x,z]`. There is no
univariate schema restriction. A sum `f(x)+g(z)` cannot generally express a joint
interaction such as `x*z`; conversely, `f(y)` in the ODE for `x` couples two states
despite having one argument. Additional intermediate variables can change the
effective dependence. Arity alone is not a scientific quality metric and no
minimum-arity rule is introduced.

## Qualification and next run

The delivered v2 plan starts fresh repair episodes from the same 15 original
drafts, with the same seeds, evidence and settings. It runs only the revised
actionable arm. Old controls and actionable results are historical references,
not newly randomized or equal-cost controls. It does not continue or reset the
budget of an existing episode. No accepted model is automatically promoted.

Offline preparation, with the repository environment active:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python scripts/saved_topology_repair.py refreeze \
  --source ORIGINAL_REPAIR_INSPECTION --root NEW_INPUTS
python scripts/saved_topology_repair.py preview \
  --root NEW_INPUTS --output PROMPT_REVIEW
```

The delivered bundle includes the sealed public inputs and serving-probe requests.
On the Jetstream2 VM, use a new root and the existing compatible local server:

```bash
sha256sum -c MANIFEST.sha256
export AF_JS2_ROOT="$HOME/af-construction"
export AF_JS2_CAMPAIGN_ROOT="$AF_JS2_ROOT/saved-actionable-repair-v2"
sudo -v
bash scripts/jetstream/run_saved_topology_repair.sh run
```

Use the same folder and exports with `status`, and then `inspect` after the worker
stops. Repeating `run` resumes only the remaining allowance. The new summary shows
diagnostic coverage separately: `checks_clear` means no tracked repair obligation
remains; unavailable or inapplicable checks do not become scientific passes.

Validation includes successful and failing record edits, exact cached resume,
completion-toggle repetition, distinct rejected alternatives, valid feedback-cycle
counterexamples, historical prompt preservation and actual saved-artifact audits.
No live calls, fitting, solver rollouts or test-data access are needed to qualify
the implementation. Scientific improvement remains to be measured in the new run.
