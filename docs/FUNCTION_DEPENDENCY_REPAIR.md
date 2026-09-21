# Local function dependency repair and saved-response audit

Milestone `local-function-dependencies-1` is opt-in. Historical construction
defaults remain strict, and the fitter, public benchmark prompts, data, process
gain policies and LLM settings are unchanged. First run the CPU-only saved-reply
audit below. Review that audit before scheduling a fresh construction confirmation;
this milestone submits no live LLM or fitting campaign automatically.

## Why the previous replies failed

The original batch and atomic prompts **did** show each grouped source set and
explicitly demand exact equality. Six of eight shared-process constructions in
v4 exhausted local repairs at that check. Three initially omitted a declared
source; three added a surveyed area. A syntax rejection is not evidence that the
proposed law is scientifically wrong, nor that its original dependency list is
right. Some replies were substantively inconsistent, including an alleged
storage-mediated transfer depending only on inflow and cancelling area factors.

The function prompt now shows the current sources, available inventory,
permitted public fixed covariates, and the exact signed consumer conversions.
No hidden labels, future target values or new numerical evidence are added.

## Opt-in behavior

Call `run_staged_functions(..., generation_granularity="equation_batch_atomic_repair",
dependency_policy="local-function-dependencies-1")` under a new frozen root.
The basin adapter also accepts that policy from a sealed plan's
`function_dependency_policy`. Existing plans omit it and retain strict behavior.
Do not modify a historical plan to opt in.

- A function can add selected, supplied **public fixed covariates** without an
  extra LLM call. The runtime records those symbols in the actual dependency list.
  It cannot silently remove sources or add new state, process or input dependencies.
- An atomic repair has one extra Boolean, `revise_dependencies`. True explicitly
  proposes replacing that term's dependencies with the symbols in its expression
  after declared parameters are removed. No redundant source list or action-type
  label is requested. The scientific choice remains the proposer's.
- The runtime checks allowed names, restricted grammar, parameter roles, equation
  closure, algebraic cycles, required driver/composition/memory pathways and the
  shared-process contract. The explicit `requires_dynamic_memory` flag is enforced
  even if the older `positive_requirements` list is empty; older protocols do not
  change. No change commits until **both** graph validation and function binding
  succeed. Rejected attempts preserve the previous draft and topology.
- A process-law correction changes its definition and declaration together.
  Consumers continue to reference the same function and parameter identities.
  Their signs, targets and conversions cannot change in this repair. Unrelated
  functions, variables and causal boundary policies are preserved.
- Every committed change includes its original expression, before/after source
  sets and topology hashes. Independent reconstruction replays this ledger before
  building the fitting handoff. Resume reuses the same content-addressed calls,
  limits and deterministic changes; it grants no extra fitting attempt.

The policy does not infer conservation, change a discharge formula to satisfy
physics, or duplicate a shared law at its consumers. A covariate appearing in both
a process and its consumer conversion is reported for review, not automatically
removed: there can be legitimate uses of the same quantity in different roles.
The prompt explicitly warns against applying the same conversion twice.

## Connectivity scope

Existing Dalla Man public checks require declared driver-to-target paths and,
where specified, mediation through dynamic memory. They reject violations and
return feedback; they do not invent a missing scientific term. A surviving
inflow-to-output path does not establish that a separately defined outlet is used.

The basin topology stage has driver/memory requirements too. Its additional
post-construction `structure` check only forbids upstream influence in the
independent control. Neither that check nor a positive driver-path result certifies
drainage, threshold behavior or water balance. This milestone reports processes
with no equation-derived target path separately. It does not guess that an
arbitrarily named process must be an outlet or automatically subtract it.

These checks use symbolic occurrences, not a proof of functional dependence.
Self-division factors are flagged by the audit; their appearance cannot be counted
as scientific evidence of a mechanism. No complete scientific certificate is added.

## Saved-response audit

`scripts/audit_function_dependencies.py` reads sealed historical topology/function
stages and original cached repair requests. Each reply is evaluated independently
in its original context. It distinguishes:

1. `already_valid`: passes the old local contract;
2. `public_covariate_addition`: becomes mechanically admissible under the new rule;
3. `requires_proposer_revision`: would pass only **if** a future proposer explicitly
   authorizes the changed dependencies; this is a counterfactual, not acceptance;
4. `blocked`: still fails grammar, closure, role, pathway or shared-law checks;
5. `unavailable`: a referenced saved repair response is missing.

Batch slots and individual repair attempts are counted separately in detailed
rows. Neither count is a recovered-model count. `whole_models_recovered` stays
zero. Historical results are not rewritten, no LLM or optimizer is called, and
trajectory values are not used. Public plan metadata can contain development data;
the audit uses only its briefs and channel contexts. No test data are accessed.

The audit seals source-file hashes and current runtime code, checkpoints each
construction, rejects source drift/symlink escapes and reproduces identical results
on resume. Missing responses and previously accepted replies newly blocked remain
explicit. Inspect these before authorizing a fresh experiment.

## ACES commands

Set `AF_COMMIT` to the supplied commit and run on ACES. Use group scratch and the
existing Python environment; no clone, new environment, GPU or model download.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set the supplied commit first}"
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/function-deps-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v4"
  export AF_OUTPUT_ROOT="$AF_GROUP/function-dependency-audit-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  mkdir -p "$AF_OUTPUT_ROOT/logs"
  sbatch --export=ALL \
    --output="$AF_OUTPUT_ROOT/logs/audit-%j.out" \
    --error="$AF_OUTPUT_ROOT/logs/audit-%j.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_function_dependency_audit_aces.sh"
)
```

After the job finishes:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/function-dependency-audit-v1
cat "$ROOT/SUMMARY.md"
jq '{classification_counts, newly_mechanically_valid,
     previously_accepted_now_blocked, unavailable_saved_attempts,
     whole_models_recovered, llm_calls, optimizer_calls}' "$ROOT/summary.json"
jq '[.constructions[] | .task as $task | .routes[] |
     {task: $task, route, connectivity}]' "$ROOT/summary.json"
```

## Local verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python -m scripts.smoke_function_dependencies \
  --output /tmp/function-dependency-smoke
```

The smoke uses prescribed replies on an isolated two-store toy. It tests explicit
dependency correction, automatic geometry inclusion, one shared law, the causal
boundary handoff, independent reconstruction and deterministic resume. It performs
no numerical optimization and is not a scientific benchmark result.
