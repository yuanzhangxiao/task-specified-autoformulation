# Signed-process construction reliability and saved-response audit

This milestone repairs a construction contract mismatch found after the basin
gain-policy pilot. It changes neither the fitter nor the benchmark data or public
task specifications. Run the saved-response audit before scheduling another live
pilot; there is no automatic follow-up.

## Construction changes

- When signed shared-process terms already populate a selected equation, the
  proposer supplies only additional terms. `terms: []` is valid in that context.
  The schema and system prompt now agree. Equations with no supplied process terms
  still require a nonempty definition. The assembled equation still goes through
  the original source, graph, memory and construction-limit checks.
- One process still has one shared function, with automatic identity uses at its
  consumers. The proposer chooses its science and signs. Explanatory prose does
  not become an executable law or a scientific certificate.
- The shared function request now retains the original process suggestion and
  unresolved conversion expressions as explicitly untrusted advisory context.
  An expression rejected as a fixed unit conversion is not silently inserted into
  the model. The function must still use its declared dependencies and restricted
  expression grammar. This does not guarantee a correct threshold law.
- New signed constructions use the `signed-process-handoff-2` marker in their
  evidence identity. Historical construction reconstruction keeps its original
  context. Do not resume an old live construction using changed numerical or
  proposal code; use a new campaign root for any later live experiment.

## What the audit establishes

`scripts/audit_process_handoff.py` reads the existing
`detention-process-pilot-3` plan, saved topology requests/replies, and published
candidate/results. It makes zero LLM calls and zero optimizer calls. It does not
read test trajectories or private reference equations. Public fixed geometry is
used for the already defined transfer-cancellation diagnostic; raw trajectory
values do not enter the diagnostics.

Each historical reply is checked in its original request context using both the
old and corrected reply schemas. Corrected additions are then assembled and
checked as an equation. A locally repaired attempt is **not** a completed model:
there may be no saved later replies, and alternative construction paths are not
invented. Counts are attempts, not independent runs or recovery rates. No candidate
is promoted and no historical result is overwritten.

The audit also records advisory facts about published equations:

1. Identical process laws, and state equations that consume several of them.
   This identifies possible redundant decompositions, not a full identifiability
   proof. In the saved seed-0 brief model, the transfer and upstream outlet both
   used `h_up`, so moving weight between them can leave the total upstream loss
   unchanged while changing the declared transfer-cancellation score.
2. Affine dependence on dynamic channels, and explicit `max`/`min`/`abs` operators
   after algebraic expansion. A name such as "overflow" is not a threshold, and
   merely containing `max` does not certify the correct threshold or direction.
3. Whether the public `initial_up` gauge reading occurs in an ongoing RHS rather
   than solely in the initial assignment. Such use is flagged for scientific
   review, not prohibited: a fixed covariate can legitimately parameterize a law.
4. Area-weighted cancellation for declared transfer terms at fitted parameters.
   This does not certify the full model's water balance or missing pathways.

These checks are scoped to the public basin vocabulary. They do not interpret
latent coordinate meanings or reject models by reading scientific descriptions.

The output freezes the source plan, read-artifact hashes, and runtime hash;
checkpoints each construction; and rejects source/runtime changes on resume.
Missing request records remain explicitly unavailable. A repeated invocation with
the same source and code is deterministic. Use a separate output directory if the
source or implementation changes.

## ACES: submit one CPU audit

Set `AF_COMMIT` to the commit supplied with this milestone, then run in the ACES
shell. The archive and audit output use group scratch to avoid the personal inode
quota. No fresh environment, GPU, model download, proposal, or fit is needed.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the supplied commit first}"
  export AF_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/process-handoff-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v3"
  export AF_OUTPUT_ROOT="$AF_GROUP/process-handoff-audit-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  [[ -f "$AF_SOURCE_ROOT/plan.json" ]]
  mkdir -p "$AF_OUTPUT_ROOT/logs"
  sbatch --parsable --account=156264627414 --partition=cpu \
    --job-name=process-handoff-audit --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=8G --time=00:20:00 --export=ALL \
    --output="$AF_OUTPUT_ROOT/logs/audit-%j.out" \
    --error="$AF_OUTPUT_ROOT/logs/audit-%j.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_process_handoff_audit_aces.sh"
)
```

Inspect the results after the job completes:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/process-handoff-audit-v1
cat "$ROOT/SUMMARY.md"
jq '{historical_status_counts, newly_valid_attempts,
     unavailable_saved_attempts, previously_accepted_now_blocked,
     llm_calls, optimizer_calls, whole_models_recovered}' "$ROOT/summary.json"
jq '.models[] | {task, status, fit,
     duplicate_laws: .diagnostics.duplicate_process_laws,
     findings: .diagnostics.findings,
     cancellation: .diagnostics.transfer_cancellation}' "$ROOT/summary.json"
```

If previously accepted attempts become blocked, examine their saved-context errors
before a live rerun. If new attempts become locally valid, a later fresh matched
pilot can test whether construction improves. That is a separate milestone; this
audit does not establish a benefit from shared processes or either gain policy.

## Local verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m scripts.smoke_signed_processes --output /tmp/signed-process-smoke
.venv/bin/python scripts/audit_process_handoff.py \
  --source /tmp/signed-process-smoke/signed --output /tmp/process-handoff-audit
```

The regression tests cover empty supplied equations, ordinary-empty rejection,
unchanged source checks, full function handoff, resume without new requests,
read-only saved audits, missing records, tampering, and advisory equation facts.
The signed-process smoke exercises construction, gain assembly, fitting, replay
and deterministic resume with offline provider fixtures.
