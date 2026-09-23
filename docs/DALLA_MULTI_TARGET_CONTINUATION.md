# Dalla Man joint-output continuation and construction repair

`review-deadline-6` is an opt-in continuation of the completed three-visit
`dalla-mechanism-pilots-v1`. It imports all six lineages at source round 2 and
allocates twelve further visits each: fifteen total visits, indexed 0–14. It
does not update the pinned checkout or artifacts used by the running pilot.
Import requires every source-round result; a Slurm completion receipt alone is
insufficient. The new preparation and submission commands may be run after the
round-2 retry finishes.

## Scientific and runtime changes

The proposer continues to supply variable kind and complete RHS expressions.
The `scientific-content-multi-revision-1` schema replaces the single
`output_expression` with `output_mappings: [{channel, expression}, ...]`.
Mappings address declared target channels. Omitted equations, mappings and
initializers remain unchanged. All required outputs are compiled and fitted
together; none is replaced by measured future values. New equation/output
parameters are declared once and resolved over all occurrences. A mapping
change that makes an observed state latent requires an explicit causal boundary.

Short evidence references retain the existing advisory policy. Valid IDs must
resolve to measurements actually present in the training packet. Unknown IDs
are saved in `citation_audit.unresolved_references`, omitted from verified
evidence IDs, and do not by themselves reject executable equations. Even valid
references do not certify a scientific claim. Grammar, available channels,
public requirements, causal initialization and parameter-role checks stay hard.

The pilot's public target contract is preserved, with its exact prompt hash and
provenance text. Its requirements are now displayed before fresh inventory/RHS
construction and on every repair or fitted revision. In T2-easy the frozen gate
expects `U` to be an algebraic output and requires a path from `Uii`. That role
was inferred by the existing contract builder from the public rate description;
it is an operational representation restriction, not a mathematical proof that
an ODE for a rate is impossible. This continuation exposes that interpretation;
it does not silently weaken the gate or change benchmark prompts. A future
comparison of representation policies would require a separately declared run.

Unfitted drafts rejected by public predicates can now receive up to three repair
calls per visit. The exact failed predicates, public contract and descriptive
training evidence are supplied. There are no fabricated residuals, fitted
parameters or verified numerical citations. A dynamic-to-algebraic conversion
requires an explicit proposer patch and is checked atomically with all other
outputs. No fit is launched until the complete public certificate passes.
Unrelated definitions remain unchanged unless explicitly revised.

Import preserves the latest recorded executable construction draft for an
unfitted lineage, including the original proposal hash. If no executable draft
exists, the next visit uses the existing staged construction budget. An
unsuccessful repair retains its executable draft for the next bounded visit;
it does not close the lineage. A fresh construction and its local repairs share
the existing 128-request/token cap; a saved-draft repair has at most three
physical calls. Every call is cached and every attempt is recorded.

Fitted lineages inherit the later controller's continuation behavior. Rejected
revisions, no-change replies and unavailable residual evidence use at most the
visit's existing incumbent-refit allocation. A failed or worse fit retains the
previous incumbent. It neither closes the lineage nor creates extra numerical
budget. Interrupted fitting preserves the incumbent or draft but cannot restart
the spent attempt in the same visit. Twelve visits remain the stopping limit.

The fitter, per-target training scales, initialization rules, numerical budgets,
validation selection, public assets and scientific-judge-off setting are inherited.
Neither test data nor private reference material is opened. Historical protocols
keep their single-output restrictions, gates, schemas and stopping behavior.

## Submission and results

Use a new independent group-scratch clone. A linked worktree can still write Git
metadata into a quota-exhausted personal-scratch repository. The commands below
read the original checkout but do not update it. Replace `REV` with the full
commit reported in the implementation response.

```bash
(
  set -euo pipefail
  module load GCCcore/13.2.0 Python/3.11.5 WebProxy
  REV=REPLACE_WITH_REPORTED_COMMIT
  AF_WORK=/scratch/group/p.nairr260351.000/u.yx126462
  AF_OLD_REPO="$AF_WORK/repos/autoformalism-dalla-pilots-4113fc1"
  export AF_REPO_ROOT="$AF_WORK/repos/autoformalism-dalla-v6-${REV:0:7}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    git clone --no-hardlinks "$AF_OLD_REPO" "$AF_REPO_ROOT"
    git -C "$AF_REPO_ROOT" remote set-url origin \
      "$(git -C "$AF_OLD_REPO" remote get-url origin)"
  fi
  git -C "$AF_REPO_ROOT" fetch origin codex/prefit-aces-v1
  git -C "$AF_REPO_ROOT" checkout --detach "$REV"
  export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
  export AF_SOURCE_ROOT="$AF_WORK/dalla-mechanism-pilots-v1"
  export AF_OUTPUT_ROOT="$AF_WORK/dalla-mechanism-continuation-v2"
  export AF_SOURCE_ROUND=2 AF_ADDITIONAL_VISITS=12
  export AF_COMPUTE_CACHE_ROOT="$AF_WORK/runtime-cache/review"
  export AF_IPC_TMP_ROOT=/tmp/af-ipc-u.yx126462
  bash "$AF_REPO_ROOT/scripts/hpc/submit_review_multi_aces.sh"
)
```

The CPU preparation job tests the controller and all-target fitter, then runs
the synthetic controller smoke with real fitting. One H100 serves each proposer
visit; separate CPU array tasks fit each lineage. In this protocol, fits require
a successful proposer job, and later dispatch requires a successful finish job.
A finish job writes the report and fails if any current task result is missing.
The runtime compiler cache defaults to group scratch; short IPC paths are under
node-local `/tmp`. A write check cannot guarantee future quota capacity.

Do not erase submission intents to force retries. A partial or uncertain
submission retains its exact scheduler replies and IDs for inspection. Repeating
a completed submission returns its receipt without duplicating jobs.

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-continuation-v2
sacct -X -j "$(jq -r '[.jobs[]] | unique | join(",")' "$AF_ROOT/submission_manifest.json")" \
  --format=JobID,JobName%30,State,ExitCode,Elapsed
jq '{planned_rounds,status_counts,rows:[.rows[] | {
  cell,seed,round,status,proposal_status,fit_trigger,
  retained_train_per_target_nmse,retained_validation_per_target_nmse,
  all_graph_requirements_certified,budget_exhausted
}]}' "$AF_ROOT/summary.json"
```

The new report contains 78 rows: six imported checkpoints plus 72 new visits.
Imported checkpoints incur no new calls or fits. Its `round` field is the global
zero-based index; inspect rounds 5, 8, 11 and 14 for total visits 6, 9, 12 and 15.
Saved `proposal.json` records contain construction-repair attempts and citation
audits. `result.json` distinguishes `construction_draft`, `selected`, original
`proposal_status`, and incumbent fallback. Low error and public graph validity
still do not prove the equations or fitted mechanisms are scientifically correct.

## Verification

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_review_multi.py tests/test_multi_target_profile.py \
  tests/test_review_continuation.py tests/test_review_parameters.py \
  tests/test_review_revision_v5.py tests/test_dalla_pilot.py
.venv/bin/python scripts/smoke_review_multi.py --output /tmp/af-review-multi-smoke
.venv/bin/python -m pytest -q -n 4
.venv/bin/ruff check .
```

The synthetic smoke repairs an unfitted three-output draft, fits it, revises an
algebraic output using a new parameter, rejects invalid symbols, refits incumbents
after rejection, and checks exact resume and source preservation. Its LLM transport
is mocked; its fits are real. It is an interface/continuation test, not evidence of
improved Dalla Man model recovery. Live confirmation remains to be run on ACES.
