# Assembly ownership and saved-v5 audit

This opt-in milestone implements `process-assembly-contract-1`. It fixes the
construction interface, not the frozen fitter, public benchmark specification,
scientific critic, or numerical budget. Historical defaults and artifacts retain
their original interpretation. Start with the one-CPU audit below; it does not
submit a live campaign or grant another fitting attempt.

## Sign ownership

The existing ordinary-term normalization was enabled in requirement repair, but
was not wired into the basin function-construction path. Shared process laws were
explicitly unrestricted and retained outer minus factors. New-policy construction
uses the same conservative normalizer for every fixed-sign ordinary slot and for
the single defining law of a signed process. The consumers continue to own their
individual signs; the shared law is normalized once before any gain assembly.

For example, `P = -(h-crest)*area` becomes `(h-crest)*area`; a negative consumer
uses `-gain*conversion*P`. This is an explicitly declared representation convention,
not algebraic equivalence. It does not impose `abs(P)` or certify nonnegative
values: `h-crest`, `exp(-h)`, `(-h)**2`, and an unrestricted ordinary function retain
their internal signs. Ordinary `-h+crest` is not reinterpreted as `-(h-crest)`.
No threshold or scientific formula is invented. Raw expressions, normalized
expressions and sign records remain in a replayable per-slot decision ledger.

## Local repair preserves process-to-target paths

Before committing a dependency change, compare equation-derived paths from every
modeled algebraic process to each public target. A previously connected process
cannot silently become disconnected in a local function repair even if the global
input-to-target path and its immediate consumers survive. The diagnostic identifies
the process and target. An intentional removal belongs to a separate topology
revision that updates the other uses as well. This milestone does not implement a
new whole-model revision controller.

Already disconnected optional processes are reported by existing connectivity
facts, not forced into the model. Empty process proposals and the existing bounded
fallback remain valid. Paths are syntactic; a zero fitted coefficient, cancelling
expression or unsupported physical interpretation is not certified by this check.

## Conversion ownership

Each process-law request now includes exact signed consumer templates and the
already declared conversions. The runtime detects duplicated **outer** covariate
factors, such as `P=f/area` with consumer conversion `1/area`, and shows the assembled
contribution in focused repair feedback. It never guesses that every repeated
covariate is a duplicate or automatically removes a conversion.

The proposer can return the intrinsic law without the consumer factor. If the
factor intentionally belongs to the scientific law, an atomic repair may instead
set `conversion_factor_is_intrinsic: true`. This is a typed decision, not an
interpretation of the explanation. The expression then survives unchanged and the
confirmation is recorded. Fixed signs and required paths still apply. No additional
unbounded repair loop is added; existing per-step and total budgets apply.

The overlap detector covers outer products/quotients of covariate symbols. Factors
inside calls, sums, powers or mixed denominators, numeric-only conversions and
general unit analysis remain outside its certificate. Symbol overlap by itself is
not evidence of scientific error. The independent-gain arm still absorbs declared
conversions into its gains; this milestone preserves that comparison policy.

## Integration and provenance

Call `run_staged_functions` with `assembly_policy="process-assembly-contract-1"`,
`dependency_policy="local-function-dependencies-1"`,
`generation_granularity="equation_batch_atomic_repair"`, and
`function_repair_policy="certified_outer_gain"`. Basin plans opt in with
`process_assembly_policy`; it is absent from historical plans. A changed policy
requires a fresh output root. The same rules apply to batch candidates, focused
repairs and independent reconstruction. Dependency edits and assembly decisions
commit only after binding succeeds. Cached resume repeats no model calls.

## Audit first

The saved-v5 audit reads sealed metadata/stages and cached function replies. It
reconstructs the source graph at the time of each request, including earlier
accepted dependency revisions; it does not use the final graph for every attempt.
It does not read trajectory files or use trajectory values in metadata. It checks
input hashes and runtime identity, checkpoints each construction and rejects drift.
It neither changes saved models nor supplies missing scientific decisions.

Classifications are `unchanged_valid`, `outer_sign_normalized`,
`conversion_review_required`, `target_path_repair_required`, `still_blocked`, and
`unavailable`. Formerly accepted replies requiring a conversion/path repair are
reported separately from unexpected acceptance regressions. Sign normalization is
counterfactual under the new convention, not an improvement to historical NMSE.
Replies are not independent models; whole-model recovery remains zero.

On ACES, use the full pushed commit supplied with this milestone:

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?set the full milestone commit}"
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/process-assembly-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v5"
  export AF_OUTPUT_ROOT="$AF_GROUP/process-assembly-audit-v1"
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
    "$AF_REPO_ROOT/scripts/hpc/run_process_assembly_audit_aces.sh"
)
```

After the CPU audit finishes:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/process-assembly-audit-v1
cat "$ROOT/SUMMARY.md"
jq '{classification_counts,
     accepted_requiring_repair: (.historically_accepted_requiring_repair | length),
     unexpected_acceptance_regressions, unavailable_saved_attempts,
     missing_constructions, llm_calls, optimizer_calls}' "$ROOT/summary.json"
```

Inspect these results before a separately frozen fresh construction comparison.
There is no automatic follow-up and no change to the historical v5 results.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_process_assembly_contract.py tests/test_process_assembly_audit.py
.venv/bin/python -m scripts.smoke_process_assembly_contract --output /tmp/process-assembly-smoke
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The smoke uses prescribed responses to exercise corrected and intentionally
retained conversion factors, shared and ordinary sign normalization, causal
initializers, independent reconstruction and zero-new-call resume. Tests also
cover lost downstream paths, rollback, historical behavior and ledger tampering.
Both gain assemblies are evaluated after reconstruction to check the direction
and magnitude of the shared contribution, including intentionally retained factors.
