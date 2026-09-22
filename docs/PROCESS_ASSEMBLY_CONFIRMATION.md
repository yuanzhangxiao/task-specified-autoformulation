# Fresh process assembly confirmation — basin v6

The saved-v5 audit passed: 45 unchanged-valid attempts, two outer signs normalized,
ten conversion questions, one lost-target-path repair and 15 still-blocked replies.
Four historical review routes replayed through the exact-hash compatibility path.
No saved attempts or constructions were missing; no unexpected acceptance
regressions occurred. These 73 function attempts are not independent models.

This milestone enables the already implemented `process-assembly-contract-1` in
a fresh, bounded construction comparison. It changes no benchmark data, public
scientific prompt, model settings, fitter algorithm or numerical budget.

## Frozen comparison

- Eight new constructions: coupled/independent basins, seeds 0/1, full/brief-only.
- Reuse exactly the v5 variable inventories, public train/validation exports,
  training-evidence settings, modeling limits and proposer budgets. There is no
  new variable-selection call. Historical equations and outcomes are report-only.
- Retain GPT-OSS-20B and the historical low-reasoning settings and serving image.
  A fresh request namespace prevents reuse of historical provider responses.
- Each construction receives the existing two gain assemblies, for sixteen fits:
  `explicit_conversion` and `independent_gains`. Both use the frozen
  `collocation-single-target-v2` fitter and existing independent solver replay.
- Enable consistent outer-sign ownership, focused duplicated-conversion feedback,
  explicit intrinsic-factor confirmation, and protection of existing
  process-to-target paths during local dependency edits.
- Empty process proposals remain valid. Optional process construction can fall
  back within the same total construction budget. Rejected models get no extra fit.

The control is the completed historical v5 campaign. Prompts and therefore sampled
models can change despite matching seeds. This tests the combined construction
interface, not a causal effect on identical equations or a new fitting method.
The experiment does not select a gain policy, promote a model, or launch extra rounds.

## Gate, checkpointing and reporting

Preparation requires the sealed successful `process-assembly-audit-v2` and complete
v5 artifacts. It verifies the exact source hashes, construction matrix, replay
certificates, absence of missing records/unexpected regressions, and matching
Python/package runtime. Expected conversion/pathway repair findings are allowed.
The audit's old runtime hash is retained as provenance, not replaced with the new
package hash. V6 pins its own package and launcher identity.

`baseline.json` snapshots the historical outcomes. The plan contains the matched
public inputs and binds the baseline; workers need only the new campaign directory.
Historical mounts can subsequently be offline. Identical preparation and completed
submission are idempotent. Partial/uncertain scheduler submissions retain receipts
and require inspection before another submission. Existing fit budget and replay
markers continue to govern resume; a restart grants no fresh numerical budget.

`COMPARISON.md` and `comparison.json` report:

- construction completion, retained process-review routes and fallback use;
- total and atomic-repair calls/tokens, once per construction, including fallback;
- committed slot sign decisions, intrinsic-factor confirmations, and rejected
  conversion/pathway attempts with their original diagnostics;
- both gain arms' statuses, train/validation NMSE, complexity, connectivity,
  fitted gains, equation diagnostics and numerical replay agreement.

Missing/unmeasured usage remains explicit. Legacy assembly decisions were not
recorded: aggregate decision counts are `null`, not reconstructed zeroes. New
slot decisions may belong to an ultimately failed construction; counts do not mean
recovered models. Connectivity and solver agreement do not certify physical
conservation or prediction accuracy. `SUMMARY.md` and `EQUATIONS.md` retain the
existing detailed fit/structure report. No private references or test data are read.

## ACES commands

Set `AF_COMMIT` to the supplied full commit, then run on ACES:

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the supplied commit first}"
  export AF_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/assembly-confirm-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v5"
  export AF_AUDIT_ROOT="$AF_GROUP/process-assembly-audit-v2"
  export AF_OUTPUT_ROOT="$AF_GROUP/detention-process-pilot-v6"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  bash "$AF_REPO_ROOT/scripts/hpc/submit_process_assembly_confirmation_aces.sh"
)
```

Four jobs: CPU preflight, one H100 proposer allocation, sixteen one-CPU fits
(at most four concurrent), and the report after the fit array terminates.

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v6
cat "$ROOT/submission_manifest.json"
JOBS=$(jq -r '.jobs | [.[]] | join(",")' "$ROOT/submission_manifest.json")
sacct -j "$JOBS" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v6
cat "$ROOT/COMPARISON.md"
jq '{historical_status_counts, current_status_counts,
     historical_accounting, current_accounting}' "$ROOT/comparison.json"
jq '[.construction_pairs[] | {task, status: .current.status,
     fallback: .current.fallback_used, assembly: .current.assembly}]' \
  "$ROOT/comparison.json"
```

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_process_assembly_confirmation.py
.venv/bin/python -m scripts.smoke_process_assembly_confirmation \
  --output /tmp/process-assembly-confirmation-smoke
bash scripts/run_tests.sh full
.venv/bin/python -m pytest -q tests/test_shared_process_pilot.py
.venv/bin/ruff check .
```

The smoke creates a separate toy with prescribed offline replies, not new basin
data. It exercises a real duplicated conversion, explicit correction and outer
sign normalization, independent reconstruction, both gain policies, two frozen
fits, numerical replay, and zero-new-work resume. Tests cover absent processes,
audit/source/baseline drift, missing results, partial reporting and bounded submit.
