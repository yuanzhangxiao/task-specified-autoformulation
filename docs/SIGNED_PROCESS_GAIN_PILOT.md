# Signed process assembly and gain-policy pilot

This milestone adds opt-in `shared-process-contract-2` and
`detention-process-pilot-3`. Historical contracts remain available. It does not
change the fitter, benchmark assets, or finalized public prompts.

## Construction

The process proposal now chooses consumers and signs in **one call**, after
variable selection and before equation topology:

```json
{
  "name": "transfer",
  "depends_on": ["h_up", "h_down"],
  "kind": "transfer",
  "uses": [
    {"target": "h_up", "sign": "negative", "conversion": "1/area_up"},
    {"target": "h_down", "sign": "positive", "conversion": "1/area_down"}
  ],
  "scientific_meaning": "A shared volumetric transfer between the basins."
}
```

The example illustrates a schema, not a prescribed law or private reference.
The LLM decides dependencies, signs and scientific meaning. Runtime never parses
the explanation to infer a relationship, conversion, or validity judgment.

- `transfer` currently means a **pairwise** transfer: exactly two different
  consumers, opposite assembly signs. This is not an exhaustive process taxonomy.
- `influence` permits arbitrary positive/negative assembly signs and one or more
  consumers. Local outlets are valid; a second consumer is not required.
- `conversion` is optional. Use `null` for unknown, `"1"` for common amount
  coordinates, or a positive product/quotient of constants and public covariates.
  It contains neither a fitted parameter nor the process symbol. Unsupported
  conversion expressions become unresolved, with the original expression logged;
  they do not discard an otherwise valid process.
- Duplicate consumers within a process remain errors. Invalid individual
  suggestions do not erase unrelated valid suggestions. No processes is valid.
- A dynamic state may drive and consume a process. An instantaneous algebraic
  cycle is rejected. For example, `P(q_out)` combined with `q_out = P(...)` cannot
  be executed by this explicit-ODE runtime. General implicit/DAE solving is out of
  scope. Longer loops introduced in subsequent topology are checked there.

The runtime inserts declared signed terms into topology. Equation calls supply
only the other terms. An exact repeated declaration is deduplicated; a different
sign or different dependency is not silently reinterpreted. There are **no
`process_uses_*` calls**.

The interaction phase generates one function for each process, with one set of
parameters. Consumer identity slots are produced by the runtime and are omitted
from function requests. Independent reconstruction checks those slots against
the signed declarations before gain compilation. A nonnegative outer magnitude
does not assert that an internal nonlinear function is everywhere positive.

If optional process construction fails, the pilot retains the existing fallback
to the original variable inventory, within the same total call/token cap. A
fallback is recorded and is not counted as a successfully bound shared process.

## Matched gain comparison

The pilot imports only the saved pre-process inventories from
`detention-process-pilot-v1`, as the previous v2 pilot did. It does not import old
fitted parameters, proposed laws, hidden labels or private reference equations.

There are eight common constructions: coupled/independent basin cases, two seeds,
and full/brief-only information. Each construction supplies the **same process
shapes, other equations, and causal initialization plan** to two CPU fitting arms:

| Policy | Pairwise transfer contribution | Influence contribution |
| --- | --- | --- |
| `explicit_conversion` | One fitted magnitude shared by the two uses, multiplied by each proposed fixed conversion | Separate fitted gain per use, with its proposed conversion when available |
| `independent_gains` | Separate effective fitted gain per use; conversion absorbed into that gain | Separate effective fitted gain per use |

Thus, for a transfer shape `phi`, the explicit arm uses
`-k*phi/area_up`, `+k*phi/area_down`. The independent arm uses
`-a*phi`, `+b*phi`. Both magnitudes are nonnegative; independent means separately
fitted, not constrained to have different numerical values.

The compiler normalizes a certifiable, process-local, direct coefficient amplitude
to one before inserting the gains. Its magnitude domain comes from the declared
consumer signs. The shared law itself retains an unrestricted sign: `-k*x`
becomes the shape `-1*x`, not `x`. This removes the simple redundant `k` in
`P=k*phi` when replacing it by effective per-consumer gains. Internal parameters
and shared parameter identities are preserved. General nonlinear scale ambiguities
are not eliminated or certified; each process records what was normalized.

If a transfer conversion is unresolved, the explicit-conversion arm is
`gain_assembly_unavailable`; the independent arm can still run. Missing conversions
are never silently replaced by a conservation claim. With no accepted processes,
the two compiled candidates are identical and labeled with zero process bindings.

When a proposed conversion is constant across training trajectories, the free
gain starts at that conversion and the explicit gain starts at one. This matches
starting training predictions while using different parameter coordinates. Missing
or varying conversions are explicitly marked unmatched and use the ordinary unit
gain. No validation observations choose these guesses. This is a comparison of
model assembly policies, not identical optimizer geometry. The fitter profile,
numerical budgets, data, other
parameter guesses and initialization maps are unchanged. No test data are opened.
If geometry changes between trajectories, one constant effective gain may not
represent the corresponding geometry-dependent conversion; that is a limitation
of the independent-gain family rather than a reason to alter the data.

Both policies reuse one LLM construction. Per-arm token fields refer to that same
construction and must not be added twice. `shared_construction_accounting` counts
it once and excludes historical variable-selection costs.

## Evaluation and limitations

Inspect construction/admission/fallback, shared-law counts, gain compilation,
train/validation NMSE and numerical replay separately. Replay agreement is
numerical consistency, not scientific validity or predictive accuracy.

The new diagnostic evaluates cancellation of **declared transfer terms** in named
physical depth coordinates `h_up` and `h_down`, using public training covariates.
Both named depths must actually be differential states in the candidate.
It tests `area_up*a == area_down*b`, including explicit conversion factors when
present. It does not require equal effective gains. Other coordinate systems are
reported unavailable; a zero transfer is uninformative. This is a conditional
term-level check, not certification of the latent state's physical meaning, the
model's complete water balance, or absence of duplicated pathways. No loss term or
model selection decision uses this diagnostic. Nothing is automatically promoted
to the full pipeline, and no subsequent experiment is automatically launched.

## ACES commands

Run on ACES. Use group scratch to avoid the personal file-count quota. The existing
base checkout provides Git transport and Python. Set the exact pushed commit.

```bash
(
  set -euo pipefail
  export AF_COMMIT=REPLACE_WITH_PUSHED_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/basin-gains-${AF_COMMIT:0:7}"
  export AF_OUTPUT_ROOT="$AF_GROUP/detention-process-pilot-v3"
  export AF_PARENT_ROOT="$AF_GROUP/detention-process-pilot-v1"
  export AF_PUBLIC_ROOT="$AF_GROUP/detention-development-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"
  export AF_CONFIG="$AF_REPO_ROOT/configs/detention_process_pilot_v3.json"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  bash "$AF_REPO_ROOT/scripts/hpc/submit_detention_process_pilot_aces.sh"
)
```

The wrapper submits a CPU preflight, one H100 for eight common constructions,
sixteen one-CPU fits (concurrency four), and a CPU summary. Every LLM call is cached;
constructed models and derived gain arms are sealed. Fits preserve consumed-attempt
markers. Repeating a completed submission returns existing job IDs; uncertain
scheduler replies retain intent rather than submitting duplicate jobs.

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v3
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SUMMARY.md"
cat "$ROOT/EQUATIONS.md"
jq '{status_counts, shared_construction_accounting,
  rows: [.rows[] | {task: .task.task_id, status, proposal_status, fallback_used,
    shared_governing_laws, training, validation, replay, transfer_cancellation,
    gain_compilation_error, gain_compilation}]}' "$ROOT/summary.json"
```

For the process-stage diagnostic (eight constructions, without duplicated arms):

```bash
jq -s '[.[] | {task: .task.task_id, status, fallback_used,
  attempts: [.attempts[] | {route, error, process_review, shared_process_contract}]}]' \
  "$ROOT"/construction/results/*/proposal.json
```

## Local verification

Implementation is in `search/signed_processes.py` (declarations and assembly),
the staged topology/function runners and reconstruction audit (preservation),
`rebuttal/process_gain_comparison.py` (gain compilation and diagnostics), and
`rebuttal/detention_process_pilot.py` (paired execution and reporting). The v3
configuration uses the existing submission scripts with an additional protocol
route. The two new test modules and smoke script exercise the complete handoff.

```bash
python -m pytest -q tests/test_signed_processes.py tests/test_process_gain_comparison.py
python -m scripts.smoke_signed_processes --output /tmp/signed-process-smoke
```

The smoke uses temporary synthetic data and prescribed responses. It checks both
gain families through the real constructors, frozen fitter and independent replay,
including deterministic resume. It tests implementation, not live LLM capability.
