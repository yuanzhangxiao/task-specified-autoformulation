# One process law reused across equations

This opt-in milestone implements `shared-process-contract-1` and the matched
`detention-process-pilot-2`. Historical advisory campaigns remain unchanged.
No fitter settings, benchmark prompts or benchmark data are modified.

## Scientific construction contract

After the variable stage, the proposer supplies four fields per process:

```json
{
  "name": "P",
  "depends_on": ["C", "D"],
  "used_in_equations_for": ["A", "B"],
  "scientific_meaning": "Transfer from A to B, controlled by C and D."
}
```

The stage lists eligible dependencies and eligible modeled equation targets
separately. An observed state may have a modeled equation; a supplied forcing
or covariate does not. One-use processes such as a local outlet are valid. Empty
proposals are valid. Each suggestion is validated independently; an invalid
outlet cannot discard an independent valid transfer. Invalid names are reported,
not guessed or silently replaced. Existing algebraic identities can be reused
without losing their structured relationships. Conflicting duplicate declarations
are rejected, while unrelated valid declarations survive.

Topology makes one joint decision for all uses of each accepted process. Each
use has a target equation, an outer assembly sign, conversion source variables,
and a scientific role. These signs are proposed by the LLM, not inferred from
process names. The subsequent equation construction must retain those terms
exactly once and completes the other terms normally. A requested different
inventory is recorded as unresolved rather than silently changing the contract.

The runtime creates one algebraic definition slot for `P`, with its declared
dependencies. Its outer sign is unrestricted because the function itself may
be signed. This is distinct from the signs of its contributions to A and B.
The interaction phase generates that function once, including its parameter
identities. Consumers refer to P; they do not regenerate independent copies of
its law. A restricted-AST check requires one multiplicative occurrence, allowing
`P`, `P/area`, or an explicitly modeled conversion/gain times P, and rejecting
`tanh(P)`, `P*P`, `P+offset`, a denominator containing P, or an inlined substitute.
Other additive contributions remain separate equation terms. Failed batch slots
receive the existing bounded atomic repair; valid slots remain intact.

Independent reconstruction revalidates the same shared-law, source and sign
contracts before handing the candidate to the unchanged fitter. The executable
candidate contains one algebraic process and one shared set of its parameters.
The smoke checks a transfer coefficient shared by two depth equations, with
distinct surveyed area conversions and a separate local outlet coefficient.

These are representation guarantees, not scientific certification. Proposer
choices can still have wrong directions, conversions, dependencies or dynamics.
Signed processes remain permitted. Consumer gains can differ for a shared
response; conservation is not inferred merely from opposite signs. The milestone
does not certify arbitrary balances, eliminate cancellation through other terms,
or permit arbitrary nonlinear consumer transformations under the shared-law label.
Dependencies/conversion choices cannot be silently revised mid-attempt; a broader
revision protocol is future work. If optional process assembly fails, the pilot
explicitly falls back to its original inventory under the same total call/token
budget, including when the proposal reused an existing process name. Fallback
models are labeled and do not count as successful bound-process constructions.

## Controlled experiment

The new 16-arm pilot uses the qualified coupled/independent basin data, two seeds,
and full/brief-only information variants. For each pair, both arms reuse the
**same saved pre-process variable inventory** from the prior review-on call.
Neither arm imports fitted parameters, old topology, old interaction laws or
hidden trajectories. The on arm makes a fresh four-field process proposal;
the off arm proceeds to the existing guidance-only topology construction.
The same downstream call/token caps, GPT-OSS-20B low configuration and frozen
single-target fitter apply. Signs, process laws and conversions remain generated
from public information. There is no pruning, scientific judge or revision loop.

The importer verifies the sealed parent plan/proposal and content-addressed
request, records the complete cache-file hash, and freezes the paired inventory
inside the new plan. It also performs a no-call, no-fit admission replay of the
old process replies under the new item-wise rules. That replay diagnoses rejected
content; it does not promote an old candidate or replace the new live proposal.

Token totals in the new pilot cover its downstream construction, including
process calls, repairs and fallback. They exclude historical variable-selection
costs, which are reused identically within each pair. Same-seed LLM generations
need not be byte-identical; an empty process proposal can still be followed by
different sampled downstream equations. Interpret such differences accordingly.

## Run on ACES

Use the pushed commit in a new archive directory on group scratch. The existing
base checkout supplies Git transport and the Python environment. Do not reuse
the old experiment output directory.

```bash
(
  set -euo pipefail
  export AF_COMMIT=REPLACE_WITH_PUSHED_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/basin-shared-${AF_COMMIT:0:7}"
  export AF_OUTPUT_ROOT="$AF_GROUP/detention-process-pilot-v2"
  export AF_PARENT_ROOT="$AF_GROUP/detention-process-pilot-v1"
  export AF_PUBLIC_ROOT="$AF_GROUP/detention-development-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"
  export AF_CONFIG="$AF_REPO_ROOT/configs/detention_process_pilot_v2.json"

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

The wrapper submits a CPU preflight, one H100 proposer allocation, 16 one-CPU
fits with concurrency four, and a summary job. Repeated successful submission
returns existing job IDs. Uncertain/partial submission retains intent and stops
instead of submitting duplicates. No automatic subsequent campaign is launched.

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v2
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SUMMARY.md"
cat "$ROOT/EQUATIONS.md"
jq '.pairs | to_entries | map({task: .key,
  previous: .value.historical_status,
  replay: .value.saved_reply_admission.status,
  accepted: .value.saved_reply_admission.suggestions,
  rejected: .value.saved_reply_admission.rejected_suggestions})' \
  "$ROOT/SAVED_REPLY_AUDIT.json"
jq '[.rows[] | {task: .task.task_id, status, proposal_status, fallback_used,
  shared_process_contract, training, validation, replay, usage}]' "$ROOT/summary.json"
```

Inspect process admission, enforced uses, fallback reasons and actual equations
before attributing NMSE changes to shared-process modeling. Completion, independent
solver agreement, predictive fit and physical mechanism compliance are separate.
No test data are opened.

## Local checks

```bash
python -m pytest -q tests/test_shared_process_contract.py tests/test_detention_bound_pilot.py
python -m scripts.smoke_shared_process_contract --output /tmp/shared-process-smoke
```

The smoke constructs separate temporary synthetic data, uses prescribed replies,
fits with the frozen fitter, checks independent replay, and verifies deterministic
resume without new calls or optimization. It tests implementation, not live-model
scientific capability.
