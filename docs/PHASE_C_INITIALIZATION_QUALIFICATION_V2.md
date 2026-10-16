# Initialization runtime qualification v2

`saved-topology-initialization-2` repeats all 15 original pre-initialization
starting drafts, sampling seeds and maximum allowances from v1. It does not
continue from the new final drafts, select the best models, fit functions or open
test data. Initializer readiness and scientific correctness remain separate.

## Changes

- **Feedback:** exact repeated structural/checklist fields appear once in a
  shared-detail table. Every finding, status, public requirement, training
  witness and current equation view remains available. Expanding the table
  recovers the corrected original packet exactly. Consumer advice checks actual
  uses: an empty consumer list cannot establish an existing contribution.
- **Admission:** reservation is verified serving input tokens + maximum output
  tokens + the existing 512-token safety margin, instead of serialized request
  bytes + maximum output. Request identity includes the policy. Actual provider
  usage replaces reservations; missing usage retains the reservation. Total
  ceilings and the 24,000-token input limit are unchanged.
- **Witness history:** retain and recheck every distinct witnessed question in
  the baseline or subsequent episode. A target/property key groups obligations
  without discarding later sample pairs. Histories absent from the frozen input
  cannot be reconstructed by this policy.
- **Measured initials:** an explicit reviewed identity between a declared state
  and a public initial measurement fixes its initialization. The typed contract
  stores the public excerpt and public-context/state-declaration hashes. Runtime
  does not infer relationships from names, trajectories or ODE arguments.
- **Rejected transactions:** separate fixed initials from editable states.
  Feedback identifies valid intended editable records in a rejected reply and
  asks the proposer to retain them when resubmitting. No partial commit occurs.

Same-quantity/same-units bindings lower to exact identity initializers in
interaction construction, without fitted coefficients or extra proposer calls.
Different coordinates/units require a reviewed transformation. Removing,
renaming, retyping or changing an attested state's description fails closed and
requires contract review. These interpretations are explicit conditions of this
narrow confirmation, not restrictions on all future coordinate choices.

For this study, both saved coupled-basin drafts explicitly use upstream depth as
the internal state. The public preparation describes `initial_up` as measured
upstream depth at the initial instant. The separately delivered reviewed records
bind `h_up(t0) = initial_up`. Benchmark prompts, data and hidden equations are
unchanged. Other latent initialization choices remain proposer-owned.

## Qualification and interpretation

V1 completed 15 episodes: 13 complete initializer inventories, 13 eligible
structures and 11 usable interaction handoffs. Four `checks_clear` labels mean no
remaining tracked repair obligation, not scientific accuracy. The audit found
two context stops, two overly conservative budget stops and nine intermediate
witness coordinates missing from final receipts. Rechecking all available
baseline/current-episode witnesses left all 15 final statuses unchanged.

Tests cover exact replay, rejected sibling edits, late states, stale public/state
attestations, exact measured identities (including aliases), tokenizer failure,
budget limits and unknown usage. A synthetic counterexample exercises a lost
early witness that incorrectly clears under the historical policy. On all 81
recorded calls, serving-tokenizer counts matched provider prompt-token usage.
This is specific to the pinned server; the safety margin is not a universal bound
for other serving implementations.

Report first-response, initializer-stage exit and final completeness; structural
eligibility; conditional conflicts and deferred checks; equations and initial
rules; context/budget stops; physical requests and median [unscaled MAD] token
cost. Compare matched starts and maximum budgets while reporting actual unequal
costs. Absence of a conditional conflict is not a fitted or mechanistic pass.

## Prepare and run

```bash
export PYTHONPATH=.:src
.venv/bin/python scripts/saved_topology_repair.py refine-initialization \
  --source /path/to/original-initialization-plan \
  --root /path/to/new-inputs \
  --measured-bindings /path/to/reviewed-measured-bindings.json
.venv/bin/python scripts/saved_topology_repair.py preview \
  --root /path/to/new-inputs --output /path/to/prompt-review
```

The Jetstream bundle contains sealed inputs and compatibility requests. With the
verified local server running, set `AF_JS2_CAMPAIGN_ROOT` to
`$HOME/af-construction/initialization-topology-v2` and run
`scripts/jetstream/run_saved_topology_repair.sh run`. Use `status` while running
and `inspect` after it stops. Repeating `run` resumes the same finite allowance.
The cap remains ten calls and 262,144 charged tokens per episode, including up to
three dedicated initializer calls. No automatic follow-up is scheduled.

Historical policies retain original payloads, schemas and admission rules; use
their pinned source to reproduce sealed campaigns. Improved live completion or
scientific repair has not yet been demonstrated for v2.
