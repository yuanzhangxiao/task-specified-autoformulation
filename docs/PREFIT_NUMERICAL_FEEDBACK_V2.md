# Routed numerical hypotheses and citation correction

The opt-in `routed-hypothesis-2` policy fixes citation handling and changes the
purpose stated to the proposer. `configs/prefit_numerical_sibling_v2.json` selects
it; the default `optional-review-1` policy retains the original prompt and strict
citation behavior. The original ACES run remains an exhausted three-call episode.

## What the feedback should say

> This visit is allocated to testing a structural hypothesis. Use the measured
> training mismatches to propose one concrete function revision. Explain the
> mismatch, why the change could address it, and which observable pattern should
> improve after refitting. The budget limit explains why optimization stopped;
> it does not explain why the error is large. Neither structural adequacy nor
> structural failure has been established. You need a testable hypothesis, not
> proof that the current model cannot fit.

This is an explicit experiment allocation, not a newly implemented general
controller rule. The original fitting status remains in the packet, with its
correct numerical scope. An unfinished fit is not presented as a reason to
decline structural exploration. `topology_revision_needed` remains available
when one existing interaction cannot express the hypothesis. `no_change` remains
available when the proposer cannot articulate a supported hypothesis, with a
request to identify missing evidence; budget exhaustion alone is not a sufficient
explanation. These are prompt instructions, not an automated scientific judge.

The new instructions request a measured observation, a conditional mechanism
hypothesis, and a predicted improvement. They do not prescribe a particular
function family or use a ground-truth answer. The unchanged residual packet
contains train-only rows, windows, detailed samples, previous-fit comparisons and
retained parameter estimates. The proposer selects evidence from that packet.
Interactive retrieval and new residual feature selection are later work.

## Citation behavior

- Exact duplicate references are removed in first-occurrence order. The raw
  cached response is preserved, and `decision.citation_normalizations` records
  the before/after lists, including for `no_change` decisions.
- `evidence_catalog` lists actual row, window and sample-detail IDs. The prompt
  asks for preferably one to three relevant references. The bounded reply schema
  still permits at most sixteen raw references.
- Missing references remain errors. Feedback gives the exact absent IDs and the
  available IDs, explains that `_samples` cannot be appended to arbitrary rows,
  and records that the equation has not yet been evaluated. Nothing silently
  replaces or drops an absent reference.
- Source/sign checks, atomic preservation, initializer contracts, requirement
  checks, request/token budgets and provider-failure handling remain in effect.

The `audit-citations` CLI reads sealed historical state and its matching packet
without running the historical campaign under changed source. It compares reply
and citation admissibility only; it does not accept a model or rewrite a result.

## ACES commands

Replace `PUBLISHED_COMMIT` with the exact commit supplied for this milestone. Each
command is a single physical line. The existing numerical sibling is untouched.

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-public-fit-convergence-v1 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-public-fit-convergence-v1 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2 PUBLISHED_COMMIT
```

First inspect the saved v1 replies with the corrected citation rules (no LLM or
fitting; prints JSON without altering the old directory):

```bash
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2/scripts/prefit_numerical_sibling.py audit-citations --root /scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v1
```

Then submit the separate v2 experiment:

```bash
AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v2 AF_CONFIG=/scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2/configs/prefit_numerical_sibling_v2.json bash /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2/scripts/hpc/submit_prefit_numerical_sibling_aces.sh
```

This retains the first-continuation parent and the frozen child fitting profile.
The explicit policy version permits one separately recorded allocation; changing
only the output directory cannot renew that allocation. Preflight runs the
focused tests and a synthetic smoke for the configured feedback policy. The
chain remains CPU preparation/replay, H100 proposal, CPU child fitting/report.

```bash
sacct -j "$(jq -r '[.prepare_job,.propose_job,.fit_job]|join(",")' /scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v2/submission_manifest.json)" --format=JobID,JobName,State,ExitCode,Elapsed
```

```bash
jq '{status,stop_reason,decision:.decision.reply,citation_normalizations:.decision.citation_normalizations,provenance:.decision.provenance,attempts:[.attempts[]|{attempt,accepted,feedback}]}' /scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v2/results/state.json
```

```bash
jq '{status,feedback_policy,proposal_outcome,physical_requests,observed_tokens,parent_training:.parent.training.normalized_mse,parent_validation:.parent.validation.normalized_mse,child_status:.child.result.status,child_training:.child.result.training.normalized_mse,child_validation:.child.result.validation.normalized_mse,child_budget_exhausted:.child.result.budget_exhausted}' /scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v2/summary.json
```

## Interpretation

This pilot changes both citation handling and task framing. It is not an isolated
prompt ablation or an old-versus-new controller comparison. An admissible revision
is a hypothesis, not a scientific conclusion; its child and parent receive
different numerical budgets. A topology request is a useful scope finding, but
topology editing is still a later milestone. No hidden reference, test data or
validation evidence enters the proposer packet.

The separate user-requested retrospective is documented in
`PREFIT_FEEDBACK_RETROSPECTIVE_2026-09-15.md`. It is not imported by the runtime,
included in the proposer prompt, or used as an acceptance criterion.
