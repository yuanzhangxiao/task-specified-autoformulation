# M38: discrepancy extraction and conditional feedback drafts

Implemented 15 October 2026, following the
[M37 verified interaction packet](PHASE_C_FITTING_INTERACTION_ADAPTER.md).
This increment translates saved numerical evidence into measured discrepancies,
conditional investigations and a readable draft. It performs no new fitting,
integration, LLM calls or model edits. The benchmark and production fitter remain
unchanged. Live construction integration is still pending.

## What the feedback says

1. **Numerical reliability and search status.** Can the retained rollout be
   trusted? Is fitting still improving, incomplete, locally stationary or solved
   within an explicitly restricted candidate/domain? There is no universal
   `fitting_done` certificate.
2. **Measured discrepancies.** Every target retains its complete training NMSE
   and signed-bias overview. Selected trajectories add response magnitude,
   direction, peak timing, recovery and residual-window comparisons when defined.
3. **Conditional investigations and alternatives.** Describe which existing
   response dependence could be inspected, alternative parameter/initialization
   explanations, and the additional evidence needed to distinguish them.
4. **A testable prediction.** State what observable discrepancy should decrease
   after a proposed revision and refit, what must remain accurate, and what result
   would fail to support the hypothesis. Aggregate NMSE improvement alone does
   not establish the proposed mechanism.
5. **Stage and preservation requirements.** Existing function forms are an
   interaction-stage question. New states or initialization rules require broader
   variable-stage review; new dependencies or shared-process consumers require
   topology review. Preserve all targets, public obligations, source/sign domains,
   shared-law consumers and the incumbent on unsuccessful revisions.

The observations are training readings, not private hidden-state truth. The
adapter does not identify sensor errors or change the measurement contract.
Equation references identify possible places to inspect, not a proven faulty
term. Authoritative per-interaction input sets, signs and obligations must still
be bound from the constructor before an edit can be applied.

## Numerical summaries and their limits

Let `r = prediction - observation` and let `s_j` be the common training scale for
target `j`, carried by the verified source. The normalized signed bias is
`mean(r / s_j)`. For comparable nonflat, same-direction excursions `a`, the
magnitude discrepancy is `(abs(a_pred) - abs(a_obs)) / s_j`; each excursion is a
departure from that trajectory's own initial level. Missing/spurious excursions
and opposite directions receive separate descriptions.

Peak-time differences require comparable unique interior peaks. Recovery uses
the first sampled half-return after that peak; it is not a fitted exponential
time constant. Flat, tied, boundary or opposite-direction peaks and censored
returns remain explicitly unavailable. Their differences are never fabricated
as zero or infinity. Time is reported in the source trajectory coordinate,
without guessing a physical unit.

Early/middle/late window NMSE uses every sampled residual in the corresponding
window. These are time thirds, not input-event-aligned causal segments. Up to two
selected examples per target come from M37; they do not establish consistency
across all trajectories or a controlled dose/history comparison. Coverage and
omitted-example/window counts remain explicit. Multiple inputs and repeated
events retain their ambiguity.

Default display thresholds are 0.05 training standard deviations, 0.05 of the
trajectory duration for timing differences, and 0.01 window NMSE. They are
versioned presentation filters, not significance tests, confidence levels or
new fit-acceptance criteria. They do not change the original diagnostic verdict.

## Routing and conditional advice

| Verified numerical evidence | Draft behavior |
| --- | --- |
| Unavailable/disagreeing rollout | Withhold function advice; identify the missing numerical check |
| Accurate output | Retain; report remaining parameter ambiguity and check public obligations separately |
| Search improving or incomplete | Prefer continued/diversified fitting; allow provisional investigations if the controller independently allocates a function-review visit |
| Retained-point local stationarity with poor fit | Permit function investigations, while retaining a genuinely different fitting start as an alternative |
| Restricted output optimum with poor fit | Permit review of this candidate family and domain; do not infer that the public specification is false |

Stronger search claims require matching witness kind, scope and retained-vector
binding. A stale or incompatible witness cannot establish stationarity or an
optimum. A successful job, budget exhaustion or low constraint violation alone
does not establish any of these conclusions.

Investigations group the measured discrepancies into response strength,
timing/recovery, balance and time-window questions. They refer only to displayed
ODE/process laws affecting the target; shared consumers remain visible. The
default allowance is three investigation families per target with omission
counts. No reachable executable law means no invented function target.

For example, a weak excursion can motivate inspecting the existing response
dependence, but it does **not** establish saturation. Controlled amplitude
contrasts would help distinguish saturation from parameter or initial-value
errors. A timing error similarly does not establish missing memory. The adapter
asks for the missing evidence rather than supplying a reference equation.

## APIs and artifacts

- `fitting/interaction_mismatches.extract(packet, policy)` makes the descriptive
  summary from an already verified M37 packet.
- `fitting/interaction_advice.draft(packet, policy, source=...)` makes the pure
  deterministic advice projection; callers must verify the packet first.
- `export_advice(diagnostics, packet, output)` verifies the diagnostic source and
  M37 packet before publishing sealed `result.json`, sealed `request.json`, and
  the exact readable `feedback.md` view.
- `completed(diagnostics, packet, output)` reconstructs and verifies all three
  artifacts without rerunning the numerical model.

The request contains the complete M37 model/evidence packet plus the draft and
an explicit proposer response schema. The response separates observed mismatch,
conditional hypothesis, alternatives, predicted improvement, disconfirmation and
requested stage. No response is executed here. Validation/test scores, reference
parameters and evaluator expectations do not enter the request. A 96,000-byte
user-payload cap fails explicitly rather than trimming targets; it is not a
token count. Exports are locked and deterministic, reject mismatched existing
artifacts, and can recreate a missing Markdown view after interrupted export.

## Verification and cost

The 194 focused fitting, integrity, response, mismatch, advice and CLI tests pass.
They include stale witnesses, unavailable evidence, censored/ambiguous response
features, multiple targets, shared consumers, source tampering, incomplete export,
prompt caps and deterministic resume. All changed Python files pass Ruff.

Replayed all four saved M37 controls without new numerical or provider work:

| Control | Draft result |
| --- | --- |
| Initial-value path | Accurate output; retain, no error-driven investigation |
| Actual piecewise kink | Accurate output; retain the weaker parameter conclusion |
| Parameter-bound case | Accurate output; retain the domain-limited conclusion |
| Stationary poor retained fit | Six descriptive mismatch rows, two unavailable timing comparisons and three conditional investigations; alternative fitting starts remain possible |

These qualify the translation, not whether LLM revisions improve benchmark
models. Source bytes and modification times were unchanged. Repeated export and
verification preserved output bytes and modification times. Initial command wall
time was 2.67–3.37 seconds, including Python startup/imports; request files were
21.5–30.0 kB. These are local replay measurements, not cluster fitting costs.
The smoke record is `output/phase-c-interaction-advice-m38-smoke/cli-smoke.json`.

Repository-wide Ruff reports 37 existing issues in unrelated `analysis/claude`
scripts. Full `pytest -q` exceeded a 300-second local verification allowance and
was stopped before completion. Its partial log contains an early failure in
`test_assessed_recovery.py::test_submission_cpu_idempotence_uncertain_reply`;
both the isolated test and all eight tests in that file pass when rerun. The
partial log lacks the failure traceback, so its cause is unconfirmed. This is
not a certification that the whole concurrently edited repository passes.

## Commands

From the pinned checkout and its existing Python environment, use an existing
verified M37 packet and its diagnostic source. Substitute actual paths:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/phase_c_interaction_feedback.py draft \
  --diagnostics /path/to/verified-diagnostics \
  --packet /path/to/interaction-packet \
  --output /path/to/new-feedback-draft
python scripts/phase_c_interaction_feedback.py verify-draft \
  --diagnostics /path/to/verified-diagnostics \
  --packet /path/to/interaction-packet \
  --output /path/to/new-feedback-draft
```

No GPU or cluster allocation is needed for drafting already verified packets.
Use a new output directory for a changed policy/code identity. Repeating the
matching command verifies/resumes the saved artifacts. If no M37 packet exists,
create it first with the existing `export --diagnostics ... --output ...` command.

Next: bind a current executable construction's authoritative edit contract, then
test a frozen paired revision/refit experiment against unchanged continuation.
Keep all outcomes and separately account for fitting, diagnostics and LLM cost.
