# Scientific judge protocols

## Scope

The comparative protocol is currently calibration-only. It does not alter the
production search controller, its historical scientific scores, or candidate
selection. Both candidates are deterministically valid and are shown without
fit metrics, trajectories, hidden equations, or mutation labels.

## Original scientific judge v2 rubric

The current production judge independently scores one candidate in six broad
categories. The runtime computes the weighted average; the LLM does not emit
the aggregate.

| Category | Original question | Weight |
|---|---|---:|
| Mechanistic coherence | Do the states, processes, equations, and mechanism tags form a coherent scientific explanation of the task? | 0.20 |
| Source-sink and balance semantics | Are production, input, utilization, elimination, transport, and outflow roles given scientifically consistent signs and balance relationships? | 0.20 |
| Dynamic plausibility | Are accumulation, decay, delay, saturation, feedback, boundary behavior, and stability scientifically plausible for the roles claimed by the candidate? | 0.20 |
| Mechanism coupling and task sufficiency | Do task-critical mechanisms actually affect the states or outputs they purport to explain, with enough coupling to support the stated scientific objective? | 0.20 |
| Nonredundancy and accounting | Are mechanisms free of duplicated fluxes, double counting, disconnected copies, or conflicting representations of the same scientific role? | 0.10 |
| Latent-state and complexity justification | Does every latent state and additional mechanism have a necessary, interpretable scientific role rather than merely increasing flexibility? | 0.10 |

The full source prompt is `_COMMON_JUDGE_PROMPT` in
`src/autoformalism/benchmarks/phase_b_public.py`.

## Calibration comparative protocol

The judge sees Candidate A and Candidate B in both orders. It answers every
question independently with `candidate_a`, `candidate_b`, `tie`,
`indeterminate`, or `not_applicable`, and must cite equation- or
dependency-level evidence.

1. **Claimed mechanisms represented:** Which candidate represents more claimed
   mechanisms with an identifiable equation term, process, or state transition?
2. **Task inputs connected to targets:** Which candidate connects more
   task-critical supplied inputs to targets through directed dependencies?
3. **Claimed processes connected to balances:** Which candidate connects more
   claimed processes to the balance or output they purport to affect?
4. **Source-term signs:** Which candidate places more claimed sources,
   production terms, or inflows with signs consistent with those roles?
5. **Sink-term signs:** Which candidate places more claimed sinks, utilization
   terms, elimination terms, or outflows with signs consistent with those roles?
6. **Flux duplication:** Which candidate repeats fewer identifiable physical
   fluxes in one balance or through algebraically duplicated pathways?
7. **Disconnected components:** Which candidate contains fewer components with
   no directed path to a requested target?
8. **Conflicting mechanisms:** Which candidate contains fewer incompatible
   components claiming to represent the same mechanism?
9. **Latent-state incoming pathways:** Which candidate gives more latent states
   an incoming driver from another state, input, or process?
10. **Latent-state outgoing influence:** Which candidate gives more latent
    states a directed influence on a target or target-driving process?
11. **Latent-accumulator relaxation:** Which candidate has fewer latent
    accumulators lacking both relaxation and a task-based reason for one-sided
    accumulation?
12. **Decay direction:** Which candidate has more claimed decay or removal terms
    whose sign opposes the accumulated quantity?
13. **Delay structure:** Which candidate gives more claimed delay states both a
    driving pathway and a relaxation or outflow pathway?
14. **Saturation structure:** Which candidate gives more claimed saturation
    terms a structurally bounded response in the variable said to saturate?

For each determined answer, an A preference is encoded as 1, a B preference as
0, and a tie as 0.5. The runtime averages these values. `indeterminate` and
`not_applicable` answers are excluded and reported separately. The judge never
emits a numeric score or an overall winner. The checklist intentionally omits
response timing, fitted timescales, realized stability, trajectory fit, and a
holistic coherence vote because those are not objective structure-only items.

The executable prompt is `ATOMIC_COMPARATIVE_PROMPT` in
`scripts/run_comparative_judge.py`.

## Calibration measurements

The experiment reports per-call and repetition-aggregated preference accuracy,
false preference and tie rates, atomic indeterminacy, A/B order consistency,
repeat ICC and standard deviation, and question- and mutation-specific results.
It also reports mutation-by-question cross-tabs so irrelevant checklist items do
not obscure the detection rate of the question targeted by each mutation.
Shard execution is protected by an exclusive file lock. The merger fails on
conflicting duplicate outcomes by default and supports an explicit, audited
first-record policy for recovery of already duplicated append-only artifacts.
The protocol should remain outside search until it demonstrates adequate
accuracy, order invariance, and repeatability on blinded calibration pairs.

## Candidate paper-appendix evidence

The first 20B structure-only study on 20 frozen baseline/mutation pairs produced
0.785 per-comparison preference accuracy, 0.035 false preference, 1.000 accuracy
after aggregation across five repetitions and both candidate orders, and 0.039
mean repeated-call standard deviation. The preceding independent continuous-score
protocol produced 0.530 paired accuracy, 0.440 false preference, and 0.248 mean
repeated-call standard deviation on the same frozen pair design. These results
are promising evidence for pairwise atomic judging, but remain provisional until
the duplicate-execution correction, complete-case reliability analysis, order
sensitivity, per-mutation results, and a same-protocol 120B comparison are all
reported together. In particular, strict A/B order consistency was only 0.690,
so the appendix must describe order reversal and aggregation as part of the
measurement protocol rather than presenting a single-order score.
