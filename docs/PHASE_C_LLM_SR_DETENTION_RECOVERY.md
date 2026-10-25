# LLM-SR detention: derivative and diagnostic recovery

Date: 2026-10-25. This is a separate, opt-in numerical-control experiment.

## Returned evidence

The first detention refit campaign recorded ten tasks. All six tasks that
entered optimization reached training NMSE below 0.001 and improved on their
model's supplied-coefficient continuous-time reference. Three of four generic
starts entered search. The four other tasks made zero optimizer evaluations:
derivative qualification prevented search, and the saved initial points were
retained. These are not demonstrated local-minimum failures.

Model 14 improved from 0.0001282615 to 0.0000084742 when initialized at the
supplied coefficients. Its generic starts reached 0.0001008075 and
0.0000084830. Model 15 improved from 0.0002329362 to 0.0000735848; its successful
generic start reached 0.0000681784. All six searches exhausted their time
allowance. All ten final diagnoses exhausted their 120-second allowance.
Independent Radau/DOP853 agreement supports the reported rollout errors, not
convergence or parameter uniqueness. No held-out performance was measured.

Three blocked starts had numerical derivative disagreements. One had zero
sampled columns but insufficient absolute-gradient precision under the search
gate. Tight retries had only 0.6--30 seconds left after the initial check,
insufficient to rebuild a complete Jacobian. Qualification and final diagnostic
work together consumed approximately 58% of measured operation wall time.

## One bounded follow-up

Keep all equations, original domains, initial-observation rules and training
events unchanged. Import all ten historical outcomes and their operation/cost
records. Use the following fixed roster:

| New task | Historical refit index | Model/start | Work |
|---|---:|---|---|
| 0 | 4 | 14 / generic 1 | Diagnose saved endpoint only |
| 1 | 8 | 15 / generic 0 | Diagnose saved endpoint only |
| 2 | 1 | 14 / perturbed 0 | Recover blocked original start |
| 3 | 6 | 15 / perturbed 0 | Recover blocked original start |
| 4 | 7 | 15 / perturbed 1 | Recover blocked original start |
| 5 | 9 | 15 / generic 1 | Recover blocked original start |

The two endpoints are representative of the previously successful generic
searches. This is a selected diagnostic follow-up, not a new estimate of
initialization success probability. A blocked start never inherits another
task's fitted coefficients. Supplied-coefficient scores remain descriptive
references only.

## Changes to numerical qualification

Each coordinate has its own identity-bound, nonrenewable operation record and
120-second allowance. A coordinate check computes only its required Jacobian
column and distinct shrinking local probes. It does not rebuild every other
column. Completed operations are reused on resume. Interrupted attempts retain
their progress and charged allowance; they do not acquire a fresh budget.

Search first checks every coordinate at standard precision. If that aggregate
does not qualify, a separate complete tight-precision pass receives its own
coordinate allowances. Do not combine columns from different endpoints,
equations, data, units, policies or integration precision. The tight pass does
not have to finish inside the small remainder of a previous pass.

The search gate permits an empirical coordinate-gradient error no larger than
the maximum of `1e-7` and one percent of that coordinate's gradient magnitude,
provided the local probe audit has not identified a failed direction. The final
diagnostic retains its stricter precision requirements. These are predeclared
numerical search settings, not confidence probabilities or a relaxation of the
model's scientific requirements.

Aggregate qualification results are also saved and authenticated. Reports
preserve the saved numerical summaries. Cross-platform checks validate raw
evidence and numerical decompositions without requiring singular vectors to
have identical signs or bits on different linear-algebra implementations.

Search qualification asks whether derivatives are sufficiently reliable to
guide useful descent. Final diagnostic qualification asks the stronger question
of whether numerical uncertainty is small enough to interpret local gradients
and parameter sensitivity. Neither uses a large nonlocal perturbation as a
derivative certificate. Genuine discrepancies remain blockers; search
eligibility alone never establishes convergence or coefficient confidence.

Each recovery search has a fresh, explicitly reported 600-second allowance,
separate from qualification. Preserve improving base-point checkpoints, verify
the original equations independently, and retain only an independently verified
endpoint from that same task. Diagnose the exact retained endpoint using a
complete tight-precision coordinate pass. Report all attempts and unresolved
directions, including failure and timeout costs.

The Delta run requests one CPU and 8 GB per array task, with two concurrent
tasks and a 90-minute scheduler ceiling. The ceiling covers the worst-case
bounded sequence; it is not a requirement to spend 90 minutes. No GPUs, LLM
calls, validation access, test access, or production-default changes are involved.

## Exact Model 15 reduction

The original parameters contain an algebraic ambiguity. Let

```
T = dt; g = k_gw; l = k_loss; k = kappa; H = h_max
b = g-l; d = g*k; M = H+T*g
```

The vector field depends on these four original parameters only through
`b,d,M`, with `T` separately retained. An eight-coordinate representation is
`C,beta,eps_s,Q_max,T,b,d,M`. A feasible original representative must satisfy
all original bounds; fixing `g` arbitrarily can shrink the model family.

For a proposed reduced point, its feasible values of `g` are the intersection
of its original interval, `[b+l_lower,b+l_upper]`,
`[(M-H_upper)/T,(M-H_lower)/T]`, and the interval implied by `d=g*k` and the
original bounds on `k`. Signed `g`, zero products and interval boundaries require
explicit handling. From any feasible witness, lift with
`l=g-b`, `H=M-T*g`, and `k=d/g` when `g` is nonzero. A zero witness requires
`d=0` and a separately feasible `k`.

This milestone implements and tests that exact feasibility/lift certificate and
records the effective combinations in diagnostics. The recovery searches retain
the original coordinates so that this experiment isolates diagnostic recovery.
A reduced-coordinate optimization comparison is a later milestone: its feasible
domain is correlated and cannot be replaced by an independent eight-dimensional
box. No individual-parameter confidence is inferred from the reduction.

## Audit and interpretation

All prior saved plan identities, receipt hashes and endpoint bindings remain
exact. Cross-platform regeneration of three original generic-start values
differed by one floating-point unit in the last place. Historical import permits
only a tightly bounded discrepancy in regenerated start values, retaining the
exact originally frozen values for execution and receipt verification. This
exception never applies to hashes, equations, policies, task rosters or costs.

Report separately: independently verified training accuracy; whether search
started; stop reason; derivative reliability; local gradient evidence; local
parameter information; exact ambiguity; and computational overhead. Budget
exhaustion is not convergence. Sampled sensitivity is not global identifiability.
Low predictive error is not scientific mechanism compliance. In particular,
the signed gain/loss terms of these LLM-SR controls are not silently repaired.

## Delta commands

Use the delivered source/input bundle in a fresh directory. The bundle contains
the historical records; the old remote experiment directory is not required.

```bash
export AF_LLMSR_DETENTION_RECOVERY_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-recovery-v1
bash scripts/hpc/start_llm_sr_detention_recovery_delta.sh run
```

Reuse the same extracted bundle and output root to inspect progress or package
the results:

```bash
bash scripts/hpc/start_llm_sr_detention_recovery_delta.sh status
bash scripts/hpc/start_llm_sr_detention_recovery_delta.sh inspect
```

The inspection archive is written beside the campaign root. Download it from
your workstation terminal, not from a shell already running on Delta.

## Local checks before submission

Six selected coordinate checks on the actual historical vectors completed in
12--58 seconds each, without fitting or held-out access. The tight `C` check at
the blocked Model 15 perturbed-0 start passed. The previously blocked zero
`h_max` direction at perturbed-1 is usable for search but remains insufficient
for strict endpoint precision. Representative directions at the two good saved
endpoints completed; their nonzero gradients do not establish convergence.
Tighter settings did not improve every direction: Model 14's perturbed-0
`h_max` tight check remains unqualified for search. The Delta experiment must
still complete all coordinate checks; these selected checks are not an overall
search or diagnostic certificate.
