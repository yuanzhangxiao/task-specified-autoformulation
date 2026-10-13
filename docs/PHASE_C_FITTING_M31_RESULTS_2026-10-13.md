# M31 results: controlled diagnoses pass; diagnostic qualification remains incomplete

Reviewed `review-20261013-213107.tar.gz` on October 13, 2026.
All sixteen tasks are recorded, all five synthetic controls give the intended
diagnoses, and all feedback packets preserve the specification boundary. The
frozen qualification nevertheless **fails**: the new alien-device diagnostic
exhausts its 180-second allowance. This is not a new fitting failure. Its
historically verified accurate incumbent remains available.

M31 changes the fitting-to-construction evidence interface, not the optimizer.
It replays eleven archived M30 entries without refitting and fits five small
synthetic controls. No new accurate constructed model or live construction
improvement is established by this experiment.

## Provenance and verification

- Protocol: `phase-c-fitting-feedback-qualification-1`; [runbook](PHASE_C_FITTING_FEEDBACK.md).
- Commit: `460242ea7ddc568223cc89bea224db6683a9cd1a`.
- Source SHA256: `642a457186e0c4692319da55d96e5806eaaaf5e1f80456fd41dc661844fc016d`.
- Plan SHA256: `c9009304371d69f1bea8f71c33e7d6651276a839ad828d4ddf7e4b55251fe556`.
- Input SHA256: `524d69db0e476863d4384b14b929f75f67889dcfecc66284ccbe58eae7a6f18a`.
- Archive SHA256: `dd6cd10f84a3708432461d28d9f36bc2066123ec5d264a2939f8e32361b14ed5`.
- Delta preparation/array/report jobs: `22874078` / `22874079` / `22874080`.
- Output: `/work/hdd/bibo/yxiao2/phase_c/fitting-feedback-v1`.
- Runtime: Python 3.12.13, NumPy 2.3.5, SciPy 1.16.3, CasADi 3.7.2,
  Pandas 2.3.3, Pydantic 2.12.5. One CPU per task, three concurrent, no GPU or LLM calls.

All 213 sealed JSON artifacts validate. Rebuilding the report reproduces the
supplied summary exactly. All fourteen emitted packets match their training
problem, original fit/backend and retained parameter vector. Two original
construction failures stay in the denominator with no fabricated feedback.
One archived candidate has no verified retained endpoint; its diagnostic is
explicitly skipped and the packet says `rollout_unverified`.

The other thirteen packets request new diagnostics: twelve complete, one times
out. `status=complete` means every task is accounted for;
`qualification_status=failed` accurately reports the unmet diagnostic gate.
Training-only feedback never uses validation/test observations or oracle labels.
Synthetic evaluator labels remain separate from the packets.
The 36 milestone regression tests pass again during review, as does scoped Ruff.
Repository-wide Ruff retains the same 37 unrelated findings in `analysis/claude/`.
No fitting or numerical replay was rerun on the downloaded archived models.

## Synthetic controls

The following NMSEs are training-only fixed-vector checks. They are noiseless
qualification examples, not five independent benchmark successes.

| Control | Training NMSE | Result |
|---|---:|---|
| Adequate drainage, `h'=g*u-k*h` | 2.73e-24 | Accurate; joint gain/rate Jacobian rank 2/2. |
| Candidate omits drainage, `h'=g*u` | 6.60084 | Restricted output problem solved; review these equations. |
| Parallel inflow gains, `h'=(g1+g2)*u-0.8*h` | 1.82e-15 | Accurate but ambiguous; rank 1/2. |
| Adequate nonlinear drainage, eight-call search | 1.01e-4 | Search incomplete; no impossibility claim. |
| Explicit synthetic no-drainage claim | 6.60084 | Same fitting diagnosis as omitted drainage; claim verdict withheld. |

The ambiguity direction is approximately `(-1, +1) / sqrt(2)`: exchanging the
two gains leaves their sum, and thus every predicted trajectory, unchanged.
This demonstrates why tiny prediction error cannot establish unique coefficients.

The omitted-drainage and false-claim controls deliberately give the fitter the
same candidate and observations. Their numerical packets therefore cannot
distinguish construction failure from a false specification. A separate
evaluator-only certificate establishes that the explicitly defined family
`h'=g*u`, `g>=0`, `u>=0` cannot produce decreasing measured depth. That certificate
does not come from fitting failure. Every packet retains
`specification_verdict=not_assessed` and disallows automatic specification revision.

## Archived models and conservative uncertainty

Five archived constructions have verified output diagnostics. Four remain
`search_incomplete`; independent basin remains
`solved_restricted_problem_poor_fit` with training NMSE 2.63299. Its exact
within-candidate optimum does not condemn the original specification. The
recorded local-stationarity witnesses from other search branches are not
misattached to the retained point. No archived packet is labeled
`stationary_poor_fit` or `search_improving` in this run.
The five controls do not provide positive end-to-end qualification of those two
routes; unit coverage alone is not experimental coverage. Some rejected witnesses
are numerically very close to the incumbent, but a matching error or nearby
coordinates cannot transfer a certificate. Recheck at the retained point instead.

| Known-structure control | Training NMSE | New diagnostic | Why parameter information remains unresolved |
|---|---:|---|---|
| Alien-device hard | 8.36e-15, historical | Unavailable after 180 s | New two-solver check did not finish. |
| Coupled basin | 1.56e-16 | Complete; Jacobian rank 2/2 | Piecewise derivative convention needs qualification. |
| CSTR hard | 3.39e-13 | Complete; Jacobian rank 9/9 | Piecewise derivative convention needs qualification. |

Basin and CSTR reproduce their saved accuracy, pass solver agreement, and have
scaled Jacobian condition numbers about 7.63 and 584.71. Their packets say
`accurate_unresolved`, because M31 intentionally declines an unqualified local
parameter conclusion for any model flagged as piecewise. This is not evidence
that those parameters are unidentifiable. The descriptive ranks are retained,
with their derivative scope explicitly marked.
The reachable CSTR equations include `max(T,250)` and `max(C,0)`; basin transfer
and outlet equations include piecewise-linear positive-part terms. The archive
does not record branch margins or crossing events, so it does not establish
whether these particular trajectories crossed a kink.

The alien diagnostic records `calls=2`. The implementation marks call two
immediately before the Radau output evaluation, following DOP853. It therefore
times out during the second output evaluation, before the third, Jacobian call.
The shared error text says "sensitivity rollout wall-clock limit reached" even
though this call requests outputs only. The tighter new replay uses relative and
absolute tolerances `1e-12` and `1e-14`. Its operation returns no diagnostic value;
completed first-solver arrays are not durably retained. The historical fit stays
accurate, while new sensitivity information stays unavailable.
The previous alien output verification used the default `1e-10`/`1e-12` tolerances
and took 96.31 seconds. Its artifact records `tight_verification=false`; those
numeric tolerances come from the pinned implementation's defaults rather than
an explicit field in that record. The more stringent replay is additional work,
not evidence of regression in the fitted equations.

Two T1 diagnostics complete and reproduce their poor scores, but do not clear
the stricter numerical gate for sensitivity. The distinction between output
agreement and confidence in derivatives must also remain visible in the next
iteration.

## Additional computation

These are sums of recorded task durations, not queue time, campaign elapsed time
or billed service units. Historical fitting costs are excluded.

- New diagnostics: **372.50 seconds**, 357.07 CPU seconds, 36 started evaluation
  calls, zero optimization calls. Accounting is complete, including the timeout.
- Alien-device alone uses 180.00 seconds, about 48% of diagnostic duration.
- Seven completed archived diagnostics use 185.00 seconds in total; their median
  is 32.39 seconds and range is 2.05–60.42 seconds.
- Five synthetic diagnostics use 7.50 seconds in total.
- Fitting those synthetic controls uses a separate 21.14 seconds, with 23 search
  calls started and 22 completed. All work is charged.

The combined new recorded duration is 393.64 seconds (6.56 minutes). The successful
small controls show useful discrimination at low cost, but this single run does
not establish a general diagnostic speedup or calibrated statistical confidence.

## Recommended next milestone

Keep the qualification failure visible; do not loosen it retrospectively or rerun
the successful fits. Focus the next milestone on diagnostic reliability:

1. Separate and checkpoint primary output replay, alternate output replay, and
   joint sensitivity. Preserve completed stages when a later stage exhausts its
   allowance. Attach stage names, solver/tolerance identities, timing and started/
   completed counts to durable results. Resume must not silently reset budgets.
2. Compare a bounded diagnostic escalation/reuse policy on the retained vectors,
   including alien-device. Reuse prior verification only when its candidate,
   parameters, data, solver contract and required numerical accuracy match.
   A looser historical check cannot stand in for a stricter requested check.
3. Add a genuine retained-point stationary control and explicit reason fields
   distinguishing a diagnostic timeout, insufficient numerical precision and a
   piecewise derivative restriction. Qualify the route without loosening exact
   witness binding or claiming a global optimum.

After diagnostic completion, qualify piecewise sensitivities locally with branch
information and independent output perturbation checks across step sizes.
Distinguish supported local derivatives from switching/weak/unavailable evidence.
Neither accepting every piecewise Jacobian nor permanently discarding all of
them is satisfactory.

Use the same saved models and five controls; report incremental cost as well as
diagnostic coverage. Keep fitting parameters fixed and no oracle values in the
diagnostic decision. Live construction integration awaits executable equations;
specification revision remains outside this milestone.
