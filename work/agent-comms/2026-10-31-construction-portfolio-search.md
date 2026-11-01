# Five-model numerical follow-up: implementation handoff

The authorized next milestone is implemented as an opt-in CPU campaign. Read
`docs/PHASE_C_CONSTRUCTION_PORTFOLIO_SEARCH.md` for the numerical scope, equations,
limitations and commands. No construction model or public benchmark was changed.

- **03:** tighten integration for its four unresolved local derivative checks;
  keep the perturbation policy unchanged. This partial diagnostic cannot grant
  full search qualification by combining old and new precision settings.
- **10/14:** import the complete saved BDF qualification, verify the start, then
  perform bounded shared-parameter/initial-root fitting using that same BDF
  objective. Independently replay endpoints with original Radau and retain only
  verified training improvements.
- **12/15:** retain the models and investigate equivalent self-relaxation/gain
  coordinates. Each timescale profile includes an unchanged-nuisance control,
  bounded nuisance refitting, and independent original-equation verification.
  Profile endpoints are diagnostic and never select the main incumbent.
- **Endpoint reporting:** independently verify and diagnose both the retained
  best and last accepted point. Accepted iteration history is used only for an
  exactly matching point in the original search chart.

The source is the reviewed continuation plan
`e92e3bf12f64431d6db74c812f849396747ed98bcb0a30ac02a60a47b3491ea1`.
Full source operations, progress and nested probes authenticate once at import.
The exported subset is pinned by original operation hashes and reviewed compact
projection hashes. It occupies 35,460,926 bytes, versus roughly 690 MB when all
old arrays and progress copies were included. No original evidence is deleted.

Local validation:

- 197 affected tests passed, followed by two additional start-gating/resume tests.
- Changed Python files pass Ruff; repository-wide Ruff still reports the same
  37 unrelated findings under `analysis/claude`.
- Models 10/14 recreate their complete 29-coordinate qualifications from the
  compact snapshots with zero new numerical solves.
- Models 12/15 preserve original predictions under the timescale chart within
  approximately `1e-15`; independent original ODE discrepancies are below `9e-10`.
- Synthetic tests cover interruption, failure gates, immutable source retention,
  receipt-only reporting, matching endpoint histories and copied Slurm workers.

The new Delta root is
`/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-search-v1`.
Five array tasks request one CPU/8 GB/four hours each, with maximum concurrency
two. The largest inclusive reservation is 8,820 seconds. The report has a
30-minute wall limit. There are no GPUs, LLM calls, or validation/test access.

No new real-model fitting has been run locally or remotely for this milestone.
No new prefix forecasts are claimed: source prefix metrics are explicitly
labelled, and any new fitted endpoint requires separate causal prefix replay.
Timescale profiles are finite-budget upper envelopes, not certified profile
minima or confidence intervals. Poor profile fits cannot exclude a timescale.
