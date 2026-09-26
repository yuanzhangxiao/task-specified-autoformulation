# CSTR benchmark: confirmed missed-input defect

Status: evidence consolidated on 2026-09-26. This note is for agents working on
benchmark integrity, model inspection and paper claims. It changes no code or
data. The companion [inspection summary](2026-09-26-model-inspection-findings-and-roadmap.md)
covers the Dalla Man findings and subsequent model repairs.

Update: the subsequent [implementation and cross-benchmark audit](2026-09-26-reference-integrity-fix-and-audit.md)
fixes the numerical generator and passes all 260 protocol/configuration checks.
Historical datasets and scores remain unchanged; see that follow-up for current
release limitations. The diagnosis below records the original finding.

## What happened

While comparing Full and No latent on CSTR seed 0, Full appeared better on
`validation_001`: Full was nearly flat, whereas No latent predicted a dip.
The saved reference was also nearly flat. We initially investigated the dip as
a model error, but then confirmed that **the reference generator had skipped
the forcing pulse during numerical integration**. The physical reference should
have a substantial dip.

This is not evidence that every method was supplied the wrong input column.
The input column recorded the pulse; the generated state/target trajectory was
inconsistent with that pulse. Models could therefore be rewarded for ignoring
a real input response.

## Confirmed example and cause

The canonical protocol is `validation_tf_mid_pulse` (exported as
`validation_001`). The feed temperature is 350 K before minute 10, 345 K from
minute 10 to 15, then 350 K again. At the reference equilibrium, the onset
changes the reactor-temperature derivative by exactly -5 K/min. A flat
temperature cannot satisfy the reference heat balance during that interval.

The generator's `_integrate` helper calls LSODA once over the entire horizon,
with `t_eval` but without forcing-boundary segmentation or an internal step
limit. **Output sample times do not force the solver to evaluate the ODE there.**
Near equilibrium, adaptive integration can take a long step across a pulse
whose value is zero at both ends.

Instrumentation of the original CSTR run recorded RHS calls at approximately
0, 0.003, 0.006, 30 and 30 minutes: none occurred inside the pulse. Re-running
that generator reproduced the saved flat temperature within 5e-10 K.

We then integrated the actual reference equations separately over each forcing
interval, carrying states continuously between intervals. All physical states
`C`, `T` and `Tj` evolved together; the historical flat auxiliary arrays were
not imposed. Radau and DOP853, each with `rtol=1e-10`, `atol=1e-12` and
`max_step=0.025` minutes, agreed within 2.19e-9 across all three state arrays.

| Quantity | Corrected diagnostic reference |
|---|---:|
| Initial reactor temperature | 365.131295254 K |
| Minimum on the 0.1-minute output grid | 360.773179343 K |
| Time of minimum | 12.7 minutes |
| Temperature decrease | **4.358115911 K** |
| Temperature at minute 15 | 360.803595886 K |
| Temperature at minute 30 | 365.131295254 K |

No latent's roughly 0.50 K dip had the right direction but was much too small.
Full's nearly absent dip matched the erroneous saved reference. Neither result
establishes accurate physical response under this intervention.

## Scope and effect on our conclusions

- The public replay found nearly flat reference temperatures in all 11 forced
  training step/pulse schedules and three validation step/pulse schedules,
  despite changing input columns. These are audit flags; the detailed
  independent physical-reference confirmation above is for the feed-temperature
  pulse. The remaining schedules still need individual checks.
- The sinusoidal validation schedule was not flat: its reference temperature
  excursion was about 12.03 K. Do not infer that every CSTR trajectory is wrong.
- The historical seed-0 pooled scores were Full train/validation
  `0.132255 / 0.0140082` and No latent `0.122933 / 0.0319746`.
  Those scores were reproduced, but the faulty reference invalidates their use
  as evidence that Full handles the pulse better. Seed 1 also reversed the
  overall validation ranking, independently limiting any general superiority
  claim.
- The easy CSTR models were conditional on supplied `C(t)` and `Tj(t)`
  auxiliaries. Correcting only the plotted `T(t)` while retaining old auxiliary
  trajectories would not create a consistent physical evaluation.
- This task did not explicitly require an extra hidden state. A No latent
  model is not automatically specification-incompatible.
- This diagnosis does not establish the same defect in Dalla Man or every
  benchmark that shares numerical infrastructure.

The original 80 model rollouts reproduced eight saved aggregate NMSEs within
6e-14. Thus the discrepancy was not introduced by our plotting/replay script.
That agreement validates reproduction of the historical scores, not their
physical correctness.

## What has and has not been fixed

Completed: public model replay, a synthetic missed-pulse reproduction, an
independent actual-CSTR reference audit, and documentation of the limitation.
The paper's benchmark appendix now flags the historical reference defect.

**A corrected benchmark release and matched reruns are not established by the
reviewed records.** The original diagnostic did not modify the single-horizon
helper, overwrite the benchmark or refit the models. The subsequent
[generator correction](../../docs/REFERENCE_INTEGRITY_AUDIT.md) replaces that
helper; it likewise preserves historical data and fitting results.

The actual-reference audit was a user-authorized post-hoc investigation.
Private reference information was not passed to proposal generation or model
selection; no benchmark test trajectories were opened in that audit.

## Recommended corrective work

1. Make reference integration respect every known discontinuity. Segment at
   step/pulse boundaries and apply any true state jumps explicitly. A suitably
   small `max_step` is a useful independent check, not a substitute for a clear
   event contract.
2. Add regression checks for a short pulse on an equilibrium system, pulses at
   the horizon boundaries, and results under changes to output-grid density.
   Compare to an analytical pulse response and independent numerical solves.
3. Audit all affected CSTR protocols and other generators using this helper.
   Regenerate targets and auxiliaries jointly, retaining exact schedule and
   data hashes. Do not assume all previously flat trajectories should be flat.
4. Preserve the historical release. Create a separately versioned correction,
   with a change manifest identifying affected cells and splits.
5. Rerun training, validation selection and final evaluation consistently for
   every compared method on that release. Merely rescoring old checkpoints
   against new curves is a diagnostic, not a corrected benchmark comparison.

## Evidence and files to share

- [Initial curve replay and synthetic integrator audit](2026-09-22-codex-cstr-curves-and-reference-audit.md).
- [Actual physical-reference confirmation](2026-09-22-codex-cstr-reference-pulse-ground-truth.md).
  This follow-up supersedes the earlier note's uncertainty about the actual
  CSTR response.
- [Replay runbook](../../docs/CSTR_CURVE_REPLAY.md) and
  [benchmark appendix](../../paper_to_revise/sections/appendix_benchmarks.tex).
- Historical public replay artifacts:
  `artifacts/cstr-curves-round12-2026-09-22/` (`trajectories.csv`,
  `inputs_auxiliaries.csv`, `trajectory_metrics.csv`, `summary.json`, models
  and figures).
- Separate private-reference diagnostic artifacts:
  `artifacts/cstr-reference-pulse-audit-2026-09-22/` (`audit.json`,
  `reference_comparison.csv`, `reference_pulse_comparison.png`).

Artifact directories are local outputs, not committed data packages. Another
agent needs those files separately to reproduce the curves.
