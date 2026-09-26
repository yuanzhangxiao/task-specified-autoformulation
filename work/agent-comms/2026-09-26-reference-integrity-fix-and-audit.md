# CSTR fix and cross-benchmark reference audit

## Outcome

The CSTR missed-pulse defect is fixed in the reference generator. The same
boundary safeguard applies to the alien device. The broader inspection found
and corrected two Dalla reference edge cases: carrying a state from the wrong
time at off-grid forcing boundaries, and using an inconsistent gastric reference
for multi-meal derivative labels.

The completed numerical audit passes **260/260 protocol/configuration instances
and 40/40 public projections**. The separate detention audit passes **18/18
trajectories**. This does not mean every historical release has been corrected
or every model-input convention is exact. Existing datasets and prompts were
preserved, and the Dalla meal-event/runtime contract still needs a separate
resolution.

## What changed

- New trusted `reference_integration.py`: explicit forcing boundaries, true
  endpoint carry, one-sided endpoint evaluation, bounded steps, tight tolerances,
  and finite/alignment/completion checks.
- CSTR and alien-device generation use this integrator for all states and
  derive output/auxiliary channels from the same solve.
- Dalla retains the gastric reference actually used in each integration segment
  for derivative export. Meal samples and mass aggregation are explicit;
  unrepresentable off-grid meals fail instead of disappearing from the input
  channel. Off-grid infusion changes remain supported.
- New manifests identify `reference-events-2` and solver settings. Writers refuse
  existing nonempty destinations. The release command uses a separate
  `phase_b_reference_events_v2` directory.
- A resumable CPU audit checks independent solvers, output-grid refinement,
  Dalla balance/event identities, public CSV values, leakage checks, file hashes
  and numeric equality across semantic variants.
- Existing intervention plotting scripts now pass explicit solver settings,
  preserving their independent-solver comparisons with the new integration API.

No fitting algorithm, LLM prompt, candidate model or historical data file was
changed. No remote session or job was opened.

## Audit results

The Phase-B audit includes T1--T4 canonical/perturbed Dalla, CSTR and alien
device. It checks every frozen train/validation/test schedule numerically;
test references are newly simulated privately, not read from existing test
datasets or used to score/select candidates. Public staging contains development
splits only. Repeated Dalla schedules across tasks are not independent systems.

| Check | Result |
|---|---:|
| Numerical protocol/configuration instances | 260/260 passed |
| Public easy/hard and semantic projections | 40/40 passed |
| Maximum scaled LSODA/DOP853 discrepancy | 1.1457904e-7 |
| Maximum scaled difference after halving output spacing | 1.7712143e-13 |
| Predeclared numerical acceptance threshold | 2e-6 |
| Detention coupled/independent train/validation cases | 18/18 passed |
| Maximum detention solver depth difference | 2.0556523e-8 m |
| Maximum detention relative volume-balance error | 1.8189894e-12 |

The first Phase-B run failed five CSTR derivative comparisons at the original
primary tolerances. Tightening tolerances resolved them; the acceptance threshold
was not relaxed. The final run uses LSODA with `rtol=1e-10`, `atol=1e-12`,
maximum step 0.5 and DOP853 with the same tolerances and maximum step 0.25.
The complete audit was resumed successfully using sealed checkpoints.

Final audit identity:
`fcb965b7c9f27399a93a9211cf9b337ddf63ce561b3e7ecd23951645c004682e`.
Local outputs: `artifacts/reference-integrity-audit-2026-09-26-v3/`.
Detention audit: `artifacts/reference-integrity-detention-2026-09-26.json`.
These are private evaluator artifacts, not proposer inputs or committed datasets.

## Comparison to the previous generator

We separately executed the exact trusted generator source from commit `f3fa8d3`
on the same protocols and compared its outputs to the correction. This is a
source-version comparison, not a claim to have inspected every remote CSV.

| Family | Finding |
|---|---|
| **CSTR** | 17/26 schedules have state differences above the scaled 2e-6 threshold: 11 training, 3 validation and 3 test. The largest temperature difference is about **15.2705 K**. The previously discussed validation pulse now has its expected **4.3581 K** dip. Historical affected scores cannot establish accurate pulse response. |
| **Alien device** | No comparable large missed-pulse error was observed in this released system. The largest state difference is about **3.38e-5**; 23/26 exceed the stringent scaled 2e-6 comparison threshold. These small differences include tighter integration tolerances and explicit boundaries. The safeguard also prevents the CSTR failure mode in future equilibrium-start systems. |
| **Dalla Man** | Across 208 task/configuration instances, maximum scaled state difference is about **1.16e-7**. Registered on-grid state trajectories remain close. Multi-meal `Qsto2`/`Qgut` derivative labels can differ by about **1866.23** in their native units because of the old gastric-reference inconsistency. Off-grid infusion endpoint carry is independently covered by a regression test. |
| **Detention basins** | Already integrate at every forcing knot, with an explicit conservation audit. No generator modification needed. |

Dalla event samples deserve a correction to the initial source-reading concern:
the old code already returned post-jump interior samples through mutation of a
shared array view. The new code makes this explicit; it does not establish that
all historical meal-time samples were wrong. The large derivative-label finding
must likewise not be described as a comparably large glucose-trajectory error.

Historical comparison artifacts:
`artifacts/reference-integrity-history-2026-09-26/` (old source snapshots,
`comparison.json` and the diagnostic comparison script).

## Remaining release work

1. **Do not overwrite the old release.** Corrected CSTR targets and auxiliaries
   must be versioned together. Every method must be retrained/selected/evaluated
   consistently if the corrected benchmark enters a reported comparison.
2. **Resolve Dalla's input semantics explicitly.** The reference uses exact
   meal-state jumps; the common runtime linearly interpolates event magnitudes.
   Time-zero events have half the interpolation area of identical interior
   events. An event-aware interface or a new mass-preserving finite-duration
   input contract is needed; silently changing one sample or using a private
   stomach mapping would be inappropriate.
3. **Treat sampled forcing as an approximation where applicable.** CSTR and
   alien reference schedules contain exact steps/pulses; generic linear
   interpolation of their sampled inputs creates short ramps. Boundary-aware
   integration fixes skipped inputs but does not make those two continuous-time
   input definitions identical. A separate physical-equation replay under the
   exact public-table interpolant found maximum target differences of **0.8568 K
   for CSTR** and **0.1827 for the alien output**, across 26 schedules each.
   These are not solver disagreement or candidate prediction errors. The
   diagnostic restarts at every interpolation knot. Its results are in
   `artifacts/reference-integrity-history-2026-09-26/input-interpolation.json`.
   A common explicit forcing contract is needed before claiming exact agreement
   between reference generation and model execution.
4. **Regenerate derivative overlays when relevant.** Any overlay containing old
   Dalla multi-meal gut derivative labels is inconsistent with the trajectory
   integration that produced it.
5. **Audit actual release hashes before replacing reported results.** The
   generator audit did not connect to ACES/Delta or certify their existing files.
   It also does not replace practical-identifiability gates or independent
   scientific assessment of the chosen reference equations.

The audit intentionally reports `release_ready=false` while the public input
contract remains unresolved. Its JSON highlights the Dalla event issue; the
additional interpolation diagnostic above broadens the release work to CSTR
and alien input semantics too. CSTR's numerical implementation is fixed; the
stronger statement “all benchmarks and historical experiments are now correct”
is not supported.

## Commands and related notes

See [the runbook](../../docs/REFERENCE_INTEGRITY_AUDIT.md) for the audit command,
deterministic resume, output interpretation and regression tests.

- [Original CSTR diagnosis](2026-09-26-cstr-reference-defect-summary.md).
- [Model inspection and future priorities](2026-09-26-model-inspection-findings-and-roadmap.md).

## Implementation verification

- Full repository test command: `bash scripts/run_tests.sh full`, with four
  parallel workers and single-threaded numerical libraries. The bulk run had
  **3489 passed, 8 skipped and 2 failures**. Both failures exposed intervention
  scripts monkeypatching the removed local `solve_ivp` name; both scripts were
  corrected to use `ReferenceSolver`. The complete affected test files then
  passed (**9/9**). No other bulk test failed.
- The full command's timing-sensitive serial group passed **66/66**. The
  original full-run log retains its failed exit status from the two subsequently
  corrected script failures; it is not relabeled as a clean rerun.
- A final focused regression run passed **101/101**, covering the new integrator,
  audit/checkpoint failure cases, Phase-B generation/public projection, both
  intervention scripts and the shared-process pilot file excluded by the bulk
  wrapper. Unchanged tests were not rerun a second time after the script fixes.
- The CSTR private-generation CLI smoke passed finite-rollout, input-excitation
  and persistence gates. Its separate scientific release gates remain pending.
- The full 260-instance/40-cell audit resumed successfully without resimulation.
- Ruff passes on every changed Python file; `ruff check .` still reports
  **37 pre-existing issues in `analysis/claude/`**. `git diff --check` passes.

Local logs: `artifacts/reference-integrity-tests-2026-09-26.log`,
`artifacts/reference-integrity-final-regressions-2026-09-26.log`,
`artifacts/reference-integrity-resume-2026-09-26.log` and
`artifacts/reference-integrity-ruff-2026-09-26.log`.
