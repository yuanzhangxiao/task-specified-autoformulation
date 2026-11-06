# Delta fitting campaign for the other 22 construction candidates

The user authorized continuing all remaining parents while Astra revises candidate 15.
The new opt-in campaign is `construction-remaining-portfolio-1`, with root
`/work/hdd/bibo/yxiao2/phase_c/construction-remaining-portfolio-v1`.
See [the runbook](../../docs/PHASE_C_REMAINING_PORTFOLIO_FITTING.md) for commands.

## Implemented scope

- All 22 original T2-easy alternatives except candidate 15 have a separate task.
- Fourteen affine models profile their 32 training initial roots at each shared-parameter evaluation. They receive warm and predetermined distinct shared-coefficient starts.
- Three nonlinear models use repeatable state-only residuals, certified reductions, three small shared-coordinate blocks and full-coordinate release. Every shared coefficient is visited before release, including saturation parameters. Each receives two starts.
- Five initially infeasible models receive five predetermined domain-legal probes. The first independently verified start enters fitting. Failure of this finite set does not establish family-wide infeasibility.
- Original-equation replay precedes retention. Source incumbents and verified intermediate improvements survive worse or failed later phases. An interrupted search can contribute its saved best point through a separately reserved replay without renewing the search.
- Descriptive feedback includes original-equation predictions, per-target and per-trajectory errors, sampled peak errors, and input-phase windows. It locates mismatches and makes no automatic scientific attribution or revision.

Each candidate has at most 1,800 numerical fitting seconds. Qualification,
verification and feedback are separate. Inclusive reservations are 5,820 seconds
for affine models and 6,300 seconds for nonlinear models. Delta requests one CPU,
8 GB and four hours per task, with concurrency two. The report has a separate
one-hour allocation. There are no GPU requests or LLM calls.

The numerical source remains the original full-training reconstruction problem.
The nonlinear precision policy explicitly tightens state-only tolerances to at
least rtol 1e-11 and atol 1e-13, consistent with the successful later nonlinear
checkpoints. This prevents the old coarse default from blocking their replay.
The compact intake preserves the original equations, parameter domains,
training split and source receipt hashes. Candidate 12's exact-zero control
remains separately recorded alongside its positive-rate starting point.

## Verification

The new campaign, numerical helpers, feedback, launcher and relevant numerical
regressions passed 213 focused tests. They include actual synthetic affine and
nonlinear fits, frozen start independence, domain failures, interrupted resume,
checkpoint retention, source tampering, report reconstruction, scheduler reply
ambiguity and Slurm spool-copy execution. No remote scheduler calls were made.

All 22 real adapters and planned starts compiled successfully. All eight
nonlinear models' active shared coordinates are covered by their three blocks.
A fresh qualification of saved candidate 10 passed repeated BDF residual solves
and original Radau replay at NMSE 1.712067856317. A separate original-equation
feedback replay of candidate 21 recovered NMSE 1.552123093659 across all sixteen
training trajectories. These bounded checks did not run shared-parameter searches
on the laptop.

Changed Python files pass Ruff, both shell wrappers pass syntax checks, and the
CLI help command works with the wrapper's declared Python path. Repository-wide
Ruff still reports 37 pre-existing findings under unrelated `analysis/claude`
files. The full repository pytest suite was not run in this milestone.

## Remaining interpretation limits

The new Delta results are pending. A completed worker means its bounded operations
terminated. It does not imply full stationarity, global optimality, parameter
confidence or scientific validity. Fresh eight-minute-prefix inference and
held-out evaluation of any improved endpoint remain separate work. No validation
or test data enter this campaign.
