# Matched unchanged-parent refit

The saved quadratic-plus-linear child improved training NMSE from 0.827381 to
0.662004 and validation NMSE from 0.928519 to 0.897582. Those numbers compare a
newly optimized child with its older parent. This diagnostic adds one unchanged
parent fit to separate an additional fitting allocation from a structural change.

The control starts from the exact original parent vector used to seed the child,
including every causal initializer coefficient. It uses the unchanged parent
equations and the same train/validation data, random seed, numerical profile,
tolerances, stage time limits and evaluation caps. Parameters remain train-fitted;
validation initial conditions are not optimized. The historical child is reused,
not fitted again. No proposer, judge, test split or private reference is opened.

The new driver runs with imports from the clean original `9196877` replay
checkout. Its existing historical source/runtime verification remains active.
The new driver is separately hashed and the comparison binds the exact child
freeze and backend-verified result. This avoids accidentally comparing an old
child with Astra's newer fitting adapter. Do not update the original checkout.

Only one control allocation is reserved per child identity. Completed results
resume exactly. An interrupted started allocation is reported as consumed, not
automatically retried. A separate output root protects all historical artifacts.
Reports verify the source, result and comparison lineage without invoking fitting.

## ACES

Create a clean checkout at the published matched-refit commit:

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1 fetch origin codex/prefit-aces-v1
git -C /scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-matched-refit-v1 PUBLISHED_COMMIT
```

Submit one CPU job (one CPU, 16 GB; no H100 or LLM server):

```bash
AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-matched-refit-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-matched-refit-v1/scripts/hpc/submit_prefit_matched_refit_aces.sh
```

The defaults use the existing original replay checkout, completed child at
`/scratch/user/u.yx126462/phase_b/prefit-parameter-replay-v1`, and output at
`/scratch/user/u.yx126462/phase_b/prefit-matched-refit-v1`. The job runs focused
regressions and a synthetic real-fit smoke before the benchmark control. All
temporary files and logs stay under the new output directory. Repeating the
submission prints its recorded job ID without submitting a duplicate.

After completion:

```bash
jq '{status,comparison,matching,refit_status:.refit.status,refit_budget_exhausted:.refit.budget_exhausted,refit_optimizer_converged:.refit.native_optimizer_converged,child_residual_calls:.child.actual_residual_calls,refit_residual_calls:.refit.actual_residual_calls,live_llm_calls,limitation}' /scratch/user/u.yx126462/phase_b/prefit-matched-refit-v1/summary.json
```

`child_minus_refit < 0` means the child has lower NMSE on that split. A similar
or better control means the extra fitting allocation can account for some or all
of the previous improvement. A better child supplies paired evidence for this
revision, not proof that its proposed scientific explanation is correct.

To regenerate and verify the report without allocating fitting:

```bash
module load GCCcore/13.2.0 Python/3.11.5
PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-matched-refit-v1/scripts/prefit_matched_refit.py report --root /scratch/user/u.yx126462/phase_b/prefit-matched-refit-v1
```

Logs are `logs/refit-JOBID.out` and `logs/refit-JOBID.err` under the output root.

## Interpretation and further rounds

This is one retrospective comparison after seeing the child's result. Equal
configured allocations do not imply equal residual calls, wall time, numerical
convergence or optimizer work. Machine load can affect a wall-clock-limited fit.
The result does not establish a population success rate or structural recovery.
No automatic model selection or next proposer round occurs in this diagnostic.

Further proposal rounds can test whether validation performance continues to
improve. They should use a new frozen experiment plan, keep the incumbent if a
revision fails or worsens validation, retain an unchanged-model refit control,
and give the proposer only training evidence. Existing completed or submitted
campaigns should not be extended in place. Astra's review campaign already has
two revision visits after construction; longer runs are separate follow-ups.

The current ACES proposal launchers request one H100 per server allocation;
fitting runs on CPUs. Extra proposal jobs therefore compete for GPU scheduling
capacity with Astra's campaign. They do not share its model server or artifacts.
Actual queue delay depends on available GPUs and account scheduling limits.
Moving proposal generation to another resource requires a separately recorded
serving/runtime identity; keep the model revision, decoding settings and fitting
protocol fixed when interpreting a continued trajectory.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_prefit_matched_refit.py tests/test_prefit_matched_refit_submission.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_matched_refit.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
bash -n scripts/hpc/submit_prefit_matched_refit_aces.sh
```
