# Final collocation comparison: rate refinement stopping policy

This closes the synthetic C+S (equality-constrained joint collocation), J+S
(joint penalty collocation), and A+S (alternating penalty collocation) comparison.
The prior rate-coordinate run repaired six severe alternating handoff failures,
but introduced one C+S regression. At that saved start a local controlled probe
reached training cost 0.205107 with `ftol=None`, versus 0.684775 with default
`ftol`. Changing Jacobian scaling alone did not fix it. This is a stopping-policy
test, not another initializer design.

## Frozen experiment

`configs/fitter_stopping_v1.json` uses a separate `fitter-rate-stopping-1`
protocol in the existing saved-start runner. Import the original **fitter-methods-v4**
directory, not the rate-refinement-v1 output. The source code, plan, task matrix,
and imported arrays retain the existing provenance checks.

All 39 saved C/J/A starts receive two refinements: `rates_default` (`ftol=1e-8`)
and `rates_no_ftol` (`ftol=None`). Both use direct rate coordinates, the same
domains, `x_scale=1`, `xtol=gtol=1e-8`, Radau sensitivities, 150 maximum optimizer
evaluations, and `600 - saved_initializer_seconds` seconds. Disabling `ftol`
retains the other stopping conditions and budget limits. No initializer is rerun.
The completed old verification retry is not repeated: 39 array tasks, 78 fits.
Rerunning both rate arms gives a contemporaneous control for accuracy and cost.

Training-only start and derivative checks precede fitting. Independent Radau/BDF
and tolerance-refinement replays remain unchanged, with 90 seconds per split.
Validation and clean-reference scores enter only the completed-fit diagnostics.
There are no benchmark/test data, hidden fitting labels, or LLM calls.

Reports distinguish native optimizer success, numerical verification, and output
recovery. They include the parameter declaration order, raw gradient infinity
norm, and native bound-scaled optimality, which should not be read as the same
quantity. Missing and failed fits remain in recovery-count denominators. Recovery
counts are grouped by C/J/A, ordinary versus stress starts, and stopping policy.
The pre-existing descriptive recovery threshold is clean train **and** validation
NMSE <= 1e-4; it does not select parameters.

## Delta execution

Use a new clean checkout and a new output directory. From that checkout:

```bash
export AF_CONFIG="$PWD/configs/fitter_stopping_v1.json"
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-stopping-v1
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-methods-v4
bash scripts/hpc/submit_fitter_rate_refinement_delta.sh
```

The launcher uses the existing Delta Python/CasADi environments: one CPU, 8 GB,
no GPU, at most two tasks concurrently, and 45 minutes per paired task. The
summary allocation is now **30 minutes**, with per-task progress printed while
checking replay hashes. Completed fits and trajectories resume without refitting;
interrupted native optimization remains explicitly marked rather than silently
receiving a new budget. Duplicate/partial submission protection is unchanged.
Do not reuse an earlier frozen output with the new code.

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-stopping-v1/summary.md
```

The compact counts at the top are sufficient for the first review; `summary.json`
retains parameters, stopping details, gradients, and independent replay evidence.

## Closing decision and limits

Joint collocation plus sensitivity refinement remains the working baseline.
Alternating optimization should replace it only if the final matched results show
a material recovery or runtime advantage without new failures. Otherwise archive
alternating as an explored option and focus on fitting actual proposed models.
This is a development comparison on already inspected synthetic cases, not a new
unseen scientific benchmark or evidence of unique latent-state identification.

No production stopping default is changed here. In particular, disabling `ftol`
cannot repair an infeasible proposed equation or an all-failure residual penalty
that produces false `gtol` success. That validity issue is a separate fitter fix.

## Local verification

The exact reported noisy separated C+S start reproduces the default failure:
12 calls, training cost 0.6847753, clean validation NMSE 0.00215148. With only
`ftol` disabled it takes 49 calls, reaches training cost 0.2051066 and clean
validation NMSE 0.0000212420. Both pass every independent Radau/BDF/tolerance
replay. This local regression used the same 75-second/80-evaluation allowance
for each arm; the Delta matrix retains the larger frozen budgets above.

`pytest -q`: 1,205 passed, three optional PyTorch tests skipped. `ruff check .`,
shell syntax checks, and checkpoint/import/paired-policy tests passed.
