# First staged-model numerical handoff

Public-scientific review of function job 2096993 selected only the anonymous
system's function seed 1 (`functional_fcdf45e854663fbd`) for a numerical probe.
All six public attempts compiled and corrected the earlier repeated outer
subtraction. Only one anonymous candidate implemented the required nonlinear
feedback; the other two remained linear. The Dalla candidates retained missing
unit conversion or unjustified sharing of dimensionally different coefficients.
Both grouped-source diagnostics still contradicted their relaxation intent.
This is a conditional review selection, not a reliable function-generation rate
or evidence of good fit. No data metrics were used to select the candidate.

The selected model has damped memory and coupling states and bounded nonlinear
feedback. Its squared response is even, its saturation scale is fixed, and it
shares a gain across the two feedback directions. These are hypotheses to test.
The current variable-projection profile has no eligible affine weight after
expanding the output's feedback into the latent dynamics. This probe therefore
uses the existing `bounded_nonlinear` fitter and adaptive `solve_ivp` integration.

## Frozen numerical contract

`configs/staged_fit_probe_v1.json` pins the function plan, task and exact result
digests plus the public benchmark prompt. Preparation verifies the original
terminal identity and exact source candidate, and matches the public scientific
brief to Sections A–E of the data release's prompt. It snapshots only
`manifest.json`, `proposer_prompt.txt`, `train.csv` and `validation.csv`, then
validates their hashes. The sealed test fingerprint remains in the manifest;
test bytes are neither copied nor opened. Source candidate domains, parameter
sharing, equations and all three fixed zero initial values remain unchanged.

The experiment uses one CPU, 8 GB memory and a 15-minute Delta allocation:

1. Replay an all-one parameter vector on complete train and validation splits,
   with a shared 30-second allowance. Any failed trajectory or nonfinite score
   records `screen_failed` and skips fitting. This failure is about that initial
   vector or budget; it does not prove no feasible parameter vector exists.
2. If the screen is finite, fit on train for at most 300 seconds. There are
   three starts and at most 100 optimizer `nfev` per start. The screened vector
   is the preferred first start; two additional starts use the fixed random
   seed. Restart selection uses training cost. Numerical Jacobian evaluations
   can exceed SciPy's reported `nfev` count and remain in the diagnostics.
3. Independently replay the fitted vector on all train and validation
   trajectories, with a shared 60-second allowance and train-only scaling.
   Every trajectory must succeed before the split receives an aggregate NMSE.

No derivative regression, CasADi initialization, fitter algorithm change,
proposer call, test evaluation or private reference is included. Fixed latent
initializers are preserved; neither train nor validation gets fitted hidden
initial values in this probe. The existing fitter's native metrics, optimizer
diagnostics and soft-constraint diagnostics are retained beside fresh replay
metrics. Compilation, initial feasibility, optimizer completion, numerical
replay and scientific assessment are separate outcomes.

The supervisor enforces a 450-second total worker cap, including a 60-second
margin for loading and numerical finalization. A timeout kills only its own
worker process group and preserves completed phase checkpoints. The checkpoints
are `source_function.json`, `candidate.json`, `default_replay.json`, `fit.json`,
`final_replay.json` and `result.json`, bound to `freeze.json`. Changed source,
dependencies, launchers or assets are rejected before reuse. A completed fit or
replay is reused after interruption; an interrupted optimizer restarts from its
seeded starts rather than resuming internal SciPy state. Terminal failures remain
in the result ledger; a changed-budget retry needs a separately frozen run.

## Delta deployment

Use an isolated clean checkout of the pushed integration commit. Historical
defaults are the Python interpreter at
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python` and public release at
`/work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3`. They need verification
after authentication; the submitter accepts explicit overrides and preflights
the exact source and public data before dispatch.

```sh
AF_REPO_ROOT=/path/to/isolated/checkout \
AF_FUNCTION_PLAN=/path/to/function-handoff/plan.json \
AF_FUNCTION_RESULTS=/path/to/function-handoff/results \
AF_OUTPUT_ROOT=/path/to/new/staged-fit-v1-output \
bash /path/to/isolated/checkout/scripts/hpc/submit_staged_fit_probe_delta.sh
```

Optional overrides are `AF_PYTHON`, `AF_PUBLIC_DATA_ROOT` and `AF_CONFIG`.
The submitter freezes inputs before submitting exactly one CPU job. An existing
submission manifest returns its job ID; an unresolved atomic submission intent
requires queue reconciliation and cannot silently create another job. The Slurm
worker verifies the pinned clean checkout before supervised execution.

The numerical CLI also supports `prepare`, `run` and its internal `worker`
independently of notebooks. Tests exercise successful synthetic parameter
recovery, absent test tables, source/terminal/prompt tampering, fixed initializer
preservation, failure before optimization, preferred-start propagation,
checkpoint identity and phase reuse, hard timeout and complete CLI resume.

The current Delta Open OnDemand shell requires NCSA Kerberos login and Duo.
Authentication is the remaining deployment prerequisite; do not record a Delta
job ID until an actual successful submission is returned.
