# Fixed-skeleton attainable-data and reference controls

This is an isolated, user-authorized numerical diagnostic. It is not a proposer
experiment or a production-default change. No reference equations, parameters,
known hidden boundaries, or reference-control results may enter proposer or judge
feedback. Test data are never opened. The four public candidates are older
proposals for one benchmark, not four known-correct generating skeletons.

## Questions and controls

For each frozen proposed model, construct noiseless observations from its exact
equations, with the original public input samples, interpolation, time grids,
and trajectory counts. A bounded deterministic search selects the first active,
finite generating point using training trajectories only. The audit records
output standard deviation, state ranges, and response to removing input. It
rejects constant/negligibly varying controls. Validation failure does not cause a
different generating point to be selected. Both Radau and BDF must verify the
chosen point before synthetic fitting proceeds. Failed generation remains a
reported result, not an optimizer failure or an omitted case.

The exact benchmark generating skeleton is constructed from an external hashed
private specification, never from hard-coded benchmark coefficients. Its 48
continuous coefficients, nonlinear scales and biases are all free during joint
fitting. Structural zeros and skew coupling parameter sharing are preserved.
It has six states, compared with four in the proposed parents. Synthetic reference
observations use the fitter's sampled-input convention, so those controls are
known attainable. Reference fitting on actual public observations is also tested.

Correct equations alone do not guarantee that the full observation protocol is
attainable. Two differences are measured explicitly:

* The original generator uses continuous forcing, whereas fitting interpolates
  the supplied input samples. Independent native-generator replay first verifies
  the specification against public train/validation outputs. Tight sampled-input
  replay at the true parameters reports the resulting discrepancy; it is a
  reference score, not a proven lower bound on the optimum.
* Some training trajectories use different hidden initial states despite identical
  public initial observations and forcing. Reference controls therefore distinguish
  **known hidden initial states** (explicit oracle covariates, no hidden trajectory
  observations) from **shared fitted initial states** (the existing public fitter
  policy). Known validation initials are privileged diagnostic information and are
  not inferred from validation outputs. A public-only calculation reports a rigorous
  training error lower bound for shared/causal initials when identical available
  conditions lead to different target trajectories. It does not apply to the oracle
  boundary controls.

## Frozen task matrix

With four public candidates and the reference export, there are five generation
tasks and **39 fitting tasks**:

| Skeleton/data | Arms | Tasks |
|---|---|---:|
| Each proposed skeleton / its generated data | Fixed parameters full/24k mesh; ordinary start 0 full/24k; ordinary start 1 at 24k; near-truth start at 24k | 24 |
| Each proposed skeleton / actual observations | Ordinary start 0 at 24k | 4 |
| Reference skeleton / generated observations, known initials | Same six synthetic arms | 6 |
| Reference skeleton / actual observations, known initials | Ordinary starts 0 and 1; near-truth start, all at 24k | 3 |
| Reference skeleton / actual observations, shared fitted initials | Ordinary starts 0 and 1 at 24k | 2 |

Fixed-parameter arms optimize only the collocation state nodes. Parameters and
physical initial values remain at the generating point. Hidden node guesses are
constant initial values, with observed output guesses; generating hidden
trajectories are not supplied. These arms report collocation objective and
constraint defects by state, trajectory and interval. They diagnose transcription
and node optimization, not parameter recovery.

Ordinary starts use the frozen candidate's guesses (or general role defaults)
and a reproducible perturbation. They do not use truth. The separately labelled
near-truth arm perturbs the generating parameters multiplicatively by log-SD 0.2;
zeros start at 0.1. This is a local recovery diagnostic, not evidence of general
recovery from an uninformed start. Each arm is independent; no oracle result is
used to select an ordinary start.

Joint fits use the existing mapped collocation and best-valid handoff/trial
rejection policy: 300 seconds / 1,000 iterations for collocation, then 600 seconds
and 480 calls for refinement. Fixed-node arms have only the 300-second stage.
These budgets are larger than the earlier mesh campaign, so its timings are not a
controlled speed baseline. Full versus 24k retains every observation and forcing
corner. A target of 24k can be exceeded to preserve input interpolation.

All joint final scores use fresh ODE rollouts and independent BDF/Radau replay.
`complete` means replay verified; synthetic `recovered` additionally requires both
train and validation NMSE <= 1e-4. Actual-observation fits have no recovery label.
Neither prediction recovery nor native optimizer success proves unique parameter
or named-latent recovery. Known-attainable success with poor actual-data fitting
supports investigating model adequacy; it does not prove a global impossibility.
For joint fits, `C success` is the existing wrapper's acceptance flag. A native
solver return followed by parameter-domain rejection can still make this false;
inspect the saved message and progress phase before diagnosing nonconvergence.

## Export and run

The prepared local reference export is a generated artifact outside Git. If it
must be recreated, run on the Mac from this pinned checkout:

```bash
PYTHONPATH=src /Users/yuanzhangxiao/Projects/autoformalism/.venv/bin/python \
  scripts/run_attainability_campaign.py export-reference \
  --source /Users/yuanzhangxiao/Projects/autoformalism/data_raw/benchmark6_alien_device/private/selected_system_spec.json \
  --output /Users/yuanzhangxiao/Projects/autoformalism/artifacts/fitter-attainability-private/reference-control-v1.json
```

Copy that single file to the diagnostic area on Delta:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/artifacts/fitter-attainability-private/reference-control-v1.json \
  delta:/work/hdd/bibo/yxiao2/phase_b/fitter-reference-control-v1.json
```

From a clean, pinned Delta checkout:

```bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-feasibility-v1
export AF_REFERENCE_FILE=/work/hdd/bibo/yxiao2/phase_b/fitter-reference-control-v1.json
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-attainability-v1
export AF_ARRAY_CONCURRENCY=2
bash scripts/hpc/submit_attainability_delta.sh
```

Preparation only freezes inputs and builds schemas. Numerical work runs on CPU
compute nodes. The launcher schedules smoke, generation, fitting, then summary;
every task requests one CPU, 16 GB, no GPU, with a 50-minute scheduler ceiling.
Generation and fitting arrays each
limit concurrency to two. No ACES work is necessary for this diagnostic.

Read results:

```bash
cat "$AF_OUTPUT_ROOT/summary.md"
"$AF_PYTHON" -S scripts/run_attainability_campaign.py summarize --output "$AF_OUTPUT_ROOT"
```

`summary.json` retains full generation audits and fit/replay records. Each fixed
arm's `fixed/progress.json` records its worst equation/interval defects, scaled
for reporting only. Joint arms retain the existing checkpoint and handoff logs.

## Checkpointing and limitations

Frozen code/runtime/input hashes prevent drift. Repeating submission preserves
the recorded jobs rather than duplicating them. Only reconcile a partial
submission after checking the queue. To resume an inactive task in the same
checkout/environment, with the original `AF_CODE_COMMIT`:

```bash
sbatch --array=INDEX --export=ALL scripts/hpc/attainability_delta.slurm run
```

Use `generate` instead of `run` for a generation index. Completed results and
saved fits/replays are reused. A native solve interrupted before saving a fit
becomes `interrupted`; resubmission does not silently grant it another budget.
The same rule applies to interrupted generation. A new numerical attempt requires
a new output root, preserving earlier evidence. Do not delete start markers.

The diagnostic exercises only this benchmark and these four proposed structures.
Generated candidate dynamics need not be as challenging as the real system;
activity gates exclude trivial controls but do not prove that every nonlinear
pathway is strongly excited or identifiable. Reference fitting is larger and
may still exhaust its bounded budget. Generation failure and numerical recovery
failure must remain separate in interpretation.
