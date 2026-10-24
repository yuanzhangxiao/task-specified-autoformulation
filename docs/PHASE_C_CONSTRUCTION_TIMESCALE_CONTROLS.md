# Construction fitting: timescale controls and supplied LLM-SR replay

23 October 2026. This milestone is opt-in and training-only. It retains the
original eight construction models; it neither promotes a revised equation nor
changes the production fitter. The two CPU campaigns below have separate roots,
receipts and conclusions.

## What the returned profiling pilot established

Archive `72122ccea7b4bf4c909d9462003f0ddd9c78e45664d11be65bf8bc7c2d642585`
contains eight authenticated terminal records. Model 6's training NMSE changed
from 1.0426742273 to 1.0426473352 with joint continuation and to 1.0426327561
with conditional multi-output profiling. Both endpoints agree with independent
original-equation integration. Both searches exhausted their optimization
allowance; neither established fitting adequacy. The profiled inner problem has
21 coordinates and full numerical column rank, but this conditional result does
not identify all 27 aggregate coordinates. The reported full local rank was 25.

The magnitude of the insulin self-decay coefficient increased from approximately
123,637 to 2,363,901 in the profiled search. Its time constant is approximately
`4.23e-7`, while observations are spaced by one time unit. Compensating forcing
coefficients also grew. Almost all the small objective improvement occurred at
the first accepted outer step. This suggests a fast-timescale limit or weak
direction worth testing; it does not prove nonidentifiability, an unattained
infimum, or that a better finite solution does not exist.

The six fixed-endpoint precision studies completed 72 column/step comparisons
without numerical failures. Multiprecision throughout parameter lifting,
initialization, propagation and residual subtraction substantially reduced
several finite-difference discrepancies. Those sampled columns cannot certify
an entire endpoint. The new diagnostic must cover every nonfixed coordinate,
check local shrinking steps and precision agreement, and preserve unresolved
outcomes if its allowance expires.

Two reporting limitations are repaired prospectively: profiled search now saves
accepted full-coordinate vectors for endpoint-bound convergence assessment, and
its residual-evaluation accounting no longer silently reports zero attempts.
Historical sealed results retain their original interpretation and hashes.

## Model 6: finite rate slices and genuinely different starts

The equations and certified domains are those in
[the multi-output profiling pilot](PHASE_C_CONSTRUCTION_MULTI_OUTPUT_PROFILE.md).
For fixed six-dimensional dynamics, bounded least squares still jointly solves
five forcing coefficients and sixteen trajectory-specific hidden initials. No
observed target is used as forcing after its permitted initialization observation.

Ten tasks are frozen before the run:

1. Continue the latest verified profiled incumbent in the unrestricted signed
   model family.
2. Run six conditional slices with insulin decay magnitude fixed at
   `1/T`, `1/sqrt(T*dt)`, `1/dt`, `10/dt`, `1000/dt`, and the saved incumbent's
   magnitude. Here `T=300` is the longest observed span and `dt=1` the smallest
   sampling interval. Each slice optimizes the other five outer coordinates and
   the 21 inner coordinates.
3. Run three different outer starts based on slow, intermediate and resolved
   timescales derived from that same observation grid. These starts use the
   original signed domains and the same inherited inner anchor.

The six slices explore the negative insulin self-coefficient branch. They do not
impose that sign on the unrestricted model. Each task gets 600 seconds of search;
qualification, original-equation verification and endpoint diagnosis have
separate bounded allowances. All consumed or reserved allowances survive resume.
Original parent costs remain separate from this additional experiment.

Reports distinguish conditional slice outcomes from the full-family diagnostic.
A finite grid of rates is not a global lower-bound certificate. A converged
conditional inner solve is not convergence of the outer search. The original
verified incumbent remains available if a new start, qualification or search
fails. Independent rollout agreement and derivative reliability still precede
stronger optimization claims.

## LLM-SR: establish the numerical control before refitting

The supplied transcription set is marked `proposed_pending_review`. Several
native row recurrences were converted to ODE filters using mean-delay or Euler
rules. Those are approximations, not exact identities between native programs
and continuous free rollouts. Native scores are therefore not assumed to be
scores of the transcribed ODEs.

The first control uses the same T2-easy cell as the construction campaign. Its
training payload hash is
`169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795`, which
exactly matches the native run's recorded training identity. The supplied
repetition-0 transcription has three target outputs, eight hidden states and
29 coefficients. Repetition 1 is retained as unavailable: no causal ODE
transcription was supplied, and the transcription review reports future-row
access. It is not replaced by another cell or repetition.

The adapter hashes the native functions as data and checks their supplied
coefficient vectors. It never executes native Python. It compiles only the
restricted equation grammar, substitutes the supplied initializers, and performs
coupled free rollouts on training observations. Targets can enter initialization
at time zero; they are not supplied as subsequent forcing. Public auxiliaries
and external inputs retain the public interpolation contract.

Two bounded solver passes check supplied-value accuracy and numerical agreement.
The report includes per-target errors, partial/failing rollouts, domain failures,
solver costs and the original transcription qualifications. An incomplete pass
cannot certify replay agreement. This is not a mechanism-compliance evaluation.

Refitting is deliberately deferred until this replay is understood and the
transcription has an explicit admissible coefficient domain. Some branches,
clamps and guards were simplified only at the supplied coefficient values.
Optimizing those values without preserving the branch conditions could silently
change the transcribed model. The current adapter rejects parameter overrides.
Once reviewed, a matched recovery experiment can separate:

- a representational advantage, if the same-data ODE replay is already accurate;
- an optimization difficulty, if our fitting cannot recover that accuracy;
- a transcription or evaluation-semantics difference, if supplied-value replay
  itself does not reproduce comparable accuracy.

No validation/test scores, native score-based model selection, new LLM calls or
new baseline discovery enter either campaign.

## Astra's directional-family handoff

The actual 23 October handoff is `ORION_HANDOFF.md` under
`output/directional-family-analysis-20261023`; the requested `.md2` path was a
typo. Its 18 revision episodes produced no accepted candidate and no new fitting
packet. Rejected candidates must not replace the original eight numerical models.

Its exact equation-family restrictions are useful even before optimization
converges: a sign/domain bound on glucose rise, identical scalar responses under
identical initial state and forcing, and a rank-one difference restriction for a
linear subsystem with only one varying hidden initial direction. These are
necessary-condition conflicts with the stated observations and numerical guard,
not global best-fit NMSE bounds. The initial-readout absolute-error bound and the
small affine-superposition defect must retain their own scopes as well. The
guard is not an authenticated simulator-error bound.

Numerical feedback should keep these distinct from optimization status. A
function-only construction scope also cannot satisfy a request for a new state
or independent initializer direction; that needs an explicit broader-scope
request, with the original conflict retained until a valid revision exists.

## Running on Delta

The delivery contains committed source, sealed training-only inputs, offline
`mpmath`, a validation receipt and a checksum manifest. Extract it into a new
directory under `/projects/bibo/yxiao2/repos`, verify `CONTENTS.sha256`, and run
from the extracted directory:

```bash
export AF_PLATFORM=delta
export AF_ACCOUNT=bibo-delta-cpu
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_TIMESCALE_CONTROLS_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-timescale-controls-v1
export AF_LLMSR_CONTROL_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-numerical-controls-v1

bash scripts/hpc/start_construction_timescale_controls.sh run
bash scripts/hpc/start_llm_sr_numerical_controls_delta.sh run
```

Both campaigns use CPU jobs, one CPU and
8 GB per task. The ten-task fitting array has concurrency two and a one-hour
task limit. The two-task supplied-value control has its own bounded replay
allowances (180 seconds for Radau and 300 seconds for DOP853), concurrency one,
and a 15-minute task limit. Independent roots allow either report to finish
without waiting for the other campaign. Keep the same environment and extracted
source directory for status and inspection:

```bash
bash scripts/hpc/start_construction_timescale_controls.sh status
bash scripts/hpc/start_llm_sr_numerical_controls_delta.sh status

# After the jobs finish:
bash scripts/hpc/start_construction_timescale_controls.sh inspect
bash scripts/hpc/start_llm_sr_numerical_controls_delta.sh inspect
```

These write `$AF_TIMESCALE_CONTROLS_ROOT/inspection.tar.gz` and
`${AF_LLMSR_CONTROL_ROOT}-inspection.tar.gz`, respectively. Download both for
review. The dependent reports run after all array tasks terminate, including
failed tasks, so a completed report alone is not proof of numerical success.

Do not remove submission intents or reset roots to recover an ambiguous Slurm
reply. Inspect accepted jobs and reconcile the saved intent. A rerun of the
launcher must preserve all earlier costs and completed task records.

## Local validation and remaining limits

The final focused suite passed 273 tests. A broader six-worker regression run was
stopped to investigate four failures after 2,236 passes and five skips; all four
failed tests passed when rerun individually without that contention. This was
not a completed green full-suite run. `ruff check .` passed in the isolated
tracked-source validation snapshot. The shared checkout retains 37 unrelated
lint findings in untracked analysis files.

The saved model-6 endpoint reproduced training NMSE 1.0426327561207533 with the
new source, with 27 aggregate coordinates and a `14448 x 6` profiled residual
Jacobian. This bounded check performed no outer optimization. A 20-second
supplied-value LLM-SR smoke completed five of sixteen training trajectories,
then correctly reported timeout and withheld a whole-split score. Those first
five trajectories did not show tiny errors, so completing the two-solver replay
is necessary before using this transcription as a successful recovery control.
Neither smoke establishes optimization adequacy or native-program equivalence.

Both campaign launchers passed schema preparation and inspection packaging
without contacting Slurm. New rate slices, different-start fits and the complete
LLM-SR replay remain Delta experiments for the user to run.
