# Local profiling recovery and instantaneous LLM-SR controls

24 October 2026. Two separate, opt-in CPU experiments. Neither revises equations,
changes the production fitter, calls an LLM, nor opens validation or test data.
Historical results retain their original receipts and interpretation.

## Which candidate is model 6?

This is zero-based index 6, the seventh of the eight saved T2-easy construction
candidates. Its construction seed is 2 and interaction proposal seed is 3:
`cell02_seed2_full_actionable_initialization_focused_equality_evidence_bookkeeping__proposal_seed3`.
It has measured glucose `Gp`, insulin `I`, utilization `U`, and one persistent
hidden insulin-memory state `I_mem`. It is not benchmark number six, and it was
not selected as the most accurate or most scientifically convincing candidate.

The aggregate equations have the form

```text
Gp' = a_meal * meal + a_EGP * EGP - U / tau_G
I'  = c_input * insulin_input + c_meal * meal - c_self * I + c_mem * I_mem
U'  = c_Uii * Uii + I_mem - lambda_U * U
I_mem' = k_act * I - lambda_mem * I_mem
```

Signs and domains remain those of the actual model; this schematic minus sign
does not impose nonnegativity on its originally signed insulin coefficient.
For fixed six dynamic coordinates, all three outputs depend jointly and affinely
on five forcing gains and sixteen trajectory-specific hidden initial values.
Bounded least squares therefore solves a 21-coordinate inner problem, leaving
only six outer coordinates. This makes the candidate useful for separating an
optimization failure from limitations of the equation family. The other seven
construction models remain in the original investigation.

## What the returned runs established

The ten-task timescale archive has SHA256
`ad60b26e0e7891ff1500a6749663a32322c551720933f4ec3b404cc2daae16be`.
All ten terminals reconstructed from their operation receipts. Only three tasks
entered search. Continuing the saved endpoint improved training NMSE from
`1.0426327561` to `1.0426308439`, while the magnitude of the insulin self
coefficient grew from about 2.36 million to 336 million. This neither certifies
convergence nor bounds the best error in the full model family.

The seven unsearched starts revealed two avoidable obstacles:

1. At a new insulin coefficient of -1, derivative perturbations inherited an
   optimizer unit of approximately 2.36 million. Even the smallest attempted
   perturbation crossed to a positive, unstable rate. These were not reliable
   local derivative tests of the new start.
2. The new dynamic coordinates were assessed with forcing coefficients inherited
   from the very fast saved model. Their large initial rollout errors were not
   minimized conditional profile errors. The linear block should be fitted first.

The two-task LLM-SR replay is separately authenticated by archive SHA256
`4f26f775a69f0b371464bc0f734e853edab9fee7a8d94bdf5985671dab07aef8`.
Both solvers failed on training trajectory 005 of the repetition-0 T2
transcription. The fifteen complete trajectories agreed numerically, but cannot
stand in for a complete training score. Repetition 1 remains unavailable.
The native selection record also reports a very large training error for the
repetition-0 program (approximately `3.54e38`). Thus its failure is not evidence
that a successful native training model was spoiled solely by transcription.

## V2 model-6 recovery

The same ten outer starts and the same original matched anchor are retained.
Every task follows:

1. Solve the bounded 21-coordinate inner problem at the declared outer start.
2. Verify that projected checkpoint using independent original-equation rollout.
3. Check local derivatives of the profiled residual on the five or six free
   outer coordinates, then search if the check permits it.
4. Independently verify a new endpoint before retaining it.
5. Diagnose the retained endpoint in all 27 aggregate coordinates. Conditional
   inner convergence and outer optimizer termination do not replace this check.

Each task still receives 600 seconds of optimization allowance: 60 for the first
inner projection and 540 shared by the outer derivative check and search.
Verification, final diagnosis and setup have separately recorded bounded
allowances. Accepted checkpoints and operation reservations survive resume;
historical costs are carried forward separately and never reset.

For a configured rate coordinate p, the local finite-difference scale is
`max(abs(p), 1/T)`, where T is the training observation span. The delay scale uses
the minimum observation interval as its floor and is additionally capped at the
magnitude of a nonzero reciprocal coordinate. These are local probe scales, not
new optimizer units, fitted parameter bounds or a new scientific prior. The
legacy default remains unchanged. The derivative check must still demonstrate
local shrinking-step consistency; a small probe is not automatically reliable.

The six fixed negative-rate slices remain conditional diagnostics. They cannot
establish a global lower bound for the unrestricted signed model family. A
failed projection, unverifiable rollout, exhausted allowance or unresolved
diagnosis remains explicit. No new endpoint automatically revises or promotes a
scientific model.

## Which LLM-SR version is the appropriate control?

Some native programs maintain row-by-row recurrences or filters. Replacing those
with continuous-time latent ODEs is an approximation, even when the intent is
similar. An instantaneous expression has a cleaner continuous-time meaning:
its right-hand side depends on the current modeled states and public inputs.
Nevertheless, changing from a native grid integrator to an adaptive integrator
can change trajectories, especially for unstable or discontinuous expressions.
Exact right-hand-side transcription alone does not guarantee identical scores.

The baseline checkout already implements `llm_sr_expression`, a restricted
one-expression variant reconstructed from the paper's “w/o Program” ablation.
Its full campaign is still proposed; limited smoke tests establish the interface,
not a successful accurate numerical control. No new discovery run is required
for the present milestone: four already saved native selections are instantaneous.

| Native task | Cell | Repetition | Native training grid NMSE | Role |
|---|---|---:|---:|---|
| 8 | CSTR easy | 0 | 9.67e20 | Divergent control |
| 9 | CSTR easy | 1 | 3.29e20 | Divergent control |
| 14 | Detention, independent inputs | 0 | 1.71e-4 | Potential positive control |
| 15 | Detention, independent inputs | 1 | 2.66e-4 | Potential positive control |

All four are retained, including failures. Their native plan, selection records,
function text, supplied coefficients, public training payloads and reviewed
transcription identities are bound before execution. Native Python is data, not
executable input. An isolated restricted expression evaluator preserves the
required real powers and piecewise operators without extending the proposer's
grammar. Unknown syntax fails closed.

Each control replays fixed supplied coefficients with Radau and DOP853 under
the public input contract, retaining failures, per-trajectory results and costs.
This is not yet coefficient recovery. Some transcription simplifications are
valid only at the supplied coefficient values or documented input-domain guards;
parameter overrides are forbidden. Any subsequent refit must first restore or
certify those branch restrictions. No claim of mechanism compliance follows
from a low replay error.

## Delta commands

Extract the delivery into a fresh directory under `/projects/bibo/yxiao2/repos`
and verify `CONTENTS.sha256`. From that directory:

```bash
export AF_PLATFORM=delta
export AF_ACCOUNT=bibo-delta-cpu
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps

bash scripts/hpc/start_construction_timescale_recovery.sh run
bash scripts/hpc/start_llm_sr_instantaneous_controls_delta.sh run
```

Both campaigns request CPUs only and use separate roots. The first defaults to
`/work/hdd/bibo/yxiao2/phase_c/construction-timescale-controls-v2`; its explicit
override is `AF_TIMESCALE_RECOVERY_ROOT`. Stale v1 root variables cannot redirect
the new wrapper. The LLM-SR launcher uses its own control root.

Use the same launchers with `status` to check progress, then `inspect` to collect
receipts after completion. `report` reconstructs summaries without fitting.
Do not resubmit an ambiguous scheduler reply: reconcile the saved intent with
Slurm first. The launchers preserve completed operations and consumed allowances.

The bundle's `VALIDATION.json` records local tests and bounded representative
checks. Local smoke results are diagnostic and do not substitute for the full
Delta campaigns.

The real-data smoke projected the resolved start, fixed inverse-interval slice,
and saved control successfully. The first two passed their six- and five-column
profiled derivative checks. The resolved start's independent verification hit
its 60-second smoke cap during original-equation Radau integration; it remains
uncertified. The larger campaign allowance is 180 seconds, not a guarantee of
completion. For LLM-SR, first-trajectory replay at supplied values executed all
four controls; task 14 had approximately `9.60e-5` NMSE. These subset checks do
not certify either full-training accuracy or native/adaptive equivalence.
