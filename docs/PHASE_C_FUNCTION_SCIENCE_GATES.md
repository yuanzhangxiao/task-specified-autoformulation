# Function construction: assembled equations and training conflicts

Audit and milestone scope, 22 October 2026. Parent–child NMSE comparisons remain
descriptive until fitting has adequate numerical evidence. This construction
milestone concerns equations and contradictions that can be established without
estimating parameters. It does not infer equation inadequacy from an unfinished
optimization or change Orion's fitter.

## What the previous feedback actually checked

The returned `construction-feedback-revision-2` run used commit `9578ad7`, eight
T2-easy parents and three revision seeds each. Its 46 replies produced 16 changed
exports, one unchanged export and seven rejected episodes. The later paired fit
used these 16 changed exports and all eight unchanged parents.

The prior model/data check was real, but narrow. In
`construction_revision_consistency.py`, it compared initial readout equality and
closed-subsystem equality under matched training conditions. It inspected the
actual candidate dependency closure and initializer graph. A relevant free
trajectory initial root correctly deferred the forced-equality argument.

- Twenty attempted replies were rejected with `MODEL_DATA_CONFLICT`, across
  eight episodes. All seven final rejections carried that diagnostic.
- Thirteen subsequent calls received that feedback: eleven repeated a conflict,
  one introduced an unknown symbol, and one introduced duplicate edits. The
  last episode later repaired its patch and exported a revised model.
- That exported child reconnected insulin to `I_mem`, whose initial root may
  vary across trajectories. The equality check therefore became deferred. This
  removed the particular contradiction; it did not establish an adequate fit.
- Among the 17 exports, 13 had deferred checks and four had no conflict found.
  These are check-coverage statuses, not scientific-compliance scores.

Children 08/09 in the paired roster were accepted on their first replies. In each,
the three initial-readout checks found no contradiction, while the three dynamic
equality checks deferred because `I_mem` has a trajectory-specific initial root.
No monotonicity or positive-invariance check existed in that protocol. The runtime
therefore never showed the proposer the disposal contradiction identified below.

The checks also did not solve simultaneous parameter-feasibility constraints in
initialization maps, enforce dimensional consistency from free-text units, or
certify signed source/sink behavior. None should be described as already covered.

## Why the composite coefficients appeared

In parent-06 and child-13, the compiled disposal equation contains both ordinary
terms and named-process uses:

```text
U' = Uii + I_mem - U/tau13 + gain4*basal_U - gain6*U_decay
basal_U = Uii
U_decay = U/tau6

therefore U' = (1+gain4)*Uii + I_mem - (1/tau13+gain6/tau6)*U.
```

`process_gains.lower_candidate_gains` introduced the nonnegative consumer gains
when assembling the process contract. It did not create the duplicate scientific
effect: the underlying candidate already included the direct contribution and a
process with the same completed law. The retained function assembly did not
consolidate these overlaps.

An audit of every RHS, process definition, observation mapping, initial condition,
initializer graph and constraint confirms that `gain4`, `gain6`, `tau13` and
`tau6` in this example affect **only this U equation**. They are not shared with
another equation or initializer. The exact parameter identifiers are
`af_process_gain_4_0`, `af_process_gain_6_0`, `tau_term_13_2_4c01e0f40a` and
`tau_term_6_0_77cba2d645`.

The effective coefficients are `c_Uii = 1+gain4 >= 1` and
`lambda_U = 1/tau13+gain6/tau6 > 0`. The first is an offset parameterization with
one free parameter; the second gives three free parameters for one effective
decay coefficient. A replacement must preserve these attainable domains. An
unrestricted new coefficient, or an arbitrary nonnegative coefficient for
`c_Uii`, would change the model family.

Function-level overlap should be shown after shared processes are expanded, along
with all places each parameter is used. Matching argument lists at topology time
still do not establish identical functions. Do not merge distinct nonlinear
functions, change cross-equation parameter ties, or discard an annotated mechanism
without accounting for its intended meaning.

## A fitting-independent conflict to catch

Children 08/09 declare an ODE, not an algebraic readout:

```text
U' = Uii + k_insulin*I_mem
I_mem' = (I-I_mem)/tau
k_insulin >= 0, tau > 0, I_mem(0) >= 0.
```

Their retained insulin equation has nonnegative production/coupling coefficients
and negative self-clearance. The forcing and observed initial insulin are
nonnegative. Hence the I/I_mem subsystem preserves nonnegativity. In public
training trajectory `train_000`, `Uii=1` and the recorded U is the constant
`1.915785980090235` throughout t=0,...,300. These models force `U' >= 1`, regardless
of the allowed root value and parameter magnitudes. The contradiction does not
depend on a numerical optimum, the meal conversion, or equality between hidden
initial states in two trajectories.

Feedback should state the displayed derivative, the domain/invariance proof, the
training witness and the resulting incompatible change. It should require an
equation repair. Merely acknowledging the limitation or deleting a diagnostic
record cannot resolve it. The proposer must choose the scientific correction;
the runtime must not silently reinterpret `U'` as `U`, add decay, or introduce a
textbook mechanism.

## Implementation and review boundary

The v3 construction-only request omits historical fitted endpoint scores and
parameter values. It uses the public task, training observations, unchanged model
definitions, actual initializer rules and deterministic evidence. Numerical starts,
derivative qualification and optimization remain Orion's responsibility; this
milestone changes no fitter policy. An available poor score is not a repair reason.

Protocol `construction-feedback-revision-3` retains the equality checks and adds
conservative affine function checks with explicit assumptions: parameter domains,
initializer domains, nonnegative invariant subsystems and uniform derivative bounds.
A finite observed increment must lie inside the implied integrated rate bounds,
including the authenticated absolute measurement error and existing numerical guard. Unknown parameter signs, unsupported expressions, uncertain
conditioning or unavailable bounds must yield a visible deferred result rather
than a fabricated proof. No finite set of these checks certifies all possible
ODE/data compatibility.

Every v3 request exposes the complete assembled equations, parameter domains
and uses, observed initial conditions and scoped latent initializers. Function
overlap and unexplained fixed conversion factors receive targeted advisory feedback;
only a qualified contradiction receives a blocking conflict label. A numeric
factor such as 1000 is not itself proof of a wrong conversion when a free
coefficient can absorb its scale.

Historical artifacts stay unchanged. New reports separate deterministic
conflicts, unresolved scientific questions, numerical fitting status and prediction
scores. Reports include equations and per-case examples, not only aggregate counts.
The standalone `construction_science_report.render` view contains the construction
audit only: equation examples, a check-coverage chart, qualified conflicts with
training witnesses, private coefficients and all parameter consumers. It deliberately
omits NMSE rankings. If a parent equation audit is supplied, the child's page also
shows that unchanged parent for equation-level inspection.

## Evidence and ownership

The historical audit is reproducible from the saved
`output/revision-checks-analysis-20261022/inspection/` result receipts and
`feedback-audit.json`, plus the `9578ad7` delivery source. Parent/child aliases and
equations are in
`output/paired-fit-analysis-20261022/426ed2c33316/returned/inputs.json`.
The paired archive SHA-256 is
`426ed2c33316d6f13cc98c368161aa8d559886b751604ace4eb706e0901b949e`.

Codex owns construction feedback and the deterministic pre-fitting evidence.
Orion owns numerical qualification/recovery. See
`docs/PHASE_C_PAIRED_CONSTRUCTION_NUMERICAL_HANDOFF.md` for the
numerical handoff and the decision to defer parent–child efficacy conclusions.

## Retrospective result and next bounded test

The same new check applied to all 24 original saved models flags nine: children
07/08/09/10/14/15 and parents 03/04/07. All nine impose an upper glucose slope of
1.952028482 on zero-meal `train_011`; its recorded first-unit rise is 2.271231699.
Children 08/09 additionally impose the positive disposal slope described above.
These proofs use the complete forcing ranges and actual initializer scopes,
never fitted coefficients or numerical endpoint scores. The other 15 are deferred,
not certified scientifically consistent.

The signed-domain gate currently handles affine state/forcing laws with supported
rational parameter coefficients and direct-state observation mappings. Nonlinear
laws, unproved initializer signs, unsupported forcing interpolation and nonidentity
observation functions remain explicitly deferred. Parameter constraints in prose,
numerical starting guesses and numerical search bounds cannot strengthen a proof.
There is no silent parameter consolidation, unit conversion or scientific repair.

`audit_construction_functions.py` rechecks a sealed, positive-schema train-only
roster and produces `audit.json` and an equation-level `REPORT.html`. It excludes
fitted scores. The existing revision CLI supports v3 in a fresh root with eight
original parents and three seeds each, at most three calls per case (72 total),
using one H100 allocation. Every attempted complete patch is rechecked. Retaining
a conflicting parent cannot export it. Existing variable types and initialization
remain fixed; a necessary wider change is reported as blocked for a subsequent
milestone. Historical v1/v2 protocols and artifacts remain unchanged.

The next run is a construction diagnostic, not a parent–child NMSE comparison.
No fitting, solver rollout, promotion or automatic follow-up is authorized by its
launcher. After collection, inspect blocking witnesses, preservation of public
pathways, parameter sharing and equations before discussing fitting eligibility.
