# Cross-stage repair and matched proposer comparison

23 October 2026. Construction milestone, following the returned directional-family
retest. This does not rank the old parent/child fits or modify Orion's numerical
work, benchmark data, public task specifications, or held-out evaluation policy.

## Decision

Use one broader repair contract for both GPT-OSS-20B and GPT-OSS-120B. This is not a
narrow-versus-wide ablation. A necessary repair may change existing equations and
dependencies, introduce a latent state, revise eligible initial conditions, or
introduce/remove algebraic processes. The runtime validates the complete revised
model before adoption; rejected patches leave the starting candidate unchanged.

The response has an explicit decision: `revise`, `retain`, or
`requires_wider_revision`. The last outcome means a necessary change is still
outside the supported contract. It stops the episode without exporting a fitting
packet or clearing the conflict. It is not an invitation to acknowledge a conflict
and accept the unchanged model. Retained models undergo the same checks.

## Supported edits and protected information

- Replace the complete RHS of an existing differential or algebraic definition.
  Parsed dependencies define the resulting topology. Unambiguous matching LHS
  notation and source-list metadata normalize without another generation call.
- Add up to eight latent differential states and sixteen algebraic processes per
  patch. New states require a complete typed initialization choice.
- Replace eligible latent initializers with a shared value, a trajectory-specific
  root, or an acyclic dependent map of allowed public/state initial values. The
  existing observation-window policy is an experiment setting, not an LLM choice.
  Provider replies use typed alias lists for dependent maps; the runtime converts
  them to the existing fitter's map representation. This avoids a strict JSON
  schema accidentally forbidding nonempty dictionaries.
- Remove explicitly named unprotected states/processes only when all references
  and required public pathways remain valid. Parameters left unused by the new
  complete equations are removed only under the existing role/constraint rules.
- Add parameters with declared roles/domains; preserve intended shared uses.

Public targets, output mappings, authenticated observed initial identities,
constraints, data-access rules and existing parameter roles/domains remain fixed.
Same-name conversion of an existing differential state into an algebraic process
is not supported by this patch contract; it is reported as requiring a wider
revision. New distinct names and complete consumer rewiring are supported where
they respect the protected public identities.

Compilation and initializer-DAG checks are followed by the existing qualified
function, matched-trajectory, initial-family and affine-family data checks.
An unrelated additional latent state cannot erase a contradiction in an unchanged
closed subsystem. Unsupported proofs are explicitly deferred, not marked passed.
There is no claim that clearing these finite checks establishes all scientific
requirements, numerical fitting adequacy, or a lower forecast NMSE.

## Feedback and review

The model view displays the editable latent initializers and the actual scope.
Feedback distinguishes model-implied bounds from observed changes, identifies
proved zero forcing channels, explains unchanged trajectory identities, and gives
concrete duplicate-declaration guidance. Repetition detection includes structural
and initializer edits; changing explanations or advisory source lists does not
make the same rejected patch new.

Witnesses are retained separately for every affected target and qualified
property. Stronger presentations may evaluate the same already-certified affine
identity at additional recorded times, keeping its exact trajectory triple,
weights/premises, observation bounds and numerical guard. The original certificate
is retained. This changes presentation, not scientific acceptance thresholds.

`preview` produces exact initial request JSON for all six starting models and a
readable `PROMPTS.html`, including an explicitly hypothetical repair-feedback
example checked against a real saved model. No generation is needed to review it.

## Matched experiment

Both arms use the same six saved T2-easy candidates, selected for defect coverage
in the previous experiment, not for their NMSE. This is a targeted repair study,
not a fresh success estimate across six benchmarks.

| Setting | Both arms |
| --- | --- |
| Starting candidates | P03-r0, P06-r2, P01-r0, P05-r2, P00-r1, P06-r1 |
| Fresh proposal seeds | 0, 1, 2 per candidate |
| Episodes | 18 |
| Maximum calls | 3 per episode; 54 per arm |
| Context / output ceiling | 32,768 / 8,192 tokens |
| Charged-token ceiling | 98,304 per episode; 1,769,472 per arm |
| Temperature / reasoning | 0.2 / low |
| Numerical fitting / test access | None |

20B revision: `6cee5e81ee83917806bbde320786a8fb61efebee`.
120B revision: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
ACES resources differ: one H100/64 GiB for 20B, two H100s/128 GiB for 120B.
The algorithmic allowances match; GPU-hours and realized token use need not.
Both allocations request two hours with a 90-minute generation window. If an arm
cannot finish, report coverage and resume its cached allowance rather than
comparing an incomplete arm as if it had the same realized opportunities.

Plans share a `comparison_sha256` over scientific inputs, code, prompt/schema,
seeds and generation policy, excluding model identity. Namespaces and output roots
are separate. Same seed does not imply identical sampling across different models
or serving hardware. Record actual model, CUDA, driver, vLLM and resource details.

Reports separate revised/retained, unresolved wider revision, rejected, unavailable
and pending outcomes; preserve all raw requests/replies, equations, checked and
deferred evidence, costs and fitting handoffs. No automatic promotion or fitting.
An oversized request is recorded as `preflight_blocked` without generation; later
cases may continue. The arm remains incomplete, and its context is not silently
truncated. Diagnostic evidence is displayed once rather than duplicated in guidance.

## Running on ACES

Use the delivered archive in group scratch, not the nearly full home/user-scratch
inode quotas. The bundle includes the pinned saved-source archive and Harmony
vocabulary. Launchers do not download model weights or install packages.

From the extracted bundle, the commands are:

```bash
bash scripts/hpc/start_structural_revision_aces.sh run 20b
bash scripts/hpc/start_structural_revision_aces.sh run 120b
```

Use `status`, `report`, `preview`, or `inspect` instead of `run` with the same
model-size argument. Default roots are
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-structural-revision-20b-v1`
and the analogous `120b-v1` directory. `inspect` creates `inspection.tar.gz` in
each root. The user-facing delivery commands supply absolute paths.

The launcher checks the scientific Python stack, complete pinned model shards and
tokenizer, image and offline Harmony before submission. Set `AF_HF_HOME`,
`AF_VLLM_IMAGE` or `AF_PYTHON` only to verified existing assets if needed.
`AF_STRUCTURAL_REVISION_ROOT` explicitly overrides a root; unrelated older
`AF_OUTPUT_ROOT` values are ignored. Missing 120B assets stop preflight before
GPU submission. An ambiguous Slurm reply is retained and never auto-resubmitted;
inspect the queue before reconciling it. Source/runtime drift blocks resume.

## Numerical qualification and search architecture

Claude's current LLM-SR implementation evaluates **negative training derivative
MSE**, using within-trajectory `numpy.gradient`, ten optimized coefficients and
up to 30 seconds of BFGS per program. Validation free-rollout NMSE does not select
the program. Its planned 10,000 programs are **per target**: 22 searches and
220,000 planned programs, with four samples per request. These are planned budgets,
not independently verified realized counts. See the separate baseline checkout's
`docs/PHASE_C_LLM_SR.md` and `src/autoformalism/rebuttal/llm_sr_driver.py`.

Its performance plausibly combines cheap dense feedback, coefficient fitting,
population selection, history-dependent expressivity, broad sampling and 120B.
Those factors have not been isolated experimentally. History programs and their
approximate ODE transcriptions must not be treated as numerically identical.

"Qualified fitting" means adequate evidence for a stated decision, not proof of
a global optimum:

1. **Score correctness:** saved-parameter predictions reproduce with an independent
   original-equation solver; initialization, forcing, weighting and evaluation
   windows match. Solver discrepancies are below a predeclared useful tolerance.
2. **Recovery competence:** on reviewed controls with a known useful parameter
   vector, fitting from default/perturbed starts recovers useful prediction error
   within the proposed search allowance. Identical parameter recovery is not
   required for nonidentifiable models.
3. **Ranking stability:** extra starts or a larger allowance do not routinely
   reverse score differences large enough to drive search. Unresolved candidates
   retain that label instead of being declared structurally inadequate.

Orion's current LLM-SR controls first replay supplied transcription coefficients
with Radau and tighter-tolerance DOP853. Their `refit_ready=false` is intentional:
coefficient domains and simplified branches need review before refitting. Successful
replay establishes a useful point, not the optimizer's ability to find it. Check
equal-trajectory weighting versus concatenated residual weighting before comparing
absolute scores; they agree only when the relevant trajectory sizes/weights agree.

The next search architecture should be a bounded population with shallow repair
and adaptive fitting: independently generate diverse valid structures; apply
cheap deterministic checks; fit each eligible model with a modest common budget;
retain credible low-error candidates and diverse unresolved alternatives; allocate
extra fitting or repair where the evidence justifies it. Preserve the best valid
incumbent. Compare breadth, depth and this hybrid by verified error and mechanism
coverage versus cumulative LLM tokens and fitting CPU, not by rounds alone.
Construction breadth can increase before numerical qualification, but aggressive
numerical ranking/pruning should wait for the decision-specific evidence above.

## Verification and remaining limits

The focused structural, feedback, campaign, launcher and existing revision suites
passed 219 tests. Additional provider-schema, alias and preflight coverage passed
with the final structural/campaign suite. Changed files pass Ruff; both shell
launchers pass Bash syntax checks. The repository-wide pytest run was started but
interrupted before completion; its two observed legacy integration failures both
passed when rerun alone. Repository-wide Ruff retains 37 findings in the unrelated
`analysis/claude/make_tables.py` file. This is not a claim of a fully green repository.

No live GPU requests, optimizer calls, solver rollouts, or test-data reads were made
for this implementation. Offline schema validation cannot establish live vLLM
grammar/120B startup compatibility; the pinned asset and serving preflights must
still pass on ACES. Neither an untested wider revision nor a deferred scientific
check is counted as a scientifically consistent model.
