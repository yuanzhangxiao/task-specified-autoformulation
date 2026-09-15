# Topology-owned sign normalization and explicit repair contracts

The `topology-owned-sign-1` requirement-repair policy restores the user's intended
division of responsibility: topology selects the destination, source set and
outer sign; the interaction reply supplies the analytic function and parameters.
The runtime normalizes redundant explicit outer minus factors and records both
the raw reply and interpreted function. It does not fit a parameter.

## Meaning of a slot

A slot is one interaction in the selected topology. For a differential `x1`,
`{x2, x3; +}, {x4; -}` corresponds to two functions assembled as
`dx1/dt = k1*f1(x2,x3) - k2*f2(x4)`. An algebraic declaration instead uses `x1`
on the left. The first function may jointly depend on `x2` and `x3`; the second
may depend on `x4`. Parameters are separately declared. A function can have more
than one scientific source. The restriction is the selected interaction's source
set, not a universal one-variable limit.

The pre-fitting repair pilot freezes these interaction signatures. A proposed
`f2(x4,x3)` changes the selected interaction. Even if `x3` already occurs elsewhere
in the equation, this changes its decomposition into functions. Such a revision
belongs to interaction-structure repair; the runtime must not silently delete the
extra dependency or assume the scientific mechanism is wrong.

## Why the strict rule existed

Commit `33de9bc` (September 8) introduced the explicit positive/negative/
unrestricted sign contract and rejected `DUPLICATED_WHOLE_OUTER_SIGN` rather than
normalizing it. Its documented caution was to preserve internal signed laws and
avoid arbitrary negative-syntax rewriting. The requirement-feedback v1 pilot
inherited that behavior. The caution remains appropriate for internal arithmetic,
but rejecting a redundant outer factor conflicts with the user's sign convention.

The shared normalizer now distinguishes these cases:

| Topology sign | Reply | Interpreted function |
| --- | --- | --- |
| negative | `-x4**2` | `x4**2`, assembled with one outer subtraction |
| positive | `-k*x` | `k*x`, assembled with the selected outer addition |
| negative | `-(x**2)/tau` | `x**2/tau` |
| negative | `(-k)*(-x)` | `k*x` |
| negative | `k*(x-y)` | unchanged |
| negative | `exp(-x/tau)` | unchanged |
| negative | `(-x)**2` | unchanged |
| unrestricted | `-k*x` | unchanged |

Only explicit minus factors reached through the outer product, quotient or unary
sign are removed. Numeric magnitudes survive. Internal sums, differences, calls
and powers are opaque; for example `-(x-y)` becomes `(x-y)` but `-x+y` stays
unchanged. This is an explicit representation convention, not symbolic
equivalence or an absolute-value transform. It does not make signed functions
nonnegative. Parsing, exact source coverage, parameter domains, local scientific
obligations and whole-model requirement checks still apply after normalization.

Each accepted normalization records the policy, original expression, normalized
expression, topology sign, count of removed minus factors and structural
certificate. Failed attempts retain the raw response, actionable error and any
sign interpretation in their feedback. The same routine lives in shared staged
function code; this milestone enables it in the new requirement-repair campaign.
Historical callers retain the explicit strict policy for reproducibility.

## Feedback and preservation

Each eligible interaction now carries a function signature, exact required source
list, assembly template and sign-ownership rule. Source-mismatch feedback names
the required, actual, missing and extra sources and explains the permitted next
action. It supplies no preferred scientific functional form. A different source
set is identified as an interaction-structure revision outside this one-function
repair scope. Another eligible interaction can still be selected on retry.

Only the chosen interaction can change. Other functions, topology, target
mappings and the causal initializer plan are preserved and recompiled as before.
No assignment or derivative notation is requested in the RHS field. Shared
fitting code, benchmark prompts and data are not changed.

## Experiment

Use `configs/prefit_requirement_feedback_v2.json` in a new output root.

CPU preparation first reconstructs the same twelve construction-audit models.
It then verifies the completed v1 requirement plan with hash
`d1b3c8c2b6583b0b5ded86a593d21be427baae1ff92aec6baa946fad99057c8c`,
checks its saved request ledgers, and compares each exact saved repair response
under strict and normalized interpretation. Cases, tasks and model settings must
match; the historical files are read only. The replay writes `replay.json` under
the new root and makes no LLM call.

The supplied traces contained eight responses: two accepted, five rejected for
source mismatch and one for a duplicated outer minus. Expected replay result:
eight saved responses, two strict acceptances, three normalized acceptances, one
newly admissible response and zero acceptance regressions. This is a prediction
to verify against the complete ACES artifacts. It does not imply a 3/3 fresh-run
success rate: fresh model responses can change with feedback.

The completed v1 live run resolved the single source gap in two of three repair
seeds, used eight calls and 30,519 tokens, and preserved all 33 control episodes
per arm without calls. Both accepted replies replaced `k3*v01` with `k3*v01**2`
under the original subtractive interaction in `dx2/dt`. The unrepaired seed ended
with `-(x2**2)/tau5` in a different subtractive interaction. These are local
repair outcomes on one model, not three independent scientific tasks.

The GPU job then performs the same three-seed repair experiment using the clearer
contract and normalization. Both arms retain already-covered/unbound controls
without calls. The new report includes `repair_policy` and
`outer_sign_normalizations` alongside gaps, first-attempt repairs, costs and
preservation. Original budgets remain three calls per repair episode, so one
natural source gap gives at most nine live calls. New cache namespaces and plan
hashes keep replay and fresh-run costs separate. Resuming terminal work makes no
new calls.

This combined live engineering change tests clearer feedback plus normalization;
it does not isolate their separate live effects. Offline replay isolates the
normalization's effect on already-saved replies. Required nonlinear syntax is
still only a necessary feature, not scientific adequacy or numerical feasibility.
The single defective source model limits generalization. Fitting, trajectory
data, test data, private references and scientific judges are not used.

## ACES commands

Use the literal commit from the completion message in place of `COMMIT` below.
Every command is a single physical line and includes explicit paths.

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v2 COMMIT
AF_RESUME=0 AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v2 AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python AF_CONFIG=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v2/configs/prefit_requirement_feedback_v2.json AF_SOURCE_ROOT=/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1 AF_PRIOR_ROOT=/scratch/user/u.yx126462/phase_b/prefit-requirements-v1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-requirements-v2 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v2/scripts/hpc/submit_prefit_requirements_aces.sh
```

After preparation, read the offline replay. After both jobs, rebuild the live
summary without additional provider calls:

```bash
jq 'del(.rows)' /scratch/user/u.yx126462/phase_b/prefit-requirements-v2/replay.json
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v2/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-requirements-v2/scripts/prefit_requirement_campaign.py summary --root /scratch/user/u.yx126462/phase_b/prefit-requirements-v2
jq 'select(.attempts | length > 0) | {episode: input_filename, stop_reason, attempts, selected_function: .final.selected_function, normalizations: .final.outer_sign_normalizations}' /scratch/user/u.yx126462/phase_b/prefit-requirements-v2/results/*/state.json
```

For interrupted, nonterminal work, repeat the submission command with
`AF_RESUME=1` after the previous jobs stop. Do not resume the old v1 root with
this changed policy or new code.

## Verification

The focused suite covers outer-product/quotient signs, meaningful internal signs,
unrestricted laws, parser failures, exact assembly, source rejection, initializer
preservation, strict replay, cached response provenance, zero-cost controls,
normalization accounting, CLI replay and deterministic resume. CPU preparation
runs these checks and both requirement smokes before launching the live worker.
All 98 focused tests passed, as did the new sign-normalization smoke, legacy
requirement smoke, Ruff and shell syntax checks. The shared checkout was held
fixed for one full regression run: 1,753 passed and three optional Torch tests
skipped in 647.48 seconds. Astra's separate recovery-bootstrap changes were
verified in the same run and are committed separately before this milestone.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_topology_owned_sign.py tests/test_staged_sign_contract.py tests/test_requirement_sign_policy.py tests/test_requirement_feedback.py tests/test_prefit_requirements.py tests/test_prefit_requirements_submission.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_sign_normalization.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_requirements.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```
