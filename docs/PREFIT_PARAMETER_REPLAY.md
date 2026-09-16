# Saved proposal replay with inherited parameter declarations

The routed sibling v2 produced the same quadratic-plus-linear revision three
times. Each response declared the new coefficient but omitted the reused old
coefficient. Legacy binding counted that undeclared identifier as a source and
rejected it. This milestone replays explicitly selected attempt 0, without new
provider requests or editing the historical experiment.

`selected-interaction-parameter-inheritance-1` accepts an exact local or canonical
parameter name belonging to the selected interaction. It inherits the canonical
role and translates the name back to its recorded local identity before binding,
avoiding another namespace suffix and preserving compatible fitted starting
values. The complete inherited declaration must match after compilation. This
does not freeze parameter values: the child fitter may optimize them.

No fuzzy names or roles are guessed. Unknown identifiers and parameters belonging
to other interactions are not inherited. Conflicting explicit roles fail. All
ordinary source, sign, nonlinear-requirement, initialization and protected-slot
checks still apply. The original response, evidence citations and hypothesis are
retained; each inherited declaration/name translation is recorded in provenance.
The option is explicit in `rebind_interaction` and `apply_revision`; historical
campaigns retain their original strict defaults and deterministic replay.

The replay verifies sealed plans, cached requests/responses, event hashes,
historical strict decisions, training evidence, original fit and continuation.
It binds the new decision to the current source/runtime and a separate output.
One reservation per historical state/policy prevents a new output path or attempt
number from silently allocating another child. Fitting is a separate command.
It reuses the unchanged child backend/profile, compatible fitted parameters and
all causal initializer parameters. Completed fits resume exactly; interrupted
started fits retain the backend's consumed-allocation behavior.

## ACES

Use the exact published commit in a clean checkout:

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-prefit-numerical-sibling-v2 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1 PUBLISHED_COMMIT
```

Submit CPU preflight tests, a synthetic real-fit smoke, and saved-response replay
(the benchmark child is not fitted in this stage):

```bash
AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1/scripts/hpc/submit_prefit_parameter_replay_aces.sh
```

After completion, inspect the selected revision and inheritance:

```bash
jq '{status,accepted,error,saved_attempt,live_llm_calls,parameter_fitting_performed,revision:.decision.reply.revision,parameter_inheritance:.decision.provenance.parameter_inheritance,protected_slots_preserved:.decision.provenance.protected_slots_preserved}' /scratch/user/u.yx126462/phase_b/prefit-parameter-replay-v1/summary.json
```

For an accepted replay, submit the frozen-profile child fit (CPU only):

```bash
AF_REPLAY_ACTION=fit AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1/scripts/hpc/submit_prefit_parameter_replay_aces.sh
```

```bash
jq '{status,accepted,live_llm_calls,parent_training:.parent.training.normalized_mse,parent_validation:.parent.validation.normalized_mse,child_status:.child.result.status,child_training:.child.result.training.normalized_mse,child_validation:.child.result.validation.normalized_mse,child_budget_exhausted:.child.result.budget_exhausted,child_optimizer_converged:.child.result.native_optimizer_converged,retained_parameters:.child.seed.retained_parameters,fresh_parameters:.child.seed.fresh_parameters}' /scratch/user/u.yx126462/phase_b/prefit-parameter-replay-v1/summary.json
```

Both submissions are idempotent and print a job ID. Logs are under the output
directory's `logs/`. No model server or GPU is used. Replay rejection is a saved
outcome and cannot allocate fitting. A numerical failure is reported separately
from successful deterministic replay. This is not a new proposer success-rate
experiment or an equal-compute parent/child comparison; it does not certify the
proposer's scientific explanation or expose test/private data.

## Local verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_prefit_parameter_replay.py tests/test_numerical_sibling.py tests/test_sibling_fit.py
PYTHONPATH=src .venv/bin/python scripts/smoke_prefit_parameter_replay.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
bash -n scripts/hpc/submit_prefit_parameter_replay_aces.sh
```
