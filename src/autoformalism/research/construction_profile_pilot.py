"""Isolated bounded multioutput profiling and fixed-endpoint precision pilot.

Parent artifacts and costs are immutable. Precision-only tasks cannot optimize,
select or promote a point; the two model-six arms share an optimization ceiling.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import poor_fit, prefix_execution, recovery_numerics
from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_multioutput_profile as profiling
from autoformalism.fitting import prefix_precision_probe as precision
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_local_recovery as history
from autoformalism.research import construction_precision_reduction as previous
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-profile-pilot-1"
INPUT_PROTOCOL = "construction-profile-pilot-inputs-1"
Digest = previous.Digest
Nonnegative = previous.previous.Nonnegative
_runtime = previous._runtime


class Policy(previous.Policy):
    """The two optimization phases sum to the unchanged joint-arm budget."""

    fixed_inner_seconds: Nonnegative = Field(default=120, gt=0, le=120)
    precision_seconds: Nonnegative = Field(default=600, gt=0, le=600)

    @model_validator(mode="after")
    def split_budget(self):
        if self.fixed_inner_seconds >= self.search_seconds:
            raise ValueError("fixed inner budget must be smaller than search budget")
        return self


class SourceModel(previous.previous.SourceModel):
    """Bind the latest full endpoint and the exact old diagnostic coordinates."""

    precision_mode: Literal["guarded", "multiprecision"]
    fixed_coordinate_names: list[str]
    fixed_parameters: dict[str, float]
    fixed_vector: list[float]
    probe_indices: list[int] = Field(min_length=1, max_length=6)
    probe_steps: dict[str, list[float]]


class Inputs(StrictSchema):
    protocol: Literal["construction-profile-pilot-inputs-1"]
    source_plan_sha256: Digest
    source_inputs_sha256: Digest
    historical_inputs: dict
    models: list[SourceModel]
    parent_tasks: list[previous.previous.ParentTask]
    old_allowances_renewed: Literal[False]
    validation_exported: Literal[False]
    test_data_opened: Literal[False]


def _probe_indices(check, dimension):
    """Freeze failed columns first, then largest saved projected uncertainty."""
    priority = set(check.get("failed_indices", [])) | set(
        check.get("inconclusive_indices", [])
    )
    errors = {
        d["index"]: float(d.get("projected_gradient_error_bound") or 0)
        for d in check.get("directions", [])
    }
    return sorted(
        range(dimension), key=lambda i: (i not in priority, -errors.get(i, 0), i)
    )[:6]


def export_inputs(source: Path, destination: Path) -> dict:
    """Authenticate every parent terminal before selecting the prescribed tasks."""
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("export must not write into source campaign")
    plan, data = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    models, parents = [], []
    for task in plan["tasks"]:
        folder = previous._folder(source, task)
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        operations = previous._operations(folder, identity, policy)
        result = previous._assemble(
            identity, data["models"][task["model"]], operations, terminal=True
        )
        previous._check_start(folder, result)
        if read_seal(folder / "result.json") != result:
            raise ValueError("parent terminal differs from operation receipts")
        endpoint = result["retained"]
        if endpoint is None or not endpoint["original_equations_verified"]:
            raise ValueError("every parent must retain a verified endpoint")
        diagnosis = operations.get("retained_diagnosis", {}).get("value") or {}
        q = operations.get("qualification", {}).get("value") or {}
        names = list(endpoint["effective_parameters"])
        # Stored mappings are canonically sorted; actual coordinate order is explicit.
        audit = diagnosis.get("recovery_audit") or q.get("recovery_audit") or {}
        names = audit.get("coordinate_reduction", {}).get("effective_names", names)
        parameters = endpoint["effective_parameters"]
        vector = endpoint["reduced_vector"]
        if [parameters[n] for n in names] != vector:
            raise ValueError("parent coordinate order differs from saved vector")
        original = data["training_inputs"]["models"][task["model"]]
        check = diagnosis.get("derivative_check") or {}
        indices = _probe_indices(check, len(names))
        steps = {
            str(direction["index"]): direction["steps"]
            for direction in check.get("directions", [])
            if direction["index"] in indices and len(direction.get("steps", [])) == 2
        }
        models.append(
            {
                "model": task["model"],
                "task_id": task["task_id"],
                "parent_task_index": task["index"],
                "problem_sha256": public.content_sha256(original["problem"]),
                "endpoint": history._endpoint(endpoint, result["retained_origin"]),
                "precision_mode": q.get("precision_mode", "guarded"),
                "fixed_coordinate_names": names,
                "fixed_parameters": parameters,
                "fixed_vector": vector,
                "probe_indices": indices,
                "probe_steps": steps,
            }
        )
        parents.append(
            {
                **{k: task[k] for k in ("index", "model", "task_id")},
                "kind": "continuation",
                "task_sha256": public.content_sha256(task),
                "terminal_sha256": public.content_sha256(result),
                "operations": history._receipts(folder, operations),
                "cost": {
                    k: v for k, v in result["cost"].items() if k in previous.COST_KEYS
                },
            }
        )
    value = {
        "protocol": INPUT_PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs_sha256": public.content_sha256(data),
        "historical_inputs": data,
        "models": models,
        "parent_tasks": parents,
        "old_allowances_renewed": False,
        "validation_exported": False,
        "test_data_opened": False,
    }
    _validate(value, Policy(population=policy.population))
    seal(destination, value)
    return {
        "inputs_sha256": public.content_sha256(value),
        "models": len(models),
        "authenticated_parent_tasks": len(parents),
        "old_allowances_renewed": False,
    }


def _validate(data, policy):
    parsed = Inputs.model_validate(data)
    previous._validate(
        parsed.historical_inputs, previous.Policy(population=policy.population)
    )
    if public.content_sha256(parsed.historical_inputs) != parsed.source_inputs_sha256:
        raise ValueError("parent input digest differs")
    originals = parsed.historical_inputs["training_inputs"]["models"]
    if len(parsed.models) != len(originals) or len(parsed.parent_tasks) != len(
        originals
    ):
        raise ValueError("all eight parent endpoints and costs required")
    for i, (model, parent, original) in enumerate(
        zip(parsed.models, parsed.parent_tasks, originals, strict=True)
    ):
        if (
            (model.model, model.parent_task_index, parent.index, parent.model)
            != (i, i, i, i)
            or model.task_id != parent.task_id
            or model.task_id != original["task_id"]
        ):
            raise ValueError("parent model identity/order differs")
        if model.problem_sha256 != public.content_sha256(original["problem"]):
            raise ValueError("model problem digest differs")
        history._validate_endpoint(model.endpoint, original)
        previous.previous._validate_cost(parent.cost)
        if set(parent.operations) - set(previous._allowances(previous.Policy())):
            raise ValueError("unknown parent operation")
        if (
            model.endpoint.origin not in parent.operations
            or parent.operations[model.endpoint.origin].status != "complete"
            or not parent.operations[model.endpoint.origin].receipt_sha256
            or not model.endpoint.original_equations_verified
        ):
            raise ValueError("unverified parent endpoint")
        names = model.fixed_coordinate_names
        if (
            len(set(names)) != len(names)
            or set(names) != set(model.fixed_parameters)
            or [model.fixed_parameters[n] for n in names] != model.fixed_vector
            or not np.isfinite(model.fixed_vector).all()
        ):
            raise ValueError("fixed coordinate/vector mismatch")
        if len(set(model.probe_indices)) != len(model.probe_indices) or any(
            i < 0 or i >= len(names) for i in model.probe_indices
        ):
            raise ValueError("invalid fixed precision columns")
        for index, pair in model.probe_steps.items():
            if (
                index not in {str(i) for i in model.probe_indices}
                or len(pair) != 2
                or not np.isfinite(pair).all()
                or min(pair) <= 0
                or not np.isclose(pair[0], 2 * pair[1], rtol=1e-12, atol=0)
            ):
                raise ValueError("invalid saved local precision pair")


def _tasks(data, policy):
    selected, studies = (
        (6, (0, 3, 4, 5, 6, 7)) if policy.population == "returned-eight" else (0, (0,))
    )
    rows = [(selected, "joint"), (selected, "profiled")] + [
        (i, "fixed_precision") for i in studies
    ]
    return [
        {"index": i, "model": m, "task_id": data["models"][m]["task_id"], "arm": arm}
        for i, (m, arm) in enumerate(rows)
    ]


def _contract():
    return {
        "profile_policy": public._jsonable(profiling.policy_receipt()),
        "precision_policy": public._jsonable(precision.policy_receipt()),
        "qualification_policy": public._jsonable(previous._adequacy_policy()),
        "diagnostic_policy": public._jsonable(
            previous._adequacy_policy(diagnostic=True)
        ),
        "verification_tolerances": [[1e-11, 1e-13]],
        "precision_tasks": "fixed endpoint; no search, selection, or promotion",
        "profile_budget": (
            "fixed inner plus outer qualification/search equals joint search allowance"
        ),
        "historical_verification_reused": False,
        "profile_diagnostic_scope": "full aggregated coordinates",
        "maximum_projection_failure_records": 8,
    }


def prepare(root, data, policy):
    """Schema-only preparation; no optimizer or numerical model construction."""
    _validate(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "numerical_contract": _contract(),
        "source_sha256": public._source_identity(),
        "runtime": _runtime(),
        "tasks": _tasks(data, policy),
        "old_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root, *, runtime=True):
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    p = Policy.model_validate(plan["policy"])
    _validate(data, p)
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, p)
        or plan["numerical_contract"] != _contract()
        or plan["llm_calls"] != 0
        or any(
            plan[k] is not False
            for k in (
                "old_allowances_renewed",
                "validation_data_opened",
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("profile pilot plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != _runtime()
    ):
        raise ValueError("profile pilot source/runtime differs")
    return plan, data


_folder = previous._folder
_verified = previous._verified


def _allowances(policy, task):
    if task["arm"] == "fixed_precision":
        return {"fixed_precision": policy.precision_seconds + 90}
    common = {
        "qualification": policy.diagnostic_seconds + 90,
        "incumbent_verification": policy.verification_seconds + 90,
    }
    if task["arm"] == "profiled":
        common |= {
            "fixed_inner": policy.fixed_inner_seconds + 60,
            "fixed_inner_verification": policy.verification_seconds + 90,
        }
    common |= {
        "continuation_search": policy.search_seconds
        - (policy.fixed_inner_seconds if task["arm"] == "profiled" else 0)
        + 150,
        "continuation_verification": policy.verification_seconds + 90,
        "retained_diagnosis": policy.diagnostic_seconds + 90,
    }
    return common


def _operations(folder, identity, policy):
    allowed = _allowances(policy, identity["task"])
    if any(
        p.is_dir() and p.name not in allowed for p in (folder / "operations").glob("*")
    ):
        raise ValueError("unknown pilot operation")
    result = {}
    for name, seconds in allowed.items():
        location = folder / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (location / "result.json").exists():
            result[name] = journal._read_operation(location, bound)
        elif (location / "started.json").exists():
            if read_seal(location / "started.json") != bound:
                raise ValueError("pilot operation reservation differs")
            result[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return result


def _assemble(identity, source, operations, *, terminal=False):
    task = identity["task"]
    base = {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "source_incumbent": source["endpoint"],
        "operations": operations,
        "cost": journal._cost(operations),
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    if task["arm"] == "fixed_precision":
        op = operations.get("fixed_precision", {})
        value = op.get("value") or op.get("observed_progress") or {}
        if value and value.get("source_parameters_sha256") != public.content_sha256(
            source["endpoint"]["parameters"]
        ):
            raise ValueError("precision study endpoint binding differs")
        return base | {
            "precision_study": value,
            "search_started": False,
            "selection_performed": False,
            "retained": None,
        }
    selected = _verified(operations.get("incumbent_verification", {}))
    retained, origin = selected, "incumbent_verification" if selected else None
    for name in ("fixed_inner_verification", "continuation_verification"):
        candidate = _verified(operations.get(name, {}))
        if candidate and (
            retained is None or recovery_numerics.improves(candidate, retained)
        ):
            retained, origin = candidate, name
    qop = operations.get("qualification", {})
    q = qop.get("value") or {}
    qualified = (
        qop.get("status") == "complete"
        and q.get("status", "complete") == "complete"
        and bool(q.get("passed"))
    )
    sop = operations.get("continuation_search", {})
    fit = previous.first._search_checkpoint(sop)
    calls = fit.get("search_calls_started", fit.get("calls_started", 0))
    dop = operations.get("retained_diagnosis", {})
    diag = dop.get("value") or {}
    if diag and (
        retained is None
        or diag.get("diagnosed_parameters_sha256")
        != public.content_sha256(retained["parameters"])
        or diag.get("verification_sha256") != public.content_sha256(retained)
    ):
        raise ValueError("diagnosis differs from exact retained endpoint")
    return base | {
        "selected_start": selected,
        "retained": retained,
        "retained_origin": origin,
        "qualification_passed": qualified,
        "search_started": bool(calls),
        "search_stop_reason": previous._termination(fit) if sop else "not_started",
        "fitting_diagnosis": diag.get("assessment")
        if dop.get("status") == "complete"
        else None,
        "optimization_status": (diag.get("assessment") or {}).get(
            "classification", "optimization_unresolved"
        ),
        "diagnostic_cost": _diagnostic_cost(operations),
        "selection_performed": retained is not None,
        "historical_verification_is_current": False,
    }


def _diagnostic_cost(operations):
    """Record the outer derivative gate without charging its search stage twice."""
    accounted = dict(operations)
    for name, operation in operations.items():
        if name != "qualification" and not name.endswith("_diagnosis"):
            continue
        value = operation.get("value") or operation.get("observed_progress") or {}
        if not (
            "derivative_passes" in value or "full_residual_precision_audit" in value
        ):
            continue  # Historical sealed reports retain their original calculation.
        records = list(value.get("derivative_passes", []))
        if "full_residual_precision_audit" in value and not any(
            record.get("backend") == "full-residual-multiprecision"
            for record in records
        ):
            # Interrupted MP work may have saved counters before producing a
            # DerivativeCheck. A completed MP check already contains these calls.
            records.append(
                {
                    "backend": "full-residual-multiprecision",
                    "status": "incomplete",
                    "derivative_check": value["full_residual_precision_audit"],
                }
            )
        checks = [record.get("derivative_check") or {} for record in records]
        complete = bool(records) and all(
            check
            and record.get("status", "complete") == "complete"
            and check.get("calls_started", 0) == check.get("calls_completed", 0)
            for record, check in zip(records, checks, strict=True)
        )
        normalized = value | {
            "passes": records,
            "status": "complete"
            if complete and value.get("status", "complete") == "complete"
            else "incomplete",
            "derivative_check": {
                key: sum(check.get(key, 0) for check in checks)
                for key in ("calls_started", "calls_completed")
            },
        }
        accounted[name] = operation | {"value": normalized}
    cost = previous.previous._diagnostic_cost(accounted)
    operation = operations.get("continuation_search", {})
    value = operation.get("value") or operation.get("observed_progress") or {}
    check = value.get("profile_derivative_check") or {}
    cost["profile_derivative_calls_started_observed"] = check.get("calls_started", 0)
    cost["profile_derivative_calls_completed_observed"] = check.get(
        "calls_completed", 0
    )
    cost["profile_derivative_seconds_observed"] = check.get("elapsed_seconds", 0)
    cost["profile_derivative_charge"] = "included in the outer optimization allowance"
    return cost


def _qualification(problem, training, settings, anchor, seconds, save):
    """Qualify the new aggregate chart, with one in-budget precision escalation."""
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {"passed": False, "precision_mode": "guarded", "passes": []}
    interrupted = None
    try:
        for mode in ("guarded", "multiprecision"):
            budget.check()
            original, oracle, _ = profiling.make_oracle(
                problem, training, settings, anchor, precision_mode=mode
            )
            result["precision_mode"] = mode
            point = oracle.reduction.reduce(original.vector(anchor))
            end = (
                min(budget.deadline(), monotonic() + seconds * 0.45)
                if mode == "guarded"
                else budget.deadline()
            )
            record = {"precision_mode": mode, "status": "started"}
            result["passes"].append(record)
            try:
                before, _ = budget.numerical(
                    lambda d, original=original, end=end: original.evaluate(
                        original.vector(anchor), min(d, end), jacobian=False
                    )
                )
                after, _ = budget.numerical(
                    lambda d, oracle=oracle, point=point, end=end: oracle.evaluate(
                        point, min(d, end), jacobian=False
                    )
                )
                gap = float(np.max(np.abs(before - after), initial=0))
                check = budget.numerical(
                    lambda d, oracle=oracle, point=point, end=end: recovery._derivative(
                        oracle, point, min(d, end), previous._adequacy_policy()
                    )
                )
                record.update(
                    status="complete",
                    derivative_check=public._jsonable(check),
                    lift_prediction_discrepancy=gap,
                    passed=bool(
                        np.isfinite(gap) and gap <= 1e-7 and check.search_qualified
                    ),
                )
                result["passed"] = record["passed"]
            except (TimeoutError, recovery_numerics.NumericalInvalidTrial) as error:
                record.update(
                    status="incomplete",
                    error=f"{type(error).__name__}: {error}",
                    passed=False,
                )
            finally:
                record.update(
                    affine_execution=deepcopy(original.affine_audit),
                    recovery_audit=deepcopy(oracle.recovery_audit),
                )
            budget.callback(
                lambda: save(deepcopy(result) | {"timing": budget.snapshot()})
            )
            if result["passed"]:
                break
    except Exception as error:
        numerical._invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        numerical._invalidate(result, error, "qualification")
    return numerical._finish(result, budget, save, "qualification", interrupted)


def _fixed_inner(oracle, profile, start, seconds, save):
    """Optimize the bounded linear block once at the frozen dynamics."""
    budget = prefix_execution._budget(seconds, seconds + 45)
    result, interrupted = {"best": None, "inner_evaluations_started": 0}, None
    try:
        result["inner_evaluations_started"] = 1
        point, residual, _, audit = budget.numerical(
            lambda d: profile.evaluate(oracle.vector(start), d, jacobian=False)
        )
        budget.check()
        loss = float(np.mean(residual**2))
        if not np.isfinite(loss):
            raise recovery_numerics.NumericalInvalidTrial("nonfinite fixed-inner loss")
        result.update(
            best={"parameters": oracle.parameters(point), "training_nmse": loss},
            projection_audit=public._jsonable(audit),
        )
    except Exception as error:
        _record_projection_failure(result, error, "fixed_inner")
        numerical._invalidate(result, error, "search")
    except BaseException as error:
        interrupted = error
        numerical._invalidate(result, error, "search")
    result["profile_audit"] = deepcopy(profile.audit)
    return numerical._finish(result, budget, save, "search", interrupted)


def _record_projection_failure(result, error, phase):
    """Keep bounded typed rank, KKT and active-face explanations for failed trials."""
    if not isinstance(error, profiling.ProfileUnavailable):
        return
    result["projection_failures_observed"] = (
        result.get("projection_failures_observed", 0) + 1
    )
    records = result.setdefault("projection_failures", [])
    if len(records) < 8:
        records.append(
            {
                "phase": phase,
                "error": str(error),
                "audit": public._jsonable(error.audit),
            }
        )


class _ProfileView:
    """Expose only the outer parameters for local derivative qualification."""

    def __init__(self, oracle, profile, point, on_failure=None):
        self.oracle, self.profile, self.anchor = oracle, profile, point.copy()
        self.on_failure = on_failure
        self.outer = list(profile.outer)
        self.units, self.lower, self.upper = (
            oracle.units[self.outer],
            oracle.lower[self.outer],
            oracle.upper[self.outer],
        )

    def local_coordinate_scales(self, point):
        full = self.anchor.copy()
        full[self.outer] = point
        return self.oracle.local_coordinate_scales(full)[self.outer]

    def evaluate(self, point, deadline, *, jacobian=True):
        full = self.anchor.copy()
        full[self.outer] = point
        try:
            _, residual, derivative, _ = self.profile.evaluate(
                full, deadline, jacobian=jacobian
            )
        except profiling.ProfileUnavailable as error:
            if self.on_failure is not None:
                self.on_failure(error)
            raise
        return residual, derivative


def _profile_search(
    oracle, profile, start, seconds, calls, save, *, preserve_full_history=False
):
    """Charge derivative qualification and outer optimization to one allowance."""
    budget = prefix_execution._budget(seconds, seconds + 120)
    state = {
        "best": None,
        "calls": 0,
        "search_calls_started": 0,
        "search_calls_completed": 0,
        "search_calls_failed": 0,
        "stop_reason": "not_started",
    }
    interrupted = None

    class TimedProfile:
        outer, inner = profile.outer, profile.inner

        def evaluate(self, point, deadline):
            state["search_calls_started"] += 1
            try:
                value = budget.numerical(
                    lambda d: profile.evaluate(point, min(d, deadline))
                )
                budget.check()
            except BaseException as error:
                state["search_calls_failed"] += 1
                _record_projection_failure(state, error, "outer_search")
                raise
            state["search_calls_completed"] += 1
            return value

    def snapshot(value):
        state.update(value)
        if preserve_full_history:
            state["search_calls_attempted"] = state["calls"]
        budget.callback(lambda: save(deepcopy(state) | {"timing": budget.snapshot()}))

    try:
        full = oracle.vector(start)
        view = _ProfileView(
            oracle,
            profile,
            full,
            lambda error: _record_projection_failure(state, error, "outer_derivative"),
        )
        check = budget.numerical(
            lambda d: recovery._derivative(
                view, full[list(profile.outer)], d, previous._adequacy_policy()
            )
        )
        state["profile_derivative_check"] = public._jsonable(check)
        snapshot({})
        if check.search_qualified:
            result = recovery_numerics.search(
                oracle,
                start,
                budget.wall + seconds + 120,
                snapshot,
                calls=calls,
                target=1e-12,
                profile=TimedProfile(),
                telemetry=True,
                record_steps=True,
                record_full_steps=preserve_full_history,
                optimizer_scaling="guarded_relative",
                objective_multiplier=1.0,
                poor_fit=poor_fit.Policy(),
                reject_invalid_trials=True,
            )
            state.update(result)
        else:
            state["stop_reason"] = "profile_derivative_qualification_not_passed"
    except Exception as error:
        numerical._invalidate(state, error, "search")
    except BaseException as error:
        interrupted = error
        numerical._invalidate(state, error, "search")
    state["profile_audit"] = deepcopy(profile.audit)
    if preserve_full_history:
        state["search_calls_attempted"] = state["calls"]
    return numerical._finish(state, budget, save, "search", interrupted)


def run_task(root, index):
    """Resume durable operations without resetting any consumed allowance."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown pilot task")
    task = plan["tasks"][index]
    source = data["models"][task["model"]]
    model = data["historical_inputs"]["training_inputs"]["models"][task["model"]]
    problem = PrefixFitProblem.model_validate(model["problem"])
    training = PublicSplit.model_validate(
        data["historical_inputs"]["training_inputs"]["training"]
    )
    settings = kernel.NumericalPolicy.model_validate(
        data["historical_inputs"]["training_inputs"]["numerical_policy"]
    )
    p = Policy.model_validate(plan["policy"])
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder, anchor = _folder(root, task), source["endpoint"]["parameters"]
    allowed = _allowances(p, task)

    def assemble(terminal=False):
        return _assemble(
            identity, source, _operations(folder, identity, p), terminal=terminal
        )

    def operate(name, work, *, mode="guarded", raw=False, fixed=False):
        def call(save):
            end = monotonic() + allowed[name]

            def check():
                if monotonic() >= end:
                    raise TimeoutError("inclusive profile pilot allowance exhausted")

            if raw:
                original = oracle = profile = None
            elif fixed:
                original, oracle = recovery.make_oracle(
                    problem,
                    training,
                    settings,
                    anchor,
                    precision_mode=mode,
                    structural_reduction=True,
                )
                profile = None
            else:
                original, oracle, profile = profiling.make_oracle(
                    problem, training, settings, anchor, precision_mode=mode
                )
            check()

            def audit(value):
                return (
                    value
                    if raw
                    else value
                    | {
                        "affine_execution": deepcopy(original.affine_audit),
                        "recovery_audit": deepcopy(oracle.recovery_audit),
                    }
                )

            def checked_save(value):
                save(audit(value))
                check()

            value = work(original, oracle, profile, checked_save)
            check()
            return audit(value)

        return journal._operation(folder, identity, name, allowed[name], call)

    def check_terminal():
        value = assemble(terminal=True)
        if read_seal(folder / "result.json") != value:
            raise ValueError("pilot terminal differs from receipts")
        return value

    with public._lock(folder):
        if (folder / "result.json").exists():
            return check_terminal()
        if task["arm"] == "fixed_precision":

            def study(original, oracle, profile, save):
                params = source["fixed_parameters"]
                if list(oracle.names) != source[
                    "fixed_coordinate_names"
                ] or not np.array_equal(oracle.vector(params), source["fixed_vector"]):
                    raise ValueError("frozen precision coordinate identity differs")
                if numerical.to_original(original, oracle, params) != anchor:
                    raise ValueError(
                        "frozen precision lift differs from original endpoint"
                    )
                binding = {"source_parameters_sha256": public.content_sha256(anchor)}
                result = precision.compare(
                    oracle,
                    params,
                    p.precision_seconds,
                    lambda v: save(v | binding),
                    indices=tuple(source["probe_indices"]),
                    steps={
                        int(i): tuple(pair) for i, pair in source["probe_steps"].items()
                    },
                )
                return result | binding

            operate("fixed_precision", study, mode=source["precision_mode"], fixed=True)
        else:
            qualified = operate(
                "qualification",
                lambda o, a, f, s: _qualification(
                    problem, training, settings, anchor, p.diagnostic_seconds, s
                ),
                raw=True,
            )
            q = qualified.get("value") or qualified.get("observed_progress") or {}
            mode = q.get("precision_mode", "guarded")
            if mode not in {"guarded", "multiprecision"}:
                raise ValueError("unknown qualification precision mode")
            operate(
                "incumbent_verification",
                lambda o, a, f, s: numerical.verify_endpoint_ladder(
                    o,
                    a,
                    numerical.to_effective(o, a, anchor),
                    p.verification_seconds,
                    s,
                    tolerances=((1e-11, 1e-13),),
                ),
                mode=mode,
            )
            current = assemble()
            selected = current["selected_start"]
            if selected is not None and task["arm"] == "profiled":
                # Derivative failure does not invalidate a fixed-dynamics convex solve.
                inner = operate(
                    "fixed_inner",
                    lambda o, a, f, s: _fixed_inner(
                        a,
                        f,
                        previous._verified_coordinates(o, a, selected),
                        p.fixed_inner_seconds,
                        s,
                    ),
                    mode=mode,
                )
                fixed = (
                    inner.get("value") or inner.get("observed_progress") or {}
                ).get("best")
                if fixed:
                    operate(
                        "fixed_inner_verification",
                        lambda o, a, f, s: numerical.verify_endpoint_ladder(
                            o,
                            a,
                            fixed["parameters"],
                            p.verification_seconds,
                            s,
                            tolerances=((1e-11, 1e-13),),
                        ),
                        mode=mode,
                    )
            if selected is not None and current["qualification_passed"]:
                start = assemble()["retained"]
                attempt = operate(
                    "continuation_search",
                    lambda o, a, f, s: _profile_search(
                        a,
                        f,
                        previous._verified_coordinates(o, a, start),
                        p.search_seconds - p.fixed_inner_seconds,
                        p.maximum_calls,
                        s,
                    )
                    if task["arm"] == "profiled"
                    else numerical.search(
                        a,
                        previous._verified_coordinates(o, a, start),
                        p.search_seconds,
                        p.maximum_calls,
                        s,
                        reject_invalid_trials=True,
                    ),
                    mode=mode,
                )
                fit = previous.first._search_checkpoint(attempt)
                if fit.get("best"):
                    operate(
                        "continuation_verification",
                        lambda o, a, f, s: numerical.verify_endpoint_ladder(
                            o,
                            a,
                            fit["best"]["parameters"],
                            p.verification_seconds,
                            s,
                            tolerances=((1e-11, 1e-13),),
                        ),
                        mode=mode,
                    )
            retained = assemble()["retained"]
            if retained:
                ops = _operations(folder, identity, p)
                candidate = _verified(ops.get("continuation_verification", {}))
                fit = previous.first._search_checkpoint(
                    ops.get("continuation_search", {})
                )
                matches = (
                    candidate is not None
                    and candidate["parameters"] == retained["parameters"]
                )
                # Outer steps do not certify full-coordinate stationarity.
                context = (
                    fit
                    if matches and task["arm"] == "joint"
                    else {
                        "stop_reason": (
                            "retained_endpoint_has_no_matching_"
                            "full_coordinate_search_history"
                        ),
                        "accepted_steps": [],
                        "budget_exhausted": bool(fit.get("budget_exhausted")),
                    }
                )
                binding = {
                    "diagnosed_parameters_sha256": public.content_sha256(
                        retained["parameters"]
                    ),
                    "verification_sha256": public.content_sha256(retained),
                    "search_history_matches_retained": matches,
                    "full_coordinate_history_available": matches
                    and task["arm"] == "joint",
                }

                def diagnosis(original, oracle, profile, save):
                    value = recovery.diagnose(
                        original,
                        oracle,
                        previous._verified_coordinates(original, oracle, retained),
                        retained,
                        context,
                        p.diagnostic_seconds,
                        lambda v: save(v | binding),
                        adequacy_policy=previous._adequacy_policy(diagnostic=True),
                    )
                    return value | binding

                operate("retained_diagnosis", diagnosis, mode=mode)
        value = assemble(terminal=True)
        seal(folder / "result.json", value)
        return value


def report(root):
    """Report both experiment roles without treating precision probes as models."""
    plan, data = verify(root, runtime=False)
    p, rows = Policy.model_validate(plan["policy"]), []
    for task in plan["tasks"]:
        source = data["models"][task["model"]]
        folder = _folder(root, task)
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        done = (folder / "result.json").exists()
        value = _assemble(
            identity, source, _operations(folder, identity, p), terminal=done
        )
        if done and read_seal(folder / "result.json") != value:
            raise ValueError("pilot terminal differs from receipts")
        rows.append(
            task
            | {
                k: v
                for k, v in value.items()
                if k
                not in {
                    "operations",
                    "identity",
                    "source_incumbent",
                    "retained",
                    "selected_start",
                }
            }
            | {
                "source_training_nmse": source["endpoint"]["training_nmse"],
                "training_nmse": (value.get("retained") or {}).get("training_nmse"),
                "currently_verified": value.get("retained") is not None,
            }
        )
    count = sum(row["status"] == "complete" for row in rows)
    old = data["historical_inputs"]
    result = {
        "protocol": PROTOCOL,
        "status": "complete" if count == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": count,
        "rows": rows,
        "historical_costs": {
            "original_and_reliability_by_model": old["earlier_cost_by_model"],
            "local_recovery_by_task": old["local_recovery_tasks"],
            "derivative_recovery_by_task": old["parent_tasks"],
            "precision_reduction_by_task": data["parent_tasks"],
        },
        "matched_optimization_seconds": p.search_seconds,
        "profile_fixed_inner_seconds": p.fixed_inner_seconds,
        "profile_outer_seconds": p.search_seconds - p.fixed_inner_seconds,
        "cost_matching_scope": (
            "optimization only; fixed-inner independent verification "
            "is additional diagnostic overhead"
        ),
        "old_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "limitation": (
            "Known frozen equations, training only; inner convex optimum does "
            "not certify outer or original parameter identifiability. "
            "Precision studies cannot select or optimize endpoints. "
            "Full-coordinate convergence requires its own evidence."
        ),
    }
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", result)
    return result


def inspect(root):
    """Archive the complete bounded evidence without running numerical work."""
    summary = report(root)
    path, temporary = root / "inspection.tar.gz", root / "inspection.tmp"
    paths = [root / "plan.json", root / "inputs.json"]
    if (root / "launcher_execution.json").exists():
        paths.append(root / "launcher_execution.json")
    for name in ("results", "report", "submissions", "logs"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="w", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for source in sorted(paths):
            if source.is_symlink() or not source.resolve().is_relative_to(
                root.resolve()
            ):
                raise ValueError("archive path escapes pilot")
            if source.name.startswith("."):
                continue
            blob = source.read_bytes()
            entry = tarfile.TarInfo(source.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}


def _full_residual_fallback(oracle, vector, budget, seconds, policy, result, save):
    """Use full-residual MP only after the guarded local evidence is insufficient."""
    end = min(budget.deadline(), monotonic() + seconds * 0.45)
    guarded = budget.numerical(
        lambda d: recovery._derivative(oracle, vector, min(d, end), policy)
    )
    result["derivative_passes"] = [
        {"backend": "guarded", "derivative_check": public._jsonable(guarded)}
    ]
    budget.callback(lambda: save(deepcopy(result) | {"timing": budget.snapshot()}))
    if guarded.search_qualified and (
        not policy.diagnostic_precision
        or (
            guarded.gradient_precision_qualified
            and guarded.parameter_information_qualified
        )
    ):
        return guarded
    if not recovery._retryable_check(guarded) and guarded.status != "passed":
        return guarded
    audit = {}
    result["full_residual_precision_audit"] = audit
    full = budget.numerical(
        lambda d: precision.derivative_check(
            oracle,
            vector,
            d,
            adequacy_policy=policy,
            audit=audit,
            checkpoint=lambda _: budget.callback(
                lambda: save(deepcopy(result) | {"timing": budget.snapshot()})
            ),
        )
    )
    result["derivative_passes"].append(
        {
            "backend": "full-residual-multiprecision",
            "derivative_check": public._jsonable(full),
        }
    )
    return full


def qualify_profile(
    problem,
    training,
    settings,
    anchor,
    seconds,
    save,
    *,
    coordinate_oracle=None,
    coordinate_parameters=None,
):
    """Opt-in aggregate qualification with a budgeted full-residual MP fallback."""
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {
        "protocol": "construction-profile-full-residual-qualification-1",
        "passed": False,
        "precision_mode": "guarded",
        "no_search_performed": True,
    }
    interrupted = None
    try:
        if coordinate_oracle is None:
            original, oracle, _ = profiling.make_oracle(
                problem, training, settings, anchor, precision_mode="guarded"
            )
        else:
            oracle, original = coordinate_oracle, coordinate_oracle.original
            if (
                public.content_sha256(original.problem)
                != public.content_sha256(problem)
                or public.content_sha256(original.training)
                != public.content_sha256(training)
                or oracle.precision_mode != "guarded"
            ):
                raise ValueError("qualification coordinate oracle identity differs")
        budget.check()
        vector = (
            oracle.reduction.reduce(original.vector(anchor))
            if coordinate_parameters is None
            else oracle.vector(coordinate_parameters)
        )
        if coordinate_parameters is not None and (
            original.parameters(oracle.reduction.lift(vector)) != anchor
        ):
            raise ValueError("qualification coordinate point differs from anchor")
        before, _ = budget.numerical(
            lambda d: original.evaluate(original.vector(anchor), d, jacobian=False)
        )
        after, _ = budget.numerical(
            lambda d: oracle.evaluate(vector, d, jacobian=False)
        )
        gap = float(np.max(np.abs(before - after), initial=0))
        result.update(
            lift_prediction_discrepancy=gap,
            recovery_audit=deepcopy(oracle.recovery_audit),
            original_coordinates=len(original.names),
            optimizer_coordinates=len(oracle.names),
        )
        check = _full_residual_fallback(
            oracle, vector, budget, seconds, previous._adequacy_policy(), result, save
        )
        result.update(
            derivative_check=public._jsonable(check),
            passed=bool(np.isfinite(gap) and gap <= 1e-7 and check.search_qualified),
            affine_execution=deepcopy(original.affine_audit),
        )
    except Exception as error:
        numerical._invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        numerical._invalidate(result, error, "qualification")
    return numerical._finish(result, budget, save, "qualification", interrupted)


def _bound_full_history(oracle, parameters, search_result):
    """Use only complete accepted vectors ending at this exact assessed endpoint."""
    steps = search_result.get("accepted_steps", [])
    matches = (
        search_result.get("accepted_history_protocol")
        == "full-coordinate-accepted-steps-1"
        and search_result.get("history_coordinate_names") == list(oracle.names)
        and search_result.get("history_coordinate_units") == oracle.units.tolist()
        and search_result.get("last_accepted_parameters") == parameters
        and (search_result.get("best") or {}).get("parameters") == parameters
        and bool(steps)
    )
    previous = None
    for step in steps if matches else []:
        try:
            point = np.asarray(step["reduced_vector"], float)
            prior = np.asarray(step["previous_reduced_vector"], float)
            if (
                point.shape != oracle.units.shape
                or prior.shape != point.shape
                or not np.isfinite(point).all()
                or not np.isfinite(prior).all()
                or (previous is not None and not np.array_equal(previous, prior))
                or step["parameters_sha256"]
                != public.content_sha256(oracle.parameters(point))
                or not np.isclose(
                    step["coordinate_step_norm"],
                    np.linalg.norm((point - prior) / oracle.units),
                    rtol=1e-12,
                    atol=0,
                )
            ):
                matches = False
                break
            previous = point
        except (KeyError, TypeError, ValueError):
            matches = False
            break
    if matches:
        matches = (
            oracle.parameters(previous) == parameters
            and steps[-1]["training_nmse"] == search_result["best"]["training_nmse"]
        )
    history = (
        [
            {
                key: step[key]
                for key in ("training_nmse", "coordinate_step_norm", "evaluation_count")
            }
            for step in steps
        ]
        if matches
        else []
    )
    return history, bool(matches)


def diagnose_profile(original, oracle, parameters, verification, fit, seconds, save):
    """Assess all aggregate coordinates with exact endpoint/history authentication."""
    if oracle.original is not original:
        raise ValueError("diagnosis oracle identity differs")
    budget = prefix_execution._budget(seconds, seconds + 60)
    policy = previous._adequacy_policy(diagnostic=True)
    steps, matches = _bound_full_history(oracle, parameters, fit)
    result = {
        "protocol": "construction-profile-full-residual-diagnosis-1",
        "assessment": None,
        "diagnosed_parameters_sha256": public.content_sha256(
            verification["parameters"]
        ),
        "verification_sha256": public.content_sha256(verification),
        "search_history_matches_retained": matches,
        "full_coordinate_history_available": matches,
        "accepted_steps_available": len(fit.get("accepted_steps", [])),
        "accepted_steps_used": len(steps),
        "recovery_audit": deepcopy(oracle.recovery_audit),
        "original_parameter_identifiability_certified": False,
    }
    interrupted = None
    try:
        vector = oracle.vector(parameters)
        derivative = _full_residual_fallback(
            oracle, vector, budget, seconds, policy, result, save
        )
        result["derivative_check"] = public._jsonable(derivative)
        residual, jacobian = budget.numerical(lambda d: oracle.evaluate(vector, d))
        stop = (
            fit.get("stop_reason")
            if matches
            else "retained_endpoint_has_no_matching_full_coordinate_search_history"
        )
        assessment = adequacy.assess(
            vector,
            residual,
            jacobian,
            oracle.lower,
            oracle.upper,
            verification={
                key: verification[key]
                for key in adequacy.OriginalVerification.model_fields
            },
            derivative=derivative,
            stop_reason=stop,
            budget_exhausted=bool(fit.get("budget_exhausted"))
            or stop in {"budget_limited", "numerical_failure", "interrupted"},
            history=steps,
            coordinate_units=oracle.units,
            policy=policy,
        )
        budget.check()
        result["assessment"] = public._jsonable(assessment)
        result["assessment_scope"] = (
            "All aggregate coordinates, including inner and any search-fixed rates; "
            "history uses full accepted vectors in the assessed coordinate units. "
            "Original parameter factorizations remain unidentified."
        )
    except Exception as error:
        numerical._invalidate(result, error, "diagnosis")
    except BaseException as error:
        interrupted = error
        numerical._invalidate(result, error, "diagnosis")
    result["affine_execution"] = deepcopy(original.affine_audit)
    return numerical._finish(result, budget, save, "diagnosis", interrupted)
