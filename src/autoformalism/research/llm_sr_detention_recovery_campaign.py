"""Bounded, resumable coordinate diagnostics for six frozen detention controls.

Historical allowances are never renewed. Coordinate operations have independent
reservations, so completed checks survive interruption without repeating full
Jacobians. Only training data and each task's own initial/incumbent vector enter.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite, ulp
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_detention_recovery as kernel
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_detention_refit_campaign as source
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256

PROTOCOL = "llm-sr-detention-recovery-1"
SELECTION = "saved generic endpoints 4/8; all four qualification-blocked starts 1/6/7/9"
ROSTER = (
    (4, "diagnosis_only"),
    (8, "diagnosis_only"),
    (1, "recover_start"),
    (6, "recover_start"),
    (7, "recover_start"),
    (9, "recover_start"),
)


class Parent(StrictSchema):
    """One complete historical terminal and its immutable operation evidence."""

    index: int = Field(ge=0, le=9)
    terminal: dict
    operations: dict


class Inputs(StrictSchema):
    """Portable training inputs plus all ten prior outcomes, including failures."""

    protocol: Literal["llm-sr-detention-recovery-input-1"] = (
        "llm-sr-detention-recovery-input-1"
    )
    source_plan: dict
    source_plan_sha256: Sha256
    source_inputs: dict
    source_report: dict
    source_report_sha256: Sha256
    parents: tuple[Parent, ...] = Field(min_length=10, max_length=10)
    validation_data_opened: Literal[False] = False
    test_data_opened: Literal[False] = False
    llm_calls: Literal[0] = 0

    @model_validator(mode="after")
    def history(self):
        _validate_history(self)
        return self


class Policy(StrictSchema):
    """Separate coordinate, optimization, and verification reservations."""

    protocol: Literal["llm-sr-detention-recovery-policy-1"] = (
        "llm-sr-detention-recovery-policy-1"
    )
    direction_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    optimization_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    verification_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    aggregation_seconds: FiniteFloat = Field(default=30, gt=0, le=30)
    accuracy_nmse_threshold: FiniteFloat = Field(default=1e-3, gt=0, le=1)
    numerical: kernel.Policy = Field(default_factory=kernel.Policy)


def _saved_starts(saved, regenerated):
    """Permit <=8 ULP only in regenerated generic-start floats, never identities.

    NumPy exp can differ by an ULP across libm/platforms. Frozen saved values are
    authoritative; this comparison does not round or rewrite any historical data.
    """
    if len(saved) != len(regenerated):
        raise ValueError("historical task roster differs")
    differences = []
    for old, new in zip(saved, regenerated, strict=True):
        before, after = deepcopy(old), deepcopy(new)
        a, b = before.pop("parameters"), after.pop("parameters")
        if before != after or set(a) != set(b):
            raise ValueError("historical task metadata differs")
        for name in a:
            if a[name] == b[name]:
                continue
            if (
                old["start_kind"] != "generic"
                or isinstance(a[name], bool)
                or not isinstance(a[name], (int, float))
                or not isinstance(b[name], (int, float))
                or not isfinite(a[name])
                or not isfinite(b[name])
                or abs(a[name] - b[name]) > 8 * max(ulp(a[name]), ulp(b[name]))
            ):
                raise ValueError("historical starting parameter differs")
            differences.append(
                {
                    "index": old["index"],
                    "parameter": name,
                    "saved": a[name],
                    "regenerated": b[name],
                }
            )
    return differences


def _validate_history(inputs: Inputs):
    """Rebuild ten terminals/report; tolerate no changes to saved receipt hashes."""
    plan = inputs.source_plan
    old = source.Inputs.model_validate(inputs.source_inputs)
    policy = source.Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != source.PROTOCOL
        or public.content_sha256(plan) != inputs.source_plan_sha256
        or plan["inputs_sha256"] != public.content_sha256(old)
        or plan["selection_rule"] != source.SELECTION
        or plan["llm_calls"] != 0
        or any(
            plan[k] is not False
            for k in (
                "historical_allowances_renewed",
                "validation_data_opened",
                "test_data_opened",
                "automatic_promotion",
            )
        )
        or public.content_sha256(inputs.source_report) != inputs.source_report_sha256
    ):
        raise ValueError("historical plan/report identity differs")
    rounding = _saved_starts(plan["tasks"], source._tasks(old, policy))
    references = {r["identity"]["task"]: r for r in old.source_report["tasks"]}
    rows = []
    allowed = source._allowances(policy)
    for index, (task, parent) in enumerate(
        zip(plan["tasks"], inputs.parents, strict=True)
    ):
        identity = {"plan_sha256": inputs.source_plan_sha256, "task": task}
        if parent.index != index or set(parent.operations) != set(allowed):
            raise ValueError("historical receipt roster differs")
        for name, operation in parent.operations.items():
            if (
                operation["identity"]
                != {
                    "campaign": identity,
                    "stage": name,
                    "allowance_seconds": allowed[name],
                }
                or operation["progress_sha256"]
                != public.content_sha256(operation.get("observed_progress", {}))
                or operation["budget_restarted"] is not False
            ):
                raise ValueError("historical operation receipt differs")
        result = source._assemble(identity, parent.operations, policy, terminal=True)
        if result != parent.terminal:
            raise ValueError("historical terminal differs from receipts")
        reference = references[task["source_task"]]
        loss = reference["stages"]["primary"].get("normalized_mse")
        if not isinstance(loss, (int, float)) or not isfinite(loss) or loss < 0:
            loss = None
        retained = result.get("retained")
        rows.append(
            result
            | {
                "supplied_reference_training_nmse": loss,
                "supplied_reference_verified": reference["solver_agreement"][
                    "verified"
                ],
                "retained_to_supplied_nmse_ratio": (
                    retained["training_nmse"] / loss
                    if retained and loss and loss > 0
                    else None
                ),
                "reference_comparison_role": (
                    "descriptive only; no optimizer seeding or promotion"
                ),
            }
        )
    expected = {
        "protocol": source.PROTOCOL,
        "status": "complete",
        "expected": 10,
        "recorded": 10,
        "tasks": rows,
        "policy": policy.model_dump(mode="json"),
        "selection_rule": source.SELECTION,
        "supplied_reference": [
            deepcopy(r)
            for r in old.source_report["tasks"]
            if r["identity"]["task"] in source.SELECTED
        ],
        "historical_costs": [
            {
                "source_task": p.terminal["identity"]["task"],
                "cost": p.terminal["stage_cost"],
            }
            for p in old.parents
        ],
        "historical_report_sha256": old.source_report_sha256,
        "model_selection_performed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    if expected != inputs.source_report:
        raise ValueError("historical report differs from authenticated receipts")
    for index, kind in ROSTER:
        terminal = inputs.parents[index].terminal
        if kind == "recover_start" and (
            terminal["optimizer_stop_reason"] != "derivatives_unqualified"
            or terminal["optimization"].get("calls") != 0
            or terminal["retained_origin"] != "start_verification"
        ):
            raise ValueError(
                "recovery roster requires the four blocked original starts"
            )
        if kind == "diagnosis_only" and not terminal.get("retained"):
            raise ValueError("diagnostic roster requires verified saved endpoints")
    return old, rounding


def export_inputs(root: Path, destination: Path) -> dict:
    """Export authenticated frozen history without regenerating or rewriting it."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("export must not write inside historical campaign")
    plan = read_seal(root / "plan.json")
    parents = []
    for index in range(10):
        directory = root / "tasks" / str(index)
        operations = {}
        for name in source._allowances(source.Policy.model_validate(plan["policy"])):
            folder = directory / "operations" / name
            operations[name] = journal._read_operation(
                folder, read_seal(folder / "started.json")
            )
        parents.append(
            {
                "index": index,
                "terminal": read_seal(directory / "result.json"),
                "operations": operations,
            }
        )
    report = public._read(root / "report/summary.json")
    inputs = Inputs(
        source_plan=plan,
        source_plan_sha256=public.content_sha256(plan),
        source_inputs=read_seal(root / "inputs.json"),
        source_report=report,
        source_report_sha256=public.content_sha256(report),
        parents=parents,
    )
    seal(destination, inputs.model_dump(mode="json"))
    return {
        "inputs_sha256": public.content_sha256(inputs),
        "historical_tasks": 10,
        "new_tasks": 6,
        "historical_allowances_renewed": False,
    }


def _tasks(inputs: Inputs):
    tasks = []
    for source_index, mode in ROSTER:
        parent = inputs.parents[source_index].terminal
        old = inputs.source_plan["tasks"][source_index]
        parameters = (
            parent["retained"]["parameters"]
            if mode == "diagnosis_only"
            else old["parameters"]
        )
        tasks.append(
            {
                "index": len(tasks),
                "source_index": source_index,
                "source_task": old["source_task"],
                "mode": mode,
                "start_id": old["start_id"],
                "parameters": deepcopy(parameters),
                "coordinate_names": old["domain_policy"]["names"],
                "source_terminal_sha256": public.content_sha256(parent),
            }
        )
    return tasks


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None):
    inputs = Inputs.model_validate(
        inputs.model_dump(mode="json") if isinstance(inputs, Inputs) else inputs
    )
    policy = Policy.model_validate((policy or Policy()).model_dump(mode="json"))
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(inputs),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "selection_rule": SELECTION,
        "historical_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_promotion": False,
    }
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            p.name != ".lock" for p in root.iterdir()
        ):
            raise ValueError("cannot prepare nonempty unsealed recovery directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 6}


def verify(root: Path, *, runtime=True):
    plan = read_seal(root / "plan.json")
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    policy = Policy.model_validate(plan["policy"])
    expected = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(inputs),
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
        "selection_rule": SELECTION,
        "historical_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_promotion": False,
    }
    if plan != expected:
        raise ValueError("detention recovery plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("detention recovery source/runtime differs")
    return plan, inputs, policy


def _allowances(task, policy):
    allowed = {"start_verification": policy.verification_seconds}
    if task["mode"] == "recover_start":
        for precision in ("standard", "tight"):
            allowed |= {
                f"search-{precision}-{i:02d}": policy.direction_seconds
                for i in range(len(task["coordinate_names"]))
            }
        allowed |= {
            "qualification-standard": policy.aggregation_seconds,
            "qualification-tight": policy.aggregation_seconds,
            "optimization": policy.optimization_seconds,
            "endpoint_verification": policy.verification_seconds,
        }
    allowed |= {
        f"diagnosis-tight-{i:02d}": policy.direction_seconds
        for i in range(len(task["coordinate_names"]))
    }
    return allowed | {"retained_diagnosis": policy.aggregation_seconds}


def _operations(directory, identity, policy):
    allowed = _allowances(identity["task"], policy)
    if any(p.name not in allowed for p in (directory / "operations").glob("*")):
        raise ValueError("unknown detention recovery operation")
    operations = {}
    for name, seconds in allowed.items():
        folder = directory / "operations" / name
        binding = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (folder / "result.json").exists():
            operations[name] = journal._read_operation(folder, binding)
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != binding:
                raise ValueError("recovery operation reservation differs")
            operations[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "budget_charge_seconds": seconds,
                "wall_seconds": None,
                "cpu_seconds": None,
                "accounting_complete": False,
            }
    return operations


def _problem(inputs, task):
    old = source.source.Inputs.model_validate(
        inputs.source_inputs["source_plan"]["inputs"]
    )
    row = next(r for r in old.tasks if r.task == task["source_task"])
    item = old.transcription_set.for_task(row.task)
    assert item is not None
    return item, row.context, row.training


def _directions(operations, task, purpose, precision):
    return [
        operations[name]["value"]
        for i in range(len(task["coordinate_names"]))
        if (name := f"{purpose}-{precision}-{i:02d}") in operations
        and operations[name].get("status") == "complete"
        and isinstance(operations[name].get("value"), dict)
    ]


def _qualifications(operations, task, problem, policy):
    """Validate frozen aggregates, preserving authoritative cross-platform numbers."""
    qualifications = {}
    if task["mode"] != "recover_start":
        return qualifications
    for precision in ("standard", "tight"):
        operation = operations.get(f"qualification-{precision}", {})
        saved = (
            operation.get("value") if operation.get("status") == "complete" else None
        )
        rows = _directions(operations, task, "search", precision)
        if saved is not None:
            kernel.validate_qualification(
                *problem,
                task["parameters"],
                rows,
                purpose="search",
                precision=precision,
                qualification=saved,
                policy=policy.numerical,
            )
            qualifications[precision] = saved
        else:
            # No accepted aggregation receipt means no usable gate. This empty
            # evidence result performs no integration or floating-point reduction.
            qualifications[precision] = kernel.qualification(
                *problem,
                task["parameters"],
                [],
                purpose="search",
                precision=precision,
                policy=policy.numerical,
            )
    return qualifications


def _assemble(identity, operations, policy, problem, *, terminal=False):
    task = identity["task"]
    fit = source._fit_checkpoint(operations.get("optimization", {}))
    retained, origin = None, None
    for name in ("start_verification", "endpoint_verification"):
        point = source._verified(operations.get(name, {}))
        expected = (
            task["parameters"]
            if name == "start_verification"
            else (fit.get("best") or {}).get("parameters")
        )
        if point and (
            point["parameters"] != expected
            or point.get("requested_parameters_sha256")
            != public.content_sha256(expected)
        ):
            raise ValueError("recovery endpoint differs from its own task")
        if point and numerics.improves(point, retained):
            retained, origin = point, name
    qualifications = _qualifications(operations, task, problem, policy)
    for precision in ("standard", "tight"):
        for row in _directions(operations, task, "search", precision):
            if row.get("identity", {}).get(
                "parameters_sha256"
            ) != public.content_sha256(task["parameters"]):
                raise ValueError("search direction differs from starting point")
    if fit:
        used = fit.get("qualification_sha256")
        if used not in {q["qualification_sha256"] for q in qualifications.values()}:
            raise ValueError("optimization qualification differs from saved checks")
    diagnosis = source._checkpoint(operations.get("retained_diagnosis", {}))
    if retained:
        for row in _directions(operations, task, "diagnosis", "tight"):
            if row.get("identity", {}).get(
                "parameters_sha256"
            ) != public.content_sha256(retained["parameters"]):
                raise ValueError("diagnostic direction differs from retained point")
    if diagnosis and (
        retained is None
        or diagnosis.get("diagnosed_parameters_sha256")
        != public.content_sha256(retained["parameters"])
        or diagnosis.get("verification_sha256") != public.content_sha256(retained)
    ):
        raise ValueError(
            "diagnosis differs from independently verified retained endpoint"
        )
    if diagnosis:
        kernel.validate_diagnosis(
            *problem,
            retained["parameters"],
            directions=_directions(operations, task, "diagnosis", "tight"),
            verification=retained,
            diagnosis={
                k: v
                for k, v in diagnosis.items()
                if k not in {"diagnosed_parameters_sha256", "verification_sha256"}
            },
            policy=policy.numerical,
        )
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "operations": operations,
        "cost": journal._cost(operations),
        "retained": retained,
        "retained_origin": origin,
        "qualification": qualifications,
        "optimization": fit,
        "optimizer_stop_reason": fit.get(
            "stop_reason",
            "not_requested" if task["mode"] == "diagnosis_only" else "not_started",
        ),
        "diagnosis": diagnosis or None,
        "diagnosis_complete": diagnosis.get("status") == "complete",
        "accuracy": (
            "unavailable"
            if retained is None
            else "within_predeclared_training_threshold"
            if retained["training_nmse"] <= policy.accuracy_nmse_threshold
            else "above_predeclared_training_threshold"
        ),
        "global_optimum_certified": False,
        "historical_allowances_renewed": False,
        "supplied_reference_used_as_fallback": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def run_task(root: Path, index: int):
    plan, inputs, policy = verify(root)
    if index not in range(6):
        raise ValueError("unknown recovery task")
    task = plan["tasks"][index]
    problem = _problem(inputs, task)
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)
    allowed = _allowances(task, policy)

    def current(terminal=False):
        return _assemble(
            identity,
            _operations(directory, identity, policy),
            policy,
            problem,
            terminal=terminal,
        )

    def operate(name, work):
        def durable(save):
            result = work(save)
            save(result)
            return result

        return journal._operation(directory, identity, name, allowed[name], durable)

    def verify_parameters(parameters, save):
        binding = {"requested_parameters_sha256": public.content_sha256(parameters)}
        value = original.verify(
            *problem,
            parameters,
            seconds=policy.verification_seconds,
            checkpoint=lambda v: save(v | binding),
            policy=policy.numerical.numerical,
        )
        return value | binding

    def direction(parameters, i, purpose, precision):
        return operate(
            f"{purpose}-{precision}-{i:02d}",
            lambda save: kernel.direction(
                *problem,
                parameters,
                i,
                purpose=purpose,
                precision=precision,
                seconds=policy.direction_seconds,
                checkpoint=save,
                policy=policy.numerical,
            ),
        )

    with public._lock(directory):
        if (directory / "result.json").exists():
            result = current(terminal=True)
            if result != read_seal(directory / "result.json"):
                raise ValueError("recovery terminal differs from receipts")
            return result
        operate(
            "start_verification",
            lambda save: verify_parameters(task["parameters"], save),
        )
        if task["mode"] == "recover_start":
            for precision in ("standard", "tight"):
                for i in range(len(task["coordinate_names"])):
                    direction(task["parameters"], i, "search", precision)
                rows = _directions(
                    _operations(directory, identity, policy), task, "search", precision
                )
                operate(
                    f"qualification-{precision}",
                    lambda save, rows=rows, precision=precision: kernel.qualification(
                        *problem,
                        task["parameters"],
                        rows,
                        purpose="search",
                        precision=precision,
                        policy=policy.numerical,
                    ),
                )
                qualification = current()["qualification"][precision]
                if qualification["search_qualified"]:
                    break
            binding = {"qualification_sha256": qualification["qualification_sha256"]}
            if qualification["search_qualified"]:
                operate(
                    "optimization",
                    lambda save: kernel.fit(
                        *problem,
                        task["parameters"],
                        qualification=qualification,
                        seconds=policy.optimization_seconds,
                        checkpoint=lambda v: save(v | binding),
                        policy=policy.numerical,
                    )
                    | binding,
                )
            else:
                operate(
                    "optimization",
                    lambda save: {
                        "stop_reason": "derivatives_unqualified",
                        "calls": 0,
                        "best": None,
                        **binding,
                    },
                )
            best = current()["optimization"].get("best")
            if best:
                operate(
                    "endpoint_verification",
                    lambda save: verify_parameters(best["parameters"], save),
                )
        retained = current()["retained"]
        if retained:
            for i in range(len(task["coordinate_names"])):
                direction(retained["parameters"], i, "diagnosis", "tight")
            rows = _directions(
                _operations(directory, identity, policy), task, "diagnosis", "tight"
            )
            binding = {
                "diagnosed_parameters_sha256": public.content_sha256(
                    retained["parameters"]
                ),
                "verification_sha256": public.content_sha256(retained),
            }
            operate(
                "retained_diagnosis",
                lambda save: kernel.diagnose(
                    *problem,
                    retained["parameters"],
                    directions=rows,
                    verification=retained,
                    policy=policy.numerical,
                )
                | binding,
            )
        result = current(terminal=True)
        seal(directory / "result.json", result)
        return result


def report(root: Path):
    plan, inputs, policy = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        directory = root / "tasks" / str(task["index"])
        done = (directory / "result.json").exists()
        row = _assemble(
            identity,
            _operations(directory, identity, policy),
            policy,
            _problem(inputs, task),
            terminal=done,
        )
        if done and row != read_seal(directory / "result.json"):
            raise ValueError("recovery terminal differs from receipts")
        rows.append(row)
    return {
        "protocol": PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": 6,
        "recorded": sum(r["status"] == "complete" for r in rows),
        "tasks": rows,
        "policy": policy.model_dump(mode="json"),
        "selection_rule": SELECTION,
        "historical_report_sha256": inputs.source_report_sha256,
        "historical_costs": [
            {"source_index": p.index, "cost": p.terminal["cost"]}
            for p in inputs.parents
        ],
        "historical_outcomes": [
            {
                "source_index": p.index,
                "optimizer_stop_reason": p.terminal["optimizer_stop_reason"],
                "training_nmse": (p.terminal.get("retained") or {}).get(
                    "training_nmse"
                ),
            }
            for p in inputs.parents
        ],
        "historical_predecessor_costs": inputs.source_report["historical_costs"],
        "historical_allowances_renewed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
