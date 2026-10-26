"""Audit corrected detention coordinates before spending another search budget.

Every task evaluates one coordinate at an unchanged historical point. Reporting
authenticates and aggregates those receipts; it performs no parameter fitting.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_detention_continuation as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_detention_continuation_campaign as source
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256

PROTOCOL = "llm-sr-detention-chart-audit-1"
MODE = "reduced_model15_consistent"
SOURCE_INDICES = (1, 3, 5)
SELECTION = "all eight corrected coordinates at continuation reduced tasks 1/3/5"


class Inputs(StrictSchema):
    """Portable complete continuation history, including unselected successes."""

    protocol: Literal["llm-sr-detention-chart-audit-input-1"] = (
        "llm-sr-detention-chart-audit-input-1"
    )
    source_plan: dict
    source_plan_sha256: Sha256
    source_inputs: dict
    source_report: dict
    source_report_sha256: Sha256
    validation_data_opened: Literal[False] = False
    test_data_opened: Literal[False] = False
    llm_calls: Literal[0] = 0
    _validated_source: source.Inputs | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def authenticate(self):
        self._validated_source = _validate_history(self)
        return self


class Policy(StrictSchema):
    """One nonrenewable derivative allowance per coordinate; no search budget."""

    protocol: Literal["llm-sr-detention-chart-audit-policy-1"] = (
        "llm-sr-detention-chart-audit-policy-1"
    )
    direction_seconds: FiniteFloat = Field(default=120, gt=0, le=120)


def _validate_history(inputs):
    old = source.Inputs.model_validate(inputs.source_inputs)
    plan = inputs.source_plan
    policy = source.Policy.model_validate(plan["policy"])
    if (
        public.content_sha256(plan) != inputs.source_plan_sha256
        or public.content_sha256(inputs.source_report) != inputs.source_report_sha256
        or plan
        != source._plan(
            old, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
        )
    ):
        raise ValueError("continuation history identity differs")
    saved_rows = inputs.source_report.get("tasks", [])
    if len(saved_rows) != 7:
        raise ValueError("complete seven-task continuation history required")
    rows = []
    for task, saved in zip(plan["tasks"], saved_rows, strict=True):
        identity = {"plan_sha256": inputs.source_plan_sha256, "task": task}
        operations = saved["operations"]
        allowed = source._allowances(task, policy)
        if set(operations) - set(allowed):
            raise ValueError("historical operation roster differs")
        for name, operation in operations.items():
            if (
                operation.get("identity")
                != {
                    "campaign": identity,
                    "stage": name,
                    "allowance_seconds": allowed[name],
                }
                or operation.get("budget_restarted") is not False
                or operation.get("progress_sha256")
                != public.content_sha256(operation.get("observed_progress", {}))
            ):
                raise ValueError("historical operation identity differs")
        parent = old.parents[task["source_index"]].terminal
        row = source._assemble(
            identity,
            operations,
            policy,
            source._problem(old, task),
            source.source._directions(
                parent["operations"], parent["identity"]["task"], "diagnosis", "tight"
            ),
            terminal=True,
        )
        if row != saved:
            raise ValueError("continuation terminal differs from receipts")
        rows.append(row)
    if source._summary(plan, old, rows) != inputs.source_report:
        raise ValueError("continuation report differs from receipts")
    for index in SOURCE_INDICES:
        task, row = plan["tasks"][index], rows[index]
        if (
            task["mode"] != "reduced_model15"
            or task["source_task"] != 15
            or row["retained"]["parameters"] != task["parameters"]
            or not source.numerics.usable(row["retained"])
        ):
            raise ValueError("audit requires the unchanged verified reduced start")
    return old


def export_inputs(root: Path, destination: Path):
    """Authenticate complete history once without altering historical files."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("export cannot write inside historical campaign")
    plan, old, _ = source.verify(root, runtime=False)
    report = source.report(root)
    if report != public._read(root / "report/summary.json"):
        raise ValueError("saved continuation report differs")
    inputs = Inputs(
        source_plan=plan,
        source_plan_sha256=public.content_sha256(plan),
        source_inputs=old.model_dump(mode="json"),
        source_report=report,
        source_report_sha256=public.content_sha256(report),
    )
    seal(destination, inputs.model_dump(mode="json"))
    return {"inputs_sha256": public.content_sha256(inputs), "tasks": 24}


def _problem(inputs, task):
    if inputs._validated_source is None:
        raise ValueError("audit history has not been authenticated")
    return source._problem(
        inputs._validated_source, inputs.source_plan["tasks"][task["source_index"]]
    )


def _numerical(inputs):
    return source.Policy.model_validate(inputs.source_plan["policy"]).numerical


def _tasks(inputs):
    result = []
    for source_index in SOURCE_INDICES:
        previous = inputs.source_plan["tasks"][source_index]
        point = previous["parameters"]
        task = {
            "source_index": source_index,
            "source_task": previous["source_task"],
            "start_id": previous["start_id"],
            "mode": MODE,
            "parameters": deepcopy(point),
            "source_terminal_sha256": public.content_sha256(
                inputs.source_report["tasks"][source_index]
            ),
        }
        domain = kernel.domain(*_problem(inputs, task), point, mode=MODE)
        if len(domain["names"]) != 8:
            raise ValueError("audit requires exactly eight reduced coordinates")
        for coordinate_index, name in enumerate(domain["names"]):
            result.append(
                task
                | {
                    "index": len(result),
                    "coordinate_index": coordinate_index,
                    "coordinate_name": name,
                    "coordinate_domain": domain,
                }
            )
    return result


def _plan(inputs, policy, *, source_sha256, runtime):
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "source_sha256": source_sha256,
        "runtime": runtime,
        "tasks": _tasks(inputs),
        "policy": policy.model_dump(mode="json"),
        "selection_rule": SELECTION,
        "historical_allowances_renewed": False,
        "optimization_performed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None):
    """Freeze twenty-four fixed-point checks; preparation performs no rollout."""
    inputs = Inputs.model_validate(
        inputs.model_dump(mode="json") if isinstance(inputs, Inputs) else inputs
    )
    policy = Policy.model_validate((policy or Policy()).model_dump(mode="json"))
    plan = _plan(
        inputs,
        policy,
        source_sha256=public._source_identity(),
        runtime=public._runtime(),
    )
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            p.name != ".lock" for p in root.iterdir()
        ):
            raise ValueError("cannot prepare nonempty unsealed audit directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 24}


def verify(root: Path, *, runtime=True):
    plan = read_seal(root / "plan.json")
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    policy = Policy.model_validate(plan["policy"])
    if plan != _plan(
        inputs, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
    ):
        raise ValueError("detention chart audit plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("detention chart audit source/runtime differs")
    return plan, inputs, policy


def _operations(directory, identity, policy):
    paths = list((directory / "operations").glob("*"))
    if any(p.name != "direction" for p in paths):
        raise ValueError("unknown chart audit operation")
    folder = directory / "operations/direction"
    binding = {
        "campaign": identity,
        "stage": "direction",
        "allowance_seconds": policy.direction_seconds,
    }
    if (folder / "result.json").exists():
        return {"direction": journal._read_operation(folder, binding)}
    if (folder / "started.json").exists():
        if read_seal(folder / "started.json") != binding:
            raise ValueError("chart audit reservation differs")
        return {
            "direction": {
                "status": "running_or_interrupted",
                "value": None,
                "budget_charge_seconds": policy.direction_seconds,
                "wall_seconds": None,
                "cpu_seconds": None,
                "accounting_complete": False,
            }
        }
    return {}


def _assemble(identity, operations, inputs, *, terminal=False):
    task = identity["task"]
    value = source._value(operations, "direction")
    gate = None
    if value is not None:
        if value.get("index") != task["coordinate_index"]:
            raise ValueError("direction receipt belongs to another coordinate")
        gate = kernel.qualification(
            *_problem(inputs, task),
            task["parameters"],
            [value],
            mode=MODE,
            policy=_numerical(inputs),
        )
    rows = (gate or {}).get("rows", [])
    coordinate = rows[0] if rows else None
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "operations": operations,
        "cost": journal._cost(operations),
        "coordinate_result": coordinate,
        "derivative_status": (value or {}).get("status", "unavailable"),
        "coordinate_search_qualified": bool(
            coordinate and coordinate["search_qualified"]
        ),
        "full_chart_qualification_claimed": False,
        "parameters_changed": False,
        "optimization_performed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def run_task(root: Path, index: int):
    """Check one coordinate once; interruptions cannot acquire a fresh budget."""
    plan, inputs, policy = verify(root)
    if isinstance(index, bool) or index not in range(24):
        raise ValueError("unknown chart audit task")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)

    def current(terminal=False):
        return _assemble(
            identity,
            _operations(directory, identity, policy),
            inputs,
            terminal=terminal,
        )

    with public._lock(directory):
        if (directory / "result.json").exists():
            result = current(terminal=True)
            if read_seal(directory / "result.json") != result:
                raise ValueError("chart audit terminal differs from receipts")
            return result

        def work(save):
            result = kernel.direction(
                *_problem(inputs, task),
                task["parameters"],
                task["coordinate_index"],
                mode=MODE,
                purpose="search",
                precision="tight",
                policy=_numerical(inputs),
                seconds=policy.direction_seconds,
                checkpoint=save,
            )
            save(result)
            return result

        journal._operation(
            directory, identity, "direction", policy.direction_seconds, work
        )
        result = current(terminal=True)
        seal(directory / "result.json", result)
        return result


def report(root: Path):
    """Validate all receipts and aggregate only matching corrected chart columns."""
    plan, inputs, policy = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        directory = root / "tasks" / str(task["index"])
        terminal = (directory / "result.json").exists()
        row = _assemble(
            identity,
            _operations(directory, identity, policy),
            inputs,
            terminal=terminal,
        )
        if terminal and row != read_seal(directory / "result.json"):
            raise ValueError("chart audit terminal differs from receipts")
        rows.append(row)
    groups = []
    for source_index in SOURCE_INDICES:
        selected = [
            r for r in rows if r["identity"]["task"]["source_index"] == source_index
        ]
        task = selected[0]["identity"]["task"]
        directions = [
            value
            for row in selected
            if (value := source._value(row["operations"], "direction")) is not None
        ]
        gate = kernel.qualification(
            *_problem(inputs, task),
            task["parameters"],
            directions,
            mode=MODE,
            policy=_numerical(inputs),
        )
        old = inputs.source_report["tasks"][source_index]
        old_gate = old["qualification"]["qualification"]
        groups.append(
            {
                "source_index": source_index,
                "start_id": task["start_id"],
                "parameters_sha256": public.content_sha256(task["parameters"]),
                "historical_training_nmse": old["retained"]["training_nmse"],
                "historical_qualification_sha256": old_gate["qualification_sha256"],
                "historical_search_qualified": old_gate["search_qualified"],
                "historical_blocked_coordinates": [
                    r["name"]
                    for r in old_gate.get("rows", [])
                    if not r["search_qualified"]
                ],
                "qualification": gate,
                "blocked_coordinates": [
                    r["name"] for r in gate["rows"] if not r["search_qualified"]
                ],
                "missing_coordinates": [
                    gate["coordinate_names"][i] for i in gate["missing_indices"]
                ],
                "comparison_scope": "new corrected audit versus saved historical audit",
                "optimization_performed": False,
                "convergence_or_parameter_confidence_claimed": False,
            }
        )
    return {
        "protocol": PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": 24,
        "recorded": sum(r["status"] == "complete" for r in rows),
        "tasks": rows,
        "groups": groups,
        "policy": policy.model_dump(mode="json"),
        "selection_rule": SELECTION,
        "source_report_sha256": inputs.source_report_sha256,
        "historical_costs": [
            {"source_index": i, "cost": r["cost"]}
            for i, r in enumerate(inputs.source_report["tasks"])
        ],
        "historical_predecessor_costs": {
            k: inputs.source_report[k]
            for k in (
                "historical_costs",
                "historical_predecessor_costs",
                "historical_earlier_predecessor_costs",
            )
        },
        "historical_allowances_renewed": False,
        "optimization_performed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
