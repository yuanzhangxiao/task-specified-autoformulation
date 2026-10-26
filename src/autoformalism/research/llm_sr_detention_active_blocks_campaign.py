"""Bounded adaptive-coordinate continuation from authenticated chart audits.

The active set is numerical search evidence, not a scientific restriction or an
identifiability claim. Every changed checkpoint gets full-coordinate rechecks;
interrupted operations cannot acquire a fresh allowance on resume.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_detention_active_blocks as active
from autoformalism.fitting import llm_sr_detention_continuation as kernel
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_detention_chart_audit_campaign as audit
from autoformalism.research import llm_sr_detention_refit_campaign as saved_fit
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256

PROTOCOL = "llm-sr-detention-active-blocks-1"
MODE = audit.MODE
SOURCE_INDICES = audit.SOURCE_INDICES


class Inputs(StrictSchema):
    """Complete audit and original continuation history, including failures."""

    protocol: Literal["llm-sr-detention-active-blocks-input-1"] = (
        "llm-sr-detention-active-blocks-input-1"
    )
    audit_plan: dict
    audit_plan_sha256: Sha256
    audit_inputs: dict
    audit_report: dict
    audit_report_sha256: Sha256
    validation_data_opened: Literal[False] = False
    test_data_opened: Literal[False] = False
    llm_calls: Literal[0] = 0
    _source: audit.Inputs | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def authenticate(self):
        self._source = _validate_history(self)
        return self


class Policy(StrictSchema):
    """Three equal bounded blocks; each check/verification has its own ledger."""

    protocol: Literal["llm-sr-detention-active-blocks-policy-1"] = (
        "llm-sr-detention-active-blocks-policy-1"
    )
    phases: Literal[3] = 3
    optimization_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    direction_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    precision_seconds: FiniteFloat = Field(default=240, gt=0, le=240)
    verification_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    aggregation_seconds: FiniteFloat = Field(default=30, gt=0, le=30)


def _validate_history(inputs):
    old = audit.Inputs.model_validate(inputs.audit_inputs)
    plan, report = inputs.audit_plan, inputs.audit_report
    policy = audit.Policy.model_validate(plan["policy"])
    if (
        public.content_sha256(plan) != inputs.audit_plan_sha256
        or public.content_sha256(report) != inputs.audit_report_sha256
        or plan
        != audit._plan(
            old, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
        )
    ):
        raise ValueError("source audit identity differs")
    saved_rows = report.get("tasks", [])
    if len(saved_rows) != 24 or report.get("status") != "complete":
        raise ValueError("complete twenty-four-coordinate audit required")
    rows = []
    for task, saved in zip(plan["tasks"], saved_rows, strict=True):
        identity = {"plan_sha256": inputs.audit_plan_sha256, "task": task}
        operations = saved["operations"]
        if set(operations) != {"direction"}:
            raise ValueError("source audit operation roster differs")
        operation = operations["direction"]
        if (
            operation.get("identity")
            != {
                "campaign": identity,
                "stage": "direction",
                "allowance_seconds": policy.direction_seconds,
            }
            or operation.get("budget_restarted") is not False
            or operation.get("progress_sha256")
            != public.content_sha256(operation.get("observed_progress", {}))
        ):
            raise ValueError("source audit operation identity differs")
        rebuilt = audit._assemble(identity, operations, old, terminal=True)
        if rebuilt != saved:
            raise ValueError("source audit terminal differs from receipts")
        rows.append(rebuilt)
    groups = []
    for source_index in SOURCE_INDICES:
        selected = [
            r for r in rows if r["identity"]["task"]["source_index"] == source_index
        ]
        task = selected[0]["identity"]["task"]
        directions = [
            audit.source._value(r["operations"], "direction") for r in selected
        ]
        gate = kernel.qualification(
            *audit._problem(old, task),
            task["parameters"],
            directions,
            mode=MODE,
            policy=audit._numerical(old),
        )
        if not gate["complete"] or len(gate["rows"]) != 8:
            raise ValueError("source audit requires complete full-coordinate evidence")
        previous = old.source_report["tasks"][source_index]
        previous_gate = previous["qualification"]["qualification"]
        groups.append(
            {
                "source_index": source_index,
                "start_id": task["start_id"],
                "parameters_sha256": public.content_sha256(task["parameters"]),
                "historical_training_nmse": previous["retained"]["training_nmse"],
                "historical_qualification_sha256": previous_gate[
                    "qualification_sha256"
                ],
                "historical_search_qualified": previous_gate["search_qualified"],
                "historical_blocked_coordinates": [
                    r["name"]
                    for r in previous_gate.get("rows", [])
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
    expected = {
        "protocol": audit.PROTOCOL,
        "status": "complete",
        "expected": 24,
        "recorded": 24,
        "tasks": rows,
        "groups": groups,
        "policy": policy.model_dump(mode="json"),
        "selection_rule": audit.SELECTION,
        "source_report_sha256": old.source_report_sha256,
        "historical_costs": [
            {"source_index": i, "cost": r["cost"]}
            for i, r in enumerate(old.source_report["tasks"])
        ],
        "historical_predecessor_costs": {
            k: old.source_report[k]
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
    if report != expected:
        raise ValueError("source audit report differs from receipts")
    return old


def export_inputs(root: Path, destination: Path):
    """Authenticate saved reports once and export all history without rollouts."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("export cannot write inside source audit")
    plan, old, _ = audit.verify(root, runtime=False)
    report = audit.report(root)
    if report != public._read(root / "report/summary.json"):
        raise ValueError("saved source audit report differs")
    inputs = Inputs(
        audit_plan=plan,
        audit_plan_sha256=public.content_sha256(plan),
        audit_inputs=old.model_dump(mode="json"),
        audit_report=report,
        audit_report_sha256=public.content_sha256(report),
    )
    seal(destination, inputs.model_dump(mode="json"))
    return {"inputs_sha256": public.content_sha256(inputs), "tasks": 5}


def _source(inputs):
    if inputs._source is None:
        raise ValueError("source audit has not been authenticated")
    return inputs._source


def _problem(inputs, task):
    return audit._problem(_source(inputs), task)


def _numerical(inputs):
    return audit._numerical(_source(inputs))


def _initial_gate(inputs, task):
    return inputs.audit_report["groups"][SOURCE_INDICES.index(task["source_index"])][
        "qualification"
    ]


def _tasks(inputs):
    tasks = []
    for source_index in (*SOURCE_INDICES, SOURCE_INDICES[0], SOURCE_INDICES[2]):
        index = len(tasks)
        previous = _source(inputs).source_plan["tasks"][source_index]
        checkpoint = _source(inputs).source_report["tasks"][source_index]["retained"]
        gate = inputs.audit_report["groups"][SOURCE_INDICES.index(source_index)][
            "qualification"
        ]
        tasks.append(
            {
                "index": index,
                "kind": "fit" if index < 3 else "precision_check",
                "source_index": source_index,
                "source_task": 15,
                "start_id": previous["start_id"],
                "mode": MODE,
                "parameters": deepcopy(previous["parameters"]),
                "parent_verified_checkpoint": deepcopy(checkpoint),
                "initial_qualification_sha256": gate["qualification_sha256"],
                "coordinate_domain": kernel.domain(
                    *_problem(inputs, previous), previous["parameters"], mode=MODE
                ),
                **(
                    {"coordinate_index": 2, "coordinate_name": "eps_s"}
                    if index >= 3
                    else {}
                ),
            }
        )
    return tasks


def _plan(inputs, policy, *, source_sha256, runtime):
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "source_sha256": source_sha256,
        "runtime": runtime,
        "tasks": _tasks(inputs),
        "policy": policy.model_dump(mode="json"),
        "historical_allowances_renewed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None):
    """Freeze three continuations and two independent precision controls."""
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
            raise ValueError("cannot prepare nonempty unsealed active-block directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 5}


def verify(root: Path, *, runtime=True):
    plan = read_seal(root / "plan.json")
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    policy = Policy.model_validate(plan["policy"])
    if plan != _plan(
        inputs, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
    ):
        raise ValueError("active-block plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("active-block source/runtime differs")
    return plan, inputs, policy


def _allowances(task, policy):
    if task["kind"] == "precision_check":
        return {"precision_direction": policy.precision_seconds}
    result = {"start_verification": policy.verification_seconds}
    for phase in range(policy.phases):
        prefix = f"phase-{phase}"
        result |= {
            f"{prefix}-selection": policy.aggregation_seconds,
            f"{prefix}-search": policy.optimization_seconds,
            f"{prefix}-verification": policy.verification_seconds,
            **{f"{prefix}-recheck-{i}": policy.direction_seconds for i in range(8)},
        }
    return result


def _operations(directory, identity, policy):
    allowed = _allowances(identity["task"], policy)
    if any(p.name not in allowed for p in (directory / "operations").glob("*")):
        raise ValueError("unknown active-block operation")
    result = {}
    for name, seconds in allowed.items():
        folder = directory / "operations" / name
        binding = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (folder / "result.json").exists():
            result[name] = journal._read_operation(folder, binding)
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != binding:
                raise ValueError("active-block reservation differs")
            progress = (
                public._read(folder / "progress.json")
                if (folder / "progress.json").exists()
                else {}
            )
            result[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "budget_charge_seconds": seconds,
                "wall_seconds": None,
                "cpu_seconds": None,
                "accounting_complete": False,
                "observed_progress": progress,
            }
    return result


def _value(operations, name):
    operation = operations.get(name, {})
    return operation.get("value") if operation.get("status") == "complete" else None


def _verified(operations, name, parameters):
    point = saved_fit._verified(operations.get(name, {}))
    if point and (
        point["parameters"] != parameters
        or point.get("requested_parameters_sha256") != public.content_sha256(parameters)
    ):
        raise ValueError("verification differs from its own requested checkpoint")
    return point


def _eligible(inputs, task, parameters, gate):
    if not gate.get("complete"):
        return []
    return list(
        active.select_active(
            *_problem(inputs, task),
            parameters,
            qualification=gate,
            policy=_numerical(inputs),
        )
    )


def _selection(inputs, task, parameters, gate):
    names = gate["coordinate_names"]
    selected = _eligible(inputs, task, parameters, gate)
    return {
        "parameters_sha256": public.content_sha256(parameters),
        "qualification_sha256": gate["qualification_sha256"],
        "active_names": selected,
        "held_names": [n for n in names if n not in selected],
        "full_coordinate_search_qualified": gate["search_qualified"],
        "full_model_convergence_claimed": False,
    }


def _state(inputs, task, operations, policy):
    """Replay each phase's retention and eligibility without numerical work."""
    problem, numerical = _problem(inputs, task), _numerical(inputs)
    retained = deepcopy(task["parent_verified_checkpoint"])
    initial = _verified(operations, "start_verification", task["parameters"])
    if initial:
        retained = initial
    origin, gate = (
        "start_verification" if initial else "parent_verified_checkpoint",
        _initial_gate(inputs, task),
    )
    phases, used, prior_active = [], set(), None
    if not initial:
        return retained, origin, gate, phases, "start_verification_unavailable", used
    for phase in range(policy.phases):
        prefix = f"phase-{phase}"
        selection = _value(operations, f"{prefix}-selection")
        if selection is None:
            if f"{prefix}-selection" in operations:
                used.add(f"{prefix}-selection")
                if (
                    operations[f"{prefix}-selection"]["status"]
                    != "running_or_interrupted"
                ):
                    return retained, origin, gate, phases, "selection_unavailable", used
            break
        used.add(f"{prefix}-selection")
        parameters = retained["parameters"]
        expected = _selection(inputs, task, parameters, gate)
        if selection != expected:
            raise ValueError("phase selection differs from verified matching evidence")
        active_names = selection["active_names"]
        phase_row = {
            "phase": phase,
            "start": retained,
            "start_origin": origin,
            "selection": selection,
            "active_names": active_names,
            "held_names": selection["held_names"],
            "released_names": []
            if prior_active is None
            else [n for n in active_names if n not in prior_active],
            "newly_held_names": []
            if prior_active is None
            else [n for n in prior_active if n not in active_names],
            "qualification_before": gate,
            "complete": False,
        }
        phases.append(phase_row)
        if not active_names:
            phase_row.update(
                complete=True,
                stop_reason="no_qualified_active_coordinates",
                retained=retained,
                qualification_after=gate,
            )
            return (
                retained,
                origin,
                gate,
                phases,
                "no_qualified_active_coordinates",
                used,
            )
        fit = saved_fit._fit_checkpoint(operations.get(f"{prefix}-search", {}))
        if not fit:
            if f"{prefix}-search" in operations:
                used.add(f"{prefix}-search")
                if operations[f"{prefix}-search"]["status"] != "running_or_interrupted":
                    phase_row.update(
                        complete=True,
                        retained=retained,
                        qualification_after=gate,
                        stop_reason="search_checkpoint_unavailable",
                    )
                    return (
                        retained,
                        origin,
                        gate,
                        phases,
                        "search_checkpoint_unavailable",
                        used,
                    )
            break
        used.add(f"{prefix}-search")
        if (
            fit.get("selection_sha256") != public.content_sha256(selection)
            or fit.get("qualification_sha256") != gate["qualification_sha256"]
            or fit.get("active_names") != active_names
            or fit.get("held_names") != selection["held_names"]
        ):
            raise ValueError("search checkpoint lacks matching active-set selection")
        phase_row["optimization"] = fit
        best = fit.get("best")
        candidate = _verified(
            operations, f"{prefix}-verification", best["parameters"] if best else None
        )
        if f"{prefix}-verification" in operations:
            used.add(f"{prefix}-verification")
        if candidate and numerics.improves(candidate, retained):
            retained, origin = candidate, f"{prefix}-verification"
        changed = retained["parameters"] != parameters
        phase_row.update(
            retained=retained, retained_origin=origin, verified_improvement=changed
        )
        attempted = not best or f"{prefix}-verification" in operations
        search_done = (
            operations[f"{prefix}-search"]["status"] != "running_or_interrupted"
        )
        if changed:
            records = []
            for i in range(8):
                name = f"{prefix}-recheck-{i}"
                if name in operations:
                    used.add(name)
                value = _value(operations, name)
                if value is not None:
                    records.append(value)
            gate = kernel.qualification(
                *problem, retained["parameters"], records, mode=MODE, policy=numerical
            )
            complete = (
                attempted
                and search_done
                and all(
                    f"{prefix}-recheck-{i}" in operations
                    and operations[f"{prefix}-recheck-{i}"]["status"]
                    != "running_or_interrupted"
                    for i in range(8)
                )
            )
            phase_row["recheck_source"] = "fresh_full_coordinate_audit"
        else:
            complete = attempted and search_done
            phase_row["recheck_source"] = "identical_point_matching_evidence_reused"
        phase_row.update(
            qualification_after=gate,
            complete=complete,
            stop_reason=fit.get("stop_reason", "interrupted"),
        )
        if not complete:
            break
        if not changed:
            return (
                retained,
                origin,
                gate,
                phases,
                "no_verified_progress_same_point",
                used,
            )
        if not gate["complete"]:
            return (
                retained,
                origin,
                gate,
                phases,
                "full_coordinate_recheck_incomplete",
                used,
            )
        prior_active = active_names
    reason = (
        "phase_budget_exhausted"
        if len(phases) == policy.phases and phases[-1]["complete"]
        else "incomplete"
    )
    return retained, origin, gate, phases, reason, used


def _assemble(identity, operations, inputs, policy, *, terminal=False):
    task = identity["task"]
    if task["kind"] == "precision_check":
        value = _value(operations, "precision_direction")
        if value is not None and value.get("index") != task["coordinate_index"]:
            raise ValueError("precision control belongs to another coordinate")
        numerical = active.precision_policy(_numerical(inputs))
        gate = kernel.qualification(
            *_problem(inputs, task),
            task["parameters"],
            [value] if value else [],
            mode=MODE,
            policy=numerical,
        )
        return {
            "identity": identity,
            "status": "complete" if terminal else "incomplete",
            "operations": operations,
            "cost": journal._cost(operations),
            "coordinate_result": gate["rows"][0] if gate["rows"] else None,
            "historical_coordinate_result": _initial_gate(inputs, task)["rows"][2],
            "numerical_policy": numerical.model_dump(mode="json"),
            "parameters_changed": False,
            "optimization_performed": False,
            "validation_data_opened": False,
            "test_data_opened": False,
            "full_model_convergence_claimed": False,
            "llm_calls": 0,
        }
    retained, origin, gate, phases, reason, used = _state(
        inputs, task, operations, policy
    )
    extra = set(operations) - used - {"start_verification"}
    if extra and (terminal or reason != "incomplete"):
        raise ValueError("operations exist beyond adaptive stopping rule")
    if terminal and reason == "incomplete":
        raise ValueError("terminal active-block task has an unfinished phase")
    active_names = _eligible(inputs, task, retained["parameters"], gate)
    searches = [operations[n] for n in operations if n.endswith("-search")]
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "operations": operations,
        "cost": journal._cost(operations),
        "retained": retained,
        "retained_origin": origin,
        "phases": phases,
        "phases_completed": sum(p["complete"] for p in phases),
        "active_names": active_names,
        "held_names": [n for n in gate["coordinate_names"] if n not in active_names],
        "final_qualification": gate,
        "stop_reason": reason,
        "full_coordinate_search_qualified": gate["search_qualified"],
        "optimization_allowance_seconds": policy.phases * policy.optimization_seconds,
        "optimization_reserved_seconds": len(searches) * policy.optimization_seconds,
        "optimization_unused_allowance_seconds": (policy.phases - len(searches))
        * policy.optimization_seconds,
        "full_model_convergence_claimed": False,
        "parameter_confidence_claimed": False,
        "historical_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def run_task(root: Path, index: int):
    """Run one durable adaptive continuation or independent precision control."""
    plan, inputs, policy = verify(root)
    if isinstance(index, bool) or not isinstance(index, int) or index not in range(5):
        raise ValueError("unknown active-block task")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)
    allowed = _allowances(task, policy)
    problem, numerical = _problem(inputs, task), _numerical(inputs)

    def operations():
        return _operations(directory, identity, policy)

    def current(terminal=False):
        return _assemble(identity, operations(), inputs, policy, terminal=terminal)

    def operate(name, work):
        def durable(save):
            result = work(save)
            save(result)
            return result

        return journal._operation(directory, identity, name, allowed[name], durable)

    def verify_parameters(parameters, save):
        binding = {"requested_parameters_sha256": public.content_sha256(parameters)}
        return (
            original.verify(
                *problem,
                parameters,
                seconds=policy.verification_seconds,
                checkpoint=lambda value: save(value | binding),
                policy=numerical.numerical,
            )
            | binding
        )

    with public._lock(directory):
        if (directory / "result.json").exists():
            result = current(terminal=True)
            if read_seal(directory / "result.json") != result:
                raise ValueError("active-block terminal differs from receipts")
            return result
        if task["kind"] == "precision_check":
            operate(
                "precision_direction",
                lambda save: kernel.direction(
                    *problem,
                    task["parameters"],
                    task["coordinate_index"],
                    mode=MODE,
                    purpose="search",
                    precision="tight",
                    seconds=policy.precision_seconds,
                    checkpoint=save,
                    policy=active.precision_policy(numerical),
                ),
            )
        else:
            operate(
                "start_verification",
                lambda save: verify_parameters(task["parameters"], save),
            )
            if _verified(operations(), "start_verification", task["parameters"]):
                for phase in range(policy.phases):
                    state = current()
                    if state["stop_reason"] not in (
                        "incomplete",
                        "phase_budget_exhausted",
                    ):
                        break
                    prefix = f"phase-{phase}"
                    # Replay gives the next incumbent after completed phases; when
                    # resuming a partial phase use its own immutable starting point.
                    row = next(
                        (p for p in state["phases"] if p["phase"] == phase), None
                    )
                    if row and row["complete"]:
                        continue
                    start = row["start"] if row else state["retained"]
                    gate = (
                        row["qualification_before"]
                        if row
                        else state["final_qualification"]
                    )
                    selection = _selection(inputs, task, start["parameters"], gate)
                    operate(
                        f"{prefix}-selection",
                        lambda save, selection=selection: selection,
                    )
                    if _value(operations(), f"{prefix}-selection") is None:
                        break
                    if not selection["active_names"]:
                        break
                    binding = {"selection_sha256": public.content_sha256(selection)}
                    operate(
                        f"{prefix}-search",
                        lambda save,
                        start=start,
                        gate=gate,
                        selection=selection,
                        binding=binding: active.fit_active(
                            *problem,
                            start["parameters"],
                            qualification=gate,
                            active_names=selection["active_names"],
                            seconds=policy.optimization_seconds,
                            checkpoint=lambda value: save(value | binding),
                            policy=numerical,
                        )
                        | binding,
                    )
                    best = saved_fit._fit_checkpoint(
                        operations().get(f"{prefix}-search", {})
                    ).get("best")
                    if best:
                        operate(
                            f"{prefix}-verification",
                            lambda save, best=best: verify_parameters(
                                best["parameters"], save
                            ),
                        )
                    retained = current()["retained"]
                    if retained["parameters"] != start["parameters"]:
                        for i in range(8):
                            operate(
                                f"{prefix}-recheck-{i}",
                                lambda save, i=i, retained=retained: kernel.direction(
                                    *problem,
                                    retained["parameters"],
                                    i,
                                    mode=MODE,
                                    purpose="search",
                                    precision="tight",
                                    seconds=policy.direction_seconds,
                                    checkpoint=save,
                                    policy=numerical,
                                ),
                            )
                    if current()["stop_reason"] != "incomplete":
                        break
        result = current(terminal=True)
        seal(directory / "result.json", result)
        return result


def report(root: Path):
    """Reconstruct phases from immutable records without resuming computation."""
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
            policy,
            terminal=terminal,
        )
        if terminal and read_seal(directory / "result.json") != row:
            raise ValueError("active-block terminal differs from receipts")
        rows.append(row)
    source = _source(inputs).source_report
    return {
        "protocol": PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": 5,
        "recorded": sum(r["status"] == "complete" for r in rows),
        "tasks": rows,
        "policy": policy.model_dump(mode="json"),
        "audit_report_sha256": inputs.audit_report_sha256,
        "historical_audit_costs": [
            {"index": r["identity"]["task"]["index"], "cost": r["cost"]}
            for r in inputs.audit_report["tasks"]
        ],
        "historical_original_arm_endpoints": [
            {
                "source_index": i,
                "start_id": source["tasks"][i]["identity"]["task"]["start_id"],
                "retained": source["tasks"][i]["retained"],
                "cost": source["tasks"][i]["cost"],
            }
            for i in (0, 2, 4)
        ],
        "historical_continuation_costs": [
            {"source_index": i, "cost": r["cost"]}
            for i, r in enumerate(source["tasks"])
        ],
        "historical_predecessor_costs": {
            k: source[k]
            for k in (
                "historical_costs",
                "historical_predecessor_costs",
                "historical_earlier_predecessor_costs",
            )
        },
        "comparison_scope": (
            "historical original-coordinate endpoints; "
            "not a fresh cost-matched comparison"
        ),
        "historical_allowances_renewed": False,
        "automatic_promotion": False,
        "full_model_convergence_claimed": False,
        "parameter_confidence_claimed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
