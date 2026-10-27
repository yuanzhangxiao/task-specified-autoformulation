"""Matched eager/lazy detention controls with shared, charged qualification.

Historical adaptive endpoints are references only. The experiment starts every
arm from the frozen original point, and never renews an earlier reservation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_detention_active_blocks as active
from autoformalism.fitting import llm_sr_detention_continuation as kernel
from autoformalism.fitting import llm_sr_detention_lazy_controls as controls
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_detention_active_blocks_campaign as source
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256

PROTOCOL = "llm-sr-detention-lazy-controls-1"
EXPECTED_SOURCE_REPORT_SHA256 = (
    "d41a220e00f9eda110c04378f31860cb1d3d34a05b3dac7a324b5d359de13c12"
)
MODES = ("original", "reduced_model15_consistent")
ARMS = (
    ("original_all_eager_contiguous", MODES[0], False, False, 1),
    ("original_all_lazy_contiguous", MODES[0], True, False, 1),
    ("reduced_all_eager_contiguous", MODES[1], False, False, 1),
    ("reduced_all_lazy_contiguous", MODES[1], True, False, 1),
    ("reduced_all_lazy_segmented", MODES[1], True, False, 3),
    ("reduced_adaptive_lazy_segmented", MODES[1], True, True, 3),
)
BOUNDARY = {
    "historical_allowances_renewed": False,
    "automatic_promotion": False,
    "full_model_convergence_claimed": False,
    "parameter_confidence_claimed": False,
    "validation_data_opened": False,
    "test_data_opened": False,
    "llm_calls": 0,
}
_AUTHENTICATED = object()


class Inputs(StrictSchema):
    """Authenticated portable history, including consumed incomplete receipts."""

    protocol: Literal["llm-sr-detention-lazy-controls-input-1"] = (
        "llm-sr-detention-lazy-controls-input-1"
    )
    source_plan: dict
    source_plan_sha256: Sha256
    source_inputs: dict
    source_report: dict
    source_report_sha256: Sha256
    incomplete_reservations: dict
    validation_data_opened: Literal[False] = False
    test_data_opened: Literal[False] = False
    llm_calls: Literal[0] = 0
    _source: source.Inputs | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def authenticate(self):
        self._source = _validate_history(self)
        return self


class Policy(StrictSchema):
    """Equal search caps, separate bounded diagnosis, and one precision escalation."""

    protocol: Literal["llm-sr-detention-lazy-controls-policy-1"] = (
        "llm-sr-detention-lazy-controls-policy-1"
    )
    optimization_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    direction_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    precision_seconds: FiniteFloat = Field(default=240, gt=0, le=240)
    verification_seconds: FiniteFloat = Field(default=120, gt=0, le=120)


@dataclass(frozen=True)
class VerifiedRun:
    """In-process authenticated invocation; serialized contexts are never accepted."""

    root: Path
    plan: dict
    inputs: Inputs
    policy: Policy
    started_wall: float
    started_cpu: float
    started_utc: str
    _authentication: object = dataclass_field(repr=False)


def _historical_summary(plan, old, rows):
    """Rebuild the prior report without opening, modifying, or rerunning its root."""
    previous = source._source(old).source_report
    return {
        "protocol": source.PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": 5,
        "recorded": sum(r["status"] == "complete" for r in rows),
        "tasks": rows,
        "policy": plan["policy"],
        "audit_report_sha256": old.audit_report_sha256,
        "historical_audit_costs": [
            {"index": r["identity"]["task"]["index"], "cost": r["cost"]}
            for r in old.audit_report["tasks"]
        ],
        "historical_original_arm_endpoints": [
            {
                "source_index": i,
                "start_id": previous["tasks"][i]["identity"]["task"]["start_id"],
                "retained": previous["tasks"][i]["retained"],
                "cost": previous["tasks"][i]["cost"],
            }
            for i in (0, 2, 4)
        ],
        "historical_continuation_costs": [
            {"source_index": i, "cost": r["cost"]}
            for i, r in enumerate(previous["tasks"])
        ],
        "historical_predecessor_costs": {
            k: previous[k]
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
        **BOUNDARY,
    }


def _validate_history(inputs):
    old = source.Inputs.model_validate(inputs.source_inputs)
    plan, report = inputs.source_plan, inputs.source_report
    policy = source.Policy.model_validate(plan["policy"])
    if (
        public.content_sha256(plan) != inputs.source_plan_sha256
        or public.content_sha256(report) != inputs.source_report_sha256
        or inputs.source_report_sha256 != EXPECTED_SOURCE_REPORT_SHA256
        or plan
        != source._plan(
            old, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
        )
    ):
        raise ValueError("historical active-block identity differs")
    if len(report.get("tasks", [])) != 5:
        raise ValueError("complete five-task historical roster required")
    rows, reservations = [], {}
    for task, saved in zip(plan["tasks"], report["tasks"], strict=True):
        identity = {"plan_sha256": inputs.source_plan_sha256, "task": task}
        allowed, operations = source._allowances(task, policy), saved["operations"]
        if set(operations) - set(allowed):
            raise ValueError("historical operation roster differs")
        for name, operation in operations.items():
            bound = {
                "campaign": identity,
                "stage": name,
                "allowance_seconds": allowed[name],
            }
            if operation["status"] == "running_or_interrupted":
                key = f"{task['index']}/{name}"
                reservations[key] = bound
                if operation != {
                    "status": "running_or_interrupted",
                    "value": None,
                    "budget_charge_seconds": allowed[name],
                    "wall_seconds": None,
                    "cpu_seconds": None,
                    "accounting_complete": False,
                    "observed_progress": operation.get("observed_progress", {}),
                }:
                    raise ValueError("historical incomplete reservation differs")
            elif (
                operation.get("identity") != bound
                or operation.get("budget_restarted") is not False
                or operation.get("progress_sha256")
                != public.content_sha256(operation.get("observed_progress", {}))
            ):
                raise ValueError("historical operation identity differs")
        row = source._assemble(
            identity, operations, old, policy, terminal=saved["status"] == "complete"
        )
        if row != saved:
            raise ValueError("historical terminal differs from receipts")
        rows.append(row)
    if reservations != inputs.incomplete_reservations:
        raise ValueError("historical consumed reservation bindings differ")
    if _historical_summary(plan, old, rows) != report:
        raise ValueError("historical report differs from receipts")
    endpoint = rows[2]
    if (
        endpoint["status"] != "incomplete"
        or endpoint["final_qualification"]["missing_indices"] != [5, 6, 7]
        or not numerics.usable(endpoint["retained"])
        or len(endpoint["phases"]) != 3
    ):
        raise ValueError(
            "expected saved generic endpoint with missing directions 5/6/7"
        )
    return old


def export_inputs(root: Path, destination: Path):
    """Export authentic history without fitting or writing inside its source root."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("export cannot write inside historical campaign")
    plan = read_seal(root / "plan.json")
    report = public._read(root / "report/summary.json")
    reservations = {}
    for row in report["tasks"]:
        index = row["identity"]["task"]["index"]
        operation_root = root / "tasks" / str(index) / "operations"
        if {p.name for p in operation_root.glob("*")} != set(row["operations"]):
            raise ValueError("saved report omits historical operation directories")
        for name, operation in row["operations"].items():
            directory = root / "tasks" / str(index) / "operations" / name
            bound = read_seal(directory / "started.json")
            if operation["status"] == "running_or_interrupted":
                reservations[f"{index}/{name}"] = bound
                if (directory / "result.json").exists():
                    raise ValueError(
                        "saved report omits a completed historical receipt"
                    )
                progress = (
                    public._read(directory / "progress.json")
                    if (directory / "progress.json").exists()
                    else {}
                )
                if progress != operation["observed_progress"]:
                    raise ValueError("historical incomplete progress differs")
            elif journal._read_operation(directory, bound) != operation:
                raise ValueError("saved report differs from historical receipt")
    inputs = Inputs(
        source_plan=plan,
        source_plan_sha256=public.content_sha256(plan),
        source_inputs=read_seal(root / "inputs.json"),
        source_report=report,
        source_report_sha256=public.content_sha256(report),
        incomplete_reservations=reservations,
    )
    for row in report["tasks"]:
        path = root / "tasks" / str(row["identity"]["task"]["index"]) / "result.json"
        if path.exists() != (row["status"] == "complete"):
            raise ValueError("historical task completion differs from saved report")
        if path.exists() and read_seal(path) != row:
            raise ValueError("historical task terminal differs")
    seal(destination, inputs.model_dump(mode="json"))
    return {
        "inputs_sha256": public.content_sha256(inputs),
        "tasks": 27,
        "fit_tasks": 18,
    }


def _old(inputs):
    if inputs._source is None:
        raise ValueError("source history has not been authenticated")
    return inputs._source


def _problem(inputs, task):
    return source._problem(_old(inputs), task)


def _numerical(inputs):
    return source._numerical(_old(inputs))


def _tasks(inputs):
    tasks, starts = [], inputs.source_plan["tasks"][:3]
    for start_index, start in enumerate(starts):
        for mode in MODES:
            tasks.append(
                {
                    "index": len(tasks),
                    "kind": "prequalification",
                    "start_index": start_index,
                    "source_index": start["source_index"],
                    "source_task": 15,
                    "start_id": start["start_id"],
                    "mode": mode,
                    "parameters": deepcopy(start["parameters"]),
                    "parent_verified_checkpoint": deepcopy(
                        start["parent_verified_checkpoint"]
                    ),
                    "coordinate_domain": kernel.domain(
                        *_problem(inputs, start), start["parameters"], mode=mode
                    ),
                }
            )
    for start_index in range(3):
        for arm, mode, lazy, adaptive, segments in ARMS:
            parent = tasks[start_index * 2 + MODES.index(mode)]
            tasks.append(
                {
                    **deepcopy(parent),
                    "index": len(tasks),
                    "kind": "fit",
                    "arm": arm,
                    "lazy_jacobian": lazy,
                    "adaptive": adaptive,
                    "segments": segments,
                    "qualification_task": parent["index"],
                }
            )
    endpoint = inputs.source_report["tasks"][2]
    oldtask = inputs.source_plan["tasks"][2]
    for coordinate in (5, 6, 7):
        tasks.append(
            {
                "index": len(tasks),
                "kind": "diagnostic_supplement",
                "source_index": oldtask["source_index"],
                "source_task": 15,
                "start_id": oldtask["start_id"],
                "mode": MODES[1],
                "parameters": deepcopy(endpoint["retained"]["parameters"]),
                "coordinate_index": coordinate,
                "coordinate_name": endpoint["final_qualification"]["coordinate_names"][
                    coordinate
                ],
                "historical_endpoint_sha256": public.content_sha256(
                    endpoint["retained"]
                ),
                "historical_operation": f"phase-2-recheck-{coordinate}",
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
        "initial_qualification_rule": (
            "shared full base audit; one full tighter audit if blocked; "
            "select complete audit with most qualified rows, prefer tighter on ties"
        ),
        "source_report_sha256": inputs.source_report_sha256,
        **BOUNDARY,
    }


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None):
    """Freeze six shared audits, eighteen controlled fits, and three supplements."""
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
            raise ValueError("cannot prepare nonempty unsealed lazy-control directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 27, "fit_tasks": 18}


def verify(root: Path, *, runtime=True):
    """Authenticate the portable source once per top-level invocation."""
    plan, inputs = (
        read_seal(root / "plan.json"),
        Inputs.model_validate(read_seal(root / "inputs.json")),
    )
    policy = Policy.model_validate(plan["policy"])
    if plan != _plan(
        inputs, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
    ):
        raise ValueError("lazy-control plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("lazy-control source/runtime differs")
    return plan, inputs, policy


def load_run(root: Path) -> VerifiedRun:
    """Start timing before authentication and share it with launcher identity checks."""
    wall, cpu = monotonic(), process_time()
    utc = datetime.now(UTC).isoformat()
    root = root.resolve()
    plan, inputs, policy = verify(root)
    return VerifiedRun(root, plan, inputs, policy, wall, cpu, utc, _AUTHENTICATED)


def _allowances(task, policy):
    if task["kind"] == "diagnostic_supplement":
        return {"supplement_direction": policy.direction_seconds}
    size = len(task["coordinate_domain"]["names"])
    if task["kind"] == "prequalification":
        return {"start_verification": policy.verification_seconds} | {
            f"{level}-{i}": seconds
            for level, seconds in (
                ("base", policy.direction_seconds),
                ("escalated", policy.precision_seconds),
            )
            for i in range(size)
        }
    return {
        name: seconds
        for segment in range(task["segments"])
        for name, seconds in {
            f"phase-{segment}-search": policy.optimization_seconds / task["segments"],
            f"phase-{segment}-verification": policy.verification_seconds,
            **{
                f"phase-{segment}-recheck-{i}": policy.direction_seconds
                for i in range(size)
            },
        }.items()
    }


def _operations(directory, identity, policy):
    allowed, result = _allowances(identity["task"], policy), {}
    if any(p.name not in allowed for p in (directory / "operations").glob("*")):
        raise ValueError("unknown lazy-control operation")
    for name, seconds in allowed.items():
        folder = directory / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (folder / "result.json").exists():
            result[name] = journal._read_operation(folder, bound)
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != bound:
                raise ValueError("lazy-control reservation differs")
            result[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "budget_charge_seconds": seconds,
                "wall_seconds": None,
                "cpu_seconds": None,
                "accounting_complete": False,
                "observed_progress": public._read(folder / "progress.json")
                if (folder / "progress.json").exists()
                else {},
            }
    return result


def _gate(inputs, task, parameters, records, numerical):
    return kernel.qualification(
        *_problem(inputs, task),
        parameters,
        records,
        mode=task["mode"],
        policy=numerical,
    )


def _records(operations, prefix, size):
    return [
        value
        for i in range(size)
        if (value := source._value(operations, f"{prefix}-{i}")) is not None
    ]


def _prequalification(inputs, task, operations):
    base = _numerical(inputs)
    size = len(task["coordinate_domain"]["names"])
    gate = _gate(
        inputs, task, task["parameters"], _records(operations, "base", size), base
    )
    attempts = [
        {
            "level": "base",
            "qualification": gate,
            "numerical_policy": base.model_dump(mode="json"),
        }
    ]
    if gate["search_qualified"] and any(n.startswith("escalated-") for n in operations):
        raise ValueError("unnecessary precision escalation beyond frozen stopping rule")
    if any(n.startswith("escalated-") for n in operations):
        tighter = active.precision_policy(base)
        escalated = _gate(
            inputs,
            task,
            task["parameters"],
            _records(operations, "escalated", size),
            tighter,
        )
        attempts.append(
            {
                "level": "escalated",
                "qualification": escalated,
                "numerical_policy": tighter.model_dump(mode="json"),
            }
        )
    selected = max(
        attempts,
        key=lambda a: (
            a["qualification"]["complete"],
            sum(r["search_qualified"] for r in a["qualification"]["rows"]),
            a["level"] == "escalated",
        ),
    )
    return {
        "attempts": attempts,
        "selected_level": selected["level"],
        "qualification": selected["qualification"],
        "numerical_policy": selected["numerical_policy"],
        "verified_start": source._verified(
            operations, "start_verification", task["parameters"]
        ),
        "evidence_shared_by_compared_arms": True,
    }


def _selection(task, parameters, gate):
    names = gate["coordinate_names"]
    eligible = (
        [r["name"] for r in gate["rows"] if r["search_qualified"]]
        if gate["complete"]
        else []
    )
    selected = (
        eligible if task["adaptive"] else names if gate["search_qualified"] else []
    )
    return {
        "parameters_sha256": public.content_sha256(parameters),
        "qualification_sha256": gate["qualification_sha256"],
        "active_names": selected,
        "held_names": [n for n in names if n not in selected],
        "full_coordinate_search_qualified": gate["search_qualified"],
        "full_model_convergence_claimed": False,
    }


def _fit_state(inputs, task, operations, policy, shared):
    """Replay at most three segments using already authenticated source objects."""
    retained = shared["verified_start"] or task["parent_verified_checkpoint"]
    gate, numerical = (
        shared["qualification"],
        kernel.Policy.model_validate(shared["numerical_policy"]),
    )
    phases, used, reason = [], set(), "initial_verification_unavailable"
    if shared["verified_start"]:
        reason = "incomplete"
        for phase in range(task["segments"]):
            prefix, start = f"phase-{phase}", retained
            selection = _selection(task, start["parameters"], gate)
            row = {
                "phase": phase,
                "start": start,
                "qualification_before": gate,
                "selection": selection,
                "complete": False,
            }
            prior = phases[-1]["selection"]["active_names"] if phases else None
            row.update(
                released_names=[]
                if prior is None
                else [n for n in selection["active_names"] if n not in prior],
                newly_held_names=[]
                if prior is None
                else [n for n in prior if n not in selection["active_names"]],
            )
            phases.append(row)
            if not selection["active_names"]:
                row.update(complete=True, retained=retained, qualification_after=gate)
                reason = "initial_or_endpoint_qualification_blocked"
                break
            name = f"{prefix}-search"
            if name not in operations:
                break
            used.add(name)
            fit = source.saved_fit._fit_checkpoint(operations[name])
            if not fit:
                reason = (
                    "search_checkpoint_unavailable"
                    if operations[name]["status"] != "running_or_interrupted"
                    else "incomplete"
                )
                break
            if any(
                fit.get(k) != v
                for k, v in _fit_binding(task, shared, selection).items()
            ):
                raise ValueError(
                    "search checkpoint differs from controlled arm selection"
                )
            row["optimization"] = fit
            best = fit.get("best")
            verification = f"{prefix}-verification"
            candidate = source._verified(
                operations, verification, best["parameters"] if best else None
            )
            if verification in operations:
                used.add(verification)
            if candidate and numerics.improves(candidate, retained):
                retained = candidate
            changed = retained["parameters"] != start["parameters"]
            ready = operations[name]["status"] != "running_or_interrupted" and (
                not best
                or (
                    verification in operations
                    and operations[verification]["status"] != "running_or_interrupted"
                )
            )
            if changed:
                size = len(gate["coordinate_names"])
                names = [f"{prefix}-recheck-{i}" for i in range(size)]
                used.update(n for n in names if n in operations)
                gate = _gate(
                    inputs,
                    task,
                    retained["parameters"],
                    _records(operations, f"{prefix}-recheck", size),
                    numerical,
                )
                ready &= all(
                    n in operations
                    and operations[n]["status"] != "running_or_interrupted"
                    for n in names
                )
            row.update(
                retained=retained,
                verified_improvement=changed,
                qualification_after=gate,
                complete=ready,
                recheck_source="fresh_full_coordinate_audit"
                if changed
                else "identical_point_matching_evidence_reused",
            )
            if not ready:
                break
            if not changed:
                reason = "no_verified_progress_same_point"
                break
            if not gate["complete"]:
                reason = "full_coordinate_recheck_incomplete"
                break
            if phase + 1 == task["segments"]:
                reason = "search_allowance_exhausted"
    if set(operations) - used:
        raise ValueError("operations exist beyond controlled stopping rule")
    return {
        "retained": retained,
        "final_qualification": gate,
        "phases": phases,
        "stop_reason": reason,
    }


def _fit_binding(task, shared, selection):
    return {
        "selection_sha256": public.content_sha256(selection),
        "shared_qualification_sha256": shared["qualification"]["qualification_sha256"],
        "qualification_sha256": selection["qualification_sha256"],
        "active_names": selection["active_names"],
        "held_names": selection["held_names"],
        "mode": task["mode"],
        "lazy_jacobian": task["lazy_jacobian"],
    }


def _assemble(
    identity, operations, inputs, policy, *, shared=None, terminal=False, overhead=None
):
    task = identity["task"]
    result = {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "operations": operations,
        "cost": journal._cost(operations),
        "invocations": overhead or [],
        **BOUNDARY,
    }
    if task["kind"] == "prequalification":
        result.update(
            _prequalification(inputs, task, operations), optimization_performed=False
        )
        if terminal:
            size = len(task["coordinate_domain"]["names"])
            required = {"start_verification", *(f"base-{i}" for i in range(size))}
            if not result["attempts"][0]["qualification"]["search_qualified"]:
                required |= {f"escalated-{i}" for i in range(size)}
            if set(operations) != required or any(
                o["status"] == "running_or_interrupted" for o in operations.values()
            ):
                raise ValueError("terminal prequalification has unfinished allowances")
    elif task["kind"] == "diagnostic_supplement":
        value = source._value(operations, "supplement_direction")
        gate = _gate(
            inputs,
            task,
            task["parameters"],
            [value] if value else [],
            _numerical(inputs),
        )
        if value and value.get("index") != task["coordinate_index"]:
            raise ValueError("supplement belongs to another coordinate")
        result.update(
            qualification=gate,
            parameters_changed=False,
            optimization_performed=False,
            new_allowance_seconds=policy.direction_seconds,
            historical_operation_receipt=inputs.source_report["tasks"][2][
                "operations"
            ].get(task["historical_operation"]),
        )
        if terminal and (
            set(operations) != {"supplement_direction"}
            or operations["supplement_direction"]["status"] == "running_or_interrupted"
        ):
            raise ValueError("terminal supplement has unfinished allowance")
    elif shared is None:
        if operations or terminal:
            raise ValueError("fit requires a terminal shared qualification")
        result.update(
            stop_reason="shared_qualification_pending", optimization_performed=False
        )
    else:
        state = _fit_state(inputs, task, operations, policy, shared)
        if terminal and state["stop_reason"] == "incomplete":
            raise ValueError("terminal fit has unfinished operations")
        searches = [n for n in operations if n.endswith("-search")]
        selected = _selection(
            task, state["retained"]["parameters"], state["final_qualification"]
        )
        result.update(
            state,
            optimization_performed=bool(searches),
            shared_qualification_sha256=shared["qualification"]["qualification_sha256"],
            numerical_policy=shared["numerical_policy"],
            active_names=selected["active_names"],
            held_names=selected["held_names"],
            final_direction_checks=state["final_qualification"]["rows"],
            missing_final_coordinates=state["final_qualification"]["missing_indices"],
            optimization_allowance_seconds=policy.optimization_seconds,
            optimization_reserved_seconds=len(searches)
            * policy.optimization_seconds
            / task["segments"],
            optimization_unused_allowance_seconds=(task["segments"] - len(searches))
            * policy.optimization_seconds
            / task["segments"],
        )
    return result


def _search_counters(row):
    """Separate native integrations from optimizer callback accounting."""
    searches = [
        operation
        for name, operation in row["operations"].items()
        if name.endswith("-search")
    ]
    records = [source.saved_fit._fit_checkpoint(operation) for operation in searches]
    counters = {}
    for field in (
        "residual_calls_started",
        "residual_calls_completed",
        "residual_calls_failed",
        "jacobian_calls",
        "residual_cache_hits",
        "trajectory_integrations",
    ):
        values = [record.get("numerical_cost", {}).get(field) for record in records]
        counters[field + "_observed"] = sum(value or 0 for value in values)
        counters[field + "_total"] = (
            sum(values)
            if all(value is not None for value in values)
            and all(op["status"] == "complete" for op in searches)
            else None
        )
    return counters | {
        "scope": "Native oracle integrations/Jacobians; backend callbacks and "
        "accepted/rejected trials remain in each search receipt and are not "
        "interpreted as integrations."
    }


def _comparisons(rows):
    """Expose matched contrasts without treating historical endpoints as controls."""
    results = []
    contrasts = (
        ("original_eager_vs_lazy", 0, 1),
        ("reduced_eager_vs_lazy", 2, 3),
        ("eager_original_vs_reduced", 0, 2),
        ("lazy_original_vs_reduced", 1, 3),
        ("reduced_lazy_contiguous_vs_segmented", 3, 4),
        ("reduced_lazy_all_vs_adaptive", 4, 5),
    )
    for start in range(3):
        for name, left, right in contrasts:
            a, b = rows[6 + start * 6 + left], rows[6 + start * 6 + right]
            left_value, right_value = a.get("retained"), b.get("retained")
            results.append(
                {
                    "contrast": name,
                    "start_id": a["identity"]["task"]["start_id"],
                    "left_task": a["identity"]["task"]["index"],
                    "right_task": b["identity"]["task"]["index"],
                    "same_original_start": a["identity"]["task"]["parameters"]
                    == b["identity"]["task"]["parameters"],
                    "same_initial_qualification": a.get("shared_qualification_sha256")
                    is not None
                    and a.get("shared_qualification_sha256")
                    == b.get("shared_qualification_sha256"),
                    "complete": a["status"] == b["status"] == "complete",
                    "right_minus_left_training_nmse": right_value["training_nmse"]
                    - left_value["training_nmse"]
                    if left_value and right_value
                    else None,
                    "left_cost": a["cost"],
                    "right_cost": b["cost"],
                    "left_search_counters": _search_counters(a),
                    "right_search_counters": _search_counters(b),
                    "full_model_convergence_claimed": False,
                }
            )
    return results


def _invocations(directory, identity):
    rows = []
    for folder in sorted(
        (directory / "invocations").glob("*"), key=lambda p: int(p.name)
    ):
        started = read_seal(folder / "started.json")
        if started["identity"] != identity:
            raise ValueError("invocation identity differs")
        if (folder / "result.json").exists():
            result = read_seal(folder / "result.json")
            if result["started_sha256"] != public.content_sha256(started):
                raise ValueError("invocation timing identity differs")
            rows.append(started | result)
        else:
            rows.append(
                started
                | {
                    "status": "interrupted_or_running",
                    "timing_accounting_complete": False,
                }
            )
    return rows


def _read_task(root, task, plan, inputs, policy, *, shared=None):
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(task["index"])
    terminal = (directory / "result.json").exists()
    row = _assemble(
        identity,
        _operations(directory, identity, policy),
        inputs,
        policy,
        shared=shared,
        terminal=terminal,
        overhead=_invocations(directory, identity),
    )
    if terminal and read_seal(directory / "result.json") != row:
        raise ValueError("lazy-control terminal differs from receipts")
    return row


def run_task(root: Path, index: int, *, context: VerifiedRun | None = None):
    """Run one stage with a cached operation map and nonrenewable reservations."""
    context = load_run(root) if context is None else context
    if (
        not isinstance(context, VerifiedRun)
        or context._authentication is not _AUTHENTICATED
        or context.root != root.resolve()
        or read_seal(root / "plan.json") != context.plan
        or context.plan["source_sha256"] != public._source_identity()
        or context.plan["runtime"] != public._runtime()
    ):
        raise ValueError("verified invocation context differs from campaign")
    began, cpu, started_at = (
        context.started_wall,
        context.started_cpu,
        context.started_utc,
    )
    plan, inputs, policy = context.plan, context.inputs, context.policy
    if isinstance(index, bool) or not isinstance(index, int) or index not in range(27):
        raise ValueError("unknown lazy-control task")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)
    shared = None
    if task["kind"] == "fit":
        shared = _read_task(
            root, plan["tasks"][task["qualification_task"]], plan, inputs, policy
        )
        if shared["status"] != "complete":
            raise ValueError(
                "shared qualification dependency is not terminal; "
                "no fit allowance spent"
            )
    with public._lock(directory):
        if (directory / "result.json").exists():
            return _read_task(root, task, plan, inputs, policy, shared=shared)
        operations = _operations(directory, identity, policy)
        allowed, problem = _allowances(task, policy), _problem(inputs, task)
        numerical = (
            kernel.Policy.model_validate(shared["numerical_policy"])
            if shared
            else _numerical(inputs)
        )
        invocation = (
            directory
            / "invocations"
            / str(len(list((directory / "invocations").glob("*"))))
        )
        start = {
            "identity": identity,
            "started_at_utc": started_at,
            "startup_wall_seconds": monotonic() - began,
            "startup_cpu_seconds": process_time() - cpu,
            "history_authentications": 1,
        }
        seal(invocation / "started.json", start)
        operation_wall, operation_cpu, numerical_wall, numerical_cpu = (
            0.0,
            0.0,
            0.0,
            0.0,
        )

        def operate(name, work):
            nonlocal operation_wall, operation_cpu, numerical_wall, numerical_cpu
            wall, proc = monotonic(), process_time()
            existing = name in operations

            def durable(save):
                result = work(save)
                save(result)
                return result

            try:
                value = journal._operation(
                    directory, identity, name, allowed[name], durable
                )
                operations[name] = value
                if not existing:
                    numerical_wall += value.get("wall_seconds") or 0
                    numerical_cpu += value.get("cpu_seconds") or 0
                return value
            except BaseException:
                # The journal seals caught interruptions before re-raising. Keep
                # their measured numerical time out of orchestration overhead.
                folder = directory / "operations" / name
                if not existing and (folder / "result.json").exists():
                    value = journal._read_operation(
                        folder,
                        {
                            "campaign": identity,
                            "stage": name,
                            "allowance_seconds": allowed[name],
                        },
                    )
                    operations[name] = value
                    numerical_wall += value.get("wall_seconds") or 0
                    numerical_cpu += value.get("cpu_seconds") or 0
                raise
            finally:
                operation_wall += monotonic() - wall
                operation_cpu += process_time() - proc

        def direction(name, parameters, coordinate, chosen_policy):
            operate(
                name,
                lambda save: kernel.direction(
                    *problem,
                    parameters,
                    coordinate,
                    mode=task["mode"],
                    purpose="search",
                    precision="tight",
                    seconds=allowed[name],
                    checkpoint=save,
                    policy=chosen_policy,
                ),
            )

        def verify_parameters(name, parameters):
            binding = {"requested_parameters_sha256": public.content_sha256(parameters)}
            return operate(
                name,
                lambda save: original.verify(
                    *problem,
                    parameters,
                    seconds=allowed[name],
                    policy=numerical.numerical,
                    checkpoint=lambda value: save(value | binding),
                )
                | binding,
            )

        completed = False
        try:
            if task["kind"] == "diagnostic_supplement":
                direction(
                    "supplement_direction",
                    task["parameters"],
                    task["coordinate_index"],
                    numerical,
                )
            elif task["kind"] == "prequalification":
                verify_parameters("start_verification", task["parameters"])
                size = len(task["coordinate_domain"]["names"])
                for i in range(size):
                    direction(f"base-{i}", task["parameters"], i, numerical)
                base = _prequalification(inputs, task, operations)
                if not base["qualification"]["search_qualified"]:
                    for i in range(size):
                        direction(
                            f"escalated-{i}",
                            task["parameters"],
                            i,
                            active.precision_policy(numerical),
                        )
            else:
                for phase in range(task["segments"]):
                    state = _fit_state(inputs, task, operations, policy, shared)
                    if state["stop_reason"] != "incomplete":
                        break
                    row = next(
                        (p for p in state["phases"] if p["phase"] == phase), None
                    )
                    if row and row["complete"]:
                        continue
                    if row is None:
                        raise ValueError("missing controlled phase state")
                    prefix, point, gate = (
                        f"phase-{phase}",
                        row["start"]["parameters"],
                        row["qualification_before"],
                    )
                    selection, binding = (
                        row["selection"],
                        _fit_binding(task, shared, row["selection"]),
                    )
                    operate(
                        f"{prefix}-search",
                        lambda save,
                        point=point,
                        selection=selection,
                        gate=gate,
                        prefix=prefix,
                        binding=binding: controls.fit_control(
                            *problem,
                            point,
                            mode=task["mode"],
                            lazy_jacobian=task["lazy_jacobian"],
                            active_names=selection["active_names"]
                            if task["adaptive"]
                            else None,
                            qualification=gate,
                            seconds=allowed[f"{prefix}-search"],
                            policy=numerical,
                            checkpoint=lambda value: save(value | binding),
                        )
                        | binding,
                    )
                    fit = source.saved_fit._fit_checkpoint(
                        operations[f"{prefix}-search"]
                    )
                    if fit and fit.get("best"):
                        verify_parameters(
                            f"{prefix}-verification", fit["best"]["parameters"]
                        )
                    state = _fit_state(inputs, task, operations, policy, shared)
                    if state["retained"]["parameters"] != point:
                        for i in range(len(gate["coordinate_names"])):
                            direction(
                                f"{prefix}-recheck-{i}",
                                state["retained"]["parameters"],
                                i,
                                numerical,
                            )
            _assemble(
                identity, operations, inputs, policy, shared=shared, terminal=True
            )
            completed = True
        finally:
            seal(
                invocation / "result.json",
                {
                    "started_sha256": public.content_sha256(start),
                    "status": "complete" if completed else "interrupted_or_failed",
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                    "elapsed_wall_seconds": monotonic() - began,
                    "elapsed_cpu_seconds": process_time() - cpu,
                    "operation_dispatch_wall_seconds": operation_wall,
                    "operation_dispatch_cpu_seconds": operation_cpu,
                    "operation_orchestration_wall_seconds": max(
                        0, operation_wall - numerical_wall
                    ),
                    "operation_orchestration_cpu_seconds": max(
                        0, operation_cpu - numerical_cpu
                    ),
                    "timing_accounting_complete": completed,
                    "scope": (
                        "Invocation includes source authentication and operations; "
                        "excludes final task sealing and scheduler wait. Dispatch "
                        "includes receipt IO; operation times are nested, not additive."
                    ),
                },
            )
        result = _assemble(
            identity,
            operations,
            inputs,
            policy,
            shared=shared,
            terminal=True,
            overhead=_invocations(directory, identity),
        )
        seal(directory / "result.json", result)
        return result


def report(root: Path):
    """Read receipts only; never qualify with a rollout or resume numerical work."""
    plan, inputs, policy = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        shared = rows[task["qualification_task"]] if task["kind"] == "fit" else None
        if shared is not None and shared["status"] != "complete":
            shared = None
        rows.append(_read_task(root, task, plan, inputs, policy, shared=shared))
    old = inputs.source_report
    endpoint = old["tasks"][2]
    records = _records(endpoint["operations"], "phase-2-recheck", 8)
    records += [
        value
        for row in rows[24:]
        if (value := source._value(row["operations"], "supplement_direction"))
        is not None
    ]
    supplement = _gate(
        inputs,
        plan["tasks"][24],
        endpoint["retained"]["parameters"],
        records,
        _numerical(inputs),
    )
    return {
        "protocol": PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": 27,
        "recorded": sum(r["status"] == "complete" for r in rows),
        "tasks": rows,
        "policy": policy.model_dump(mode="json"),
        "source_report_sha256": inputs.source_report_sha256,
        "historical_active_endpoints": [
            {"index": i, "retained": row["retained"], "cost": row["cost"]}
            for i, row in enumerate(old["tasks"][:3])
        ],
        "historical_costs": [
            {"index": i, "cost": row["cost"]} for i, row in enumerate(old["tasks"])
        ],
        "historical_predecessor_costs": {
            k: old[k]
            for k in (
                "historical_audit_costs",
                "historical_original_arm_endpoints",
                "historical_continuation_costs",
                "historical_predecessor_costs",
            )
        },
        "historical_generic_endpoint_supplement": supplement,
        "shared_qualification_costs": [
            {"index": i, "cost": rows[i]["cost"]} for i in range(6)
        ],
        "controlled_comparisons": _comparisons(rows),
        "comparison_scope": (
            "fresh six-arm fixed-start controls; equal search "
            "allowance, diagnostic and orchestration costs reported "
            "separately; historical endpoints are references"
        ),
        **BOUNDARY,
    }
