"""Matched detention continuations with immutable history and original-model checks.

The two Model 15 arms start from the same retained vector. Model 14 temporarily
holds one coordinate, then explicitly rechecks the full model before release.
No equation, original parameter domain, data split or historical budget changes.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_detention_continuation as kernel
from autoformalism.fitting import llm_sr_detention_recovery as diagnostic
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.fitting.llm_sr_detention_stationarity import enrich_diagnosis
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_detention_recovery_campaign as source
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256

PROTOCOL = "llm-sr-detention-continuation-1"
SELECTION = (
    "recovery 3/4/5 paired original/reduced; recovery 2 temporary subset/release"
)
ROSTER = (
    *((index, mode) for index in (3, 4, 5) for mode in ("original", "reduced_model15")),
    (2, "subset_model14"),
)
ASSESSMENT_KEYS = {"stationarity_assessment", "rank_assessment"}


class Parent(StrictSchema):
    """One authenticated recovery terminal and its complete operation records."""

    index: int = Field(ge=0, le=5)
    terminal: dict
    operations: dict


class Inputs(StrictSchema):
    """Portable training-only history; includes successes and failures alike."""

    protocol: Literal["llm-sr-detention-continuation-input-1"] = (
        "llm-sr-detention-continuation-input-1"
    )
    source_plan: dict
    source_plan_sha256: Sha256
    source_inputs: dict
    source_report: dict
    source_report_sha256: Sha256
    parents: tuple[Parent, ...] = Field(min_length=6, max_length=6)
    validation_data_opened: Literal[False] = False
    test_data_opened: Literal[False] = False
    llm_calls: Literal[0] = 0
    _validated_source: source.Inputs | None = PrivateAttr(default=None)
    _problems: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def authenticated_history(self):
        self._validated_source = _validate_history(self)
        return self


class Policy(StrictSchema):
    """Equal main search allowances; a separately reported conditional release."""

    protocol: Literal["llm-sr-detention-continuation-policy-1"] = (
        "llm-sr-detention-continuation-policy-1"
    )
    optimization_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    release_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    direction_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    verification_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    aggregation_seconds: FiniteFloat = Field(default=30, gt=0, le=30)
    accuracy_nmse_threshold: FiniteFloat = Field(default=1e-3, gt=0, le=1)
    numerical: diagnostic.Policy = Field(default_factory=diagnostic.Policy)


def _source_report(plan, old, rows):
    """Reconstruct the previous report, including all predecessor accounting."""
    return {
        "protocol": source.PROTOCOL,
        "status": "complete",
        "expected": 6,
        "recorded": 6,
        "tasks": rows,
        "policy": plan["policy"],
        "selection_rule": source.SELECTION,
        "historical_report_sha256": old.source_report_sha256,
        "historical_costs": [
            {"source_index": p.index, "cost": p.terminal["cost"]} for p in old.parents
        ],
        "historical_outcomes": [
            {
                "source_index": p.index,
                "optimizer_stop_reason": p.terminal["optimizer_stop_reason"],
                "training_nmse": (p.terminal.get("retained") or {}).get(
                    "training_nmse"
                ),
            }
            for p in old.parents
        ],
        "historical_predecessor_costs": old.source_report["historical_costs"],
        "historical_allowances_renewed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def _validate_history(inputs):
    old = source.Inputs.model_validate(inputs.source_inputs)
    plan = inputs.source_plan
    policy = source.Policy.model_validate(plan["policy"])
    expected_plan = {
        "protocol": source.PROTOCOL,
        "inputs_sha256": public.content_sha256(old),
        "policy": policy.model_dump(mode="json"),
        "tasks": source._tasks(old),
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
        "selection_rule": source.SELECTION,
        "historical_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_promotion": False,
    }
    if (
        plan != expected_plan
        or public.content_sha256(plan) != inputs.source_plan_sha256
        or public.content_sha256(inputs.source_report) != inputs.source_report_sha256
    ):
        raise ValueError("recovery history plan/report identity differs")
    rows = []
    for index, (task, parent) in enumerate(
        zip(plan["tasks"], inputs.parents, strict=True)
    ):
        identity = {"plan_sha256": inputs.source_plan_sha256, "task": task}
        allowed = source._allowances(task, policy)
        if parent.index != index or set(parent.operations) - set(allowed):
            raise ValueError("recovery history operation roster differs")
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
                raise ValueError("recovery history operation differs")
        result = source._assemble(
            identity,
            parent.operations,
            policy,
            source._problem(old, task),
            terminal=True,
        )
        if result != parent.terminal:
            raise ValueError("recovery history terminal differs from receipts")
        rows.append(result)
    if _source_report(plan, old, rows) != inputs.source_report:
        raise ValueError("recovery history report differs from receipts")
    for index, _ in ROSTER:
        point = inputs.parents[index].terminal.get("retained")
        if point is None or not numerics.usable(point):
            raise ValueError("continuation requires independently verified endpoints")
    return old


def export_inputs(root: Path, destination: Path) -> dict:
    """Export six authenticated terminals without writing inside their campaign."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("export cannot write inside the historical campaign")
    plan, old, policy = source.verify(root, runtime=False)
    parents = []
    for task in plan["tasks"]:
        index = task["index"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "tasks" / str(index)
        parents.append(
            Parent(
                index=index,
                terminal=read_seal(folder / "result.json"),
                operations=source._operations(folder, identity, policy),
            )
        )
    report = public._read(root / "report/summary.json")
    inputs = Inputs(
        source_plan=plan,
        source_plan_sha256=public.content_sha256(plan),
        source_inputs=old.model_dump(mode="json"),
        source_report=report,
        source_report_sha256=public.content_sha256(report),
        parents=parents,
    )
    seal(destination, inputs.model_dump(mode="json"))
    return {
        "inputs_sha256": public.content_sha256(inputs),
        "historical_tasks": 6,
        "tasks": 7,
    }


def _problem(inputs, task):
    index = task["source_index"]
    if index not in inputs._problems:
        old = inputs._validated_source
        if old is None:
            raise ValueError("continuation inputs have not been authenticated")
        inputs._problems[index] = source._problem(
            old, inputs.source_plan["tasks"][index]
        )
    return inputs._problems[index]


def _tasks(inputs):
    tasks = []
    for source_index, mode in ROSTER:
        parent = inputs.parents[source_index].terminal
        old = inputs.source_plan["tasks"][source_index]
        task = {
            "index": len(tasks),
            "source_index": source_index,
            "source_task": old["source_task"],
            "start_id": old["start_id"],
            "mode": mode,
            "parameters": deepcopy(parent["retained"]["parameters"]),
            "parent_verified_checkpoint": deepcopy(parent["retained"]),
            "source_terminal_sha256": public.content_sha256(parent),
            "original_coordinate_names": old["coordinate_names"],
        }
        task["coordinate_domain"] = kernel.domain(
            *_problem(inputs, task), task["parameters"], mode=mode
        )
        tasks.append(task)
    return tasks


def _plan(inputs, policy, *, source_sha256, runtime):
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(inputs),
        "source_sha256": source_sha256,
        "runtime": runtime,
        "selection_rule": SELECTION,
        "historical_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_promotion": False,
    }


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None):
    """Freeze a fresh matched experiment; never renew old task allowances."""
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
            raise ValueError("cannot prepare nonempty unsealed continuation directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 7}


def verify(root: Path, *, runtime=True):
    plan = read_seal(root / "plan.json")
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    policy = Policy.model_validate(plan["policy"])
    if plan != _plan(
        inputs, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
    ):
        raise ValueError("detention continuation plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("detention continuation source/runtime differs")
    return plan, inputs, policy


def _allowances(task, policy):
    n = len(task["coordinate_domain"]["names"])
    full = len(task["original_coordinate_names"])
    result = {
        "start_verification": policy.verification_seconds,
        "import_directions": policy.aggregation_seconds,
        "qualification": policy.aggregation_seconds,
        "optimization": policy.optimization_seconds,
        "endpoint_verification": policy.verification_seconds,
        "retained_diagnosis": policy.aggregation_seconds,
        **{f"search-{i:02d}": policy.direction_seconds for i in range(n)},
        **{f"diagnosis-{i:02d}": policy.direction_seconds for i in range(full)},
    }
    if task["mode"] == "subset_model14":
        result |= {
            **{f"full-recheck-{i:02d}": policy.direction_seconds for i in range(full)},
            "release_import_directions": policy.aggregation_seconds,
            "release_qualification": policy.aggregation_seconds,
            "release_optimization": policy.release_seconds,
            "release_verification": policy.verification_seconds,
        }
    return result


def _operations(directory, identity, policy):
    allowed = _allowances(identity["task"], policy)
    if any(p.name not in allowed for p in (directory / "operations").glob("*")):
        raise ValueError("unknown detention continuation operation")
    operations = {}
    for name, seconds in allowed.items():
        folder = directory / "operations" / name
        binding = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (folder / "result.json").exists():
            operations[name] = journal._read_operation(folder, binding)
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != binding:
                raise ValueError("continuation operation reservation differs")
            operations[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "budget_charge_seconds": seconds,
                "wall_seconds": None,
                "cpu_seconds": None,
                "accounting_complete": False,
            }
    return operations


def _value(operations, name):
    row = operations.get(name, {})
    return row.get("value") if row.get("status") == "complete" else None


def _directions(operations, prefix, count):
    return [
        value
        for i in range(count)
        if isinstance(value := _value(operations, f"{prefix}-{i:02d}"), dict)
    ]


def _retained(task, operations, *, release=True):
    retained = task["parent_verified_checkpoint"]
    origin = "parent_verified_checkpoint"
    fit = source.source._fit_checkpoint(operations.get("optimization", {}))
    released = source.source._fit_checkpoint(operations.get("release_optimization", {}))
    expected = {
        "start_verification": task["parameters"],
        "endpoint_verification": (fit.get("best") or {}).get("parameters"),
        "release_verification": (released.get("best") or {}).get("parameters"),
    }
    for name, parameters in expected.items():
        if name == "release_verification" and not release:
            continue
        point = source.source._verified(operations.get(name, {}))
        if point and (
            point["parameters"] != parameters
            or point.get("requested_parameters_sha256")
            != public.content_sha256(parameters)
        ):
            raise ValueError(
                "verified continuation endpoint differs from its own search"
            )
        if point and (
            name == "start_verification" or numerics.improves(point, retained)
        ):
            retained, origin = point, name
    return retained, origin


def _search_directions(task, operations, *, release=False):
    imported = _value(
        operations, "release_import_directions" if release else "import_directions"
    )
    if isinstance(imported, dict):
        return imported.get("directions", [])
    return _directions(operations, "search", len(task["coordinate_domain"]["names"]))


def _final_directions(task, operations, retained):
    count = len(task["original_coordinate_names"])
    rows = _directions(operations, "full-recheck", count)
    if len(rows) == count and all(
        r.get("parameters") == retained["parameters"] for r in rows
    ):
        return rows
    return _directions(operations, "diagnosis", count)


def _enriched(problem, parameters, diagnosis, policy):
    return enrich_diagnosis(
        diagnosis,
        parameters=parameters,
        bounds=original.domain_policy(*problem)["bounds"],
        gradient_tolerance=policy.numerical.numerical.derivative.gradient_tolerance,
        bound_tolerance=policy.numerical.numerical.derivative.bound_tolerance,
    )


def _assemble(
    identity, operations, policy, problem, parent_directions, *, terminal=False
):
    task = identity["task"]
    retained, origin = _retained(task, operations)
    fit = source.source._fit_checkpoint(operations.get("optimization", {}))
    release_fit = source.source._fit_checkpoint(
        operations.get("release_optimization", {})
    )
    for release in (False, True):
        name = "release_import_directions" if release else "import_directions"
        imported = _value(operations, name)
        if imported is None:
            continue
        point = _retained(task, operations, release=False)[0] if release else None
        parameters = point["parameters"] if point else task["parameters"]
        rows = (
            _directions(
                operations, "full-recheck", len(task["original_coordinate_names"])
            )
            if release
            else parent_directions
        )
        expected = {
            "directions": kernel.import_original_directions(
                *problem,
                parameters,
                rows,
                mode="original" if release else task["mode"],
                policy=policy.numerical,
            ),
            "original_evidence_reused": True,
            "new_integrations": 0,
        }
        if imported != expected:
            raise ValueError("imported derivative evidence differs from its source")
    qualifications = {}
    for release in (False, True):
        name = "release_qualification" if release else "qualification"
        saved = _value(operations, name)
        if saved is None:
            continue
        start = _retained(task, operations, release=False)[0] if release else None
        parameters = start["parameters"] if release and start else task["parameters"]
        kernel.validate_qualification(
            *problem,
            parameters,
            _search_directions(task, operations, release=release),
            mode="original" if release else task["mode"],
            precision="tight",
            purpose="search",
            qualification=saved,
            policy=policy.numerical,
        )
        qualifications[name] = saved
        fitted = release_fit if release else fit
        if (
            fitted
            and fitted.get("qualification_sha256") != saved["qualification_sha256"]
        ):
            raise ValueError(
                "continuation search gate differs from saved qualification"
            )
    if fit and "qualification" not in qualifications:
        raise ValueError("continuation fit lacks its qualification receipt")
    if release_fit and "release_qualification" not in qualifications:
        raise ValueError("released fit lacks its full-coordinate qualification")
    diagnosis = _value(operations, "retained_diagnosis")
    if diagnosis:
        binding = (
            {
                "diagnosed_parameters_sha256": public.content_sha256(
                    retained["parameters"]
                ),
                "verification_sha256": public.content_sha256(retained),
            }
            if retained
            else {}
        )
        if retained is None or any(diagnosis.get(k) != v for k, v in binding.items()):
            raise ValueError("continuation diagnosis differs from retained endpoint")
        raw = {
            k: v
            for k, v in diagnosis.items()
            if k not in ASSESSMENT_KEYS | set(binding)
        }
        diagnostic.validate_diagnosis(
            *problem,
            retained["parameters"],
            directions=_final_directions(task, operations, retained),
            verification=retained,
            diagnosis=raw,
            policy=policy.numerical,
        )
        if (
            diagnosis
            != _enriched(problem, retained["parameters"], raw, policy) | binding
        ):
            raise ValueError("continuation diagnostic interpretation differs")
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "operations": operations,
        "cost": journal._cost(operations),
        "retained": retained,
        "retained_origin": origin,
        "fresh_start_verification_usable": source.source._verified(
            operations.get("start_verification", {})
        )
        is not None,
        "qualification": qualifications,
        "optimization": fit,
        "release_optimization": release_fit,
        "optimizer_stop_reason": fit.get("stop_reason", "not_started"),
        "release_stop_reason": release_fit.get("stop_reason", "not_requested"),
        "diagnosis": diagnosis,
        "diagnosis_complete": bool(diagnosis and diagnosis.get("status") == "complete"),
        "accuracy": "unavailable"
        if retained is None
        else "within_predeclared_training_threshold"
        if retained["training_nmse"] <= policy.accuracy_nmse_threshold
        else "above_predeclared_training_threshold",
        "global_optimum_certified": False,
        "historical_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def run_task(root: Path, index: int):
    """Execute stages once, reusing authenticated matching evidence only."""
    plan, inputs, policy = verify(root)
    if isinstance(index, bool) or index not in range(7):
        raise ValueError("unknown detention continuation task")
    task = plan["tasks"][index]
    problem = _problem(inputs, task)
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)
    allowed = _allowances(task, policy)
    parent = inputs.parents[task["source_index"]].terminal
    parent_directions = source._directions(
        parent["operations"], parent["identity"]["task"], "diagnosis", "tight"
    )

    def operations():
        return _operations(directory, identity, policy)

    def current(terminal=False):
        return _assemble(
            identity,
            operations(),
            policy,
            problem,
            parent_directions,
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
        return (
            original.verify(
                *problem,
                parameters,
                seconds=policy.verification_seconds,
                checkpoint=lambda v: save(v | binding),
                policy=policy.numerical.numerical,
            )
            | binding
        )

    def import_directions(name, parameters, rows, mode):
        return operate(
            name,
            lambda save: {
                "directions": kernel.import_original_directions(
                    *problem,
                    parameters,
                    rows,
                    mode=mode,
                    policy=policy.numerical,
                ),
                "original_evidence_reused": True,
                "new_integrations": 0,
            },
        )

    def qualify(name, parameters, rows, mode):
        operate(
            name,
            lambda save: kernel.qualification(
                *problem,
                parameters,
                rows,
                mode=mode,
                precision="tight",
                purpose="search",
                policy=policy.numerical,
            ),
        )
        return _value(operations(), name)

    def search(name, parameters, mode, qualification, seconds):
        binding = {"qualification_sha256": qualification["qualification_sha256"]}
        operate(
            name,
            lambda save: kernel.fit(
                *problem,
                parameters,
                mode=mode,
                qualification=qualification,
                seconds=seconds,
                checkpoint=lambda v: save(v | binding),
                policy=policy.numerical,
            )
            | binding,
        )

    def original_directions(prefix, parameters):
        for i in range(len(task["original_coordinate_names"])):
            operate(
                f"{prefix}-{i:02d}",
                lambda save, i=i: diagnostic.direction(
                    *problem,
                    parameters,
                    i,
                    purpose="diagnosis",
                    precision="tight",
                    seconds=policy.direction_seconds,
                    checkpoint=save,
                    policy=policy.numerical,
                ),
            )
        return _directions(operations(), prefix, len(task["original_coordinate_names"]))

    with public._lock(directory):
        if (directory / "result.json").exists():
            result = current(terminal=True)
            if result != read_seal(directory / "result.json"):
                raise ValueError("continuation terminal differs from receipts")
            return result
        operate(
            "start_verification",
            lambda save: verify_parameters(task["parameters"], save),
        )
        if source.source._verified(operations().get("start_verification", {})) is None:
            result = current(terminal=True)
            seal(directory / "result.json", result)
            return result
        if task["mode"] == "reduced_model15":
            for i in range(len(task["coordinate_domain"]["names"])):
                operate(
                    f"search-{i:02d}",
                    lambda save, i=i: kernel.direction(
                        *problem,
                        task["parameters"],
                        i,
                        mode=task["mode"],
                        purpose="search",
                        precision="tight",
                        seconds=policy.direction_seconds,
                        checkpoint=save,
                        policy=policy.numerical,
                    ),
                )
        else:
            import_directions(
                "import_directions", task["parameters"], parent_directions, task["mode"]
            )
        gate = qualify(
            "qualification",
            task["parameters"],
            _search_directions(task, operations()),
            task["mode"],
        )
        if gate is not None:
            search(
                "optimization",
                task["parameters"],
                task["mode"],
                gate,
                policy.optimization_seconds,
            )
            best = current()["optimization"].get("best")
            if best:
                operate(
                    "endpoint_verification",
                    lambda save: verify_parameters(best["parameters"], save),
                )
        retained = current()["retained"]
        if retained and task["mode"] == "subset_model14":
            rows = original_directions("full-recheck", retained["parameters"])
            import_directions(
                "release_import_directions", retained["parameters"], rows, "original"
            )
            gate = qualify(
                "release_qualification",
                retained["parameters"],
                _search_directions(task, operations(), release=True),
                "original",
            )
            if gate is not None and gate["search_qualified"]:
                search(
                    "release_optimization",
                    retained["parameters"],
                    "original",
                    gate,
                    policy.release_seconds,
                )
                best = current()["release_optimization"].get("best")
                if best:
                    operate(
                        "release_verification",
                        lambda save: verify_parameters(best["parameters"], save),
                    )
        retained = current()["retained"]
        if retained:
            rows = _final_directions(task, operations(), retained)
            if len(rows) != len(task["original_coordinate_names"]):
                rows = original_directions("diagnosis", retained["parameters"])
            binding = {
                "diagnosed_parameters_sha256": public.content_sha256(
                    retained["parameters"]
                ),
                "verification_sha256": public.content_sha256(retained),
            }
            operate(
                "retained_diagnosis",
                lambda save: _enriched(
                    problem,
                    retained["parameters"],
                    diagnostic.diagnose(
                        *problem,
                        retained["parameters"],
                        directions=rows,
                        verification=retained,
                        policy=policy.numerical,
                    ),
                    policy,
                )
                | binding,
            )
        result = current(terminal=True)
        seal(directory / "result.json", result)
        return result


def report(root: Path):
    """Reconstruct every terminal and paired comparison without numerical fitting."""
    plan, inputs, policy = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        directory = root / "tasks" / str(task["index"])
        done = (directory / "result.json").exists()
        parent = inputs.parents[task["source_index"]].terminal
        row = _assemble(
            identity,
            _operations(directory, identity, policy),
            policy,
            _problem(inputs, task),
            source._directions(
                parent["operations"], parent["identity"]["task"], "diagnosis", "tight"
            ),
            terminal=done,
        )
        if done and row != read_seal(directory / "result.json"):
            raise ValueError("continuation terminal differs from receipts")
        rows.append(row)
    comparisons = []
    for a, b in zip(rows[0:6:2], rows[1:6:2], strict=True):
        started = [
            row["optimization"].get(
                "calls_started", row["optimization"].get("calls", 0)
            )
            > 0
            for row in (a, b)
        ]
        comparisons.append(
            {
                "source_index": a["identity"]["task"]["source_index"],
                "same_start": a["identity"]["task"]["parameters"]
                == b["identity"]["task"]["parameters"],
                "optimization_allowance_seconds_per_arm": policy.optimization_seconds,
                "both_search_started": all(started),
                "original_search_started": started[0],
                "reduced_search_started": started[1],
                "original_search_wall_seconds": (
                    a["operations"].get("optimization", {}).get("wall_seconds")
                    if started[0]
                    else 0.0
                ),
                "reduced_search_wall_seconds": (
                    b["operations"].get("optimization", {}).get("wall_seconds")
                    if started[1]
                    else 0.0
                ),
                "original_training_nmse": (a.get("retained") or {}).get(
                    "training_nmse"
                ),
                "reduced_training_nmse": (b.get("retained") or {}).get("training_nmse"),
                "cost_scope": (
                    "equal search allowances; qualification/diagnosis overhead "
                    "reported separately"
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": 7,
        "recorded": sum(r["status"] == "complete" for r in rows),
        "tasks": rows,
        "paired_comparisons": comparisons,
        "policy": policy.model_dump(mode="json"),
        "selection_rule": SELECTION,
        "historical_report_sha256": inputs.source_report_sha256,
        "historical_costs": [
            {"source_index": p.index, "cost": p.terminal["cost"]}
            for p in inputs.parents
        ],
        "historical_predecessor_costs": inputs.source_report["historical_costs"],
        "historical_earlier_predecessor_costs": inputs.source_report[
            "historical_predecessor_costs"
        ],
        "historical_allowances_renewed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
