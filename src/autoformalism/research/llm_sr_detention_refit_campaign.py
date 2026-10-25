"""Train-only refitting controls with authenticated supplied-value history.

Both detention models are selected by their predeclared control role. Every
start owns its own verified incumbents and a nonrenewable optimization attempt.
The supplied endpoint is comparison evidence, never another start's fallback.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_detention_refit as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_instantaneous_campaign as source
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256

PROTOCOL = "llm-sr-detention-refit-1"
INPUT_PROTOCOL = "llm-sr-detention-refit-input-1"
SELECTION = "both detention current-value numerical controls; no score-based selection"
SELECTED = (14, 15)


class Parent(StrictSchema):
    """One original terminal and the complete authenticated operation checkpoints."""

    index: int = Field(ge=0, le=3)
    terminal: dict
    operations: dict


class Inputs(StrictSchema):
    """Portable public training and all four immutable predecessor outcomes."""

    protocol: Literal["llm-sr-detention-refit-input-1"] = INPUT_PROTOCOL
    source_plan: dict
    source_plan_sha256: Sha256
    source_report: dict
    source_report_sha256: Sha256
    parents: tuple[Parent, Parent, Parent, Parent]
    selection_rule: Literal[
        "both detention current-value numerical controls; no score-based selection"
    ] = SELECTION
    validation_data_opened: Literal[False] = False
    test_data_opened: Literal[False] = False
    llm_calls: Literal[0] = 0

    @model_validator(mode="after")
    def authenticated_history(self) -> Inputs:
        """Reconstruct the original report, not just its declared checksum."""
        _validate_history(self)
        return self


class Policy(StrictSchema):
    """One matched optimization allowance per start, with separate verification."""

    protocol: Literal["llm-sr-detention-refit-policy-1"] = (
        "llm-sr-detention-refit-policy-1"
    )
    optimization_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    verification_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    diagnostic_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    start_seed: Literal[20261025] = 20261025
    accuracy_nmse_threshold: FiniteFloat = Field(default=1e-3, gt=0, le=1)
    numerical: kernel.Policy = Field(default_factory=kernel.Policy)


def _historical_stages(operations: dict) -> dict:
    """Mirror the read-only predecessor report's stage summaries exactly."""
    stages = {}
    for name, operation in operations.items():
        value = operation.get("value") or {}
        stages[name] = {
            "operation_status": operation["status"],
            "replay_status": value.get("status"),
            "normalized_mse": value.get("normalized_mse"),
            "per_target_normalized_mse": value.get("per_target_normalized_mse", {}),
            "failed_trajectories": value.get("failed_trajectories", {}),
            "budget_exhausted": value.get("budget_exhausted"),
            **{
                key: operation[key]
                for key in (
                    "wall_seconds",
                    "cpu_seconds",
                    "budget_charge_seconds",
                    "budget_restarted",
                )
            },
            "per_trajectory_diagnostics": [
                {key: value for key, value in row.items() if key != "predictions"}
                for row in value.get("trajectories", [])
            ],
        }
    return stages


def _validate_history(inputs: Inputs) -> source.Inputs:
    """Authenticate the complete prior plan, terminals, progress and cost chain."""
    plan = inputs.source_plan
    old = source.Inputs.model_validate(plan["inputs"])
    policy = source.Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != source.PROTOCOL
        or public.content_sha256(plan) != inputs.source_plan_sha256
        or plan["input_sha256"] != public.content_sha256(old)
        or plan["selection_rule"]
        != "all four supplied programs with current-value rewrites"
        or plan["training_content_sha256"]
        != {str(t.task): public.content_sha256(t.training) for t in old.tasks}
        or public.content_sha256(inputs.source_report) != inputs.source_report_sha256
    ):
        raise ValueError("source plan/report identity differs")
    rows = []
    for index, (task, parent) in enumerate(zip(old.tasks, inputs.parents, strict=True)):
        identity = source.common._identity(plan, task)
        terminal, operations = parent.terminal, parent.operations
        if parent.index != index or terminal["identity"] != identity:
            raise ValueError("source parent roster differs")
        if set(operations) != {"primary", "verification"} or set(
            terminal["operations"]
        ) != set(operations):
            raise ValueError("all source replay receipts required")
        for name, operation in operations.items():
            bound = {
                "campaign": identity,
                "stage": name,
                "allowance_seconds": getattr(policy, name).seconds,
            }
            if (
                operation["identity"] != bound
                or operation["progress_sha256"]
                != public.content_sha256(operation.get("observed_progress", {}))
                or terminal["operations"][name] != public.content_sha256(operation)
                or operation["budget_restarted"] is not False
            ):
                raise ValueError("source operation receipt differs")
        agreement = source.common._agreement(operations, policy)
        if (
            terminal["solver_agreement"] != agreement
            or terminal["stage_cost"] != journal._cost(operations)
            or terminal["status"]
            != ("replay_verified" if agreement["verified"] else "replay_unverified")
            or terminal["optimizer_calls"] != 0
            or any(
                terminal[key] is not False
                for key in (
                    "native_program_executed",
                    "native_equivalence_certified",
                    "validation_or_test_access",
                    "model_selected",
                    "refit_ready",
                )
            )
        ):
            raise ValueError("source terminal evidence differs")
        rows.append(terminal | {"stages": _historical_stages(operations)})
    expected = {
        "protocol": source.PROTOCOL,
        "plan_sha256": inputs.source_plan_sha256,
        "training_content_sha256": plan["training_content_sha256"],
        "policy": policy.model_dump(mode="json"),
        "tasks": rows,
        "status": "complete",
        "optimizer_calls": 0,
        "model_selection_performed": False,
        "validation_or_test_access": False,
    }
    if expected != inputs.source_report:
        raise ValueError("source report differs from its authenticated receipts")
    return old


def export_inputs(root: Path, destination: Path) -> dict:
    """Read all four completed source receipts without rewriting their report."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("export must not write inside source campaign")
    plan, _, _ = source.verify(root, runtime=False)
    reconstructed = source.report(root)
    report = public._read(root / "report/summary.json")
    if reconstructed != report or report["status"] != "complete":
        raise ValueError("complete unchanged source report required")
    parents = []
    for index, _task in enumerate(source.Inputs.model_validate(plan["inputs"]).tasks):
        directory = root / "tasks" / str(index)
        terminal = read_seal(directory / "result.json")
        operations = {}
        for name in ("primary", "verification"):
            folder = directory / "operations" / name
            bound = read_seal(folder / "started.json")
            operations[name] = journal._read_operation(folder, bound)
        parents.append({"index": index, "terminal": terminal, "operations": operations})
    inputs = Inputs(
        source_plan=plan,
        source_plan_sha256=public.content_sha256(plan),
        source_report=report,
        source_report_sha256=public.content_sha256(report),
        parents=parents,
    )
    seal(destination, inputs.model_dump(mode="json"))
    return {
        "inputs_sha256": public.content_sha256(inputs),
        "authenticated_parent_tasks": 4,
        "selected_source_tasks": list(SELECTED),
        "selection_rule": SELECTION,
        "historical_costs_preserved": True,
        "validation_data_opened": False,
        "test_data_opened": False,
    }


def _tasks(inputs: Inputs, policy: Policy) -> list[dict]:
    """Freeze all ten starts before inspecting any new optimization outcome."""
    old = source.Inputs.model_validate(inputs.source_plan["inputs"])
    tasks = []
    for task in old.tasks:
        if task.task not in SELECTED:
            continue
        item = old.transcription_set.for_task(task.task)
        assert item is not None
        starts = kernel.deterministic_starts(
            item, task.context, task.training, seed=policy.start_seed
        )
        if len(starts) != 5 or [s["start_kind"] for s in starts] != [
            "supplied",
            "perturbed",
            "perturbed",
            "generic",
            "generic",
        ]:
            raise ValueError(
                "expected one supplied, two perturbed and two generic starts"
            )
        if len({s["start_id"] for s in starts}) != 5:
            raise ValueError("start identifiers must be distinct")
        for start in starts:
            tasks.append(
                {
                    "index": len(tasks),
                    "source_task": task.task,
                    "benchmark_id": task.benchmark_id,
                    "repetition": task.repetition,
                    **start,
                    "domain_policy": kernel.domain_policy(
                        item, task.context, task.training
                    ),
                }
            )
    if len(tasks) != 10:
        raise ValueError("both detention numerical controls required")
    return public._jsonable(tasks)


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None) -> dict:
    """Seal public history, fixed start roster and source/runtime identities."""
    inputs = Inputs.model_validate(
        inputs.model_dump(mode="json") if isinstance(inputs, Inputs) else inputs
    )
    policy = Policy.model_validate((policy or Policy()).model_dump(mode="json"))
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(inputs, policy),
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
            path.name != ".lock" for path in root.iterdir()
        ):
            raise ValueError("cannot prepare nonempty unsealed refit directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, Inputs, Policy]:
    """Reject changed data, starts, domains, source or runtime before execution."""
    plan = read_seal(root / "plan.json")
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(inputs)
        or plan["tasks"] != _tasks(inputs, policy)
        or plan["selection_rule"] != SELECTION
        or plan["llm_calls"] != 0
        or any(
            plan[key] is not False
            for key in (
                "historical_allowances_renewed",
                "validation_data_opened",
                "test_data_opened",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("detention refit plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("detention refit source/runtime differs")
    return plan, inputs, policy


def _allowances(policy: Policy) -> dict[str, float]:
    return {
        "start_verification": policy.verification_seconds,
        "optimization": policy.optimization_seconds,
        "endpoint_verification": policy.verification_seconds,
        "retained_diagnosis": policy.diagnostic_seconds,
    }


def _operations(directory: Path, identity: dict, policy: Policy) -> dict:
    """Authenticate completed progress; keep unclosed reservation costs visible."""
    allowed, operations = _allowances(policy), {}
    if any(path.name not in allowed for path in (directory / "operations").glob("*")):
        raise ValueError("unknown detention refit operation")
    for name, seconds in allowed.items():
        folder = directory / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (folder / "result.json").exists():
            operations[name] = journal._read_operation(folder, bound)
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != bound:
                raise ValueError("detention operation reservation differs")
            operations[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return operations


def _checkpoint(operation: dict) -> dict:
    value = operation.get("value")
    return value if isinstance(value, dict) else operation.get("observed_progress", {})


def _fit_checkpoint(operation: dict) -> dict:
    """Preserve the latest accepted search point across a hard interruption."""
    value = _checkpoint(operation)
    search = value.get("search")
    if not isinstance(search, dict):
        return value
    result = value | search
    if value.get("best") and (
        not result.get("best")
        or value["best"]["training_nmse"] < result["best"]["training_nmse"]
    ):
        result["best"] = value["best"]
    return result


def _verified(operation: dict) -> dict | None:
    value = _checkpoint(operation)
    if not value.get("usable_for_retention"):
        return None
    score = value.get("training_nmse")
    if (
        not isinstance(score, (int, float))
        or not isfinite(score)
        or score < 0
        or not isinstance(value.get("parameters"), dict)
    ):
        raise ValueError("usable verification lacks finite training evidence")
    if not numerics.usable(value):
        raise ValueError("usable verification lacks independent solver agreement")
    return value


def _assemble(
    identity: dict, operations: dict, policy: Policy, *, terminal=False
) -> dict:
    """Select only this start's independently verified endpoints; never reference."""
    retained, origin = None, None
    fit = _fit_checkpoint(operations.get("optimization", {}))
    for name in ("start_verification", "endpoint_verification"):
        point = _verified(operations.get(name, {}))
        if point:
            expected = (
                identity["task"]["parameters"]
                if name == "start_verification"
                else (fit.get("best") or {}).get("parameters")
            )
            if point.get("parameters") != expected or point.get(
                "requested_parameters_sha256"
            ) != public.content_sha256(expected):
                raise ValueError(
                    "verified parameters differ from their own start/search"
                )
        if point and numerics.improves(point, retained):
            retained, origin = point, name
    diagnosis = _checkpoint(operations.get("retained_diagnosis", {}))
    if diagnosis and (
        retained is None
        or diagnosis.get("diagnosed_parameters_sha256")
        != public.content_sha256(retained["parameters"])
        or diagnosis.get("verification_sha256") != public.content_sha256(retained)
    ):
        raise ValueError("diagnosis differs from retained endpoint")
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "operations": operations,
        "cost": journal._cost(operations),
        "retained": retained,
        "retained_origin": origin,
        "accuracy": "unavailable"
        if retained is None
        else (
            "within_predeclared_training_threshold"
            if retained["training_nmse"] <= policy.accuracy_nmse_threshold
            else "above_predeclared_training_threshold"
        ),
        "optimization": fit,
        "optimizer_stop_reason": fit.get("stop_reason", "not_started"),
        "fitting_adequacy": "not_certified_by_this_control",
        "diagnosis": diagnosis or None,
        "diagnosis_complete": (
            operations.get("retained_diagnosis", {}).get("status") == "complete"
            and diagnosis.get("status") == "complete"
        ),
        "convergence_is_accuracy": False,
        "global_optimum_certified": False,
        "supplied_reference_used_as_fallback": False,
        "historical_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def run_task(root: Path, index: int) -> dict:
    """Run each stage once, preserving checkpointed work after interruption."""
    plan, inputs, policy = verify(root)
    if index not in range(len(plan["tasks"])):
        raise ValueError("unknown detention refit task index")
    task = plan["tasks"][index]
    old = source.Inputs.model_validate(inputs.source_plan["inputs"])
    source_task = next(t for t in old.tasks if t.task == task["source_task"])
    item = old.transcription_set.for_task(source_task.task)
    assert item is not None
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)
    allowed = _allowances(policy)

    def current(terminal=False):
        return _assemble(
            identity,
            _operations(directory, identity, policy),
            policy,
            terminal=terminal,
        )

    def operate(name, work):
        def durable(save):
            result = work(save)
            save(result)
            return result

        return journal._operation(directory, identity, name, allowed[name], durable)

    with public._lock(directory):
        if (directory / "result.json").exists():
            result = current(terminal=True)
            if read_seal(directory / "result.json") != result:
                raise ValueError("detention terminal differs from receipts")
            return result

        def verify_parameters(parameters, save):
            binding = {"requested_parameters_sha256": public.content_sha256(parameters)}
            value = kernel.verify(
                item,
                source_task.context,
                source_task.training,
                parameters,
                seconds=policy.verification_seconds,
                checkpoint=lambda value: save(value | binding),
                policy=policy.numerical,
            )
            return value | binding

        operate(
            "start_verification",
            lambda save: verify_parameters(task["parameters"], save),
        )
        attempt = operate(
            "optimization",
            lambda save: kernel.fit(
                item,
                source_task.context,
                source_task.training,
                task["parameters"],
                seconds=policy.optimization_seconds,
                checkpoint=save,
                policy=policy.numerical,
            ),
        )
        best = _fit_checkpoint(attempt).get("best")
        if best:
            operate(
                "endpoint_verification",
                lambda save: verify_parameters(best["parameters"], save),
            )
        retained = current()["retained"]
        if retained:
            binding = {
                "diagnosed_parameters_sha256": public.content_sha256(
                    retained["parameters"]
                ),
                "verification_sha256": public.content_sha256(retained),
            }

            def diagnose(save):
                result = kernel.diagnose(
                    item,
                    source_task.context,
                    source_task.training,
                    retained["parameters"],
                    seconds=policy.diagnostic_seconds,
                    checkpoint=lambda value: save(value | binding),
                    policy=policy.numerical,
                )
                return result | binding

            operate("retained_diagnosis", diagnose)
        result = current(terminal=True)
        seal(directory / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Keep all starts and all historical costs; no cross-start/model promotion."""
    plan, inputs, policy = verify(root, runtime=False)
    rows = []
    references = {row["identity"]["task"]: row for row in inputs.source_report["tasks"]}
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        directory = root / "tasks" / str(task["index"])
        done = (directory / "result.json").exists()
        result = _assemble(
            identity, _operations(directory, identity, policy), policy, terminal=done
        )
        if done and read_seal(directory / "result.json") != result:
            raise ValueError("detention terminal differs from receipts")
        reference_row = references[task["source_task"]]
        reference = reference_row["stages"]["primary"].get("normalized_mse")
        if (
            not isinstance(reference, (int, float))
            or not isfinite(reference)
            or reference < 0
        ):
            reference = None
        retained = result.get("retained")
        rows.append(
            result
            | {
                "supplied_reference_training_nmse": reference,
                "supplied_reference_verified": reference_row["solver_agreement"][
                    "verified"
                ],
                "retained_to_supplied_nmse_ratio": (
                    retained["training_nmse"] / reference
                    if retained and reference is not None and reference > 0
                    else None
                ),
                "reference_comparison_role": (
                    "descriptive only; no optimizer seeding or promotion"
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] == "complete" for r in rows),
        "tasks": rows,
        "policy": policy.model_dump(mode="json"),
        "selection_rule": SELECTION,
        "supplied_reference": [
            deepcopy(row)
            for row in inputs.source_report["tasks"]
            if row["identity"]["task"] in SELECTED
        ],
        "historical_costs": [
            {
                "source_task": parent.terminal["identity"]["task"],
                "cost": parent.terminal["stage_cost"],
            }
            for parent in inputs.parents
        ],
        "historical_report_sha256": inputs.source_report_sha256,
        "model_selection_performed": False,
        "automatic_promotion": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
