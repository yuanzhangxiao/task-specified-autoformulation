"""Training-only local recovery of frozen models and independent numerical probes.

Every model gets one reduced-coordinate continuation from its best newly verified
historical endpoint. Old disputed vectors are a separate verification-only roster:
they never participate in start selection, fitting or automatic model promotion.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_affine
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_continuation as original_campaign
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_fit_reliability as previous
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-local-recovery-1"
INPUT_PROTOCOL = "construction-local-recovery-inputs-1"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonnegative = Annotated[FiniteFloat, Field(ge=0)]
ARMS = original_campaign.ARMS
COST_KEYS = previous.COST_KEYS


class Policy(StrictSchema):
    """One fresh bounded continuation per model, not per historical arm."""

    population: Literal["returned-eight", "synthetic-smoke"] = "returned-eight"
    search_seconds: Nonnegative = Field(default=600, gt=0, le=600)
    diagnostic_seconds: Nonnegative = Field(default=300, gt=0, le=300)
    verification_seconds: Nonnegative = Field(default=180, gt=0, le=180)
    maximum_calls: int = Field(default=1000, ge=1, le=1000)


class Endpoint(StrictSchema):
    """Original-coordinate endpoint with historical, not current, verification."""

    origin: str = Field(min_length=1, max_length=100)
    parameters: dict[str, FiniteFloat]
    parameters_sha256: Digest
    original_equations_verified: bool
    training_nmse: Nonnegative | None = None
    alternate_training_nmse: Nonnegative | None = None
    solver_loss_discrepancy: Nonnegative | None = None
    solver_prediction_discrepancy: Nonnegative | None = None
    per_target_normalized_mse: dict[str, Nonnegative] | None = None


class SourceArm(StrictSchema):
    """Authenticates one returned arm and keeps two historical phases separate."""

    arm: Literal["affine_original", "affine_reduced"]
    terminal_sha256: Digest
    operations: dict[str, previous.OperationEvidence]
    endpoint: Endpoint
    original_cost_known: bool
    original_cost: dict[str, Nonnegative | bool | None] | None
    reliability_cost: dict[str, Nonnegative | bool | None]


class ModelHistory(StrictSchema):
    """Each model owns its history once; replay rows never duplicate its costs."""

    model: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    problem_sha256: Digest
    arms: list[SourceArm] = Field(min_length=2, max_length=2)


class Replay(StrictSchema):
    """Predeclared numerical discrepancy probe, excluded from search selection."""

    model: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    source_arm: Literal["affine_original", "affine_reduced"]
    source_ordinal: int = Field(ge=0, le=3)
    source_receipt_sha256: Digest
    endpoint: Endpoint


class Inputs(StrictSchema):
    """No nested campaign history, free-form responses, or held-out arrays."""

    protocol: Literal["construction-local-recovery-inputs-1"]
    source_plan_sha256: Digest
    source_inputs_sha256: Digest
    training_inputs_sha256: Digest
    training_inputs: dict
    models: list[ModelHistory]
    replays: list[Replay]
    test_data_opened: Literal[False]
    validation_exported: Literal[False]
    old_allowances_renewed: Literal[False]


def _endpoint(value: dict, origin: str) -> dict:
    fields = Endpoint.model_fields.keys() - {"origin", "parameters_sha256"}
    point = {key: value.get(key) for key in fields}
    point["original_equations_verified"] = bool(
        value.get("original_equations_verified")
    )
    point.update(
        origin=origin,
        parameters_sha256=public.content_sha256(value["parameters"]),
    )
    return Endpoint.model_validate(point).model_dump(mode="json")


def _receipts(folder: Path, operations: dict) -> dict:
    """Project identities only after the old journal has authenticated progress."""
    evidence = {}
    for name, operation in operations.items():
        location = folder / "operations" / name
        receipt = (
            read_seal(location / "result.json")
            if (location / "result.json").exists()
            else None
        )
        evidence[name] = {
            "status": operation["status"],
            "receipt_sha256": public.content_sha256(receipt) if receipt else None,
            "reservation_sha256": public.content_sha256(
                read_seal(location / "started.json")
            ),
            "progress_sha256": receipt["progress_sha256"] if receipt else None,
        }
    return evidence


def export_inputs(source: Path, destination: Path) -> dict:
    """Authenticate all source receipts read-only and export strict projections."""
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("local recovery export must not write into source campaign")
    plan, old = previous.verify(source, runtime=False)
    settings = previous.Policy.model_validate(plan["policy"])
    histories, replays = [], []
    for model_index, model in enumerate(old["source_inputs"]["models"]):
        arms = []
        for arm_index, arm_name in enumerate(ARMS):
            index = 2 * model_index + arm_index
            task, arm = plan["tasks"][index], old["arms"][index]
            identity = {"plan_sha256": public.content_sha256(plan), "task": task}
            folder = source / "results" / task["task_id"] / arm_name
            permitted = previous._allowances(settings, arm)
            if any(
                p.is_dir() and p.name not in permitted
                for p in (folder / "operations").glob("*")
            ):
                raise ValueError("unknown source reliability operation")
            operations = previous._operations(folder, identity, settings, arm)
            value = previous._assemble(identity, arm, operations, terminal=True)
            previous._check_selected_start(folder, value)
            terminal = read_seal(folder / "result.json")
            if terminal != value:
                raise ValueError(
                    "source terminal differs from authenticated operations"
                )
            arms.append(
                {
                    "arm": arm_name,
                    "terminal_sha256": public.content_sha256(terminal),
                    "operations": _receipts(folder, operations),
                    "endpoint": _endpoint(value["retained"], value["retained_origin"]),
                    "original_cost_known": arm["source_cost_known"],
                    "original_cost": arm["source_cost"],
                    "reliability_cost": {
                        k: v for k, v in value["cost"].items() if k in COST_KEYS
                    },
                }
            )
            for ordinal, point in enumerate(arm["failed_endpoints"]):
                replays.append(
                    {
                        "model": model_index,
                        "task_id": model["task_id"],
                        "source_arm": arm_name,
                        "source_ordinal": ordinal,
                        "source_receipt_sha256": arm["operations"][point["origin"]][
                            "receipt_sha256"
                        ],
                        "endpoint": _endpoint(point, point["origin"]),
                    }
                )
        histories.append(
            {
                "model": model_index,
                "task_id": model["task_id"],
                "problem_sha256": public.content_sha256(model["problem"]),
                "arms": arms,
            }
        )
    data = {
        "protocol": INPUT_PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs_sha256": public.content_sha256(old),
        "training_inputs_sha256": public.content_sha256(old["source_inputs"]),
        "training_inputs": old["source_inputs"],
        "models": histories,
        "replays": replays,
        "test_data_opened": False,
        "validation_exported": False,
        "old_allowances_renewed": False,
    }
    _validate(data, Policy(population=settings.population))
    seal(destination, data)
    return {
        "inputs_sha256": public.content_sha256(data),
        "models": len(histories),
        "main_tasks": len(histories),
        "verification_only_tasks": len(replays),
    }


def _validate_endpoint(point: Endpoint, model: dict) -> None:
    """Reject mismatched coordinates, channels and false historical certificates."""
    if set(point.parameters) != set(model["parameters"]) or (
        point.parameters_sha256 != public.content_sha256(point.parameters)
    ):
        raise ValueError("endpoint coordinates or digest differ")
    if point.per_target_normalized_mse is not None and set(
        point.per_target_normalized_mse
    ) != set(model["problem"]["context"]["targets"]):
        raise ValueError("endpoint target identities differ")
    if point.original_equations_verified:
        numbers = [
            point.training_nmse,
            point.alternate_training_nmse,
            point.solver_loss_discrepancy,
            point.solver_prediction_discrepancy,
        ]
        if any(v is None for v in numbers):
            raise ValueError("historical verification evidence missing")
        tolerance = max(1e-12, 1e-8 * max(numbers[:2]))
        if (
            point.solver_prediction_discrepancy > 1e-7
            or point.solver_loss_discrepancy > tolerance
            or abs(numbers[0] - numbers[1]) > tolerance
        ):
            raise ValueError("historical verification discrepancy exceeds contract")


def _validate(data: dict, policy: Policy) -> None:
    parsed = Inputs.model_validate(data)
    original_campaign._validate(
        parsed.training_inputs, original_campaign.Policy(population=policy.population)
    )
    if parsed.training_inputs_sha256 != public.content_sha256(parsed.training_inputs):
        raise ValueError("training input identity differs")
    models = parsed.training_inputs["models"]
    if len(models) != len(parsed.models):
        raise ValueError("all source models required")
    allowed_ops = previous._allowances(
        previous.Policy(), {"failed_endpoints": [None] * 4}
    )
    for index, (history, model) in enumerate(zip(parsed.models, models, strict=True)):
        if (history.model, history.task_id, history.problem_sha256) != (
            index,
            model["task_id"],
            public.content_sha256(model["problem"]),
        ):
            raise ValueError("model history identity/order differs")
        if [arm.arm for arm in history.arms] != list(ARMS):
            raise ValueError("both historical coordinate arms required in order")
        for arm in history.arms:
            if set(arm.operations) - set(allowed_ops):
                raise ValueError("unknown historical operation evidence")
            if set(arm.reliability_cost) != COST_KEYS or (
                arm.original_cost is not None and set(arm.original_cost) != COST_KEYS
            ):
                raise ValueError("historical cost schema differs")
            if arm.original_cost_known and (
                arm.original_cost is None
                or not arm.original_cost["accounting_complete"]
            ):
                raise ValueError("unknown source cost cannot be certified")
            _validate_endpoint(arm.endpoint, model)
            origin = arm.endpoint.origin
            if origin != "source_incumbent" and (
                origin not in arm.operations
                or arm.operations[origin].receipt_sha256 is None
                or (
                    arm.endpoint.original_equations_verified
                    and arm.operations[origin].status != "complete"
                )
            ):
                raise ValueError("retained endpoint lacks a historical receipt")
    seen = set()
    previous_key = (-1, -1, -1)
    for replay in parsed.replays:
        if (
            replay.model >= len(models)
            or replay.task_id != models[replay.model]["task_id"]
        ):
            raise ValueError("replay model identity differs")
        _validate_endpoint(replay.endpoint, models[replay.model])
        key = (replay.model, ARMS.index(replay.source_arm), replay.source_ordinal)
        duplicate = (replay.model, replay.source_arm, replay.endpoint.parameters_sha256)
        if key <= previous_key or duplicate in seen:
            raise ValueError("replay roster duplicated or reordered")
        if replay.endpoint.original_equations_verified:
            raise ValueError("replay roster requires disputed old endpoints")
        seen.add(duplicate)
        previous_key = key


def _tasks(data: dict) -> list[dict]:
    tasks = [
        {
            "model": h["model"],
            "task_id": h["task_id"],
            "kind": "continuation",
            "arm": "affine_reduced",
        }
        for h in data["models"]
    ]
    tasks.extend(
        {
            "model": r["model"],
            "task_id": r["task_id"],
            "kind": "verification_only",
            "arm": "affine_reduced",
            "replay": i,
        }
        for i, r in enumerate(data["replays"])
    )
    return [{"index": i, **task} for i, task in enumerate(tasks)]


def _precision_policy():
    return prefix_affine.AffinePrecisionPolicy()


def _numerical_contract() -> dict:
    """Freeze the opt-in numerical changes independently of user-facing budgets."""
    return {
        "affine_precision": public._jsonable(_precision_policy()),
        "derivative_policy": public._jsonable(
            adequacy.Policy(precision_aware=True, local_adaptive=True)
        ),
        "verification_tolerances": [[1e-11, 1e-13]],
        "maximum_normalized_prediction_gap": 1e-7,
    }


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Seal a new experiment; preparation performs no numerical model evaluation."""
    _validate(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "numerical_contract": _numerical_contract(),
        "tasks": _tasks(data),
        "start_rule": "best_newly_verified_cross_arm_endpoint_then_one_reduced_search",
        "replay_selection_allowed": False,
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    _validate(data, Policy.model_validate(plan["policy"]))
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data)
        or plan["numerical_contract"] != _numerical_contract()
        or plan["start_rule"]
        != "best_newly_verified_cross_arm_endpoint_then_one_reduced_search"
        or plan["llm_calls"] != 0
        or any(
            plan[key] is not False
            for key in (
                "replay_selection_allowed",
                "old_allowances_renewed",
                "test_data_opened",
                "validation_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("local recovery plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("local recovery source/runtime differs")
    return plan, data


def _folder(root: Path, task: dict) -> Path:
    return root / "results" / f"task_{task['index']:02d}"


def _allowances(policy: Policy, task: dict) -> dict:
    if task["kind"] == "verification_only":
        return {"disputed_verification": policy.verification_seconds + 90}
    return {
        "source_original_verification": policy.verification_seconds + 90,
        "source_reduced_verification": policy.verification_seconds + 90,
        "qualification": policy.diagnostic_seconds + 90,
        "continuation_search": policy.search_seconds + 150,
        "continuation_verification": policy.verification_seconds + 90,
        "continuation_diagnosis": policy.diagnostic_seconds + 90,
    }


def _operations(folder: Path, identity: dict, policy: Policy) -> dict:
    operations = {}
    allowed = _allowances(policy, identity["task"])
    if any(
        p.is_dir() and p.name not in allowed for p in (folder / "operations").glob("*")
    ):
        raise ValueError("unknown local recovery operation")
    for name, seconds in allowed.items():
        location = folder / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (location / "result.json").exists():
            operations[name] = journal._read_operation(location, bound)
        elif (location / "started.json").exists():
            if read_seal(location / "started.json") != bound:
                raise ValueError("local recovery reservation differs")
            operations[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return operations


def _selected(operations: dict) -> dict | None:
    chosen = None
    for arm, name in zip(
        ARMS,
        ("source_original_verification", "source_reduced_verification"),
        strict=True,
    ):
        point = previous._verified(operations.get(name, {}))
        if point is not None and (
            chosen is None or previous._improves(point, chosen["verification"])
        ):
            chosen = {
                "source_arm": arm,
                "origin": name,
                "parameters": deepcopy(point["parameters"]),
                "verification": point,
            }
    return chosen


def _assemble(
    identity: dict, history: dict, operations: dict, *, terminal=False
) -> dict:
    task = identity["task"]
    selected = _selected(operations) if task["kind"] == "continuation" else None
    retained = deepcopy(selected["verification"]) if selected else None
    origin = selected["origin"] if selected else None
    candidate = previous._verified(operations.get("continuation_verification", {}))
    if candidate is not None and (
        retained is None or previous._improves(candidate, retained)
    ):
        retained, origin = candidate, "continuation_verification"
    diagnosis_op = operations.get("continuation_diagnosis", {})
    diagnosis = (
        (diagnosis_op.get("value") or {}).get("assessment")
        if candidate is not None
        and retained is not None
        and candidate["parameters"] == retained["parameters"]
        and diagnosis_op.get("status") == "complete"
        else None
    )
    search = operations.get("continuation_search", {})
    checkpoint = original_campaign._search_checkpoint(search)
    started = checkpoint.get("search_calls_started", checkpoint.get("calls_started", 0))
    qualification = operations.get("qualification", {})
    qualified = qualification.get("status") == "complete" and bool(
        (qualification.get("value") or {}).get("passed")
    )
    if task["kind"] == "verification_only":
        reason = "verification_only_by_design"
    elif selected is None:
        reason = "no_currently_verified_source_endpoint"
    elif not qualified:
        reason = "qualification_" + qualification.get("status", "not_recorded")
        if qualification.get("status") == "complete":
            reason = "derivative_or_coordinate_qualification_not_passed"
    elif not started:
        reason = "search_" + search.get("status", "not_recorded")
    else:
        reason = None
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "selected_start": selected,
        "retained": retained,
        "retained_origin": origin,
        "historical_endpoints": [
            {"arm": a["arm"], "endpoint": a["endpoint"]} for a in history["arms"]
        ],
        "historical_verification_is_current": False,
        "qualification_passed": qualified,
        "search_started": bool(started),
        "search_not_started_reason": reason,
        "fitting_diagnosis": diagnosis,
        "optimization_status": (diagnosis or {}).get(
            "classification", "optimization_unresolved"
        )
        if task["kind"] == "continuation"
        else "not_assessed_verification_only",
        "disputed_endpoint": previous._verified(
            operations.get("disputed_verification", {})
        ),
        "operations": operations,
        "cost": journal._cost(operations),
        "diagnostic_cost": original_campaign._diagnostic_cost(operations),
        "old_allowances_renewed": False,
        "automatic_revision": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
    }


def _check_start(folder: Path, value: dict) -> None:
    if (folder / "selected_start.json").exists() and read_seal(
        folder / "selected_start.json"
    ) != value["selected_start"]:
        raise ValueError("selected start differs from current verification receipts")


def run_task(root: Path, index: int) -> dict:
    """Reverify same-model incumbents, then continue once in reduced coordinates."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown local recovery task index")
    task = plan["tasks"][index]
    history = data["models"][task["model"]]
    model = data["training_inputs"]["models"][task["model"]]
    policy = Policy.model_validate(plan["policy"])
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = _folder(root, task)
    problem = PrefixFitProblem.model_validate(model["problem"])
    training = PublicSplit.model_validate(data["training_inputs"]["training"])
    settings = kernel.NumericalPolicy.model_validate(
        data["training_inputs"]["numerical_policy"]
    )
    allowances = _allowances(policy, task)

    def operate(name, work):
        def call(save):
            deadline = monotonic() + allowances[name]

            def check():
                if monotonic() >= deadline:
                    raise TimeoutError("inclusive local recovery allowance exhausted")

            def checked_save(value):
                save(value | {"affine_execution": deepcopy(original.affine_audit)})
                check()

            original, oracle = numerical.make_oracle(
                problem,
                training,
                settings,
                "affine_reduced",
                precision_policy=_precision_policy(),
            )
            check()
            result = work(original, oracle, checked_save)
            check()
            return result | {"affine_execution": deepcopy(original.affine_audit)}

        return journal._operation(folder, identity, name, allowances[name], call)

    def verify_original(original, oracle, parameters, save):
        effective = numerical.to_effective(original, oracle, parameters)
        return numerical.verify_endpoint_ladder(
            original,
            oracle,
            effective,
            policy.verification_seconds,
            save,
            tolerances=((1e-11, 1e-13),),
        )

    with public._lock(folder):
        terminal = folder / "result.json"
        if terminal.exists():
            saved = read_seal(terminal)
            rebuilt = _assemble(
                identity, history, _operations(folder, identity, policy), terminal=True
            )
            _check_start(folder, rebuilt)
            if saved != rebuilt:
                raise ValueError("local recovery terminal differs from receipts")
            return saved
        if task["kind"] == "verification_only":
            point = data["replays"][task["replay"]]["endpoint"]
            operate(
                "disputed_verification",
                lambda o, a, save: verify_original(o, a, point["parameters"], save),
            )
        else:
            for arm, name in zip(
                history["arms"],
                ("source_original_verification", "source_reduced_verification"),
                strict=True,
            ):
                operate(
                    name,
                    lambda o, a, save, p=arm["endpoint"]["parameters"]: verify_original(
                        o, a, p, save
                    ),
                )
            selected = _selected(_operations(folder, identity, policy))
            seal(folder / "selected_start.json", selected)
            if selected is not None:
                qualified = operate(
                    "qualification",
                    lambda o, a, save: numerical.qualification(
                        o,
                        a,
                        selected["parameters"],
                        policy.diagnostic_seconds,
                        save,
                        adequacy_policy=adequacy.Policy(
                            precision_aware=True, local_adaptive=True
                        ),
                    ),
                )
                if qualified["status"] == "complete" and (
                    qualified.get("value") or {}
                ).get("passed"):
                    attempt = operate(
                        "continuation_search",
                        lambda o, a, save: numerical.search(
                            a,
                            numerical.to_effective(o, a, selected["parameters"]),
                            policy.search_seconds,
                            policy.maximum_calls,
                            save,
                            reject_invalid_trials=True,
                        ),
                    )
                    fit = original_campaign._search_checkpoint(attempt)
                    if fit.get("best"):
                        checked = operate(
                            "continuation_verification",
                            lambda o, a, save: numerical.verify_endpoint_ladder(
                                o,
                                a,
                                fit["best"]["parameters"],
                                policy.verification_seconds,
                                save,
                                tolerances=((1e-11, 1e-13),),
                            ),
                        )
                        point = previous._verified(checked)
                        if point is not None:
                            operate(
                                "continuation_diagnosis",
                                lambda o, a, save: numerical.diagnose(
                                    a,
                                    numerical.to_effective(o, a, point["parameters"]),
                                    point,
                                    fit,
                                    policy.diagnostic_seconds,
                                    save,
                                    adequacy_policy=adequacy.Policy(
                                        precision_aware=True, local_adaptive=True
                                    ),
                                ),
                            )
        result = _assemble(
            identity, history, _operations(folder, identity, policy), terminal=True
        )
        _check_start(folder, result)
        seal(terminal, result)
        return result


def report(root: Path) -> dict:
    """Reconstruct all tasks and emit historical costs once per source model."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = _folder(root, task)
        terminal = (folder / "result.json").exists()
        value = _assemble(
            identity,
            data["models"][task["model"]],
            _operations(folder, identity, policy),
            terminal=terminal,
        )
        _check_start(folder, value)
        if terminal and read_seal(folder / "result.json") != value:
            raise ValueError("local recovery terminal differs from receipts")
        rows.append(
            task
            | {
                key: value[key]
                for key in (
                    "status",
                    "selected_start",
                    "retained_origin",
                    "qualification_passed",
                    "search_started",
                    "search_not_started_reason",
                    "optimization_status",
                    "fitting_diagnosis",
                    "cost",
                    "diagnostic_cost",
                )
            }
            | {
                "training_nmse": (value["retained"] or {}).get("training_nmse"),
                "currently_verified": value["retained"] is not None,
                "replay_verified": value["disputed_endpoint"] is not None,
            }
        )
    count = sum(row["status"] == "complete" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "status": "complete" if count == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": count,
        "rows": rows,
        "historical_cost_by_model": [
            {
                "model": h["model"],
                "task_id": h["task_id"],
                "arms": [
                    {
                        k: a[k]
                        for k in (
                            "arm",
                            "original_cost_known",
                            "original_cost",
                            "reliability_cost",
                        )
                    }
                    for a in h["arms"]
                ],
            }
            for h in data["models"]
        ],
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "limitation": (
            "Training-only numerical recovery, not a fresh coordinate-arm comparison "
            "or a global-optimum/scientific-validity certificate. Disputed endpoint "
            "probes do not select starts. Historical costs appear once per model "
            "and remain separate from new operations."
        ),
    }
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", summary)
    return summary


def inspect(root: Path) -> dict:
    """Package immutable training provenance and receipts without numerical work."""
    summary = report(root)
    path, temporary = root / "inspection.tar.gz", root / "inspection.tmp"
    paths = [root / "plan.json", root / "inputs.json"]
    if (root / "launcher_execution.json").exists():
        paths.append(root / "launcher_execution.json")
    for name in ("results", "report", "submissions", "logs"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for source in sorted(paths):
            if source.is_symlink() or not source.resolve().is_relative_to(
                root.resolve()
            ):
                raise ValueError("archive path escapes local recovery campaign")
            if source.name.startswith("."):
                continue
            blob = source.read_bytes()
            entry = tarfile.TarInfo(source.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}
