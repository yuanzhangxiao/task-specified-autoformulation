"""Resume frozen construction incumbents after bounded derivative-check repair.

Only the eight previously selected endpoints enter fitting. Six completed
verification-only probes retain their historical receipts and costs, but cannot
enter selection or acquire another numerical allowance in this campaign.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_continuation as first
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_local_recovery as previous
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-derivative-recovery-1"
INPUT_PROTOCOL = "construction-derivative-recovery-inputs-1"
Digest = previous.Digest
Nonnegative = previous.Nonnegative
COST_KEYS = previous.COST_KEYS


class Policy(previous.Policy):
    """Separate finite allowances for one continuation of each selected model."""

    diagnostic_seconds: Nonnegative = Field(default=600, gt=0, le=600)


class ParentTask(StrictSchema):
    """Authenticated source task provenance and costs, without old fitted arrays."""

    index: int = Field(ge=0)
    model: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    kind: Literal["continuation", "verification_only"]
    task_sha256: Digest
    terminal_sha256: Digest
    operations: dict[str, previous.previous.OperationEvidence]
    cost: dict[str, Nonnegative | bool | None]


class EarlierArmCost(StrictSchema):
    arm: Literal["affine_original", "affine_reduced"]
    original_cost_known: bool
    original_cost: dict[str, Nonnegative | bool | None] | None
    reliability_cost: dict[str, Nonnegative | bool | None]


class EarlierModelCost(StrictSchema):
    model: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    arms: list[EarlierArmCost] = Field(min_length=2, max_length=2)


class SourceModel(StrictSchema):
    """Only a source model's selected current-verified endpoint may continue."""

    model: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    problem_sha256: Digest
    parent_task_index: int = Field(ge=0)
    endpoint: previous.Endpoint


class Inputs(StrictSchema):
    """Positive projection of training data, selected endpoints and cost history."""

    protocol: Literal["construction-derivative-recovery-inputs-1"]
    source_plan_sha256: Digest
    source_inputs_sha256: Digest
    training_inputs_sha256: Digest
    training_inputs: dict
    models: list[SourceModel]
    parent_tasks: list[ParentTask]
    earlier_cost_by_model: list[EarlierModelCost]
    old_allowances_renewed: Literal[False]
    verification_only_endpoints_exported: Literal[False]
    validation_exported: Literal[False]
    test_data_opened: Literal[False]


def export_inputs(source: Path, destination: Path) -> dict:
    """Authenticate all returned parent tasks without rewriting any source file."""
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError(
            "derivative recovery export must not write into source campaign"
        )
    plan, data = previous.verify(source, runtime=False)
    settings = previous.Policy.model_validate(plan["policy"])
    models, parents = [], []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = previous._folder(source, task)
        operations = previous._operations(folder, identity, settings)
        value = previous._assemble(
            identity, data["models"][task["model"]], operations, terminal=True
        )
        previous._check_start(folder, value)
        if read_seal(folder / "result.json") != value:
            raise ValueError("source terminal differs from authenticated operations")
        parents.append(
            {
                **{key: task[key] for key in ("index", "model", "task_id", "kind")},
                "task_sha256": public.content_sha256(task),
                "terminal_sha256": public.content_sha256(value),
                "operations": previous._receipts(folder, operations),
                "cost": {
                    key: val for key, val in value["cost"].items() if key in COST_KEYS
                },
            }
        )
        if task["kind"] != "continuation":
            continue
        endpoint = value["retained"]
        if endpoint is None or not endpoint["original_equations_verified"]:
            raise ValueError(
                "every source model requires a selected verified incumbent"
            )
        model = data["training_inputs"]["models"][task["model"]]
        models.append(
            {
                "model": task["model"],
                "task_id": task["task_id"],
                "problem_sha256": public.content_sha256(model["problem"]),
                "parent_task_index": task["index"],
                "endpoint": previous._endpoint(endpoint, value["retained_origin"]),
            }
        )
    payload = {
        "protocol": INPUT_PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs_sha256": public.content_sha256(data),
        "training_inputs_sha256": public.content_sha256(data["training_inputs"]),
        "training_inputs": data["training_inputs"],
        "models": models,
        "parent_tasks": parents,
        "earlier_cost_by_model": [
            {
                "model": history["model"],
                "task_id": history["task_id"],
                "arms": [
                    {
                        key: arm[key]
                        for key in (
                            "arm",
                            "original_cost_known",
                            "original_cost",
                            "reliability_cost",
                        )
                    }
                    for arm in history["arms"]
                ],
            }
            for history in data["models"]
        ],
        "old_allowances_renewed": False,
        "verification_only_endpoints_exported": False,
        "validation_exported": False,
        "test_data_opened": False,
    }
    _validate(payload, Policy(population=settings.population))
    seal(destination, payload)
    return {
        "inputs_sha256": public.content_sha256(payload),
        "tasks": len(models),
        "authenticated_parent_tasks": len(parents),
        "verification_only_tasks_repeated": 0,
    }


def _validate_cost(cost: dict | None) -> None:
    if cost is not None and set(cost) != COST_KEYS:
        raise ValueError("historical cost schema differs")


def _validate(data: dict, policy: Policy) -> None:
    parsed = Inputs.model_validate(data)
    first._validate(parsed.training_inputs, first.Policy(population=policy.population))
    if parsed.training_inputs_sha256 != public.content_sha256(parsed.training_inputs):
        raise ValueError("training input digest differs")
    originals = parsed.training_inputs["models"]
    count = len(originals)
    if len(parsed.models) != count or len(parsed.earlier_cost_by_model) != count:
        raise ValueError("all source models and cost histories required")
    if len(parsed.parent_tasks) < count or (
        policy.population == "returned-eight" and len(parsed.parent_tasks) != 14
    ):
        raise ValueError("complete parent task roster required")
    for index, task in enumerate(parsed.parent_tasks):
        if (
            task.index != index
            or task.model >= count
            or task.task_id != originals[task.model]["task_id"]
        ):
            raise ValueError("parent task identity/order differs")
        expected_kind = "continuation" if index < count else "verification_only"
        if task.kind != expected_kind or (index < count and task.model != index):
            raise ValueError("parent continuation/replay roster differs")
        allowed = previous._allowances(previous.Policy(), {"kind": task.kind})
        if set(task.operations) - set(allowed):
            raise ValueError("unknown parent operation evidence")
        _validate_cost(task.cost)
    for index, (source, original, costs) in enumerate(
        zip(parsed.models, originals, parsed.earlier_cost_by_model, strict=True)
    ):
        identity = (index, original["task_id"])
        if (source.model, source.task_id) != identity or (
            costs.model,
            costs.task_id,
        ) != identity:
            raise ValueError("model history identity/order differs")
        if (
            source.parent_task_index != index
            or source.problem_sha256 != public.content_sha256(original["problem"])
        ):
            raise ValueError("selected source problem identity differs")
        previous._validate_endpoint(source.endpoint, original)
        if not source.endpoint.original_equations_verified:
            raise ValueError("selected source endpoint must be historically verified")
        operations = parsed.parent_tasks[index].operations
        if (
            source.endpoint.origin not in operations
            or operations[source.endpoint.origin].status != "complete"
            or operations[source.endpoint.origin].receipt_sha256 is None
        ):
            raise ValueError("selected source endpoint lacks verified parent receipt")
        if [arm.arm for arm in costs.arms] != list(previous.ARMS):
            raise ValueError("both original historical arm costs required")
        for arm in costs.arms:
            _validate_cost(arm.original_cost)
            _validate_cost(arm.reliability_cost)
            if arm.original_cost_known and (
                arm.original_cost is None
                or not arm.original_cost["accounting_complete"]
            ):
                raise ValueError("unknown historical cost cannot be certified")


def _tasks(data: dict) -> list[dict]:
    return [
        {
            "index": index,
            "model": model["model"],
            "task_id": model["task_id"],
            "arm": "affine_reduced",
        }
        for index, model in enumerate(data["models"])
    ]


def _adequacy_policy():
    return adequacy.Policy(
        precision_aware=True, local_adaptive=True, local_bounds_aware=True
    )


def _numerical_contract() -> dict:
    """Typed opt-in policies; policy content forms part of the sealed plan."""
    return {
        "derivative_policy": public._jsonable(_adequacy_policy()),
        "recovery_policy": public._jsonable(recovery.policy_receipt()),
        "verification_tolerances": [[1e-11, 1e-13]],
        "maximum_normalized_prediction_gap": 1e-7,
        "propagation": "guarded_then_bounded_forced_multiprecision_qualification",
        "inactive_coordinates": "proved_initializer_only_constants_on_training",
        "qualification_precedes_search": True,
        "qualification_selects_incumbent_verification_precision": True,
    }


def _runtime() -> dict:
    """Bind direct-library execution to the multiprecision dependency as well."""
    runtime = public._runtime()
    try:
        precision_version = version("mpmath")
    except PackageNotFoundError:
        precision_version = None
    return runtime | {"precision_packages": {"mpmath": precision_version}}


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Freeze eight continuation tasks without model evaluation on the login node."""
    _validate(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "numerical_contract": _numerical_contract(),
        "source_sha256": public._source_identity(),
        "runtime": _runtime(),
        "tasks": _tasks(data),
        "start_rule": "one_frozen_selected_incumbent_per_model",
        "old_allowances_renewed": False,
        "replay_selection_allowed": False,
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
        or plan["start_rule"] != "one_frozen_selected_incumbent_per_model"
        or plan["llm_calls"] != 0
        or any(
            plan[key] is not False
            for key in (
                "old_allowances_renewed",
                "replay_selection_allowed",
                "test_data_opened",
                "validation_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("derivative recovery plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != _runtime()
    ):
        raise ValueError("derivative recovery source/runtime differs")
    return plan, data


def _folder(root: Path, task: dict) -> Path:
    return root / "results" / f"task_{task['index']:02d}"


def _allowances(policy: Policy) -> dict:
    return {
        "qualification": policy.diagnostic_seconds + 90,
        "incumbent_verification": policy.verification_seconds + 90,
        "continuation_search": policy.search_seconds + 150,
        "continuation_verification": policy.verification_seconds + 90,
        "continuation_diagnosis": policy.diagnostic_seconds + 90,
    }


def _operations(folder: Path, identity: dict, policy: Policy) -> dict:
    allowed = _allowances(policy)
    if any(
        p.is_dir() and p.name not in allowed for p in (folder / "operations").glob("*")
    ):
        raise ValueError("unknown derivative recovery operation")
    operations = {}
    for name, seconds in allowed.items():
        directory = folder / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (directory / "result.json").exists():
            operations[name] = journal._read_operation(directory, bound)
        elif (directory / "started.json").exists():
            if read_seal(directory / "started.json") != bound:
                raise ValueError("derivative recovery reservation differs")
            operations[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return operations


def _verified(operation: dict) -> dict | None:
    return previous.previous._verified(operation)


def _diagnostic_cost(operations: dict) -> dict:
    """Count each precision pass once; partial or absent audit is a lower bound."""
    stages = {
        name: op
        for name, op in operations.items()
        if name == "qualification" or name.endswith("_diagnosis")
    }
    checks, complete = [], bool(stages)
    for name, op in stages.items():
        value = op.get("value") or op.get("observed_progress") or {}
        records = value.get("passes", []) if name == "qualification" else [value]
        complete = complete and op["status"] == "complete" and bool(records)
        for record in records:
            check = record.get("derivative_check") or {}
            checks.append(check)
            complete = (
                complete
                and bool(check)
                and (
                    record.get("status", "complete") == "complete"
                    and check.get("calls_started", 0) == check.get("calls_completed", 0)
                )
            )
    return {
        "derivative_calls_started_observed": sum(
            check.get("calls_started", 0) for check in checks
        ),
        "derivative_calls_completed_observed": sum(
            check.get("calls_completed", 0) for check in checks
        ),
        "stage_wall_seconds_observed": sum(
            op.get("wall_seconds") or 0 for op in stages.values()
        ),
        "complete": complete,
        "precision_passes_observed": len(
            (
                stages.get("qualification", {}).get("value")
                or stages.get("qualification", {}).get("observed_progress")
                or {}
            ).get("passes", [])
        ),
        "scope": (
            "Every guarded and multiprecision derivative pass is counted once. "
            "Stage wall includes failed passes, setup and assessment. Nested in "
            "new total cost, not another charge; missing audits are lower bounds."
        ),
    }


def _assemble(
    identity: dict, source: dict, operations: dict, *, terminal=False
) -> dict:
    selected = _verified(operations.get("incumbent_verification", {}))
    retained, origin = selected, "incumbent_verification" if selected else None
    candidate = _verified(operations.get("continuation_verification", {}))
    if candidate is not None and (
        retained is None or previous.previous._improves(candidate, retained)
    ):
        retained, origin = candidate, "continuation_verification"
    qop = operations.get("qualification", {})
    qval = qop.get("value") or {}
    qualified = (
        qop.get("status") == "complete"
        and qval.get("status", "complete") == "complete"
        and bool(qval.get("passed"))
    )
    sop = operations.get("continuation_search", {})
    fit = first._search_checkpoint(sop)
    calls = fit.get("search_calls_started", fit.get("calls_started", 0))
    if selected is None:
        reason = "incumbent_current_verification_unavailable"
    elif not qualified:
        reason = "qualification_" + (
            qval.get("status") or qop.get("status", "not_recorded")
        )
        if (
            qval.get("status", "complete") == "complete"
            and qop.get("status") == "complete"
        ):
            reason = "derivative_or_coordinate_qualification_not_passed"
    elif not calls:
        reason = "search_" + sop.get("status", "not_recorded")
    else:
        reason = None
    dop = operations.get("continuation_diagnosis", {})
    diagnosis = (
        (dop.get("value") or {}).get("assessment")
        if candidate is not None
        and retained is not None
        and candidate["parameters"] == retained["parameters"]
        and dop.get("status") == "complete"
        else None
    )
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "source_incumbent": source["endpoint"],
        "selected_start": selected,
        "retained": retained,
        "retained_origin": origin,
        "historical_verification_is_current": False,
        "qualification_passed": qualified,
        "search_started": bool(calls),
        "search_not_started_reason": reason,
        "search_stop_reason": fit.get("stop_reason") if sop else "not_started",
        "fitting_diagnosis": diagnosis,
        "fitting_diagnosis_scope": (
            "Remaining active training coordinates only; anchored initializer "
            "coefficients and removed gain factorizations are not identified."
            if diagnosis is not None
            else None
        ),
        "original_parameter_identifiability_certified": False,
        "optimization_status": (diagnosis or {}).get(
            "classification", "optimization_unresolved"
        ),
        "operations": operations,
        "cost": journal._cost(operations),
        "diagnostic_cost": _diagnostic_cost(operations),
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
    }


def _check_start(folder: Path, value: dict) -> None:
    if (folder / "selected_start.json").exists() and read_seal(
        folder / "selected_start.json"
    ) != value["selected_start"]:
        raise ValueError("selected start differs from incumbent verification")


def _make_oracles(problem, training, settings, anchor, precision_mode):
    """Apply numerical policies without changing equations or parameter domains."""
    return recovery.make_oracle(
        problem, training, settings, anchor, precision_mode=precision_mode
    )


def run_task(root: Path, index: int) -> dict:
    """Continue one independently reverified incumbent, with immutable stage budgets."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown derivative recovery task index")
    task, source = plan["tasks"][index], data["models"][index]
    model = data["training_inputs"]["models"][task["model"]]
    policy = Policy.model_validate(plan["policy"])
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = _folder(root, task)
    problem = PrefixFitProblem.model_validate(model["problem"])
    training = PublicSplit.model_validate(data["training_inputs"]["training"])
    settings = kernel.NumericalPolicy.model_validate(
        data["training_inputs"]["numerical_policy"]
    )
    allowances = _allowances(policy)

    anchor = source["endpoint"]["parameters"]

    def operate(name, work, *, precision_mode="guarded", raw=False):
        def call(save):
            deadline = monotonic() + allowances[name]

            def check():
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "inclusive derivative recovery allowance exhausted"
                    )

            original, oracle = (
                (None, None)
                if raw
                else _make_oracles(problem, training, settings, anchor, precision_mode)
            )
            check()

            def with_audit(value):
                if raw:
                    return value
                return value | {
                    "affine_execution": deepcopy(original.affine_audit),
                    "recovery_audit": deepcopy(getattr(oracle, "recovery_audit", {})),
                }

            def checked_save(value):
                save(with_audit(value))
                check()

            result = work(original, oracle, checked_save)
            check()
            return with_audit(result)

        return journal._operation(folder, identity, name, allowances[name], call)

    with public._lock(folder):
        terminal = folder / "result.json"
        if terminal.exists():
            saved = read_seal(terminal)
            rebuilt = _assemble(
                identity, source, _operations(folder, identity, policy), terminal=True
            )
            _check_start(folder, rebuilt)
            if saved != rebuilt:
                raise ValueError("derivative recovery terminal differs from receipts")
            return saved
        qualified = operate(
            "qualification",
            lambda o, a, save: recovery.qualification(
                problem,
                training,
                settings,
                anchor,
                policy.diagnostic_seconds,
                save,
                adequacy_policy=_adequacy_policy(),
            ),
            raw=True,
        )
        qval = qualified.get("value") or qualified.get("observed_progress") or {}
        precision_mode = qval.get("precision_mode", "guarded")
        if precision_mode not in {"guarded", "multiprecision"}:
            raise ValueError("qualification returned unknown precision mode")
        operate(
            "incumbent_verification",
            lambda o, a, save: numerical.verify_endpoint_ladder(
                o,
                a,
                numerical.to_effective(o, a, source["endpoint"]["parameters"]),
                policy.verification_seconds,
                save,
                tolerances=((1e-11, 1e-13),),
            ),
            precision_mode=precision_mode,
        )
        selected = _verified(
            _operations(folder, identity, policy).get("incumbent_verification", {})
        )
        seal(folder / "selected_start.json", selected)
        if (
            selected is not None
            and qualified["status"] == "complete"
            and qval.get("status", "complete") == "complete"
            and qval.get("passed")
        ):
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
                precision_mode=precision_mode,
            )
            fit = first._search_checkpoint(attempt)
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
                    precision_mode=precision_mode,
                )
                point = _verified(checked)
                if point is not None:
                    operate(
                        "continuation_diagnosis",
                        lambda o, a, save: recovery.diagnose(
                            o,
                            a,
                            numerical.to_effective(o, a, point["parameters"]),
                            point,
                            fit,
                            policy.diagnostic_seconds,
                            save,
                            adequacy_policy=_adequacy_policy(),
                        ),
                        precision_mode=precision_mode,
                    )
        result = _assemble(
            identity, source, _operations(folder, identity, policy), terminal=True
        )
        _check_start(folder, result)
        seal(terminal, result)
        return result


def report(root: Path) -> dict:
    """Report attempted optimization separately from completed operation bookkeeping."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows = []
    for task, source in zip(plan["tasks"], data["models"], strict=True):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = _folder(root, task)
        terminal = (folder / "result.json").exists()
        value = _assemble(
            identity, source, _operations(folder, identity, policy), terminal=terminal
        )
        _check_start(folder, value)
        if terminal and read_seal(folder / "result.json") != value:
            raise ValueError("derivative recovery terminal differs from receipts")
        rows.append(
            task
            | {
                key: value[key]
                for key in (
                    "status",
                    "retained_origin",
                    "qualification_passed",
                    "search_started",
                    "search_not_started_reason",
                    "search_stop_reason",
                    "optimization_status",
                    "fitting_diagnosis",
                    "fitting_diagnosis_scope",
                    "original_parameter_identifiability_certified",
                    "cost",
                    "diagnostic_cost",
                )
            }
            | {
                "source_training_nmse": source["endpoint"]["training_nmse"],
                "training_nmse": (value["retained"] or {}).get("training_nmse"),
                "currently_verified": value["retained"] is not None,
            }
        )
    count = sum(row["status"] == "complete" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "status": "complete" if count == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": count,
        "rows": rows,
        "earlier_cost_by_model": data["earlier_cost_by_model"],
        "parent_cost_by_task": [
            {key: task[key] for key in ("index", "model", "task_id", "kind", "cost")}
            for task in data["parent_tasks"]
        ],
        "verification_only_tasks_repeated": 0,
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "limitation": (
            "Training-only continuation from frozen selected incumbents. A complete "
            "task does not establish convergence; absent current verification does "
            "not erase historical evidence. No global-optimum or scientific-validity "
            "certificate. Earlier and parent costs appear once, separate from new work."
        ),
    }
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", summary)
    return summary


def inspect(root: Path) -> dict:
    """Archive immutable training provenance and current numerical receipts."""
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
                raise ValueError("archive path escapes derivative recovery campaign")
            if source.name.startswith("."):
                continue
            blob = source.read_bytes()
            entry = tarfile.TarInfo(source.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}
