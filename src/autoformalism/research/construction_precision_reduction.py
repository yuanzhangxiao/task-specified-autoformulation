"""Bounded precision and exact-coordinate continuation of eight frozen models.

Every retained endpoint receives its own diagnosis. Historical operation receipts
and costs remain separate; no replay-only endpoint is selected or refitted.
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

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_derivative_recovery as previous
from autoformalism.research import construction_fit_continuation as first
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-precision-reduction-1"
INPUT_PROTOCOL = "construction-precision-reduction-inputs-1"
Policy = previous.Policy
Digest = previous.Digest
COST_KEYS = previous.COST_KEYS
_runtime = previous._runtime


class Inputs(StrictSchema):
    """Strict projection of current endpoints, training inputs and prior costs."""

    protocol: Literal["construction-precision-reduction-inputs-1"]
    source_plan_sha256: Digest
    source_inputs_sha256: Digest
    training_inputs_sha256: Digest
    training_inputs: dict
    models: list[previous.SourceModel]
    parent_tasks: list[previous.ParentTask]
    local_recovery_source_plan_sha256: Digest
    local_recovery_source_inputs_sha256: Digest
    local_recovery_tasks: list[previous.ParentTask]
    earlier_cost_by_model: list[previous.EarlierModelCost]
    old_allowances_renewed: Literal[False]
    verification_only_endpoints_exported: Literal[False]
    validation_exported: Literal[False]
    test_data_opened: Literal[False]


def export_inputs(source: Path, destination: Path) -> dict:
    """Authenticate all eight terminals and export only their retained endpoints."""
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("precision export must not write into source campaign")
    plan, data = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    models, parents = [], []
    for task in plan["tasks"]:
        folder = previous._folder(source, task)
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        operations = previous._operations(folder, identity, policy)
        value = previous._assemble(
            identity, data["models"][task["model"]], operations, terminal=True
        )
        previous._check_start(folder, value)
        if read_seal(folder / "result.json") != value:
            raise ValueError("source terminal differs from authenticated operations")
        endpoint = value["retained"]
        if endpoint is None or not endpoint["original_equations_verified"]:
            raise ValueError("every source model requires a verified incumbent")
        parents.append(
            {
                **{key: task[key] for key in ("index", "model", "task_id")},
                "kind": "continuation",
                "task_sha256": public.content_sha256(task),
                "terminal_sha256": public.content_sha256(value),
                "operations": previous.previous._receipts(folder, operations),
                "cost": {
                    key: val for key, val in value["cost"].items() if key in COST_KEYS
                },
            }
        )
        original = data["training_inputs"]["models"][task["model"]]
        models.append(
            {
                "model": task["model"],
                "task_id": task["task_id"],
                "problem_sha256": public.content_sha256(original["problem"]),
                "parent_task_index": task["index"],
                "endpoint": previous.previous._endpoint(
                    endpoint, value["retained_origin"]
                ),
            }
        )
    payload = {
        "protocol": INPUT_PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs_sha256": public.content_sha256(data),
        "training_inputs_sha256": data["training_inputs_sha256"],
        "training_inputs": data["training_inputs"],
        "models": models,
        "parent_tasks": parents,
        "local_recovery_source_plan_sha256": data["source_plan_sha256"],
        "local_recovery_source_inputs_sha256": data["source_inputs_sha256"],
        "local_recovery_tasks": data["parent_tasks"],
        "earlier_cost_by_model": data["earlier_cost_by_model"],
        "old_allowances_renewed": False,
        "verification_only_endpoints_exported": False,
        "validation_exported": False,
        "test_data_opened": False,
    }
    _validate(payload, Policy(population=policy.population))
    seal(destination, payload)
    return {
        "inputs_sha256": public.content_sha256(payload),
        "tasks": len(models),
        "authenticated_parent_tasks": len(parents),
        "historical_local_tasks_preserved": len(data["parent_tasks"]),
        "verification_only_tasks_repeated": 0,
    }


def _validate(data: dict, policy: Policy) -> None:
    parsed = Inputs.model_validate(data)
    first._validate(parsed.training_inputs, first.Policy(population=policy.population))
    if parsed.training_inputs_sha256 != public.content_sha256(parsed.training_inputs):
        raise ValueError("training input digest differs")
    originals = parsed.training_inputs["models"]
    count = len(originals)
    if any(
        len(records) != count
        for records in (
            parsed.models,
            parsed.parent_tasks,
            parsed.earlier_cost_by_model,
        )
    ):
        raise ValueError("all selected models and historical costs required")
    if len(parsed.local_recovery_tasks) < count or (
        policy.population == "returned-eight" and len(parsed.local_recovery_tasks) != 14
    ):
        raise ValueError("complete local-recovery history required")
    for index, task in enumerate(parsed.local_recovery_tasks):
        if (
            task.index != index
            or task.model >= count
            or task.task_id != originals[task.model]["task_id"]
            or task.kind != ("continuation" if index < count else "verification_only")
            or (index < count and task.model != index)
        ):
            raise ValueError("local-recovery history identity/order differs")
        allowed = previous.previous._allowances(
            previous.previous.Policy(), {"kind": task.kind}
        )
        if set(task.operations) - set(allowed):
            raise ValueError("unknown local-recovery operation")
        previous._validate_cost(task.cost)
    for index, (source, parent, original, costs) in enumerate(
        zip(
            parsed.models,
            parsed.parent_tasks,
            originals,
            parsed.earlier_cost_by_model,
            strict=True,
        )
    ):
        identity = (index, original["task_id"])
        if (
            (source.model, source.task_id) != identity
            or (parent.model, parent.task_id) != identity
            or (costs.model, costs.task_id) != identity
            or parent.index != index
            or parent.kind != "continuation"
            or source.parent_task_index != index
            or source.problem_sha256 != public.content_sha256(original["problem"])
        ):
            raise ValueError("selected model or parent identity/order differs")
        if set(parent.operations) - set(previous._allowances(previous.Policy())):
            raise ValueError("unknown derivative-recovery operation")
        previous._validate_cost(parent.cost)
        previous.previous._validate_endpoint(source.endpoint, original)
        if (
            not source.endpoint.original_equations_verified
            or source.endpoint.origin
            not in {"incumbent_verification", "continuation_verification"}
            or source.endpoint.origin not in parent.operations
            or parent.operations[source.endpoint.origin].status != "complete"
            or parent.operations[source.endpoint.origin].receipt_sha256 is None
        ):
            raise ValueError("selected endpoint lacks verified parent receipt")
        if [arm.arm for arm in costs.arms] != list(previous.previous.ARMS):
            raise ValueError("both original historical arm costs required")
        for arm in costs.arms:
            previous._validate_cost(arm.original_cost)
            previous._validate_cost(arm.reliability_cost)
            if arm.original_cost_known and (
                arm.original_cost is None
                or not arm.original_cost["accounting_complete"]
            ):
                raise ValueError("unknown historical cost cannot be certified")


def _tasks(data: dict) -> list[dict]:
    return [
        {
            "index": i,
            "model": row["model"],
            "task_id": row["task_id"],
            "arm": "certified_reduced",
        }
        for i, row in enumerate(data["models"])
    ]


def _adequacy_policy(*, diagnostic=False):
    options = (
        {"diagnostic_precision": True, "local_max_scales": 12} if diagnostic else {}
    )
    return adequacy.Policy(
        precision_aware=True, local_adaptive=True, local_bounds_aware=True, **options
    )


def _numerical_contract() -> dict:
    return {
        "qualification_policy": public._jsonable(_adequacy_policy()),
        "diagnosis_policy": public._jsonable(_adequacy_policy(diagnostic=True)),
        "recovery_policy": public._jsonable(
            recovery.policy_receipt(structural_reduction=True)
        ),
        "verification_tolerances": [[1e-11, 1e-13]],
        "maximum_normalized_prediction_gap": 1e-7,
        "start_rule": "one_frozen_selected_incumbent_per_model",
        "reduction": "certified_domain_preserving_with_original_equation_lift",
        "diagnosis_endpoint": "exact_current_retained_including_unchanged_incumbent",
        "qualification_precedes_search": True,
        "historical_verification_reused": False,
    }


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Seal training-only tasks without evaluating models on the login node."""
    _validate(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "numerical_contract": _numerical_contract(),
        "source_sha256": public._source_identity(),
        "runtime": _runtime(),
        "tasks": _tasks(data),
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


def verify(root: Path, *, runtime=True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    _validate(data, Policy.model_validate(plan["policy"]))
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data)
        or plan["numerical_contract"] != _numerical_contract()
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
        raise ValueError("precision reduction plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != _runtime()
    ):
        raise ValueError("precision reduction source/runtime differs")
    return plan, data


_folder = previous._folder
_verified = previous._verified


def _allowances(policy: Policy) -> dict:
    return {
        "qualification": policy.diagnostic_seconds + 90,
        "incumbent_verification": policy.verification_seconds + 90,
        "continuation_search": policy.search_seconds + 150,
        "continuation_verification": policy.verification_seconds + 90,
        "retained_diagnosis": policy.diagnostic_seconds + 90,
    }


def _operations(folder: Path, identity: dict, policy: Policy) -> dict:
    allowed = _allowances(policy)
    if any(
        p.is_dir() and p.name not in allowed for p in (folder / "operations").glob("*")
    ):
        raise ValueError("unknown precision reduction operation")
    operations = {}
    for name, seconds in allowed.items():
        directory = folder / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (directory / "result.json").exists():
            operations[name] = journal._read_operation(directory, bound)
        elif (directory / "started.json").exists():
            if read_seal(directory / "started.json") != bound:
                raise ValueError("precision reduction reservation differs")
            operations[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return operations


def _termination(fit: dict) -> str:
    """Report the actual optimizer condition without claiming gradient convergence."""
    message = fit.get("message", "")
    named = [name for name in ("ftol", "xtol", "gtol") if name in message]
    if named:
        return "_and_".join(named)
    return fit.get("stop_reason", "not_started")


def _assemble(identity, source, operations, *, terminal=False):
    selected = _verified(operations.get("incumbent_verification", {}))
    retained, origin = selected, "incumbent_verification" if selected else None
    candidate = _verified(operations.get("continuation_verification", {}))
    if candidate is not None and (
        retained is None or previous.previous.previous._improves(candidate, retained)
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
        if qval.get("status", "complete") == qop.get("status") == "complete":
            reason = "derivative_or_coordinate_qualification_not_passed"
    elif not calls:
        reason = "search_" + sop.get("status", "not_recorded")
    else:
        reason = None
    dop = operations.get("retained_diagnosis", {})
    dval = dop.get("value") or {}
    if dval and (
        retained is None
        or dval.get("diagnosed_parameters_sha256")
        != public.content_sha256(retained["parameters"])
        or dval.get("verification_sha256") != public.content_sha256(retained)
    ):
        raise ValueError("diagnosis differs from exact retained endpoint")
    diagnosis = dval.get("assessment") if dop.get("status") == "complete" else None
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
        "search_stop_reason": _termination(fit) if sop else "not_started",
        "optimizer_termination": _termination(fit) if sop else "not_started",
        "fitting_diagnosis": diagnosis,
        "diagnostic_gradient_precision_qualified": (
            (dval.get("derivative_check") or {}).get("gradient_precision_qualified")
            if dop.get("status") == "complete"
            else None
        ),
        "fitting_diagnosis_scope": (
            "Exact retained endpoint in certified active training coordinates; "
            "removed original parameters are not identified."
            if diagnosis is not None
            else None
        ),
        "original_parameter_identifiability_certified": False,
        "optimization_status": (diagnosis or {}).get(
            "classification", "optimization_unresolved"
        ),
        "operations": operations,
        "cost": journal._cost(operations),
        "diagnostic_cost": previous._diagnostic_cost(operations),
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
    }


_check_start = previous._check_start


def _make_oracles(problem, training, settings, anchor, precision_mode):
    return recovery.make_oracle(
        problem,
        training,
        settings,
        anchor,
        precision_mode=precision_mode,
        structural_reduction=True,
    )


def _verified_coordinates(original, oracle, endpoint):
    """Reuse the exact verified vector, avoiding a rounded reduce/lift round trip."""
    parameters = endpoint["effective_parameters"]
    vector = oracle.vector(parameters)
    if not np.array_equal(vector, np.asarray(endpoint["reduced_vector"])):
        raise ValueError("verified coordinate vector differs from endpoint receipt")
    if numerical.to_original(original, oracle, parameters) != endpoint["parameters"]:
        raise ValueError("verified coordinate lift differs from retained parameters")
    return deepcopy(parameters)


def run_task(root: Path, index: int) -> dict:
    """Qualify, continue and diagnose one immutable incumbent under finite budgets."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown precision reduction task index")
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
    anchor, allowances = source["endpoint"]["parameters"], _allowances(policy)

    def operate(name, work, *, precision_mode="guarded", raw=False):
        def call(save):
            deadline = monotonic() + allowances[name]

            def check():
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "inclusive precision reduction allowance exhausted"
                    )

            original, oracle = (
                (None, None)
                if raw
                else _make_oracles(problem, training, settings, anchor, precision_mode)
            )
            check()

            def with_audit(value):
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
                save(with_audit(value))
                check()

            result = work(original, oracle, checked_save)
            check()
            return with_audit(result)

        return journal._operation(folder, identity, name, allowances[name], call)

    def assembled():
        return _assemble(
            identity, source, _operations(folder, identity, policy), terminal=True
        )

    with public._lock(folder):
        terminal = folder / "result.json"
        if terminal.exists():
            value = assembled()
            _check_start(folder, value)
            if read_seal(terminal) != value:
                raise ValueError("precision reduction terminal differs from receipts")
            return value
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
                structural_reduction=True,
            ),
            raw=True,
        )
        qval = qualified.get("value") or qualified.get("observed_progress") or {}
        mode = qval.get("precision_mode", "guarded")
        if mode not in {"guarded", "multiprecision"}:
            raise ValueError("qualification returned unknown precision mode")
        operate(
            "incumbent_verification",
            lambda o, a, save: numerical.verify_endpoint_ladder(
                o,
                a,
                numerical.to_effective(o, a, anchor),
                policy.verification_seconds,
                save,
                tolerances=((1e-11, 1e-13),),
            ),
            precision_mode=mode,
        )
        selected = assembled()["selected_start"]
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
                    _verified_coordinates(o, a, selected),
                    policy.search_seconds,
                    policy.maximum_calls,
                    save,
                    reject_invalid_trials=True,
                ),
                precision_mode=mode,
            )
            fit = first._search_checkpoint(attempt)
            if fit.get("best"):
                operate(
                    "continuation_verification",
                    lambda o, a, save: numerical.verify_endpoint_ladder(
                        o,
                        a,
                        fit["best"]["parameters"],
                        policy.verification_seconds,
                        save,
                        tolerances=((1e-11, 1e-13),),
                    ),
                    precision_mode=mode,
                )
        retained = assembled()["retained"]
        if retained is not None:
            operations = _operations(folder, identity, policy)
            candidate = _verified(operations.get("continuation_verification", {}))
            fit = first._search_checkpoint(operations.get("continuation_search", {}))
            history_matches = (
                candidate is not None
                and candidate["parameters"] == retained["parameters"]
            )
            context = (
                fit
                if history_matches
                else {
                    "stop_reason": "retained_endpoint_has_no_matching_search_history",
                    "accepted_steps": [],
                    "budget_exhausted": bool(fit.get("budget_exhausted")),
                }
            )
            binding = {
                "diagnosed_parameters_sha256": public.content_sha256(
                    retained["parameters"]
                ),
                "verification_sha256": public.content_sha256(retained),
                "search_history_matches_retained": history_matches,
            }

            def diagnosis(original, oracle, save):
                def bound_save(value):
                    save(value | binding)

                value = recovery.diagnose(
                    original,
                    oracle,
                    _verified_coordinates(original, oracle, retained),
                    retained,
                    context,
                    policy.diagnostic_seconds,
                    bound_save,
                    adequacy_policy=_adequacy_policy(diagnostic=True),
                )
                return value | binding

            operate("retained_diagnosis", diagnosis, precision_mode=mode)
        value = assembled()
        _check_start(folder, value)
        seal(terminal, value)
        return value


def report(root: Path) -> dict:
    """Reconstruct selection, diagnosis and costs only from authenticated receipts."""
    plan, data = verify(root, runtime=False)
    policy, rows = Policy.model_validate(plan["policy"]), []
    for task, source in zip(plan["tasks"], data["models"], strict=True):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = _folder(root, task)
        terminal = (folder / "result.json").exists()
        value = _assemble(
            identity, source, _operations(folder, identity, policy), terminal=terminal
        )
        _check_start(folder, value)
        if terminal and read_seal(folder / "result.json") != value:
            raise ValueError("precision reduction terminal differs from receipts")
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
                    "optimizer_termination",
                    "optimization_status",
                    "fitting_diagnosis",
                    "diagnostic_gradient_precision_qualified",
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
    recorded = sum(row["status"] == "complete" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "rows": rows,
        "earlier_cost_by_model": data["earlier_cost_by_model"],
        "local_recovery_cost_by_task": [
            {key: task[key] for key in ("index", "model", "task_id", "kind", "cost")}
            for task in data["local_recovery_tasks"]
        ],
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
        "limitation": "Training-only recovery of frozen equations; task completion and "
        "optimizer xtol do not establish convergence. Diagnosis is bound to the exact "
        "retained endpoint; certified quotients do not identify removed parameters. "
        "Historical costs are separate and never charged again. No global-optimum "
        "or scientific-validity certificate.",
    }
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", summary)
    return summary


def inspect(root: Path) -> dict:
    """Create a portable receipt archive without any numerical evaluation."""
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
                raise ValueError("archive path escapes precision reduction campaign")
            if source.name.startswith("."):
                continue
            blob = source.read_bytes()
            entry = tarfile.TarInfo(source.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}
