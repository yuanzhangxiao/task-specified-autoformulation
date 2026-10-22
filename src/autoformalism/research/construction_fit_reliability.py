"""Separately budgeted numerical reliability for frozen construction continuations.

The source campaign is read-only. Saved failed verifications are replayed before
choosing one current-incumbent continuation per arm. Historical missing costs
stay unknown, and an unresolved attempt never replaces the exact old incumbent.
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
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_continuation as previous
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-fitting-reliability-1"
INPUT_PROTOCOL = "construction-fitting-reliability-inputs-1"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonnegative = Annotated[FiniteFloat, Field(ge=0)]
OLD_VERIFICATIONS = tuple(
    ["retained_control_verification"]
    + [f"{name}_verification" for name in previous.STARTS]
)
COST_KEYS = frozenset(journal._cost({})) - {"scope"}


class Policy(StrictSchema):
    """One new search per arm; every verification ladder has a total finite cap."""

    population: Literal["returned-eight", "synthetic-smoke"] = "returned-eight"
    search_seconds: Nonnegative = Field(default=600, gt=0, le=600)
    diagnostic_seconds: Nonnegative = Field(default=120, gt=0, le=120)
    verification_seconds: Nonnegative = Field(default=180, gt=0, le=180)
    replay_seconds: Nonnegative = Field(default=180, gt=0, le=180)
    maximum_calls: int = Field(default=1000, ge=1, le=1000)


class Endpoint(StrictSchema):
    """Only original-coordinate parameters and training verification evidence."""

    origin: Literal[
        "original_saved_vector",
        "retained_control_verification",
        "inherited_verification",
        "declared_verification",
        "perturbed_declared_verification",
    ]
    parameters: dict[str, FiniteFloat]
    parameters_sha256: Digest
    original_equations_verified: bool
    training_nmse: Nonnegative | None = None
    alternate_training_nmse: Nonnegative | None = None
    solver_loss_discrepancy: Nonnegative | None = None
    solver_prediction_discrepancy: Nonnegative | None = None
    per_target_normalized_mse: dict[str, Nonnegative] | None = None


class OperationEvidence(StrictSchema):
    """Hashes bind authenticated old receipts without importing free-form logs."""

    status: Literal["complete", "unavailable", "interrupted", "running_or_interrupted"]
    receipt_sha256: Digest | None
    reservation_sha256: Digest
    progress_sha256: Digest | None


class ArmEvidence(StrictSchema):
    """One expected original/reduced arm, including missing historical evidence."""

    index: int = Field(ge=0)
    model: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    arm: Literal["affine_original", "affine_reduced"]
    source_status: Literal["complete", "partial", "source_missing"]
    source_terminal_sha256: Digest | None
    source_qualification_passed: bool
    source_cost_known: bool
    source_cost: dict[str, Nonnegative | bool | None] | None
    operations: dict[str, OperationEvidence]
    incumbent: Endpoint
    failed_endpoints: list[Endpoint] = Field(max_length=4)


class Inputs(StrictSchema):
    """Positive allowlist; the nested old input is separately schema checked."""

    protocol: Literal["construction-fitting-reliability-inputs-1"]
    source_plan_sha256: Digest
    source_inputs_sha256: Digest
    source_inputs: dict
    arms: list[ArmEvidence]
    test_data_opened: Literal[False]
    validation_exported: Literal[False]
    old_allowances_renewed: Literal[False]


def _endpoint(point: dict, origin: str) -> dict:
    """Project authenticated verification evidence into a bounded source schema."""
    payload = {
        "origin": origin,
        "parameters": point["parameters"],
        "parameters_sha256": public.content_sha256(point["parameters"]),
        "original_equations_verified": point.get("original_equations_verified", False),
        **{
            key: point.get(key)
            for key in (
                "training_nmse",
                "alternate_training_nmse",
                "solver_loss_discrepancy",
                "solver_prediction_discrepancy",
                "per_target_normalized_mse",
            )
        },
    }
    return Endpoint.model_validate(payload).model_dump(mode="json")


def _source_arm(source: Path, plan: dict, data: dict, index: int) -> dict:
    """Authenticate old journals and their terminal reconstruction without fitting."""
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = source / "results" / task["task_id"] / task["arm"]
    policy = previous.Policy.model_validate(plan["policy"])
    allowed = previous._allowances(policy)
    if any(
        p.is_dir() and p.name not in allowed for p in (folder / "operations").glob("*")
    ):
        raise ValueError("unknown source operation")
    operations = previous._operations(folder, identity, policy)
    terminal = (folder / "result.json").exists()
    assembled = previous._assemble(identity, operations, terminal=terminal)
    saved = read_seal(folder / "result.json") if terminal else None
    if terminal and saved != assembled:
        raise ValueError("source terminal differs from authenticated operations")
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
    retained = assembled["retained"]
    incumbent = _endpoint(
        retained or {"parameters": data["models"][task["model"]]["parameters"]},
        assembled["retained_origin"] + "_verification"
        if retained
        else "original_saved_vector",
    )
    failed, seen = [], set()
    for name in OLD_VERIFICATIONS:
        operation = operations.get(name, {})
        # Only a sealed result binds its progress checkpoint. A bare interrupted
        # progress file has no authenticated endpoint and cannot enter selection.
        if not (folder / "operations" / name / "result.json").exists():
            continue
        point = operation.get("value") or operation.get("observed_progress") or {}
        qualified = (
            operation.get("status") == "complete"
            and point.get("status", "complete") == "complete"
            and point.get("original_equations_verified")
        )
        if point.get("parameters") and not qualified:
            endpoint = _endpoint(point | {"original_equations_verified": False}, name)
            if endpoint["parameters_sha256"] not in seen:
                seen.add(endpoint["parameters_sha256"])
                failed.append(endpoint)
    cost = (
        {key: value for key, value in assembled["cost"].items() if key in COST_KEYS}
        if operations
        else None
    )
    return {
        "index": index,
        **task,
        "source_status": "complete"
        if terminal
        else "partial"
        if operations
        else "source_missing",
        "source_terminal_sha256": public.content_sha256(saved) if saved else None,
        "source_qualification_passed": bool(
            (operations.get("qualification", {}).get("value") or {}).get("passed")
        ),
        "source_cost_known": bool(operations)
        and assembled["cost"]["accounting_complete"],
        "source_cost": cost,
        "operations": evidence,
        "incumbent": incumbent,
        "failed_endpoints": failed,
    }


def export_inputs(source: Path, destination: Path) -> dict:
    """Export training-only old inputs and authenticated endpoint/cost projections."""
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("reliability export must not write into its source campaign")
    plan, data = previous.verify(source, runtime=False)
    payload = {
        "protocol": INPUT_PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs_sha256": public.content_sha256(data),
        "source_inputs": data,
        "arms": [_source_arm(source, plan, data, i) for i in range(len(plan["tasks"]))],
        "test_data_opened": False,
        "validation_exported": False,
        "old_allowances_renewed": False,
    }
    policy = Policy(population=plan["policy"]["population"])
    _validate(payload, policy)
    seal(destination, payload)
    return {
        "inputs_sha256": public.content_sha256(payload),
        "tasks": len(payload["arms"]),
        "source_missing": [
            a["index"]
            for a in payload["arms"]
            if a["source_status"] == "source_missing"
        ],
    }


def _validate(data: dict, policy: Policy) -> None:
    """Validate every projected field before any numerical construction."""
    parsed = Inputs.model_validate(data)
    old = parsed.source_inputs
    previous._validate(old, previous.Policy(population=policy.population))
    if parsed.source_inputs_sha256 != public.content_sha256(old):
        raise ValueError("source inputs digest differs")
    expected = [
        (i, model, arm)
        for i, model in enumerate(old["models"])
        for arm in previous.ARMS
    ]
    if len(parsed.arms) != len(expected):
        raise ValueError("all expected reliability arms required")
    for index, (item, (model_index, model, arm)) in enumerate(
        zip(parsed.arms, expected, strict=True)
    ):
        if (item.index, item.model, item.task_id, item.arm) != (
            index,
            model_index,
            model["task_id"],
            arm,
        ):
            raise ValueError("source arm identity/order differs")
        if set(item.operations) - set(previous._allowances(previous.Policy())):
            raise ValueError("unknown source operation evidence")
        missing = item.source_status == "source_missing"
        if missing and (
            item.operations
            or item.source_cost is not None
            or item.source_cost_known
            or item.source_terminal_sha256
            or item.source_qualification_passed
            or item.failed_endpoints
        ):
            raise ValueError("missing source cannot claim known historical work")
        if item.source_status == "complete" and item.source_terminal_sha256 is None:
            raise ValueError("complete source requires terminal identity")
        if missing and (
            item.incumbent.origin != "original_saved_vector"
            or item.incumbent.original_equations_verified
        ):
            raise ValueError(
                "missing source requires an uncertified saved-vector fallback"
            )
        if item.source_cost is not None and set(item.source_cost) != COST_KEYS:
            raise ValueError("source cost schema differs")
        if item.source_cost_known and (
            item.source_cost is None or not item.source_cost["accounting_complete"]
        ):
            raise ValueError("source cost certainty differs")
        hashes = []
        for endpoint in [item.incumbent, *item.failed_endpoints]:
            if set(endpoint.parameters) != set(
                model["parameters"]
            ) or endpoint.parameters_sha256 != public.content_sha256(
                endpoint.parameters
            ):
                raise ValueError("source endpoint coordinates or digest differ")
            if endpoint.per_target_normalized_mse is not None and set(
                endpoint.per_target_normalized_mse
            ) != set(model["problem"]["context"]["targets"]):
                raise ValueError("source endpoint targets differ")
            if (
                endpoint.origin != "original_saved_vector"
                and endpoint.origin not in item.operations
            ):
                raise ValueError("endpoint lacks source operation evidence")
            if endpoint.origin != "original_saved_vector" and (
                item.operations[endpoint.origin].receipt_sha256 is None
                or (
                    endpoint.original_equations_verified
                    and item.operations[endpoint.origin].status != "complete"
                )
            ):
                raise ValueError("endpoint lacks a qualifying sealed source receipt")
            if (
                endpoint.origin == "original_saved_vector"
                and endpoint.original_equations_verified
            ):
                raise ValueError(
                    "saved-vector fallback has no verification certificate"
                )
            if endpoint.original_equations_verified and any(
                getattr(endpoint, key) is None
                for key in (
                    "training_nmse",
                    "alternate_training_nmse",
                    "solver_loss_discrepancy",
                    "solver_prediction_discrepancy",
                )
            ):
                raise ValueError("verified source endpoint lacks numerical evidence")
            if endpoint.original_equations_verified:
                tolerance = max(
                    1e-12,
                    1e-8
                    * max(endpoint.training_nmse, endpoint.alternate_training_nmse),
                )
                if (
                    endpoint.solver_prediction_discrepancy > 1e-7
                    or endpoint.solver_loss_discrepancy > tolerance
                    or abs(endpoint.training_nmse - endpoint.alternate_training_nmse)
                    > tolerance
                ):
                    raise ValueError("source verification discrepancy exceeds contract")
        if (
            item.incumbent.origin == "original_saved_vector"
            and item.incumbent.parameters != model["parameters"]
        ):
            raise ValueError("fallback differs from exact original saved vector")
        for endpoint in item.failed_endpoints:
            if (
                endpoint.original_equations_verified
                or endpoint.origin == "original_saved_vector"
            ):
                raise ValueError("replay endpoint is not a saved failed verification")
            hashes.append(endpoint.parameters_sha256)
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate source replay endpoints")


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Freeze schemas and separate allowances; no solver work on a login node."""
    _validate(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "tasks": [
            {key: arm[key] for key in ("index", "model", "task_id", "arm")}
            for arm in data["arms"]
        ],
        "start_rule": (
            "replay_saved_failed_endpoints_then_one_verified_current_incumbent"
        ),
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
    """Verify a new campaign without requiring any historical runtime to exist."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    _validate(data, Policy.model_validate(plan["policy"]))
    expected = [
        {key: arm[key] for key in ("index", "model", "task_id", "arm")}
        for arm in data["arms"]
    ]
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != expected
        or plan["llm_calls"] != 0
        or any(
            plan[key] is not False
            for key in (
                "old_allowances_renewed",
                "test_data_opened",
                "validation_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("reliability plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("reliability source/runtime differs")
    return plan, data


def _allowances(policy: Policy, arm: dict) -> dict:
    """New finite reservations include separate setup/checkpoint overhead."""
    return {
        "incumbent_verification": policy.verification_seconds + 90,
        **{
            f"replay_{i}_verification": policy.replay_seconds + 90
            for i in range(len(arm["failed_endpoints"]))
        },
        "qualification": policy.diagnostic_seconds + 90,
        "continuation_search": policy.search_seconds + 150,
        "continuation_verification": policy.verification_seconds + 90,
        "continuation_diagnosis": policy.diagnostic_seconds + 90,
    }


def _operations(folder: Path, identity: dict, policy: Policy, arm: dict) -> dict:
    """Read new immutable reservations without renewing interrupted stages."""
    operations = {}
    for name, seconds in _allowances(policy, arm).items():
        location = folder / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (location / "result.json").exists():
            operations[name] = journal._read_operation(location, bound)
        elif (location / "started.json").exists():
            if read_seal(location / "started.json") != bound:
                raise ValueError("reliability reservation differs")
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
    """Only completed, positively verified stages can support new retention."""
    value = operation.get("value") or {}
    return (
        value
        if operation.get("status") == "complete"
        and value.get("original_equations_verified")
        and value.get("status", "complete") == "complete"
        else None
    )


def _improves(candidate: dict, incumbent: dict) -> bool:
    """Require strict improvement beyond the two-solver score envelope."""
    return not incumbent.get("original_equations_verified") or max(
        candidate["training_nmse"], candidate["alternate_training_nmse"]
    ) < min(incumbent["training_nmse"], incumbent["alternate_training_nmse"])


def _selected(arm: dict, operations: dict) -> dict:
    """Selection happens after replay and retains exact original coordinates."""
    retained, origin = deepcopy(arm["incumbent"]), "source_incumbent"
    current = _verified(operations.get("incumbent_verification", {}))
    # A replay of the exact old vector updates its numerical receipt, not factors.
    if current is not None:
        retained, origin = current, "incumbent_verification"
    for i in range(len(arm["failed_endpoints"])):
        name = f"replay_{i}_verification"
        point = _verified(operations.get(name, {}))
        if point is not None and _improves(point, retained):
            retained, origin = point, name
    return {
        "origin": origin,
        "parameters": deepcopy(retained["parameters"]),
        "verification": retained,
        "currently_verified": origin != "source_incumbent",
    }


def _assemble(identity: dict, arm: dict, operations: dict, *, terminal=False) -> dict:
    """Keep source/new costs separate and show unresolved numerical evidence."""
    selected = _selected(arm, operations)
    retained, origin = selected["verification"], selected["origin"]
    candidate = _verified(operations.get("continuation_verification", {}))
    if candidate is not None and _improves(candidate, retained):
        retained, origin = candidate, "continuation_verification"
    diagnosis_operation = operations.get("continuation_diagnosis", {})
    # Equal loss at a different vector cannot transfer its optimizer history.
    diagnosis = (
        (diagnosis_operation.get("value") or {})
        if candidate is not None
        and candidate["parameters"] == retained["parameters"]
        and diagnosis_operation.get("status") == "complete"
        else {}
    )
    assessment = diagnosis.get("assessment")
    replay = [
        {
            "source_origin": endpoint["origin"],
            "parameters_sha256": endpoint["parameters_sha256"],
            "operation_status": operations.get(f"replay_{i}_verification", {}).get(
                "status", "not_recorded"
            ),
            "verified": _verified(operations.get(f"replay_{i}_verification", {}))
            is not None,
        }
        for i, endpoint in enumerate(arm["failed_endpoints"])
    ]
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "source_status": arm["source_status"],
        "source_cost_known": arm["source_cost_known"],
        "source_cost": arm["source_cost"],
        "source_incumbent": arm["incumbent"],
        "selected_start": selected,
        "retained_origin": origin,
        "retained": retained,
        "before_after": {
            "source_training_nmse": arm["incumbent"]["training_nmse"],
            "selected_start_training_nmse": selected["verification"].get(
                "training_nmse"
            ),
            "retained_training_nmse": retained.get("training_nmse"),
        },
        "qualification_passed": bool(
            (operations.get("qualification", {}).get("value") or {}).get("passed")
        ),
        "saved_endpoint_replay": replay,
        "fitting_diagnosis": assessment,
        "optimization_status": (assessment or {}).get(
            "classification", "optimization_unresolved"
        ),
        "operations": operations,
        "cost": journal._cost(operations),
        "diagnostic_cost": previous._diagnostic_cost(operations),
        "old_allowances_renewed": False,
        "automatic_revision": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
    }


def _check_selected_start(folder: Path, result: dict) -> None:
    """Bind the persisted start to its reconstructed pre-search replay choice."""
    path = folder / "selected_start.json"
    if path.exists() and read_seal(path) != result["selected_start"]:
        raise ValueError("selected start differs from verification receipts")


def run_task(root: Path, index: int) -> dict:
    """Replay before choice, then continue once with separately reserved budgets."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown reliability task index")
    task, arm = plan["tasks"][index], data["arms"][index]
    model = data["source_inputs"]["models"][task["model"]]
    policy = Policy.model_validate(plan["policy"])
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"] / task["arm"]
    problem = PrefixFitProblem.model_validate(model["problem"])
    training = PublicSplit.model_validate(data["source_inputs"]["training"])
    settings = kernel.NumericalPolicy.model_validate(
        data["source_inputs"]["numerical_policy"]
    )
    allowances = _allowances(policy, arm)

    def operate(name, work, *, original_only=False):
        def call(save):
            deadline = monotonic() + allowances[name]

            def check():
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "inclusive reliability stage allowance exhausted"
                    )

            def checked_save(value):
                save(value)
                check()

            original, oracle = numerical.make_oracle(
                problem,
                training,
                settings,
                "affine_original" if original_only else task["arm"],
            )
            check()
            result = work(original, oracle, checked_save)
            check()
            return result

        return journal._operation(folder, identity, name, allowances[name], call)

    with public._lock(folder):
        terminal = folder / "result.json"
        if terminal.exists():
            saved = read_seal(terminal)
            rebuilt = _assemble(
                identity, arm, _operations(folder, identity, policy, arm), terminal=True
            )
            _check_selected_start(folder, rebuilt)
            if saved != rebuilt:
                raise ValueError("reliability terminal differs from stage receipts")
            return saved
        operate(
            "incumbent_verification",
            lambda o, a, save: numerical.verify_endpoint_ladder(
                o, o, arm["incumbent"]["parameters"], policy.verification_seconds, save
            ),
            original_only=True,
        )
        for i, endpoint in enumerate(arm["failed_endpoints"]):
            operate(
                f"replay_{i}_verification",
                lambda o,
                a,
                save,
                p=endpoint["parameters"]: numerical.verify_endpoint_ladder(
                    o, o, p, policy.replay_seconds, save
                ),
                original_only=True,
            )
        selected = _selected(arm, _operations(folder, identity, policy, arm))
        seal(folder / "selected_start.json", selected)
        if selected["currently_verified"]:
            qualified = operate(
                "qualification",
                lambda o, a, save: numerical.qualification(
                    o,
                    a,
                    selected["parameters"],
                    policy.diagnostic_seconds,
                    save,
                    adequacy_policy=adequacy.Policy(precision_aware=True),
                ),
            )
            if qualified["status"] == "complete" and (qualified.get("value") or {}).get(
                "passed"
            ):

                def search(o, a, save):
                    start = numerical.to_effective(o, a, selected["parameters"])
                    return numerical.search(
                        a,
                        start,
                        policy.search_seconds,
                        policy.maximum_calls,
                        save,
                        reject_invalid_trials=True,
                    )

                attempt = operate("continuation_search", search)
                fit = previous._search_checkpoint(attempt)
                if fit.get("best"):
                    parameters = fit["best"]["parameters"]
                    checked = operate(
                        "continuation_verification",
                        lambda o, a, save: numerical.verify_endpoint_ladder(
                            o, a, parameters, policy.verification_seconds, save
                        ),
                    )
                    point = _verified(checked)
                    if point is not None:
                        operate(
                            "continuation_diagnosis",
                            lambda o, a, save: numerical.diagnose(
                                a,
                                parameters,
                                point,
                                fit,
                                policy.diagnostic_seconds,
                                save,
                                adequacy_policy=adequacy.Policy(precision_aware=True),
                            ),
                        )
        result = _assemble(
            identity, arm, _operations(folder, identity, policy, arm), terminal=True
        )
        seal(terminal, result)
        return result


def report(root: Path) -> dict:
    """Reconstruct all expected arms; old missing numerical costs remain unknown."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows = []
    for task, arm in zip(plan["tasks"], data["arms"], strict=True):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"] / task["arm"]
        terminal = (folder / "result.json").exists()
        value = _assemble(
            identity, arm, _operations(folder, identity, policy, arm), terminal=terminal
        )
        _check_selected_start(folder, value)
        if terminal and read_seal(folder / "result.json") != value:
            raise ValueError("reliability terminal differs from operations")
        rows.append(
            task
            | {
                key: value[key]
                for key in (
                    "status",
                    "source_status",
                    "source_cost_known",
                    "source_cost",
                    "selected_start",
                    "retained_origin",
                    "before_after",
                    "qualification_passed",
                    "saved_endpoint_replay",
                    "optimization_status",
                    "fitting_diagnosis",
                    "cost",
                    "diagnostic_cost",
                )
            }
            | {
                "training_nmse": value["retained"].get("training_nmse"),
                "per_target_normalized_mse": value["retained"].get(
                    "per_target_normalized_mse"
                ),
            }
        )
    count = sum(row["status"] == "complete" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "status": "complete" if count == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": count,
        "rows": rows,
        "source_missing": [
            row["index"] for row in rows if row["source_status"] == "source_missing"
        ],
        "source_cost_complete": all(row["source_cost_known"] for row in rows),
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "limitation": (
            "Local numerical evidence only; no global optimum, scientific-correctness "
            "or structural-revision certificate. Source and new costs are separate; "
            "missing source cost is unknown."
        ),
    }
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", summary)
    return summary


def inspect(root: Path) -> dict:
    """Archive sealed training provenance and receipts without numerical work."""
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
                raise ValueError("archive path escapes reliability campaign")
            if source.name.startswith("."):
                continue
            blob = source.read_bytes()
            entry = tarfile.TarInfo(source.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}
