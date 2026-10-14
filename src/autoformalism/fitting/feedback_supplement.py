"""M33: separately budgeted continuation of a frozen diagnostic experiment.

A supplement never edits its source. Successful evaluations retain their exact
identities; every earlier failed or interrupted attempt remains in the cost
ledger. Only missing evaluations of the original perturbation points can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import portable_feedback as portable
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import staged_feedback as feedback
from autoformalism.schemas.base import StrictSchema


class Policy(StrictSchema):
    """A new incremental allowance, never a change to an existing allowance."""

    extra_seconds: float = Field(default=1200, ge=1, le=7200)
    per_call_seconds: float = Field(default=120, ge=1, le=1800)


class Identity(StrictSchema):
    """Bind a new allowance to one exact predecessor and frozen experiment."""

    protocol: Literal["fitting-feedback-supplement-1"] = "fitting-feedback-supplement-1"
    predecessor_sha256: str
    original_identity_sha256: str
    perturbation_plan_sha256: str
    policy_sha256: str
    source_sha256: str
    runtime_sha256: str


class OperationReference(StrictSchema):
    """An exact inherited or newly executed operation, without relabeling it."""

    key: str
    origin_sha256: str
    operation_sha256: str
    identity: dict
    stage: feedback.StageRecord


class SupplementPlan(StrictSchema):
    """Reviewable frozen inputs and the one-attempt-per-missing-key schedule."""

    identity: Identity
    source_path: str
    policy: Policy
    runtime: dict
    original_identity: feedback.legacy.Binding
    perturbation_plan: list[feedback.piecewise.Perturbation]
    reused_operations: list[OperationReference]
    missing_keys: list[str]


class SupplementResult(StrictSchema):
    """Both diagnoses and all original/incremental attempts and costs."""

    protocol: Literal["fitting-feedback-supplement-1"] = "fitting-feedback-supplement-1"
    identity: Identity
    original_packet: feedback.FeedbackPacket
    predecessor_packet: feedback.FeedbackPacket
    packet: feedback.FeedbackPacket
    original_cost: feedback.DiagnosticCost
    inherited_cost: feedback.DiagnosticCost
    new_cost: feedback.DiagnosticCost
    combined_cost: feedback.DiagnosticCost
    ledger: list[OperationReference]
    reused_operations: list[OperationReference]
    attempted_operations: list[OperationReference]
    required_calls: int
    new_calls_started: int | None
    new_calls_completed: int | None
    remaining_keys: list[str]
    additional_budget_exhausted: bool
    parameters_changed: Literal[False] = False
    optimization_calls: Literal[0] = 0
    validation_used: Literal[False] = False
    test_used: Literal[False] = False
    llm_calls: Literal[0] = 0


@dataclass(frozen=True)
class _Source:
    """Verified source lineage and successful views used only for assembly."""

    snapshot: portable.Snapshot
    digest: str
    packet: feedback.FeedbackPacket
    original_packet: feedback.FeedbackPacket
    operations: dict
    ledger: list[OperationReference]


def _reference(key: str, operation: dict, origin: str) -> OperationReference:
    return OperationReference(
        key=key,
        origin_sha256=origin,
        operation_sha256=public.content_sha256(operation),
        identity=operation["identity"],
        stage=feedback._record(key, operation),
    )


def _cost(references: list[OperationReference]) -> feedback.DiagnosticCost:
    return feedback._cost([item.stage for item in references])


def _targets(source: _Source) -> dict[str, tuple[dict, str]]:
    """Use only the points actually frozen by the original experiment."""
    snapshot = source.snapshot
    primary = feedback._value(source.operations, "primary")
    if snapshot.fit.parameters is None or not primary:
        raise ValueError("supplement requires a retained point and primary replay")
    names = primary["metadata"]["names"]
    targets = {}
    for item in snapshot.plan:
        if item.in_domain:
            parameters = dict(zip(names, item.vector, strict=True))
            for suffix, solver in (("primary", "DOP853"), ("alternate", "Radau")):
                targets[f"perturb/{item.key}/{suffix}"] = (parameters, solver)
    return targets


def _missing(source: _Source) -> list[str]:
    return [
        key
        for key in _targets(source)
        if source.operations.get(key, {}).get("status") != "complete"
    ]


def _source(folder: Path, seen: frozenset[Path] = frozenset()) -> _Source:
    """Read and verify a completed M32/M33 source without writing to it."""
    folder = folder.resolve()
    if folder in seen:
        raise ValueError("cyclic supplemental source lineage")
    if (folder / "supplement.json").exists():
        plan = SupplementPlan.model_validate(read_seal(folder / "supplement.json"))
        predecessor = _source(folder / plan.source_path, seen | {folder})
        result, operations = _completed(folder, plan, predecessor)
        digest = public.content_sha256(
            {
                "plan": plan.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "stages": read_seal(folder / "stages.json"),
            }
        )
        selected = dict(predecessor.operations)
        for key, operation in operations.items():
            if selected.get(key, {}).get("status") != "complete":
                selected[key] = operation
        return _Source(
            predecessor.snapshot,
            digest,
            result.packet,
            result.original_packet,
            selected,
            result.ledger,
        )
    snapshot = portable.load_snapshot(folder)
    return _Source(
        snapshot,
        snapshot.source_sha256,
        snapshot.packet,
        snapshot.packet,
        snapshot.operations,
        [
            _reference(key, op, snapshot.source_sha256)
            for key, op in snapshot.operations.items()
        ],
    )


def _plan(source: _Source, path: Path, folder: Path, policy: Policy) -> SupplementPlan:
    runtime = {
        "packages": public._runtime(),
        "numerical_environment": portable.numerical_environment(),
    }
    original = source.original_packet.identity
    identity = Identity(
        predecessor_sha256=source.digest,
        original_identity_sha256=public.content_sha256(
            original.model_dump(mode="json")
        ),
        perturbation_plan_sha256=public.content_sha256(
            [item.model_dump(mode="json") for item in source.snapshot.plan]
        ),
        policy_sha256=public.content_sha256(policy.model_dump(mode="json")),
        source_sha256=public._source_identity(),
        runtime_sha256=public.content_sha256(runtime),
    )
    return SupplementPlan(
        identity=identity,
        source_path=relpath(path.resolve(), folder.resolve()),
        policy=policy,
        runtime=runtime,
        original_identity=original,
        perturbation_plan=list(source.snapshot.plan),
        reused_operations=[r for r in source.ledger if r.stage.status == "complete"],
        missing_keys=_missing(source),
    )


def _paths(source: Path, folder: Path) -> tuple[Path, Path]:
    source, folder = source.resolve(), folder.resolve()
    if source.is_relative_to(folder) or folder.is_relative_to(source):
        raise ValueError("supplement and immutable source require separate directories")
    return source, folder


def _policy(folder: Path, policy: Policy | None) -> Policy:
    """An omitted override resumes the already frozen incremental allowance."""
    if policy is None and (folder / "supplement.json").exists():
        return SupplementPlan.model_validate(
            read_seal(folder / "supplement.json")
        ).policy
    return policy or Policy()


def prepare(source: Path, folder: Path, policy: Policy | None = None) -> SupplementPlan:
    """Seal a distinct, reviewable allowance without running any evaluation."""
    source, folder = _paths(source, folder)
    inherited = _source(source)
    with public._lock(folder):
        plan = _plan(inherited, source, folder, _policy(folder, policy))
        seal(folder / "supplement.json", plan.model_dump(mode="json"))
        return plan


def _validate_plan(plan: SupplementPlan, source: _Source, folder: Path) -> None:
    """Bind every input, including a predecessor's failed historical attempts."""
    expected = _plan(source, folder / plan.source_path, folder, plan.policy)
    # Inspection is portable; execution separately insists on the current runtime.
    expected = expected.model_copy(
        update={
            "identity": expected.identity.model_copy(
                update={
                    "source_sha256": plan.identity.source_sha256,
                    "runtime_sha256": plan.identity.runtime_sha256,
                }
            ),
            "runtime": plan.runtime,
        }
    )
    if (
        plan != expected
        or public.content_sha256(plan.runtime) != plan.identity.runtime_sha256
    ):
        raise ValueError("supplement source, policy or frozen point identity differs")


def _stage_identity(plan: SupplementPlan, source: _Source, key: str) -> dict:
    parameters, solver = _targets(source)[key]
    policy = source.snapshot.policy
    return plan.identity.model_dump(mode="json") | {
        "stage": key,
        "point_sha256": public.content_sha256(parameters),
        "solver": solver,
        "sensitivities": False,
        "relative_tolerance": policy.relative_tolerance,
        "absolute_tolerance": policy.absolute_tolerance,
    }


def _envelope(path: Path, operation: dict) -> dict:
    if (path / "record.json").exists():
        envelope = read_seal(path / "record.json")
        if {k: v for k, v in envelope.items() if k != "progress"} != operation:
            raise ValueError("supplement stage progress envelope differs")
        return envelope
    progress = (
        public._read(path / "progress.json")
        if (path / "progress.json").exists()
        else {}
    )
    envelope = operation | {"progress": progress}
    seal(path / "record.json", envelope)
    return envelope


def _result(
    source: _Source, plan: SupplementPlan, operations: dict
) -> SupplementResult:
    """Recombine successful evidence while retaining every attempt's exact cost."""
    selected = dict(source.operations)
    selected.update(operations)
    snapshot = source.snapshot
    packet = feedback._assemble(
        snapshot.problem,
        snapshot.fit,
        snapshot.backend,
        snapshot.policy,
        source.original_packet.identity,
        selected,
        perturbation_plan=snapshot.plan,
    )
    origin = public.content_sha256(plan.identity.model_dump(mode="json"))
    attempted = [_reference(key, op, origin) for key, op in operations.items()]
    ledger = source.ledger + attempted
    combined_cost, new_cost = _cost(ledger), _cost(attempted)
    reasons = set(packet.diagnostics.reason_codes)
    reasons.discard("PERTURBATION_BUDGET_EXHAUSTED")
    if (
        "PERTURBATION_BUDGET_EXHAUSTED"
        in source.original_packet.diagnostics.reason_codes
    ):
        reasons.add("HISTORICAL_PERTURBATION_BUDGET_EXHAUSTED")
    if any(item.stage.status != "complete" for item in source.ledger):
        reasons.add("HISTORICAL_DIAGNOSTIC_FAILURES_RETAINED")
    if new_cost.budget_charge_seconds >= plan.policy.extra_seconds:
        reasons.add("SUPPLEMENT_BUDGET_EXHAUSTED")
    # The assessment uses one available result per planned key. Its audit and
    # accounting use *all* attempts, so a successful retry cannot erase a failure.
    diagnostics = packet.diagnostics.model_copy(
        update={
            "stages": [item.stage for item in ledger],
            "reason_codes": sorted(reasons),
            "artifact_sha256": public.content_sha256(
                [item.model_dump(mode="json") for item in ledger]
            ),
        }
    )
    packet = packet.model_copy(
        update={"diagnostics": diagnostics, "cost": combined_cost}
    )
    return SupplementResult(
        identity=plan.identity,
        original_packet=source.original_packet,
        predecessor_packet=source.packet,
        packet=packet,
        original_cost=source.original_packet.cost,
        inherited_cost=_cost(source.ledger),
        new_cost=new_cost,
        combined_cost=combined_cost,
        ledger=ledger,
        reused_operations=plan.reused_operations,
        attempted_operations=attempted,
        required_calls=len(plan.missing_keys),
        new_calls_started=new_cost.calls,
        new_calls_completed=new_cost.calls_completed,
        remaining_keys=[
            key
            for key in _targets(source)
            if selected.get(key, {}).get("status") != "complete"
        ],
        additional_budget_exhausted=new_cost.budget_charge_seconds
        >= plan.policy.extra_seconds,
    )


def _validate_operation(operation: dict, allowance: float, residual_size: int) -> None:
    """Require truthful completion states and retained measured/reserved charges."""
    charge = operation["budget_charge_seconds"]
    if not np.isfinite(charge) or charge < 0:
        raise ValueError("invalid supplemental budget charge")
    if operation.get("budget_restarted") is not False:
        raise ValueError("supplemental operation budget was restarted")
    if operation["status"] == "interrupted":
        if (
            operation["value"] is not None
            or operation["accounting_complete"] is not False
            or any(
                operation[field] is not None
                for field in ("wall_seconds", "cpu_seconds", "calls")
            )
            or charge != allowance
        ):
            raise ValueError("interrupted supplemental accounting differs")
        return
    if (
        operation["status"] not in {"complete", "unavailable"}
        or operation["accounting_complete"] is not True
        or operation["calls"] not in (0, 1)
        or any(
            not isinstance(operation[field], (int, float))
            or not np.isfinite(operation[field])
            or operation[field] < 0
            for field in ("wall_seconds", "cpu_seconds")
        )
        or charge != operation["wall_seconds"]
    ):
        raise ValueError("supplement measured operation cost differs")
    value = operation["value"]
    if operation["status"] == "complete":
        if (
            not isinstance(value, dict)
            or operation["calls"] != 1
            or operation["progress"].get("calls_completed") != 1
        ):
            raise ValueError("complete supplemental stage lacks completed evidence")
        residual = np.asarray(value["residual"], dtype=float)
        if residual.shape != (residual_size,) or not np.isfinite(residual).all():
            raise ValueError("invalid supplemental residual")
        if value["jacobian"] is not None:
            raise ValueError("supplemental output stage contains unexpected Jacobian")
    elif value is not None or not operation.get("error"):
        raise ValueError("unavailable supplemental stage contains completed evidence")


def _read_operations(folder: Path, plan: SupplementPlan, source: _Source) -> dict:
    operations, charge = {}, 0.0
    manifest = read_seal(folder / "stages.json")
    for index, (key, digest) in enumerate(manifest):
        if index >= len(plan.missing_keys) or key != plan.missing_keys[index]:
            raise ValueError("supplement stages differ from frozen plan")
        remaining = plan.policy.extra_seconds - charge
        if remaining <= 0:
            raise ValueError("supplement stage exceeds incremental budget")
        path = folder / "stages" / key
        operation = read_seal(path / "record.json")
        expected = _stage_identity(plan, source, key) | {
            "allowance_seconds": min(plan.policy.per_call_seconds, remaining)
        }
        if (
            public.content_sha256(operation) != digest
            or operation["identity"] != expected
            or read_seal(path / "started.json") != expected
            or {k: v for k, v in operation.items() if k != "progress"}
            != read_seal(path / "result.json")
        ):
            raise ValueError("supplement operation identity or envelope differs")
        _validate_operation(
            operation,
            expected["allowance_seconds"],
            len(feedback._value(source.operations, "primary")["residual"]),
        )
        operations[key] = operation
        charge += operation["budget_charge_seconds"]
    started_keys = {
        str(path.parent.relative_to(folder / "stages"))
        for path in (folder / "stages").rglob("started.json")
    }
    if started_keys != set(operations):
        raise ValueError("supplement ledger omits a started attempt")
    if len(operations) < len(plan.missing_keys) and charge < plan.policy.extra_seconds:
        raise ValueError("supplement ended with unstarted work and remaining allowance")
    return operations


def _completed(
    folder: Path, plan: SupplementPlan, source: _Source
) -> tuple[SupplementResult, dict]:
    _validate_plan(plan, source, folder)
    operations = _read_operations(folder, plan, source)
    saved = SupplementResult.model_validate(read_seal(folder / "result.json"))
    recomputed = _result(source, plan, operations)
    selected = source.operations | operations
    portable.verify_packet(saved.packet, recomputed.packet, selected)
    # Original and predecessor packets are exact source artifacts. Only the new
    # numerical assembly can vary with the inspecting linear algebra runtime.
    if saved.model_copy(update={"packet": recomputed.packet}) != recomputed:
        raise ValueError("supplement evidence or historical cost differs")
    return saved, operations


def completed(folder: Path) -> SupplementResult:
    """Verify sealed evidence and lineage without evaluations or source edits."""
    folder = folder.resolve()
    plan = SupplementPlan.model_validate(read_seal(folder / "supplement.json"))
    source_path, _ = _paths(folder / plan.source_path, folder)
    return _completed(folder, plan, _source(source_path, frozenset({folder})))[0]


def run(source: Path, folder: Path, policy: Policy | None = None) -> SupplementResult:
    """Run each missing original point at most once under a new fixed allowance."""
    source, folder = _paths(source, folder)
    inherited = _source(source)
    with public._lock(folder):
        plan = _plan(inherited, source, folder, _policy(folder, policy))
        seal(folder / "supplement.json", plan.model_dump(mode="json"))
        if (folder / "result.json").exists():
            return _completed(folder, plan, inherited)[0]
        operations, oracle = {}, None
        targets = _targets(inherited)
        for key in plan.missing_keys:
            charged = sum(op["budget_charge_seconds"] for op in operations.values())
            remaining = plan.policy.extra_seconds - charged
            if remaining <= 0:
                break
            parameters, method = targets[key]

            def evaluate(deadline, checkpoint, parameters=parameters, method=method):
                nonlocal oracle
                checkpoint(
                    {"calls": 0, "calls_completed": 0, "phase": "symbolic_setup"}
                )
                if oracle is None:
                    oracle = checks.Rollouts(
                        inherited.snapshot.problem,
                        checks.ConfidencePolicy(
                            relative_tolerance=inherited.snapshot.policy.relative_tolerance,
                            absolute_tolerance=inherited.snapshot.policy.absolute_tolerance,
                        ),
                    )
                return feedback.evaluate_stage(
                    oracle,
                    oracle.vector(parameters),
                    deadline,
                    checkpoint,
                    method,
                    False,
                )

            path = folder / "stages" / key
            operation = checks.operation(
                path,
                _stage_identity(plan, inherited, key),
                min(plan.policy.per_call_seconds, remaining),
                evaluate,
            )
            operations[key] = _envelope(path, operation)
        result = _result(inherited, plan, operations)
        seal(
            folder / "stages.json",
            [[key, public.content_sha256(op)] for key, op in operations.items()],
        )
        seal(folder / "result.json", result.model_dump(mode="json"))
        return result
