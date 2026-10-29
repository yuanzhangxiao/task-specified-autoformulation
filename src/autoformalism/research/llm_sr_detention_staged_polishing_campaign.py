"""One nonrenewable long segment and staged audits for two detention endpoints.

Portable inputs retain one training problem and two complete endpoint audits,
with historical provenance/cost digests instead of recursive campaign histories.
Report reconstruction performs arithmetic and receipt authentication only.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_polishing as polishing
from autoformalism.fitting import llm_sr_detention_probe_recovery as recovery
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import llm_sr_detention_staged_polishing as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.fitting.llm_sr_numerical_control import Transcription
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_detention_polishing_campaign as source
from autoformalism.research import (
    llm_sr_detention_probe_recovery_campaign as accounting,
)
from autoformalism.research import llm_sr_detention_refit_campaign as saved_fit
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "llm-sr-detention-staged-polishing-1"
EXPECTED_SOURCE_REPORT_SHA256 = (
    "a0fdef201d58dcaa70b524ee3374fbef874fd68807535a904fd3e9085d487591"
)
EXPECTED_SOURCE_REPORT_FILE_SHA256 = (
    "87c7e3d5805ad594ee82a5d863e572b72d37d033d2c9f2ffe96d5101215db535"
)
EXPECTED_INPUTS_SHA256 = (
    "91ee51ad094d53e40fcb6727dbca95ae01cbfa56395ed4b9e1eda13e219ea616"
)
STAGED_SOURCE_REPORT_SHA256 = (
    "2e367f471fe781f2dd3461e48f6e9e51af27590c74944aff5638e69486e6bb00"
)
STAGED_SOURCE_REPORT_FILE_SHA256 = (
    "e84c3863bfcaca65c1fba8cc2bc9c6411fa259865361fca2c0aad6159f67a9ac"
)
STAGED_INPUTS_SHA256 = (
    "5606f66b19db3ba14fcd5d93a0f55b70c4c4720d1e51e6546d01548822964737"
)
SourceStage = Literal["polishing", "staged"]
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


def _source_pins(source_stage: SourceStage) -> tuple[str, str, str]:
    """Keep historical pins while authorizing exactly one returned continuation."""
    if source_stage == "polishing":
        return (
            EXPECTED_SOURCE_REPORT_SHA256,
            EXPECTED_SOURCE_REPORT_FILE_SHA256,
            EXPECTED_INPUTS_SHA256,
        )
    if source_stage == "staged":
        return (
            STAGED_SOURCE_REPORT_SHA256,
            STAGED_SOURCE_REPORT_FILE_SHA256,
            STAGED_INPUTS_SHA256,
        )
    raise ValueError("unknown polishing source stage")


class Inputs(StrictSchema):
    """Exact compact extraction of the two approved final retained endpoints."""

    protocol: Literal[
        "llm-sr-detention-staged-polishing-input-1",
        "llm-sr-detention-staged-polishing-input-2",
    ] = "llm-sr-detention-staged-polishing-input-1"
    model: Transcription
    context: ValidationContext
    training: PublicSplit
    cases: list[dict]
    provenance: dict
    historical_costs: dict
    validation_data_opened: Literal[False] = False
    test_data_opened: Literal[False] = False
    llm_calls: Literal[0] = 0

    @model_validator(mode="after")
    def authenticate(self):
        stage = "staged" if self.protocol.endswith("-2") else "polishing"
        report_sha, _, inputs_sha = _source_pins(stage)
        if public.content_sha256(self) != inputs_sha:
            raise ValueError("compact polishing source extraction differs")
        if self.provenance["source_report_sha256"] != report_sha:
            raise ValueError("polishing source report differs")
        if len(self.cases) != 2:
            raise ValueError("polishing roster differs")
        for index, case in enumerate(self.cases):
            if case["index"] != index or case["source_index"] != index:
                raise ValueError("polishing case order differs")
            checkpoint, gate = case["parent_verified_checkpoint"], case["qualification"]
            if (
                checkpoint.get("status") != "complete"
                or not checkpoint.get("usable_for_retention")
                or not numerics.usable(checkpoint)
                or checkpoint["parameters"] != case["parameters"]
                or checkpoint.get("requested_parameters_sha256")
                != public.content_sha256(case["parameters"])
                or case["source_verification_sha256"]
                != public.content_sha256(checkpoint)
            ):
                raise ValueError(
                    "source endpoint lacks matching independent verification"
                )
            _validate_qualification(self, case, gate)
            if (
                not gate["complete"]
                or not gate["search_qualified"]
                or gate["coordinate_names"] != case["coordinate_names"]
            ):
                raise ValueError("source endpoint lacks full search qualification")
        return self


class Policy(StrictSchema):
    """One long search, full search audit, and conditionally executed strict audit."""

    protocol: Literal["llm-sr-detention-staged-polishing-policy-1"] = (
        "llm-sr-detention-staged-polishing-policy-1"
    )
    segments: Literal[1] = 1
    segment_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    direction_seconds: FiniteFloat = Field(default=360, ge=240, le=360)
    verification_seconds: FiniteFloat = Field(default=120, gt=0, le=120)


@dataclass(frozen=True)
class VerifiedRun:
    """Authenticated invocation shared by the launcher and one task worker."""

    root: Path
    plan: dict
    inputs: Inputs
    policy: Policy
    started_wall: float
    started_cpu: float
    started_utc: str
    _authentication: object = field(repr=False)


def _problem(inputs):
    return inputs.model, inputs.context, inputs.training


def _numerical(case):
    return chart.Policy.model_validate(case["numerical_policy"])


def _validate_qualification(inputs, case, qualification):
    """Authenticate raw evidence, allowing only documented summation roundoff."""
    chart.validate_qualification(
        *_problem(inputs),
        case["parameters"],
        qualification["directions"],
        mode=case["mode"],
        qualification=qualification,
        policy=_numerical(case),
    )


def _gate(inputs, task, parameters, directions):
    return chart.qualification(
        *_problem(inputs),
        parameters,
        directions,
        mode=task["mode"],
        policy=_numerical(task),
    )


def _historical_costs(value, path="historical_costs"):
    """Flatten separately labeled prior charges without copying endpoint histories."""
    rows = []
    if isinstance(value, dict):
        if "budget_charge_seconds" in value:
            fields = {
                key: value[key]
                for key in (
                    "wall_seconds",
                    "cpu_seconds",
                    "budget_charge_seconds",
                    "accounting_complete",
                    "new_operation_count",
                    "scope",
                )
                if key in value
            }
            rows.append(
                {
                    "source_path": path,
                    "cost": fields,
                    "source_entry_sha256": public.content_sha256(value),
                }
            )
        else:
            for key, child in sorted(value.items()):
                rows.extend(_historical_costs(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_historical_costs(child, f"{path}[{index}]"))
    return rows


def _extract(source_root: Path, *, source_stage: SourceStage = "polishing") -> dict:
    """Authenticate the saved source report against every original task receipt."""
    report_sha, file_sha, _ = _source_pins(source_stage)
    source_report = report if source_stage == "staged" else source.report
    source_verify = verify if source_stage == "staged" else source.verify
    print(
        "Authenticating source report and operation receipts...",
        file=sys.stderr,
        flush=True,
    )
    saved_path = source_root / "report/summary.json"
    if sha256(saved_path.read_bytes()).hexdigest() != file_sha:
        raise ValueError("saved source report file differs")
    saved = public._read(saved_path)
    if public.content_sha256(saved) != report_sha:
        raise ValueError("saved source report content differs")
    if source_report(source_root) != saved:
        raise ValueError("source report does not exactly reconstruct from receipts")
    if saved.get("status") != "complete" or saved.get("recorded") != 2:
        raise ValueError("source continuation is not complete")
    plan, old, _ = source_verify(source_root, runtime=False)
    cases = []
    for index in range(2):
        row = saved["tasks"][index]
        task, point = row["identity"]["task"], row["retained"]
        cases.append(
            {
                "index": index,
                "source_index": index,
                "start_id": task["start_id"],
                "mode": task["mode"],
                "parameters": deepcopy(point["parameters"]),
                "parent_verified_checkpoint": deepcopy(point),
                "source_verification_sha256": public.content_sha256(point),
                "numerical_policy": deepcopy(task["numerical_policy"]),
                "coordinate_names": list(
                    row["final_qualification"]["coordinate_names"]
                ),
                "qualification": deepcopy(row["final_qualification"]),
                "source_receipts": {
                    "row_sha256": public.content_sha256(row),
                    "operations_sha256": public.content_sha256(row["operations"]),
                    "qualification_sha256": row["final_qualification"][
                        "qualification_sha256"
                    ],
                },
                "historical_cost": deepcopy(row["cost"]),
            }
        )
    return {
        "protocol": "llm-sr-detention-staged-polishing-input-"
        + ("2" if source_stage == "staged" else "1"),
        "model": old.model.model_dump(mode="json"),
        "context": old.context.model_dump(mode="json"),
        "training": old.training.model_dump(mode="json"),
        "cases": cases,
        "provenance": {
            "source_plan_sha256": public.content_sha256(plan),
            "source_inputs_sha256": plan["inputs_sha256"],
            "source_report_sha256": report_sha,
            "source_report_file_sha256": file_sha,
            "source_report_exactly_reconstructed": True,
            "extraction_scope": "one training problem and final source tasks 0 and 1",
            "authentication_scope": "content/provenance pins; not a signature",
        },
        "historical_costs": {
            "source_tasks": [
                {"source_index": i, "cost": row["cost"]}
                for i, row in enumerate(saved["tasks"])
            ],
            "earlier_cost_entries": _historical_costs(saved["historical_costs"]),
            "earlier_costs_sha256": public.content_sha256(saved["historical_costs"]),
            "accounting_rule": "separately labeled source scopes; not additive totals",
        },
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def export_inputs(
    source_root: Path, destination: Path, *, source_stage: SourceStage = "polishing"
) -> dict:
    """Create a compact digest-pinned export without numerical work."""
    inputs = Inputs.model_validate(_extract(source_root, source_stage=source_stage))
    seal(destination, inputs.model_dump(mode="json"))
    return {
        "inputs": str(destination),
        "sha256": public.content_sha256(inputs),
        "tasks": 2,
    }


def _tasks(inputs):
    return [
        {
            **{
                k: deepcopy(case[k])
                for k in (
                    "index",
                    "source_index",
                    "start_id",
                    "mode",
                    "coordinate_names",
                    "numerical_policy",
                )
            },
            "kind": "staged_polishing",
            "source_case_sha256": public.content_sha256(case),
            "initial_parameters_sha256": public.content_sha256(case["parameters"]),
            "initial_qualification_sha256": case["qualification"][
                "qualification_sha256"
            ],
        }
        for case in inputs.cases
    ]


def _allowances(task, policy):
    return {
        "search": policy.segment_seconds,
        "verification": policy.verification_seconds,
        **{
            f"{stage}-direction-{i}": policy.direction_seconds
            for stage in ("search", "strict")
            for i in range(len(task["coordinate_names"]))
        },
    }


def _plan(inputs, policy, *, source_sha256, runtime):
    tasks = _tasks(inputs)
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "source_sha256": source_sha256,
        "runtime": runtime,
        "tasks": tasks,
        "policy": policy.model_dump(mode="json"),
        "source_report_sha256": inputs.provenance["source_report_sha256"],
        "operation_allowances": {
            str(t["index"]): _allowances(t, policy) for t in tasks
        },
        "maximum_new_operation_budget_seconds": sum(
            sum(_allowances(t, policy).values()) for t in tasks
        ),
        "evidence_rule": (
            "reuse authenticated initial audit; "
            "full search audit before selective strict audit"
        ),
        "retention_rule": (
            "each start independently; require original-equation verification"
        ),
        **BOUNDARY,
    }


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None):
    """Freeze exactly two starts and their complete nonrenewable operation roster."""
    inputs = Inputs.model_validate(
        inputs.model_dump(mode="json") if isinstance(inputs, Inputs) else inputs
    )
    policy = policy or Policy()
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
            raise ValueError("cannot prepare nonempty unsealed polishing directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 2, "fit_tasks": 2}


def verify(root: Path, *, runtime=True):
    """Authenticate compact inputs and policy without opening the source campaign."""
    plan = read_seal(root / "plan.json")
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    policy = Policy.model_validate(plan["policy"])
    if plan != _plan(
        inputs, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
    ):
        raise ValueError("polishing plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("polishing source/runtime differs")
    return plan, inputs, policy


def load_run(root: Path) -> VerifiedRun:
    """Authenticate once and retain launcher-inclusive invocation timing."""
    began, cpu, utc = monotonic(), process_time(), datetime.now(UTC).isoformat()
    root = root.resolve()
    plan, inputs, policy = verify(root)
    return VerifiedRun(root, plan, inputs, policy, began, cpu, utc, _AUTHENTICATED)


def _operations(directory, identity, policy):
    """Bind every receipt/progress file to its one allowed reservation."""
    allowed, result = _allowances(identity["task"], policy), {}
    if any(p.name not in allowed for p in (directory / "operations").glob("*")):
        raise ValueError("unknown polishing operation")
    for name, seconds in allowed.items():
        folder = directory / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (folder / "result.json").exists():
            result[name] = journal._read_operation(folder, bound)
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != bound:
                raise ValueError("polishing reservation differs")
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
        elif folder.exists() and any(p.name != ".lock" for p in folder.iterdir()):
            raise ValueError("polishing operation lacks reservation")
    return result


def _search_binding(case):
    return {
        "source_parameters_sha256": public.content_sha256(case["parameters"]),
        "source_qualification_sha256": case["qualification"]["qualification_sha256"],
        "mode": case["mode"],
        "segments": 1,
        "protocol": PROTOCOL,
    }


def _audit_binding(parameters, index, *, strict):
    return {
        "parameters_sha256": public.content_sha256(parameters),
        "index": index,
        "search_only": not strict,
        "strict_endpoint_audit": strict,
    }


def _audit_names(task, stage):
    return [f"{stage}-direction-{i}" for i in range(len(task["coordinate_names"]))]


def _consumed(operations, names):
    return all(
        name in operations and operations[name]["status"] != "running_or_interrupted"
        for name in names
    )


def _audit_directions(task, operations, parameters, *, strict):
    """Bind fresh audit decisions to their stage, coordinate, point and policy."""
    directions = []
    for i, name in enumerate(_audit_names(task, "strict" if strict else "search")):
        if name not in operations:
            continue
        saved = saved_fit._checkpoint(operations[name])
        if not saved:
            continue
        if saved.get("binding") != _audit_binding(parameters, i, strict=strict):
            raise ValueError("endpoint audit point or policy differs")
        record = saved.get("direction", {})
        if not (record.get("identity") or record.get("status") == "complete"):
            continue
        if record.get("index") != i:
            raise ValueError("endpoint audit coordinate differs")
        metadata = record.get("probe_recovery", {})
        if (
            metadata.get("protocol") != recovery.PROTOCOL
            or "search_error_limits" not in metadata
        ):
            raise ValueError("endpoint receipt lacks recovery policy provenance")
        limits = metadata["search_error_limits"]
        numerical = _numerical(task)
        if strict and limits is not None:
            raise ValueError("endpoint receipt did not use strict auditing")
        if limits not in (
            None,
            [
                numerical.search_gradient_absolute_error,
                numerical.search_gradient_relative_error,
            ],
        ):
            raise ValueError("endpoint audit derivative thresholds differ")
        directions.append(record)
    return directions


def _state(inputs, task, operations):
    """Reconstruct retention and conditional audit stages from immutable receipts."""
    case = inputs.cases[task["index"]]
    retained, fit, candidate = case["parent_verified_checkpoint"], {}, None
    used, ready = set(), False
    if "search" in operations:
        used.add("search")
        fit = saved_fit._fit_checkpoint(operations["search"])
        if fit and any(fit.get(k) != v for k, v in _search_binding(case).items()):
            raise ValueError("search checkpoint differs from approved initial endpoint")
        best = fit.get("best")
        ready = operations["search"]["status"] != "running_or_interrupted"
        if best:
            if "verification" in operations:
                if not ready:
                    raise ValueError("verification precedes consumed search")
                used.add("verification")
                verification = saved_fit._checkpoint(operations["verification"])
                if verification and verification.get("requested_parameters_sha256") != (
                    public.content_sha256(best["parameters"])
                ):
                    raise ValueError(
                        "verification differs from requested search endpoint"
                    )
                candidate = saved_fit._verified(operations["verification"])
                if candidate and candidate.get("status") != "complete":
                    raise ValueError(
                        "retention requires complete independent verification"
                    )
                if candidate and candidate["parameters"] != best["parameters"]:
                    raise ValueError("verification returned a different endpoint")
                if candidate and numerics.improves(candidate, retained):
                    retained = candidate
            ready = (
                ready
                and "verification" in operations
                and (operations["verification"]["status"] != "running_or_interrupted")
            )
    search_names = _audit_names(task, "search")
    strict_names = _audit_names(task, "strict")
    if set(search_names) & operations.keys() and not ready:
        raise ValueError("endpoint audit precedes consumed search and verification")
    used.update(set(search_names) & operations.keys())
    search_directions = _audit_directions(
        task, operations, retained["parameters"], strict=False
    )
    search_gate = _gate(inputs, task, retained["parameters"], search_directions)
    search_complete = ready and _consumed(operations, search_names)
    decision = (
        kernel.audit_decision(
            *_problem(inputs),
            retained["parameters"],
            qualification=search_gate,
            verification=retained,
            mode=task["mode"],
            policy=_numerical(task),
        )
        if search_complete
        else None
    )
    required = bool(decision and decision["strict_audit_required"])
    if set(strict_names) & operations.keys() and not required:
        raise ValueError("strict audit precedes a qualified escalation decision")
    used.update(set(strict_names) & operations.keys())
    strict_directions = _audit_directions(
        task, operations, retained["parameters"], strict=True
    )
    strict_gate = (
        _gate(inputs, task, retained["parameters"], strict_directions)
        if required
        else None
    )
    if set(operations) - used:
        raise ValueError("operations exist beyond the fixed polishing sequence")
    gate = strict_gate if required else search_gate
    assessment = polishing.assess_endpoint(
        *_problem(inputs),
        retained["parameters"],
        qualification=gate,
        verification=retained,
        mode=task["mode"],
        policy=_numerical(task),
    )
    # A search-purpose stage cannot establish the campaign's strict conclusion.
    if not required:
        assessment = assessment | {"qualified_full_chart_stationarity": False}
        if assessment["status"] == "qualified_local_stationarity":
            assessment["status"] = "unresolved"
            assessment["reason"] = "strict endpoint audit was not executed"
    strict_complete = required and _consumed(operations, strict_names)
    complete = search_complete and (not required or strict_complete)
    strict_reserved = bool(set(strict_names) & operations.keys())
    strict_executed = any(
        operations[name].get("wall_seconds") is not None
        or saved_fit._checkpoint(operations[name]).get("direction", {}).get("identity")
        or saved_fit._checkpoint(operations[name]).get("direction", {}).get("status")
        == "complete"
        for name in strict_names
        if name in operations
    )
    strict_audit = {
        "status": "complete"
        if strict_complete
        else "required"
        if required
        else "skipped"
        if search_complete
        else "pending",
        "performed": True if strict_executed else None if strict_reserved else False,
        "reserved": strict_reserved,
        "required": required if search_complete else None,
        "skipped_reason": decision["strict_audit_reason"]
        if search_complete and not required
        else None,
        "all_coordinate_allowances_consumed": strict_complete,
    }
    return {
        "retained": retained,
        "candidate_verification": candidate,
        "verified_improvement": retained["parameters"] != case["parameters"],
        "optimization": fit or None,
        "initial_qualification_sha256": case["qualification"]["qualification_sha256"],
        "search_qualification": search_gate,
        "search_audit_complete": search_complete,
        "audit_decision": decision,
        "strict_audit": strict_audit,
        "strict_qualification": strict_gate,
        "final_qualification": gate,
        "endpoint_assessment": assessment,
        "full_model_stationarity_qualified": bool(
            strict_complete and assessment["qualified_full_chart_stationarity"]
        ),
        "stop_reason": "single_search_allowance_consumed" if complete else "incomplete",
        "complete": complete,
    }


def _cost(operations, task, policy):
    """Separate measured charges, reservations and prior/reused work."""
    allowed = _allowances(task, policy)
    consumed = sum(allowed[name] for name in operations)
    unexecuted = [name for name in allowed if name not in operations]
    strict_names = _audit_names(task, "strict")
    return accounting._cost(operations) | {
        "reserved_allowance_seconds": consumed,
        "consumed_allowance_seconds": consumed,
        "unexecuted_allowance_seconds": sum(allowed[name] for name in unexecuted),
        "unexecuted_operations": unexecuted,
        "potential_strict_allowance_seconds": sum(allowed[n] for n in strict_names),
        "strict_consumed_allowance_seconds": sum(
            allowed[n] for n in strict_names if n in operations
        ),
        "strict_unexecuted_allowance_seconds": sum(
            allowed[n] for n in strict_names if n not in operations
        ),
        "maximum_new_allowance_seconds": sum(allowed.values()),
        "available_initial_qualification_count": 1,
        "available_initial_verification_count": 1,
        "initial_evidence_authentication_integration_seconds": 0,
    }


def _search_performed(fit):
    """Distinguish numerical evidence from a binding-only interrupted checkpoint."""
    if not fit:
        return False
    return bool(
        fit.get("best")
        or any(
            fit.get(key, 0) > 0
            for key in ("calls", "search_calls_started", "search_calls_attempted")
        )
        or fit.get("numerical_cost", {}).get("residual_calls_started", 0) > 0
        or fit.get("initial_sample_reuse", {}).get("residual_reuse_hits", 0) > 0
    )


def _assemble(identity, operations, inputs, policy, *, terminal=False):
    task = identity["task"]
    state = _state(inputs, task, operations)
    if terminal and not state["complete"]:
        raise ValueError("terminal polishing task has unfinished operations")
    searched, reserved = (
        _search_performed(state["optimization"]),
        "search" in operations,
    )
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        **state,
        "operations": operations,
        "cost": _cost(operations, task, policy),
        "optimization_status": "searched"
        if searched
        else ("search_reserved_execution_unconfirmed" if reserved else "pending"),
        "optimization_performed": True if searched else None if reserved else False,
        "search_reserved": reserved,
        "diagnostic_status": state["endpoint_assessment"]["status"],
        "historical_cost": inputs.cases[task["index"]]["historical_cost"],
        "initial_training_nmse": inputs.cases[task["index"]][
            "parent_verified_checkpoint"
        ]["training_nmse"],
        "initial_sample_reuse": (state["optimization"] or {}).get(
            "initial_sample_reuse", {}
        ),
        "reused_start_qualification": True,
        "strict_endpoint_audit_requested": bool(
            (state["audit_decision"] or {}).get("strict_audit_required")
        ),
        **BOUNDARY,
    }


def _read_task(root, task, plan, inputs, policy):
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(task["index"])
    terminal = (directory / "result.json").exists()
    result = _assemble(
        identity,
        _operations(directory, identity, policy),
        inputs,
        policy,
        terminal=terminal,
    )
    if terminal and read_seal(directory / "result.json") != result:
        raise ValueError("terminal task differs from authenticated receipts")
    return result


def run_task(root: Path, index: int, *, context: VerifiedRun | None = None):
    """Execute once, consume interruptions, and audit the retained endpoint."""
    context = context or load_run(root)
    if (
        not isinstance(context, VerifiedRun)
        or context._authentication is not _AUTHENTICATED
        or context.root != root.resolve()
        or read_seal(root / "plan.json") != context.plan
        or read_seal(root / "inputs.json") != context.inputs.model_dump(mode="json")
        or context.plan["source_sha256"] != public._source_identity()
        or context.plan["runtime"] != public._runtime()
    ):
        raise ValueError("verified invocation differs from polishing campaign")
    if isinstance(index, bool) or not isinstance(index, int) or index not in range(2):
        raise ValueError("unknown polishing task")
    plan, inputs, policy = context.plan, context.inputs, context.policy
    task, case = plan["tasks"][index], inputs.cases[index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)
    with public._lock(directory):
        if (directory / "result.json").exists():
            return _read_task(root, task, plan, inputs, policy)
        operations, allowed = (
            _operations(directory, identity, policy),
            _allowances(task, policy),
        )
        # Fail closed on orphan/stale downstream receipts before executing any work.
        _state(inputs, task, operations)
        invocation = (
            directory
            / "invocations"
            / str(len(list((directory / "invocations").glob("*"))))
        )
        start = {
            "identity": identity,
            "started_at_utc": context.started_utc,
            "authentication_wall_seconds": monotonic() - context.started_wall,
            "authentication_cpu_seconds": process_time() - context.started_cpu,
        }
        seal(invocation / "started.json", start)

        def operate(name, work):
            print(
                f"Staged polishing task {index}: {name} ({allowed[name]:g}s allowance)",
                file=sys.stderr,
                flush=True,
            )

            def durable(save):
                result = work(save)
                save(result)
                return result

            operations[name] = journal._operation(
                directory, identity, name, allowed[name], durable
            )

        completed = False
        try:
            binding = _search_binding(case)
            operate(
                "search",
                lambda save: kernel.fit(
                    *_problem(inputs),
                    case["parameters"],
                    qualification=case["qualification"],
                    mode=task["mode"],
                    seconds=policy.segment_seconds,
                    policy=_numerical(task),
                    checkpoint=lambda value: save(value | binding),
                )
                | binding,
            )
            fit = saved_fit._fit_checkpoint(operations["search"])
            if fit and fit.get("best"):
                parameters = fit["best"]["parameters"]
                binding = {
                    "requested_parameters_sha256": public.content_sha256(parameters)
                }
                operate(
                    "verification",
                    lambda save: original.verify(
                        *_problem(inputs),
                        parameters,
                        seconds=policy.verification_seconds,
                        policy=_numerical(task).numerical,
                        checkpoint=lambda value: save(value | binding),
                    )
                    | binding,
                )
            state = _state(inputs, task, operations)
            parameters = state["retained"]["parameters"]
            seeds = (
                case["qualification"]["directions"]
                if parameters == case["parameters"]
                else []
            )
            cache_root = (
                directory
                / "probe-cache"
                / public.content_sha256(
                    {
                        "parameters": parameters,
                        "mode": task["mode"],
                        "numerical_policy": task["numerical_policy"],
                    }
                )
            )

            def audit(*, strict, seeds):
                """Run each stage once with shared exact-point raw samples."""
                for coordinate, name in enumerate(
                    _audit_names(task, "strict" if strict else "search")
                ):
                    binding = _audit_binding(parameters, coordinate, strict=strict)
                    operate(
                        name,
                        lambda save, coordinate=coordinate, binding=binding: {
                            "binding": binding,
                            "direction": recovery.direction(
                                *_problem(inputs),
                                parameters,
                                coordinate,
                                mode=task["mode"],
                                policy=_numerical(task),
                                seconds=policy.direction_seconds,
                                cache_root=cache_root,
                                seed_directions=seeds,
                                search_only=not strict,
                                checkpoint=lambda value: save(
                                    {"binding": binding, "direction": value}
                                ),
                            ),
                        },
                    )

            audit(strict=False, seeds=seeds)
            state = _state(inputs, task, operations)
            if state["audit_decision"]["strict_audit_required"]:
                audit(strict=True, seeds=state["search_qualification"]["directions"])
            result = _assemble(identity, operations, inputs, policy, terminal=True)
            seal(directory / "result.json", result)
            completed = True
            return result
        finally:
            seal(
                invocation / "result.json",
                {
                    "started_sha256": public.content_sha256(start),
                    "status": "complete" if completed else "interrupted_or_failed",
                    "elapsed_wall_seconds": monotonic() - context.started_wall,
                    "elapsed_cpu_seconds": process_time() - context.started_cpu,
                    "scope": (
                        "authentication/orchestration/operations; "
                        "nested timings not additive"
                    ),
                },
            )


def report(root: Path):
    """Reconstruct two endpoint outcomes using receipts, never integration/search."""
    plan, inputs, policy = verify(root, runtime=False)
    rows = [_read_task(root, task, plan, inputs, policy) for task in plan["tasks"]]
    recorded = sum(row["status"] == "complete" for row in rows)
    return {
        "protocol": PROTOCOL,
        "status": "complete" if recorded == 2 else "incomplete",
        "expected": 2,
        "recorded": recorded,
        "tasks": rows,
        "policy": policy.model_dump(mode="json"),
        "historical_costs": inputs.historical_costs,
        "source_report_sha256": inputs.provenance["source_report_sha256"],
        "maximum_new_operation_budget_seconds": plan[
            "maximum_new_operation_budget_seconds"
        ],
        "comparison_scope": "two fixed starts reported separately; training only",
        "completion_interpretation": (
            "consumed allowances do not imply convergence or confidence"
        ),
        **BOUNDARY,
    }
