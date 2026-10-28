"""Recover saved detention derivative evidence and continue two incumbents.

The portable export is a digest-pinned extraction of an exactly reconstructed
historical report. It contains one public training problem, selected endpoints,
raw derivative evidence and provenance receipts, rather than recursively copying
previous campaigns. Every new operation has one nonrenewable allowance.
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
from autoformalism.fitting import llm_sr_detention_continuation as kernel
from autoformalism.fitting import llm_sr_detention_lazy_controls as controls
from autoformalism.fitting import llm_sr_detention_probe_recovery as recovery
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.fitting.llm_sr_numerical_control import Transcription
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_detention_lazy_controls_campaign as source
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "llm-sr-detention-probe-recovery-1"
EXPECTED_SOURCE_REPORT_SHA256 = (
    "6527f1b9c843009690ca4e323d75407272439370da4a2a45d8e17d73639d2435"
)
EXPECTED_SOURCE_REPORT_FILE_SHA256 = (
    "6717d370ff13ba9e3876dfafc4ce766cf899434fe5d8ce77c32c659e9f03bb03"
)
# Filled from the verified, deterministic export of the approved returned run.
EXPECTED_INPUTS_SHA256 = (
    "d89e1a11cf604b9ce0ab45ff820961dee9ac751c62459742fc8bbc3d686f7cd2"
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
    """Compact exact extraction; the digest pins all problem and evidence fields."""

    protocol: Literal["llm-sr-detention-probe-recovery-input-1"] = (
        "llm-sr-detention-probe-recovery-input-1"
    )
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
        if public.content_sha256(self) != EXPECTED_INPUTS_SHA256:
            raise ValueError("compact detention source extraction differs")
        if self.provenance["source_report_sha256"] != EXPECTED_SOURCE_REPORT_SHA256:
            raise ValueError("detention source report differs")
        if len(self.cases) != 6:
            raise ValueError("detention recovery roster differs")
        for case in self.cases:
            if not numerics.usable(case["parent_verified_checkpoint"]):
                raise ValueError("source checkpoint is not independently verified")
            if case["parameters"] != case["parent_verified_checkpoint"]["parameters"]:
                raise ValueError("source point differs from verified checkpoint")
            # Recompute qualification from raw evidence under the exact policy.
            # This is arithmetic only: no integration or optimization takes place.
            _gate(self, case, case["parameters"], case["seed_directions"])
        return self


class Policy(StrictSchema):
    """Three bounded searches and larger, separately charged derivative checks."""

    protocol: Literal["llm-sr-detention-probe-recovery-policy-1"] = (
        "llm-sr-detention-probe-recovery-policy-1"
    )
    segments: Literal[3] = 3
    segment_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    direction_seconds: FiniteFloat = Field(default=360, ge=240, le=360)
    verification_seconds: FiniteFloat = Field(default=120, gt=0, le=120)


@dataclass(frozen=True)
class VerifiedRun:
    """Authenticated top-level invocation, reusable by launcher and worker."""

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


def _gate(inputs, task, parameters, directions):
    return kernel.qualification(
        *_problem(inputs),
        parameters,
        directions,
        mode=task["mode"],
        policy=kernel.Policy.model_validate(task["numerical_policy"]),
    )


def _extract(source_root: Path):
    """Reconstruct the original report once, before extracting portable evidence."""
    print(
        "Authenticating original source plan, receipts and report...",
        flush=True,
        file=sys.stderr,
    )
    saved_path = source_root / "report/summary.json"
    if (
        sha256(saved_path.read_bytes()).hexdigest()
        != EXPECTED_SOURCE_REPORT_FILE_SHA256
    ):
        raise ValueError("saved source report file differs")
    saved = public._read(saved_path)
    if public.content_sha256(saved) != EXPECTED_SOURCE_REPORT_SHA256:
        raise ValueError("saved source report content differs")
    reconstructed = source.report(source_root)
    if reconstructed != saved:
        raise ValueError("source report does not exactly reconstruct from receipts")
    print(
        "Original report reconstructed exactly; extracting approved endpoints...",
        flush=True,
        file=sys.stderr,
    )
    plan, old, _ = source.verify(source_root, runtime=False)
    problem = source._problem(old, plan["tasks"][7])
    cases = []
    for index, (source_indices, kind) in enumerate(
        [
            ([7], "continuation"),
            ([23], "continuation"),
            ([18, 19], "diagnostic"),
            ([17], "diagnostic"),
            ([1], "diagnostic"),
            ([], "historical_supplement"),
        ]
    ):
        if source_indices:
            row = saved["tasks"][source_indices[0]]
            checkpoint = row.get("retained", row.get("verified_start"))
            gate = row.get("final_qualification", row.get("qualification"))
            numerical = row["numerical_policy"]
            for other in source_indices[1:]:
                sibling = saved["tasks"][other]
                if (
                    sibling["retained"]["parameters"] != checkpoint["parameters"]
                    or sibling["numerical_policy"] != numerical
                    or sibling["final_qualification"]["identity"] != gate["identity"]
                ):
                    raise ValueError("supposed duplicate source endpoints differ")
            costs = [
                {"source_index": i, "cost": saved["tasks"][i]["cost"]}
                for i in source_indices
            ]
            receipts = [
                {
                    "source_index": i,
                    "row_sha256": public.content_sha256(saved["tasks"][i]),
                    "operations_sha256": public.content_sha256(
                        saved["tasks"][i]["operations"]
                    ),
                }
                for i in source_indices
            ]
            start_id = row["identity"]["task"]["start_id"]
        else:
            gate = saved["historical_generic_endpoint_supplement"]
            checkpoint = saved["historical_active_endpoints"][2]["retained"]
            numerical = source._numerical(old).model_dump(mode="json")
            costs = [saved["historical_active_endpoints"][2]]
            receipts = [{"supplement_sha256": public.content_sha256(gate)}]
            start_id = "historical_generic_1"
        cases.append(
            {
                "index": index,
                "kind": kind,
                "start_id": start_id,
                "source_indices": source_indices,
                "mode": gate["mode"],
                "parameters": deepcopy(checkpoint["parameters"]),
                "parent_verified_checkpoint": deepcopy(checkpoint),
                "numerical_policy": numerical,
                "coordinate_names": gate["coordinate_names"],
                "seed_directions": deepcopy(gate["directions"]),
                "source_qualification_sha256": gate["qualification_sha256"],
                "source_receipts": receipts,
                "historical_costs": costs,
            }
        )
    return {
        "protocol": "llm-sr-detention-probe-recovery-input-1",
        "model": problem[0].model_dump(mode="json"),
        "context": problem[1].model_dump(mode="json"),
        "training": problem[2].model_dump(mode="json"),
        "cases": cases,
        "provenance": {
            "source_plan_sha256": public.content_sha256(plan),
            "source_inputs_sha256": plan["inputs_sha256"],
            "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
            "source_report_file_sha256": EXPECTED_SOURCE_REPORT_FILE_SHA256,
            "source_report_exactly_reconstructed": True,
            "extraction_scope": "one training problem and six approved endpoint cases",
            "authentication_scope": (
                "content and provenance pins; not an independent signature"
            ),
        },
        "historical_costs": {
            "source_tasks": [
                {"index": i, "cost": row["cost"]}
                for i, row in enumerate(saved["tasks"])
            ],
            "earlier_campaigns": saved["historical_costs"],
            "predecessors": saved["historical_predecessor_costs"],
        },
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def export_inputs(source_root: Path, destination: Path):
    """Authenticate once and export a compact, immutable train-only payload."""
    inputs = Inputs.model_validate(_extract(source_root))
    seal(destination, inputs.model_dump(mode="json"))
    return {
        "inputs": str(destination),
        "sha256": public.content_sha256(inputs),
        "tasks": len(inputs.cases),
    }


def _tasks(inputs):
    tasks = []
    for case in inputs.cases:
        gate = _gate(inputs, case, case["parameters"], case["seed_directions"])
        passed = [r["index"] for r in gate["rows"] if r["search_qualified"]]
        tasks.append(
            {
                k: deepcopy(v)
                for k, v in case.items()
                if k not in {"seed_directions", "historical_costs", "source_receipts"}
            }
            | {
                "source_case_sha256": public.content_sha256(case),
                "reused_direction_indices": passed,
                "recovery_direction_indices": [
                    i for i in range(len(case["coordinate_names"])) if i not in passed
                ],
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
        "source_report_sha256": inputs.provenance["source_report_sha256"],
        "active_rule": (
            "complete full audit required; hold failed directions, never missing ones"
        ),
        "evidence_rule": (
            "passed historical rows reused; "
            "failed and missing rows explicitly reevaluated"
        ),
        **BOUNDARY,
    }


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None):
    """Freeze a new recovery root, never overwrite the historical campaign."""
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
            raise ValueError("cannot prepare nonempty unsealed recovery directory")
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 6, "fit_tasks": 2}


def verify(root: Path, *, runtime=True):
    """Authenticate compact inputs and current code/runtime without historical IO."""
    plan = read_seal(root / "plan.json")
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    policy = Policy.model_validate(plan["policy"])
    if plan != _plan(
        inputs, policy, source_sha256=plan["source_sha256"], runtime=plan["runtime"]
    ):
        raise ValueError("probe-recovery plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("probe-recovery source/runtime differs")
    return plan, inputs, policy


def load_run(root: Path) -> VerifiedRun:
    """Time authentication once and share the verified context with the launcher."""
    began, cpu, utc = monotonic(), process_time(), datetime.now(UTC).isoformat()
    print("Authenticating compact recovery inputs...", flush=True, file=sys.stderr)
    root = root.resolve()
    plan, inputs, policy = verify(root)
    return VerifiedRun(root, plan, inputs, policy, began, cpu, utc, _AUTHENTICATED)


def _allowances(task, policy):
    result = {
        f"initial-direction-{i}": policy.direction_seconds
        for i in task["recovery_direction_indices"]
    }
    if task["kind"] == "continuation":
        for phase in range(policy.segments):
            result |= {
                f"phase-{phase}-search": policy.segment_seconds,
                f"phase-{phase}-verification": policy.verification_seconds,
            }
            result |= {
                f"phase-{phase}-recheck-{i}": policy.direction_seconds
                for i in range(len(task["coordinate_names"]))
            }
    return result


def _operations(directory, identity, policy):
    """Use the established nonrenewable receipt reader with this roster's caps."""
    allowed = _allowances(identity["task"], policy)
    result = {}
    if any(p.name not in allowed for p in (directory / "operations").glob("*")):
        raise ValueError("unknown probe-recovery operation")
    for name, seconds in allowed.items():
        folder = directory / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (folder / "result.json").exists():
            result[name] = journal._read_operation(folder, bound)
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != bound:
                raise ValueError("probe-recovery reservation differs")
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


def _value(operations, name):
    return source.source._value(operations, name)


def _initial_gate(inputs, task, operations):
    case = inputs.cases[task["index"]]
    passed = set(task["reused_direction_indices"])
    directions = [d for d in case["seed_directions"] if d["index"] in passed]
    directions += [
        v
        for i in task["recovery_direction_indices"]
        if (v := _value(operations, f"initial-direction-{i}")) is not None
    ]
    return _gate(inputs, task, task["parameters"], directions)


def _selection(parameters, gate):
    active = (
        [r["name"] for r in gate["rows"] if r["search_qualified"]]
        if gate["complete"]
        else []
    )
    return {
        "parameters_sha256": public.content_sha256(parameters),
        "qualification_sha256": gate["qualification_sha256"],
        "active_names": active,
        "held_names": [n for n in gate["coordinate_names"] if n not in active],
        "full_coordinate_search_qualified": gate["search_qualified"],
        "full_model_convergence_claimed": False,
    }


def _fit_binding(task, selection):
    return {
        "selection_sha256": public.content_sha256(selection),
        "qualification_sha256": selection["qualification_sha256"],
        "active_names": selection["active_names"],
        "held_names": selection["held_names"],
        "mode": task["mode"],
        "lazy_jacobian": True,
    }


def _state(inputs, task, operations, policy):
    """Replay only valid operations; preserve independently verified incumbents."""
    retained = task["parent_verified_checkpoint"]
    gate = _initial_gate(inputs, task, operations)
    initial = {f"initial-direction-{i}" for i in task["recovery_direction_indices"]}
    used = initial & operations.keys()

    def done(names):
        return all(
            n in operations and operations[n]["status"] != "running_or_interrupted"
            for n in names
        )

    phases, reason = [], "incomplete"
    if done(initial):
        if task["kind"] != "continuation":
            reason = "diagnostic_allowance_completed"
        else:
            for phase in range(policy.segments):
                start = retained
                selection = _selection(start["parameters"], gate)
                row = {
                    "phase": phase,
                    "start": start,
                    "qualification_before": gate,
                    "selection": selection,
                    "complete": False,
                }
                phases.append(row)
                if not selection["active_names"]:
                    reason = "initial_or_endpoint_qualification_blocked"
                    row.update(
                        complete=True, retained=retained, qualification_after=gate
                    )
                    break
                prefix = f"phase-{phase}"
                search = f"{prefix}-search"
                if search not in operations:
                    break
                used.add(search)
                fit = source.source.saved_fit._fit_checkpoint(operations[search])
                if not fit:
                    if done([search]):
                        reason = "search_checkpoint_unavailable"
                    break
                if any(
                    fit.get(k) != v for k, v in _fit_binding(task, selection).items()
                ):
                    raise ValueError(
                        "search checkpoint differs from frozen active selection"
                    )
                row["optimization"] = fit
                best = fit.get("best")
                verification = f"{prefix}-verification"
                if best and verification in operations:
                    used.add(verification)
                candidate = source.source._verified(
                    operations, verification, best["parameters"] if best else None
                )
                if candidate and numerics.improves(candidate, retained):
                    retained = candidate
                changed = retained["parameters"] != start["parameters"]
                ready = done([search]) and (not best or done([verification]))
                if changed:
                    names = [
                        f"{prefix}-recheck-{i}"
                        for i in range(len(task["coordinate_names"]))
                    ]
                    used.update(n for n in names if n in operations)
                    gate = _gate(
                        inputs,
                        task,
                        retained["parameters"],
                        [v for n in names if (v := _value(operations, n)) is not None],
                    )
                    ready = ready and done(names)
                row.update(
                    retained=retained,
                    verified_improvement=changed,
                    qualification_after=gate,
                    complete=ready,
                )
                if not ready:
                    break
                if not changed:
                    reason = "no_verified_progress_same_point"
                    break
                if not gate["complete"]:
                    reason = "full_coordinate_recheck_incomplete"
                    break
                if phase + 1 == policy.segments:
                    reason = "search_allowance_exhausted"
    if set(operations) - used:
        raise ValueError("operations exist beyond the frozen stopping rule")
    return {
        "retained": retained,
        "initial_qualification": _initial_gate(inputs, task, operations),
        "final_qualification": gate,
        "phases": phases,
        "stop_reason": reason,
    }


def _cost(operations):
    """Missing measured time remains unknown; allowances are separately charged."""

    def measured(field):
        values = [o.get(field) for o in operations.values()]
        return sum(values) if all(v is not None for v in values) else None

    return {
        "new_operation_count": len(operations),
        "wall_seconds": measured("wall_seconds"),
        "cpu_seconds": measured("cpu_seconds"),
        "budget_charge_seconds": sum(
            o["budget_charge_seconds"] for o in operations.values()
        ),
        "accounting_complete": all(
            o.get("accounting_complete", False) for o in operations.values()
        ),
        "callback_wall_seconds": None,
        "numerical_active_wall_seconds": None,
        "subcategory_timing_availability": (
            "unavailable; do not infer from wall minus CPU"
        ),
        "scope": (
            "new operations only; old measurements and reused evidence are separate"
        ),
    }


def _assemble(identity, operations, inputs, policy, *, terminal=False):
    task = identity["task"]
    state = _state(inputs, task, operations, policy)
    if terminal and state["stop_reason"] == "incomplete":
        raise ValueError("terminal task has unfinished operations")
    gate = state["final_qualification"]
    diagnostic = (
        "incomplete"
        if not gate["complete"]
        else "failed"
        if gate["failed_indices"]
        else "passed_for_search"
        if gate["search_qualified"]
        else "search_precision_unqualified"
    )
    directions = gate["directions"]
    if not gate["complete"] and any(
        d.get("budget_exhausted") or d.get("diagnostic_outcome") == "timed_out"
        for d in directions
    ):
        diagnostic = "timed_out"
    searches = {n: o for n, o in operations.items() if n.endswith("-search")}
    searched = any(
        source.source.saved_fit._fit_checkpoint(o) for o in searches.values()
    )
    performed = True if searched else None if searches else False
    optimization = (
        "not_requested"
        if task["kind"] != "continuation"
        else "searched"
        if searched
        else "search_reserved_execution_unconfirmed"
        if searches
        else "blocked_before_search"
        if terminal
        else "pending"
    )
    final_selection = _selection(state["retained"]["parameters"], gate)
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        **state,
        "active_names": final_selection["active_names"],
        "held_names": final_selection["held_names"],
        "phases_completed": sum(p["complete"] for p in state["phases"]),
        "operations": operations,
        "cost": _cost(operations),
        "optimization_status": optimization,
        "optimization_performed": performed,
        "search_reserved": bool(searches),
        "search_work_evidence": (
            "saved numerical checkpoint"
            if searched
            else "execution unconfirmed"
            if searches
            else "no search reserved"
        ),
        "operation_status_counts": {
            status: sum(o["status"] == status for o in operations.values())
            for status in sorted({o["status"] for o in operations.values()})
        },
        "diagnostic_status": diagnostic,
        "historical_costs": inputs.cases[task["index"]]["historical_costs"],
        "reused_direction_indices": task["reused_direction_indices"],
        "explicitly_reevaluated_direction_indices": task["recovery_direction_indices"],
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
    """Run one frozen case with durable samples and nonrenewable phase budgets."""
    context = context or load_run(root)
    if (
        not isinstance(context, VerifiedRun)
        or context._authentication is not _AUTHENTICATED
        or context.root != root.resolve()
        or read_seal(root / "plan.json") != context.plan
        or context.plan["source_sha256"] != public._source_identity()
        or context.plan["runtime"] != public._runtime()
    ):
        raise ValueError("verified invocation differs from recovery campaign")
    if isinstance(index, bool) or not isinstance(index, int) or index not in range(6):
        raise ValueError("unknown recovery task")
    plan, inputs, policy = context.plan, context.inputs, context.policy
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "tasks" / str(index)
    with public._lock(directory):
        if (directory / "result.json").exists():
            return _read_task(root, task, plan, inputs, policy)
        operations = _operations(directory, identity, policy)
        allowed = _allowances(task, policy)
        problem = _problem(inputs)
        numerical = kernel.Policy.model_validate(task["numerical_policy"])
        case = inputs.cases[index]
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
                f"Recovery task {index}: {name} ({allowed[name]:g}s allowance)",
                flush=True,
                file=sys.stderr,
            )

            def durable(save):
                result = work(save)
                save(result)
                return result

            receipt = journal._operation(
                directory, identity, name, allowed[name], durable
            )
            operations[name] = receipt
            return receipt

        def direction(name, parameters, coordinate, seeds=()):
            cache_identity = {
                "parameters": parameters,
                "mode": task["mode"],
                "numerical_policy": task["numerical_policy"],
            }
            cache_root = (
                directory / "probe-cache" / public.content_sha256(cache_identity)
            )
            operate(
                name,
                lambda save: recovery.direction(
                    *problem,
                    parameters,
                    coordinate,
                    mode=task["mode"],
                    policy=numerical,
                    seconds=allowed[name],
                    cache_root=cache_root,
                    seed_directions=seeds,
                    checkpoint=save,
                ),
            )

        def verify_parameters(name, parameters):
            binding = {"requested_parameters_sha256": public.content_sha256(parameters)}
            operate(
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
            for i in task["recovery_direction_indices"]:
                direction(
                    f"initial-direction-{i}",
                    task["parameters"],
                    i,
                    case["seed_directions"],
                )
            if task["kind"] == "continuation":
                for phase in range(policy.segments):
                    state = _state(inputs, task, operations, policy)
                    if state["stop_reason"] != "incomplete":
                        break
                    row = next(
                        (r for r in state["phases"] if r["phase"] == phase), None
                    )
                    if row and row["complete"]:
                        continue
                    if row is None:
                        raise ValueError("missing continuation phase")
                    point, gate, selection = (
                        row["start"]["parameters"],
                        row["qualification_before"],
                        row["selection"],
                    )
                    binding, prefix = _fit_binding(task, selection), f"phase-{phase}"
                    operate(
                        f"{prefix}-search",
                        lambda save,
                        point=point,
                        gate=gate,
                        selection=selection,
                        binding=binding: controls.fit_control(
                            *problem,
                            point,
                            mode=task["mode"],
                            lazy_jacobian=True,
                            active_names=selection["active_names"],
                            qualification=gate,
                            seconds=policy.segment_seconds,
                            policy=numerical,
                            checkpoint=lambda value: save(value | binding),
                        )
                        | binding,
                    )
                    fit = source.source.saved_fit._fit_checkpoint(
                        operations[f"{prefix}-search"]
                    )
                    if fit and fit.get("best"):
                        verify_parameters(
                            f"{prefix}-verification", fit["best"]["parameters"]
                        )
                    state = _state(inputs, task, operations, policy)
                    if state["retained"]["parameters"] != point:
                        for i in range(len(task["coordinate_names"])):
                            direction(
                                f"{prefix}-recheck-{i}",
                                state["retained"]["parameters"],
                                i,
                            )
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
                        "authentication, orchestration and operations; "
                        "nested timings are not additive"
                    ),
                },
            )


def report(root: Path):
    """Reconstruct six results using receipts only, without numerical execution."""
    print(
        "Authenticating compact inputs and reconstructing recovery receipts...",
        flush=True,
        file=sys.stderr,
    )
    plan, inputs, policy = verify(root, runtime=False)
    rows = [_read_task(root, t, plan, inputs, policy) for t in plan["tasks"]]
    recorded = sum(r["status"] == "complete" for r in rows)
    return {
        "protocol": PROTOCOL,
        "status": "complete" if recorded == 6 else "incomplete",
        "expected": 6,
        "recorded": recorded,
        "tasks": rows,
        "optimization_status_counts": {
            s: sum(r["optimization_status"] == s for r in rows)
            for s in sorted({r["optimization_status"] for r in rows})
        },
        "diagnostic_status_counts": {
            s: sum(r["diagnostic_status"] == s for r in rows)
            for s in sorted({r["diagnostic_status"] for r in rows})
        },
        "policy": policy.model_dump(mode="json"),
        "historical_costs": inputs.historical_costs,
        "source_report_sha256": inputs.provenance["source_report_sha256"],
        "comparison_scope": (
            "targeted continuation and diagnostic recovery; "
            "not a matched method comparison"
        ),
        "completion_interpretation": (
            "terminal receipts do not imply search, convergence or confidence"
        ),
        **BOUNDARY,
    }
