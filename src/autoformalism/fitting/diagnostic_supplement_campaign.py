"""M33: portable M32 audit and a separately budgeted CSTR diagnostic supplement."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import feedback_supplement as supplement
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import staged_feedback_campaign as previous
from autoformalism.fitting.feedback_supplement import Policy as SupplementPolicy
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-diagnostic-supplement-1"
TARGET = "archive__control_cstr_hard"


class Policy(StrictSchema):
    """Freeze one target, its incremental budget and the unchanged M32 roster."""

    protocol: Literal["phase-c-diagnostic-supplement-1"] = PROTOCOL
    target: Literal["archive__control_cstr_hard"] = TARGET
    supplement: SupplementPolicy = Field(default_factory=SupplementPolicy)


def _files(source: Path) -> dict[str, str]:
    """Inventory only M32 experiment JSON, excluding logs and scheduler artifacts."""
    paths = [source / "plan.json", source / "inputs.json"]
    paths.extend(sorted((source / "results").rglob("*.json")))
    result = {}
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(source.resolve())
        ):
            raise ValueError("source snapshot must contain regular, contained files")
        result[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def audit(source: Path) -> dict:
    """Reassemble all saved M32 rows without integrating, fitting or rewriting them."""
    plan, data = previous.verify(source, runtime=False)
    expected = previous.ARCHIVED + tuple(
        "fixture__" + item["case_id"] for item in previous.fixtures.roster()
    )
    if (
        data["fixtures"] != previous.fixtures.roster()
        or tuple(item["task_id"] for item in plan["tasks"]) != expected
    ):
        raise ValueError("M33 requires the exact 23-outcome M32 roster")
    records = {item["task_id"]: item for item in data["records"]}
    rows = []
    for task in plan["tasks"]:
        folder = source / "results" / task["task_id"]
        if not (folder / "result.json").exists():
            raise ValueError(
                "M33 requires a fully recorded M32 source, including failures"
            )
        rows.append(
            previous.completed(
                folder,
                {"plan_sha256": public.content_sha256(plan), "task": task},
                records.get(task["task_id"]),
            )
        )
    qualification = previous._qualification(rows)
    if qualification["qualification_status"] != "passed":
        raise ValueError("source M32 qualification does not pass portable verification")
    return {
        "status": "passed",
        "expected": len(rows),
        "verified": len(rows),
        "source_plan_sha256": public.content_sha256(plan),
        "qualification_gates": qualification["qualification_gates"],
        "rows": [
            {
                "task_id": row["task_id"],
                "status": row["status"],
                "result_sha256": public.content_sha256(row),
                "diagnosis": (row.get("feedback") or {}).get("diagnosis"),
            }
            for row in rows
        ],
        "numerical_calls": 0,
        "fitting_performed": False,
    }


def prepare(root: Path, inputs: Path, policy: Policy) -> dict:
    """Copy a verified immutable M32 snapshot; never edit its original campaign."""
    root, inputs = root.resolve(), inputs.resolve()
    if root.is_relative_to(inputs) or inputs.is_relative_to(root):
        raise ValueError("supplement campaign and M32 source must be separate")
    manifest = _files(inputs)
    with public._lock(root):
        if (root / "plan.json").exists():
            plan, data = verify(root)
            if plan["policy"] != policy.model_dump(mode="json") or data != manifest:
                raise ValueError("supplement campaign source/policy differs")
            return {"identity": public.content_sha256(plan), "tasks": 1}
        source_audit = audit(inputs)
        snapshot = root / "source_snapshot"
        for name, digest in manifest.items():
            payload = (inputs / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("M32 source changed during snapshot")
            output = snapshot / name
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and output.read_bytes() != payload:
                raise ValueError("partial source snapshot differs")
            if not output.exists():
                temporary = output.with_suffix(".copying")
                temporary.write_bytes(payload)
                temporary.replace(output)
        if _files(inputs) != manifest or _files(snapshot) != manifest:
            raise ValueError("M32 source/snapshot changed during preparation")
        plan = {
            "protocol": PROTOCOL,
            "policy": policy.model_dump(mode="json"),
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "inputs_sha256": public.content_sha256(manifest),
            "audit_sha256": public.content_sha256(source_audit),
            "tasks": [{"task_id": policy.target}],
            "test_data_opened": False,
            "live_llm_calls": 0,
            "fitting_performed": False,
        }
        seal(root / "inputs.json", manifest)
        seal(root / "portable_audit.json", source_audit)
        seal(root / "plan.json", plan)
    return {
        "identity": public.content_sha256(plan),
        "tasks": 1,
        "portable_audit": source_audit["status"],
        "verified": source_audit["verified"],
    }


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Bind every copied byte; report reconstruction may use another platform."""
    plan, manifest = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != [{"task_id": policy.target}]
        or plan["inputs_sha256"] != public.content_sha256(manifest)
        or _files(root / "source_snapshot") != manifest
        or plan["audit_sha256"]
        != public.content_sha256(read_seal(root / "portable_audit.json"))
    ):
        raise ValueError("supplement campaign snapshot/identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("supplement campaign source/runtime differs")
    return plan, manifest


def run_task(root: Path, index: int) -> dict:
    """Complete only missing probes of the saved CSTR vector, under a new budget."""
    plan, _ = verify(root)
    if index != 0:
        raise ValueError("unknown supplement task index")
    policy = Policy.model_validate(plan["policy"])
    source = root / "source_snapshot/results" / policy.target / "diagnostics"
    folder = root / "results" / policy.target
    return supplement.run(source, folder, policy.supplement).model_dump(mode="json")


def report(root: Path) -> dict:
    """Separate the portability gate, completed work and scientific diagnosis."""
    plan, _ = verify(root, runtime=False)
    source_audit = audit(root / "source_snapshot")
    if source_audit != read_seal(root / "portable_audit.json"):
        raise ValueError("portable audit changed")
    policy = Policy.model_validate(plan["policy"])
    folder = root / "results" / policy.target
    result = supplement.completed(folder) if (folder / "result.json").exists() else None
    if result:
        subplan = supplement.SupplementPlan.model_validate(
            read_seal(folder / "supplement.json")
        )
        source = root / "source_snapshot/results" / policy.target / "diagnostics"
        if (folder / subplan.source_path).resolve() != source.resolve() or (
            subplan.runtime["packages"] != plan["runtime"]
        ):
            raise ValueError("supplement predecessor/runtime differ from campaign")
        original = read_seal(
            root / "source_snapshot/results" / policy.target / "feedback.json"
        )
        if (
            result.identity.policy_sha256
            != public.content_sha256(policy.supplement.model_dump(mode="json"))
            or result.original_packet.model_dump(mode="json") != original
            or result.identity.source_sha256 != plan["source_sha256"]
        ):
            raise ValueError("supplement inputs/policy/source differ from campaign")
    packet = result.packet if result else None
    assessment = packet.diagnostics.piecewise_assessment if packet else None
    gates = {
        "portable_audit_passed": source_audit["status"] == "passed",
        "supplement_recorded": result is not None,
        "original_endpoint_unchanged": not result.parameters_changed
        if result
        else None,
        "output_accuracy_preserved": packet.diagnostics.output_status == "verified"
        and packet.diagnosis.startswith("accurate_")
        if packet
        else None,
        "cost_accounting_complete": result.combined_cost.accounting_complete
        if result
        else None,
    }
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if result else "incomplete",
        "qualification_status": "passed"
        if all(gates.values())
        else "pending"
        if result is None
        else "failed",
        "qualification_gates": gates,
        "portable_audit": source_audit,
        "target": policy.target,
        "original_diagnosis": result.original_packet.diagnosis if result else None,
        "diagnosis": packet.diagnosis if packet else None,
        "piecewise_status": assessment.status if assessment else None,
        "required_calls": result.required_calls if result else None,
        "new_calls_started": result.new_calls_started if result else None,
        "new_calls_completed": result.new_calls_completed if result else None,
        "remaining_keys": result.remaining_keys if result else None,
        "original_cost": result.original_cost.model_dump(mode="json")
        if result
        else None,
        "new_cost": result.new_cost.model_dump(mode="json") if result else None,
        "combined_cost": result.combined_cost.model_dump(mode="json")
        if result
        else None,
        "fitting_performed": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Same fitted vector, data, perturbations and thresholds. "
        "A recorded supplement may remain numerically unresolved. Local sensitivity "
        "support does not establish global identifiability or scientific correctness.",
    }
    public._write(root / "summary.json", summary)
    return summary
