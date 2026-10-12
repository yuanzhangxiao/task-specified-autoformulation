"""M29: frozen public constructor handoffs plus known-structure fitting controls."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import verified_replay
from autoformalism.fitting.transcription_campaign import coefficient_metrics
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.rebuttal.review_deadline_pipeline import request_for
from autoformalism.research.construction_baseline import read_outcome
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-public-fitting-integration-1"
INPUT_PROTOCOL = "phase-c-public-fitting-integration-inputs-1"
CONSTRUCTED = tuple(f"cell{i:02d}_seed0_full" for i in range(8))
CONTROLS = ("alien_hard", "basin_coupled", "cstr_hard")


class Policy(StrictSchema):
    """One frozen guarded search per handoff; post-fit evaluator is separate."""

    protocol: Literal["phase-c-public-fitting-integration-1"] = PROTOCOL
    fitting: bridge.Policy = Field(
        default_factory=lambda: bridge.Policy(allow_piecewise=True)
    )
    replay: verified_replay.ReplayPolicy = Field(
        default_factory=verified_replay.ReplayPolicy
    )


def export_inputs(construction: Path, m28_inputs: Path, output: Path) -> dict:
    """Take Full/seed 0 by cell identity, never by construction or fitting score."""
    plan = sealed_read(construction / "plan.json")
    if (
        plan["protocol"] != "phase-c-construction-baseline-1"
        or plan["test_data_opened"]
    ):
        raise ValueError("expected the frozen public Phase C construction pilot")
    tasks = {t["task_id"]: t for t in plan["tasks"]}
    records = {}
    for name in CONSTRUCTED:
        task = tasks[name]
        if task["arm"] != "full" or task["seed"] != 0:
            raise ValueError("construction task identity differs")
        cell = plan["cells"][task["benchmark_id"]]
        proposal = read_outcome(
            construction / "results" / name / "proposal.json", plan, task
        )
        request = None
        if proposal and proposal["status"] == "constructed":
            request = request_for(proposal["bundle"], plan, task, 0).model_dump(
                mode="json"
            )
            path = construction / "results" / name / "fit/freeze.json"
            if path.exists():
                frozen = public._read(path)
                digest = public.content_sha256(
                    {k: v for k, v in frozen.items() if k != "identity"}
                )
                if digest != frozen["identity"] or request != frozen["request"]:
                    raise ValueError("archived public request identity differs")
        records[name] = {
            "kind": "constructed",
            "case": task["benchmark_id"],
            "source_status": proposal["status"] if proposal else "missing",
            "source_proposal_sha256": proposal.get("artifact_sha256")
            if proposal
            else None,
            "request": request,
            "training": cell["training"],
            "validation": cell["validation"],
            "public_specification_sha256": public.content_sha256(
                cell["public_specification"]
            ),
        }
    old = read_seal(m28_inputs)
    if old["protocol"] != "phase-c-rollout-adapter-inputs-1" or old["test_data_opened"]:
        raise ValueError("expected frozen M28 development inputs")
    for name in CONTROLS:
        case = old["development"]["cases"][name]
        request = old["development"]["commons"][name + "_s0"]["request"]
        records["control_" + name] = {
            "kind": "known_structure",
            "case": name,
            "source_status": "available",
            "request": request,
            "training": case["training"],
            "validation": case["validation"],
            "reference_parameters": case["reference_parameters"],
        }
    data = {
        "protocol": INPUT_PROTOCOL,
        "records": records,
        "construction_plan_sha256": plan["artifact_sha256"],
        "m28_inputs_sha256": public.content_sha256(old),
        "selection_rule": "all_eight_cells_full_seed0_plus_three_known_controls_seed0",
        "historical_fits_used": False,
        "test_data_opened": False,
    }
    tasks = roster(data)
    seal(output, data)
    return {
        "inputs_sha256": public.content_sha256(data),
        "tasks": len(tasks),
        "construction_status_counts": dict(
            Counter(records[n]["source_status"] for n in CONSTRUCTED)
        ),
    }


def roster(data: dict) -> list[dict]:
    """Keep missing/failed constructions in the fixed denominator."""
    expected = (*CONSTRUCTED, *("control_" + n for n in CONTROLS))
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or data["historical_fits_used"] is not False
        or set(data["records"]) != set(expected)
    ):
        raise ValueError("public integration roster differs")
    tasks = []
    for name in expected:
        record = data["records"][name]
        kind = "constructed" if name in CONSTRUCTED else "known_structure"
        if record["kind"] != kind:
            raise ValueError("integration cohort differs")
        available = record["request"] is not None
        if available != (record["source_status"] in {"constructed", "available"}):
            raise ValueError("source status/request coverage differs")
        if kind == "known_structure" and not available:
            raise ValueError("known control request missing")
        if available:
            request = PublicFitRequest.model_validate(record["request"])
            train, val = [
                PublicSplit.model_validate(record[n])
                for n in ("training", "validation")
            ]
            public._check_data(request, train, val)
        tasks.append({"task_id": name, "kind": kind, "case": record["case"]})
    return tasks


def prepare(root: Path, inputs: Path, policy: Policy) -> dict:
    """Freeze the original requests and public arrays without benchmark generation."""
    data = read_seal(inputs)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(data),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime=True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster(data)
        or plan["inputs_sha256"] != public.content_sha256(data)
    ):
        raise ValueError("integration plan/input identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("integration source/runtime differs")
    return plan, data


def evaluate(record, parameters, policy, folder) -> dict:
    """Validation and known reference coefficients enter only after fit sealing."""
    if parameters is None:
        return {
            "status": "no_verified_parameters",
            "solver_agreement_passed": False,
            "accounting_complete": True,
            "wall_seconds": 0,
            "budget_charge_seconds": 0,
        }
    result = verified_replay.run(
        PublicFitRequest.model_validate(record["request"]),
        parameters,
        PublicSplit.model_validate(record["training"]),
        PublicSplit.model_validate(record["validation"]),
        policy,
        folder,
    )
    per_target = {}
    if result["solver_agreement_passed"]:
        rows = result["attempts"][-1]["value"]["rows"]
        for split in ("train", "val"):
            selected = [r for r in rows if r["split"] == split]
            per_target[split] = {
                c: sum(r["samples"] * r["nmse"][c] for r in selected)
                / sum(r["samples"] for r in selected)
                for c in record["request"]["context"]["targets"]
            }
    result["per_target_nmse"] = per_target
    if record["kind"] == "known_structure":
        result["coefficient_recovery"] = coefficient_metrics(
            record, record["request"], parameters
        )
    return result


def completed(folder: Path, identity: dict) -> dict:
    """No receipt may silently substitute another request or fitted endpoint."""
    row = read_seal(folder / "result.json")
    if row["identity"] != identity:
        raise ValueError("integration task identity differs")
    if (
        row["fit"] is not None
        and bridge.completed(folder / "fit").model_dump(mode="json") != row["fit"]
    ):
        raise ValueError("integration fitted handoff differs")
    if row["evaluation"] != read_seal(folder / "evaluation.json"):
        raise ValueError("integration evaluation differs")
    return row


def run_task(root: Path, index: int) -> dict:
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown integration task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(folder, identity)
        record, policy = (
            data["records"][task["task_id"]],
            Policy.model_validate(plan["policy"]),
        )
        fitted, parameters = None, None
        if record["request"] is not None:
            fitted = bridge.run(
                bridge.Inputs(request=record["request"], training=record["training"]),
                policy.fitting,
                folder / "fit",
            )
            parameters = fitted.fit.parameters if fitted.fit else None
        if not (folder / "evaluation.json").exists():
            seal(
                folder / "evaluation.json",
                evaluate(record, parameters, policy.replay, folder / "evaluation"),
            )
        result = task | {
            "identity": identity,
            "status": fitted.status if fitted else "source_unavailable",
            "source_status": record["source_status"],
            "fit": fitted.model_dump(mode="json") if fitted else None,
            "evaluation": read_seal(folder / "evaluation.json"),
        }
        seal(folder / "result.json", result)
        return result


def costs(row: dict) -> dict:
    """Setup, search and evaluator time are all charged, including failures."""
    outer = row.get("fit") or {}
    parts = [
        outer.get("preparation") or {},
        (outer.get("fit") or {}).get("cost") or {},
        row.get("evaluation") or {},
    ]
    return {
        "wall_seconds": sum(p.get("wall_seconds") or 0 for p in parts),
        "budget_charge_seconds": sum(p.get("budget_charge_seconds", 0) for p in parts),
        "accounting_complete": row["status"] != "missing"
        and all(p.get("accounting_complete", True) for p in parts),
    }


def report(root: Path) -> dict:
    plan, _ = verify(root, runtime=False)
    rows = [
        completed(
            root / "results" / t["task_id"],
            {"plan_sha256": public.content_sha256(plan), "task": t},
        )
        if (root / "results" / t["task_id"] / "result.json").exists()
        else t | {"status": "missing"}
        for t in plan["tasks"]
    ]
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "incomplete"
        if any(r["status"] == "missing" for r in rows)
        else "complete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "costs": {r["task_id"]: costs(r) for r in rows},
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Archived construction handoffs, not the latest constructor "
        "or a paired legacy fitter comparison. Missing sources remain in the roster. "
        "Poor fit does not prove structural infeasibility. Known coefficient errors "
        "are evaluator-only and absent for discovered structures. Piecewise "
        "sensitivities are branch-local; uncertainty is not calibrated confidence.",
    }
    public._write(root / "summary.json", result)
    return result
