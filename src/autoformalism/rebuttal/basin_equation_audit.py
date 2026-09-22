"""Immutable, public-only saved-v7 physics audit and prospective feedback packets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.basin_equation_checks import POLICY, assess
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_gain_comparison import compile_bundle
from autoformalism.rebuttal.review_deadline_pipeline import _bundle
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.staged_topology import content_hash

PROTOCOL = "basin-saved-equation-audit-1"
SURVEY_NAMES = ("area_up", "area_down", "crest_up", "crest_down", "warning_depth")
REPO = Path(__file__).resolve().parents[3]


def _read(source: Path, relative: str) -> dict | None:
    path = source / relative
    if not path.resolve().is_relative_to(source):
        raise ValueError("source artifact escapes campaign")
    return sealed_read(path) if path.exists() else None


def _paths(plan: dict) -> list[str]:
    paths = {"plan.json"}
    seen = set()
    for task in plan["tasks"]:
        name, common = task["task_id"], task["construction_task"]["task_id"]
        if any(not re.fullmatch(r"[A-Za-z0-9_]+", n) for n in (name, common)):
            raise ValueError("invalid source task path")
        if name in seen or task["case"] not in {"coupled", "independent"}:
            raise ValueError("duplicate task or unknown basin contract")
        seen.add(name)
        paths.update(f"results/{name}/{f}.json" for f in ("proposal", "result"))
        paths.add(f"construction/results/{common}/proposal.json")
        paths.update(
            f"construction/results/{common}/{route}/{stage}_stage.json"
            for route in ("review", "fallback")
            for stage in ("topology", "function")
        )
    return sorted(paths)


def _snapshot(source: Path, paths: list[str]) -> tuple[dict, dict]:
    """Include absence in the frozen inventory; a later result needs a fresh audit."""
    values, hashes = {}, {}
    for relative in paths:
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("source artifact escapes campaign")
        raw = path.read_bytes() if path.exists() else None
        value = json.loads(raw) if raw is not None else None
        if value is not None and value.get("artifact_sha256") != content_hash(
            {k: v for k, v in value.items() if k != "artifact_sha256"}
        ):
            raise ValueError("source artifact digest differs")
        values[relative] = value
        hashes[relative] = hashlib.sha256(raw).hexdigest() if raw is not None else None
    return values, hashes


def _surveys(cell: dict) -> list[dict]:
    """Use training geometry only, without input/output signals or validation."""
    surveys = {}
    for row in cell["training"]["rows"]:
        geometry = {n: row["fixed_covariates"][n] for n in SURVEY_NAMES}
        surveys[content_hash(geometry)] = geometry
    return [surveys[k] for k in sorted(surveys)]


def _verify_common(task, cell, common, records):
    """Reconstruct the selected construction using its saved decision ledger."""
    name = task["construction_task"]["task_id"]
    if common["task"] != task["construction_task"]:
        raise ValueError("common construction identity differs")
    route = common["attempts"][-1]["route"]
    if route not in {"review", "fallback"}:
        raise ValueError("unknown saved route")
    base = f"construction/results/{name}/{route}"
    topology, functions = (
        records[f"{base}/{s}_stage.json"] for s in ("topology", "function")
    )
    if topology is None or functions is None:
        return False
    rebuilt = _bundle(
        cell,
        common["task"],
        PublicScientificBrief.model_validate(cell["brief"]),
        topology["result"],
        functions["result"],
    )
    if rebuilt != common["bundle"]:
        raise ValueError(
            "saved common model differs from independently reconstructed ledger"
        )
    return True


def _row(task: dict, cell: dict, records: dict, verified: set) -> dict:
    name, common_name = task["task_id"], task["construction_task"]["task_id"]
    proposal = records[f"results/{name}/proposal.json"]
    result = records[f"results/{name}/result.json"]
    common = records[f"construction/results/{common_name}/proposal.json"]
    row = {
        "task": name,
        "construction": common_name,
        "case": task["case"],
        "gain_policy": task["gain_policy"],
        "status": "unavailable",
        "assessment": None,
        "public_specification": cell["brief"]["scientific_context"],
    }
    if not proposal or not common:
        return {**row, "reason": "saved proposal or common construction missing"}
    if (
        proposal["task"] != task
        or proposal["common_proposal_sha256"] != common["artifact_sha256"]
    ):
        raise ValueError("saved proposal identity differs")
    if result and (
        result["task"] != task
        or result["proposal_sha256"] != proposal["artifact_sha256"]
    ):
        raise ValueError("saved fitted result belongs to another model")
    row.update(
        source_proposal_sha256=proposal["artifact_sha256"],
        source_result_sha256=result["artifact_sha256"] if result else None,
        fallback_used=common.get("fallback_used"),
        construction_errors=[
            {"route": a["route"], "error": a.get("error")}
            for a in common.get("attempts", [])
            if a.get("error")
        ],
    )
    bundle = proposal.get("bundle")
    if not bundle or not common.get("bundle"):
        return {
            **row,
            "reason": "no assembled model",
            "historical_status": proposal["status"],
        }
    if common_name not in verified:
        if not _verify_common(task, cell, common, records):
            return {**row, "reason": "selected construction ledger missing"}
        verified.add(common_name)
    # Gain lowering reads only training fixed covariates, never the signal arrays.
    geometry_only = {
        "rows": [
            {"fixed_covariates": r["fixed_covariates"]}
            for r in cell["training"]["rows"]
        ]
    }
    rebuilt = compile_bundle(
        common["bundle"],
        common.get("shared_process_contract"),
        task["gain_policy"],
        geometry_only,
    )
    if rebuilt != bundle:
        raise ValueError("saved gain assembly differs from common construction")
    fit = (result or {}).get("fit") or {}
    parameters = fit.get("parameters")
    if parameters is not None:
        parameters = dict(parameters)
    candidate = CandidateModel.model_validate(bundle["candidate"])
    try:
        assessment = assess(
            candidate,
            ValidationContext.model_validate(bundle["initialization"]["context"]),
            task["case"],
            _surveys(cell),
            parameters,
        )
    except (ValueError, ModelValidationError) as exc:
        return {**row, "status": "invalid_or_unsupported", "reason": str(exc)}
    return {
        **row,
        "status": "assessed",
        "assessment": assessment,
        "saved_fit_status": (result or {}).get("status", "unavailable"),
        "parameter_sha256": content_hash(parameters)
        if parameters is not None
        else None,
        "equations": {
            "states": [e.model_dump(mode="json") for e in candidate.state_equations],
            "processes": [
                p.model_dump(include={"name", "expression"})
                for p in candidate.processes
            ],
        },
    }


def feedback(row: dict) -> dict:
    """A future repair/critic input, without fit scores, observed series or verdicts."""
    assessment = row.get("assessment") or {}
    return {
        "protocol": "basin-equation-feedback-1",
        "task": row["task"],
        "case": row["case"],
        "public_specification": row["public_specification"],
        "candidate_sha256": assessment.get("candidate_sha256"),
        "source_proposal_sha256": row.get("source_proposal_sha256"),
        "parameter_sha256": row.get("parameter_sha256"),
        "equations": row.get("equations"),
        "assumptions": assessment.get("assumptions", []),
        "findings": assessment.get("checks", []),
        "data_boundary": "public_training_only",
        "instructions": "These are scoped deterministic observations, not scientific "
        "certification. The proposer owns scientific changes; no automatic sign, "
        "conversion or topology rewrite is authorized by this packet. Unverified "
        "checks are not failed requirements. Passing probes is not a global proof. "
        "Failure at saved parameters does not prove that the skeleton "
        "cannot be fitted.",
        "critic_called": False,
        "automatic_rejection": False,
        "model_promoted": False,
    }


def _reports(output: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# Saved basin equation audit",
        "",
        "Public equations and training survey geometry only. No fitting, solver "
        "rollouts, LLM calls, model changes or promotion. Sixteen gain arms are not "
        "sixteen independent constructions. Pass/fail refers to each scoped check, "
        "not overall mechanism compliance.",
        "",
        f"Row status: {summary['status_counts']}",
        "",
        "| Check | Pass | Fail | Unverified |",
        "| --- | ---: | ---: | ---: |",
    ]
    for code, counts in summary["checks_by_code"].items():
        lines.append(
            f"| {code} | {counts.get('pass', 0)} | {counts.get('fail', 0)} "
            f"| {counts.get('unverified', 0)} |"
        )
    details = [
        "# Equation findings",
        "",
        "Named upstream-depth meaning is conditional. Missing mappings are "
        "unverified. Numeric probes use saved fitted parameters and survey "
        "covariates, never measured output trajectories. Equivalent inline and "
        "named laws use the same checks.",
    ]
    for row in rows:
        details += ["", f"## {row['task']}", "", f"Status: {row['status']}"]
        if row.get("reason"):
            details.append(row["reason"])
        for f in (row.get("assessment") or {}).get("checks", []):
            details += ["", f"- **{f['code']} — {f['status']}**: {f['message']}"]
    for name, content in (("SUMMARY.md", lines), ("FINDINGS.md", details)):
        (output / name).write_text("\n".join(content) + "\n")


def audit(source: Path, output: Path) -> dict:
    """Freeze exact source bytes, reconstruct, checkpoint, and publish evidence."""
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("audit output must be separate from source")
    plan = _read(source, "plan.json")
    if (
        not plan
        or plan.get("protocol") != "detention-process-pilot-3"
        or plan.get("function_delivery_policy") != "identified-function-delivery-1"
        or plan.get("test_data_opened")
        or plan.get("private_reference_opened")
    ):
        raise ValueError("requires public identified-delivery v7 source")
    paths = _paths(plan)
    for case in {task["case"] for task in plan["tasks"]}:
        expected = (REPO / f"configs/detention_prompts/{case}.md").read_text()
        supplied = plan["cells"][case]["brief"]["scientific_context"]
        if supplied.strip() != expected.strip():
            raise ValueError(
                "public specification differs from the audited rule profile"
            )
    records, hashes = _snapshot(source, paths)
    if records["plan.json"] != plan:
        raise ValueError("source plan changed during snapshot")
    frozen = {
        "protocol": PROTOCOL,
        "policy": POLICY,
        "source": str(source),
        "source_plan_sha256": plan["artifact_sha256"],
        "files": hashes,
        "runtime_sha256": runtime_source_hash(),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
        "trajectory_values_used": False,
        "validation_used": False,
    }
    with public._lock(output):
        sealed_write(output / "freeze.json", frozen)
        rows, verified = [], set()
        for task in plan["tasks"]:
            checkpoint = output / "checkpoints" / f"{task['task_id']}.json"
            if checkpoint.exists():
                saved = sealed_read(checkpoint)
                if saved["audit_identity"] != content_hash(frozen):
                    raise ValueError("checkpoint audit identity differs")
                row = saved["row"]
            else:
                row = _row(task, plan["cells"][task["case"]], records, verified)
                sealed_write(
                    checkpoint, {"audit_identity": content_hash(frozen), "row": row}
                )
            rows.append(row)
            sealed_write(output / "feedback" / f"{task['task_id']}.json", feedback(row))
        if _snapshot(source, paths)[1] != hashes:
            raise ValueError("source changed during audit")
        counts = defaultdict(Counter)
        for row in rows:
            for f in (row.get("assessment") or {}).get("checks", []):
                counts[f["code"]][f["status"]] += 1
        summary = {
            **{k: v for k, v in frozen.items() if k not in {"files", "runtime_sha256"}},
            "identity": content_hash(frozen),
            "rows": rows,
            "planned_arms": len(rows),
            "planned_constructions": len({r["construction"] for r in rows}),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "checks_by_code": {k: dict(v) for k, v in sorted(counts.items())},
            "scientific_compliance_score": None,
            "automatic_followup": False,
            "model_changes": 0,
            "critic_calls": 0,
        }
        saved = sealed_write(output / "summary.json", summary)
        _reports(output, rows, summary)
        return saved
