"""Offline replay of parent-role preservation over frozen revision replies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    ComponentRevisionReply,
    apply_component_revision,
)
from autoformalism.rebuttal.staged_topology_campaign import public_validation_context
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash


def replay_multiround_role_repairs(
    source_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Revalidate stored round-one replies without calls, fitting, or judging."""
    plan_path = source_root / "plan.json"
    summary_path = source_root / "results" / "summary.json"
    if not summary_path.is_file():
        summary_path = source_root / "summary.json"
    plan = _read_object(plan_path)
    summary = _read_object(summary_path)
    plan_digest = content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan_digest != plan.get("plan_sha256"):
        raise ValueError("source multiround plan digest differs")
    if summary.get("status") != "complete":
        raise ValueError("source multiround campaign is not complete")
    for key in (
        "validation_used_for_parameter_fitting",
        "scientific_judge_called",
        "test_data_opened",
        "private_reference_opened",
        "automatic_winner_defined",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"source public-only boundary failed: {key}")

    results_root = summary_path.parent
    rows: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        candidate_path = source_root / task["candidate_path"]
        if _sha256(candidate_path) != task["candidate_file_sha256"]:
            raise ValueError(f"source candidate differs: {task['task_id']}")
        candidate = CandidateModel.model_validate_json(candidate_path.read_text())
        context = public_validation_context(task["benchmark_id"])
        records = [
            _read_object(path)
            for path in (results_root / task["task_id"] / "calls").glob("*.json")
        ]
        records = sorted(
            (
                item
                for item in records
                if item.get("step") == "round_1_function_revision"
                and item.get("status") == "responded"
            ),
            key=lambda item: int(item["attempt"]),
        )
        for record in records:
            request = json.loads(record["request"]["body"]["messages"][1]["content"])
            selected = tuple(
                item["component"] for item in request["selected_components"]
            )
            response = visible_response(record)
            reply = ComponentRevisionReply.model_validate(response)
            try:
                revised, audit = apply_component_revision(
                    candidate,
                    reply,
                    context,
                    selected=selected,
                    route="function_revision",
                )
                error = None
            except (ValueError, TypeError, KeyError) as exc:
                revised = None
                audit = None
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "task_id": task["task_id"],
                    "seed": task["seed"],
                    "attempt": record["attempt"],
                    "request_hash": record["request_hash"],
                    "selected_components": list(selected),
                    "accepted_after_role_policy": revised is not None,
                    "error": error,
                    "parameter_role_derivations": (
                        audit["parameter_role_derivations"] if audit else []
                    ),
                    "candidate_equations": (
                        {
                            **{
                                item.state: item.rhs
                                for item in revised.state_equations
                            },
                            **{
                                item.name: item.expression
                                for item in revised.processes
                            },
                        }
                        if revised
                        else None
                    ),
                }
            )
    accepted = sum(row["accepted_after_role_policy"] for row in rows)
    repaired = sum(bool(row["parameter_role_derivations"]) for row in rows)
    report = {
        "schema_version": "staged-multiround-role-replay-1",
        "status": "pass" if rows and accepted == len(rows) else "fail",
        "source_root": str(source_root),
        "source_plan_sha256": plan_digest,
        "source_summary_file_sha256": _sha256(summary_path),
        "stored_response_count": len(rows),
        "accepted_after_role_policy_count": accepted,
        "responses_with_role_derivation_count": repaired,
        "new_llm_calls_made": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
        "rows": rows,
    }
    atomic_json(output, report)
    return report


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["replay_multiround_role_repairs"]
