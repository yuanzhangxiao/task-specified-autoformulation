#!/usr/bin/env python3
"""Audit historical v5 replies through new edit validation, without new calls."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.repair_transactions import (
    RepairAction,
    commit_action,
    default_initialization,
)
from autoformalism.rebuttal.staged_topology_campaign import public_validation_context
from autoformalism.schemas import CandidateModel, ParameterSpec
from autoformalism.staged_topology import content_hash


def replay(source: Path, output: Path):
    plan = json.loads((source / "plan.json").read_text())
    if (
        content_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
        != plan["plan_sha256"]
    ):
        raise ValueError("source plan differs")
    if (
        plan.get("test_data_opened") is not False
        or plan.get("private_reference_opened") is not False
    ):
        raise ValueError("source is not public-only")
    rows = []
    for task in plan["tasks"]:
        path = source / task["candidate_path"]
        if not path.resolve().is_relative_to((source / "frozen/candidates").resolve()):
            raise ValueError("candidate path outside public source")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != task["candidate_file_sha256"]
        ):
            raise ValueError("source candidate differs")
        original = CandidateModel.model_validate_json(path.read_text())
        context = public_validation_context(task["benchmark_id"])
        for call in sorted(
            (source / "results" / task["task_id"] / "calls").glob("*.json")
        ):
            record = json.loads(call.read_text())
            row = {
                "task": task["task_id"],
                "request_hash": record["request_hash"],
                "source_call_sha256": hashlib.sha256(call.read_bytes()).hexdigest(),
            }
            try:
                request = json.loads(
                    record["request"]["body"]["messages"][1]["content"]
                )
                payload = original.model_dump(mode="json")
                for field, name, law in (
                    ("state_equations", "state", "rhs"),
                    ("processes", "name", "expression"),
                ):
                    for entry in payload[field]:
                        entry[law] = request["all_current_equations"][entry[name]]
                payload["parameters"] = [
                    ParameterSpec(name=n, scope="global", role=r).model_dump(
                        mode="json"
                    )
                    for n, r in request["available_parent_parameters"].items()
                ]
                parent = CandidateModel.model_validate(payload)
                initial = default_initialization(parent, context)
                reply = visible_response(record)
                # Only syntax adaptation: historical scope and equations stay intact.
                action = RepairAction.model_validate(
                    {
                        "scope": "model"
                        if request["route"] == "topology_revision"
                        else "function",
                        "equations": reply.get("revisions", []),
                        "keep": reply.get("keep_components", []),
                    }
                )
                _, _, audit = commit_action(
                    parent, initial, action, context, ("v01",), ("v01",)
                )
                row.update(
                    status=audit["status"],
                    domain_findings=audit.get("domain_findings", []),
                )
            except (ValueError, ModelValidationError) as exc:
                row.update(
                    status="rejected",
                    code=getattr(exc, "code", "ACTION_CONTRACT"),
                    message=str(exc),
                    details=getattr(exc, "details", {}),
                )
            except (KeyError, TypeError) as exc:
                row.update(status="unsupported_record", message=str(exc))
            rows.append(row)
    result = {
        "schema_version": "repair-transaction-replay-1",
        "source_plan_sha256": plan["plan_sha256"],
        "status": "complete",
        "stored_reply_count": len(rows),
        "outcome_counts": dict(Counter(r["status"] for r in rows)),
        "diagnostic_counts": dict(Counter(r["code"] for r in rows if "code" in r)),
        "new_llm_calls": 0,
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "counterfactual_search_claimed": False,
        "rows": rows,
    }
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.source_root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if not result["stored_reply_count"] or result["outcome_counts"].get(
        "unsupported_record"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
