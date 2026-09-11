#!/usr/bin/env python3
"""Audit every stored v5 reply at its original public decision point, without calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.revision_actions import request_revision_action
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    RevisionContractError,
)
from autoformalism.rebuttal.staged_topology_campaign import public_validation_context
from autoformalism.schemas import CandidateModel, ParameterSpec
from autoformalism.staged_topology import content_hash


def replay(source: Path, output: Path) -> dict:
    """Reconstruct the exact provisional equations in each cached public request.

    This is a same-decision-point audit, not a simulated counterfactual search:
    historical replies were generated from old prompts. All raw replies remain
    unchanged and are accounted for, including transport/schema failures.
    """
    plan = json.loads((source / "plan.json").read_text())
    if (
        content_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
        != plan["plan_sha256"]
    ):
        raise ValueError("source plan hash differs")
    if (
        plan.get("test_data_opened") is not False
        or plan.get("private_reference_opened") is not False
    ):
        raise ValueError("source is not a public-only campaign")
    rows = []
    for task in plan["tasks"]:
        path = source / task["candidate_path"]
        if not path.resolve().is_relative_to((source / "frozen/candidates").resolve()):
            raise ValueError("candidate path is outside frozen public candidates")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != task["candidate_file_sha256"]
        ):
            raise ValueError("frozen source candidate differs")
        original = CandidateModel.model_validate_json(path.read_text())
        task_root = source / "results" / task["task_id"]
        for call_path in sorted((task_root / "calls").glob("*.json")):
            record = json.loads(call_path.read_text())
            row = {
                "task_id": task["task_id"],
                "seed": task["seed"],
                "step": record.get("step"),
                "attempt": record.get("attempt"),
                "source_call_sha256": hashlib.sha256(
                    call_path.read_bytes()
                ).hexdigest(),
            }
            with tempfile.TemporaryDirectory(
                prefix="revision-action-replay-"
            ) as directory:
                try:
                    request = json.loads(
                        record["request"]["body"]["messages"][1]["content"]
                    )
                    payload = original.model_dump(mode="json")
                    equations = request["all_current_equations"]
                    for equation in payload["state_equations"]:
                        equation["rhs"] = equations[equation["state"]]
                    for process in payload["processes"]:
                        process["expression"] = equations[process["name"]]
                    payload["parameters"] = [
                        ParameterSpec(name=name, scope="global", role=role).model_dump(
                            mode="json"
                        )
                        for name, role in request["available_parent_parameters"].items()
                    ]
                    candidate = CandidateModel.model_validate(payload)

                    class RecordedClient:
                        settings = SimpleNamespace(attempts_per_step=1)

                        def __init__(self, stored):
                            self.stored = stored

                        def call(self, **_kwargs):
                            return self.stored

                    _, action = request_revision_action(
                        client=RecordedClient(record),
                        route=request["route"],
                        candidate=candidate,
                        selected=tuple(
                            item["component"]
                            for item in request["selected_pending_components"]
                        ),
                        context=public_validation_context(task["benchmark_id"]),
                        scientific_context=request["scientific_context"],
                        numerical_feedback=request["numerical_feedback"],
                        round_index=1,
                        failure_checkpoint=Path(directory) / "failures.json",
                        transaction_path=Path(directory) / "transaction.json",
                        nonlinear_targets=("v01",),
                    )
                    row.update(
                        status=action["status"],
                        kept_components=action["kept_components"],
                    )
                except RevisionContractError as exc:
                    row.update(status="rejected", diagnostic=exc.diagnostic())
                except (ValueError, TypeError, KeyError) as exc:
                    row.update(
                        status="replay_error",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                transaction_path = Path(directory) / "transaction.json"
                transaction = (
                    json.loads(transaction_path.read_text())
                    if transaction_path.exists()
                    else {}
                )
                row["retained_components"] = sorted(transaction.get("accepted", {}))
                row["diagnostics"] = (transaction.get("diagnostic") or {}).get(
                    "component_failures", []
                )
                rows.append(row)
    counts = Counter(row["status"] for row in rows)
    report = {
        "schema_version": "staged-revision-action-replay-1",
        "status": "complete",
        "source_plan_sha256": plan["plan_sha256"],
        "stored_reply_count": len(rows),
        "outcome_counts": dict(counts),
        "diagnostic_counts": dict(
            Counter(item["code"] for row in rows for item in row["diagnostics"])
        ),
        "new_llm_calls": 0,
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "counterfactual_search_claimed": False,
        "rows": rows,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    """Save full diagnostics; print only a short screen-friendly summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.source_root, args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"}, indent=2
        )
    )
    if not result["stored_reply_count"] or result["outcome_counts"].get("replay_error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
