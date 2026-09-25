#!/usr/bin/env python3
"""Synthetic assisted-sign handoff, real paired fits and deterministic resume."""

import json
import tempfile
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_sign_diagnostic as campaign
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.search import sign_review
from autoformalism.staged_topology import content_hash
from scripts.smoke_dalla_sign_repair import fixture as source_fixture


def fixture(root: Path) -> tuple[Path, Path, Path]:
    """An explicit toy clearance decision; no provider or benchmark inputs."""
    source, _, output = source_fixture(root)
    packet = public._read(source)
    entry = packet["models"][0]
    value = entry["result"]
    request = PublicFitRequest.model_validate(value["selected_request"])
    brief = packet["cells"][entry["task"]["cell"]]["brief"]
    decisions = root / "explicit-decisions.json"
    public._write(
        decisions,
        {
            "protocol": campaign.PROTOCOL,
            "decision_source": campaign.LABEL,
            "task_id": entry["task"]["task_id"],
            "source_result_sha256": value["artifact_sha256"],
            "request_sha256": content_hash(request.model_dump(mode="json")),
            "public_context_sha256": content_hash(sign_review.context(request, brief)),
            "assumptions": [
                "Explicit synthetic clearance hypothesis; not LLM evidence."
            ],
            "review": {
                "decisions": [
                    {
                        "slot_id": slot["slot_id"],
                        "outer_weight_sign": "negative",
                        "mechanism_role": "sink",
                        "basis": "public_task",
                        "public_quote": "The output has clearance.",
                        "rationale": "Interpret the toy self-term as a clearance sink.",
                    }
                    for slot in sign_review.slots(request)
                ]
            },
        },
    )
    return source, decisions, output


def run(root: Path) -> dict:
    source, decisions, output = fixture(root)
    campaign.freeze(source, decisions, output)
    assert campaign.run_one(output, 0)["sign_integrity"]["passed"]
    assert campaign.report(output)["status"] == "partial"
    campaign.run_one(output, 1)
    result = campaign.report(output)
    assert (
        result["status"] == "complete" and result["decision_source"] == campaign.LABEL
    )
    assert result["rows"][0]["validation_nmse"] < 1e-5
    before = {str(p): p.read_bytes() for p in output.rglob("*.json")}
    campaign.run_one(output, 0)
    campaign.run_one(output, 1)
    assert result == campaign.report(output)
    assert before == {str(p): p.read_bytes() for p in output.rglob("*.json")}
    return {
        "status": "passed",
        "protocol": campaign.PROTOCOL,
        "arms": 2,
        "decision_source": campaign.LABEL,
        "live_llm_calls": 0,
        "benchmark_data_used": False,
        "test_data_opened": False,
        "exact_resume": True,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="manual-sign-smoke-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
