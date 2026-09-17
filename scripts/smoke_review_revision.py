#!/usr/bin/env python3
"""Offline larger-model revision, real fitting, fallback and deterministic resume."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting

SPEC = importlib.util.spec_from_file_location(
    "continuation_smoke", Path(__file__).with_name("smoke_review_continuation.py")
)
PREVIOUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS)


def run(directory: Path) -> dict:
    PREVIOUS.run(directory / "previous")
    source, root = directory / "previous/continuation", directory / "revision"
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    plan = continuation.prepare(source, root, 4, 2, protocol=io.REVISION_PROTOCOL)
    task = next(t for t in plan["tasks"] if t["arm"] == "full")
    calls = []
    for index in (1, 2):

        def transport(url, body, timeout, visit=index):
            payload = json.loads(body["messages"][1]["content"])
            calls.append(payload)
            assert payload["protocol"] == "scientific-content-revision-5"
            assert "validation" not in payload and "test" not in payload
            assert "limits" not in payload["public_brief"]
            target = next(
                e for e in payload["model"]["equations"] if e["component"] == "v01"
            )
            raw = {
                "hypothesis": "Exercise larger-model revision on synthetic data.",
                "equations": [
                    {
                        "component": "v01",
                        "expression": target["expression"] + "+0.001*u01" * 9
                        if visit == 1
                        else "unknown*v01",
                    }
                ],
                "new_parameters": [{"name": "unused_metadata"}],
            }
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
                ],
                "usage": {"total_tokens": 100},
            }

        client = pipeline._client(
            root, plan, task, index, "http://offline", lambda: True, transport
        )
        proposal = pipeline.propose_one(root, plan, task, index, client)
        result = pipeline.fit_one(root, plan, task, index)
        assert (
            result["selected"] is not None
            and result["trial"]["fit"]["training"]["available"]
        )
        if index == 1:
            assert proposal["status"] == "committed", proposal
            audit = proposal["decision"]["provenance"]
            assert audit["size_audit"]["after"]["exceeded_references"]
            assert len(audit["unused_new_declarations_removed"]) == 1
        else:
            assert proposal["status"] == "revision_failed"
            assert len(proposal["attempts"]) == 3
            assert result["fit_trigger"] == "incumbent_fallback"
        assert pipeline.propose_one(root, plan, task, index, None) == proposal
        assert pipeline.fit_one(root, plan, task, index) == result
    reporting.report(root)
    diagnostics = json.loads((root / "revision_diagnostics.json").read_text())
    assert diagnostics["accepted_revision_visits"] == 1
    assert diagnostics["trial_visits_above_original_size_references"] >= 1
    assert sum(r["visits"] for r in diagnostics["terminal_errors"]) == 1
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    return {
        "status": "passed",
        "real_larger_model_fit": True,
        "real_fallback_fit": True,
        "unused_metadata_removed": True,
        "exact_resume": True,
        "source_preserved": True,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "recorded_calls": len(calls),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="review-revision-smoke-") as tmp:
        print(json.dumps(run(Path(tmp))))
