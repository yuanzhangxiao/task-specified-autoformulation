#!/usr/bin/env python3
"""Offline round-2 import, real edited fit, failure fallback and exact resume."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_write

SPEC = importlib.util.spec_from_file_location(
    "v2_smoke", Path(__file__).with_name("smoke_review_deadline_v2.py")
)
V2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V2)


def run(directory):
    source, root = directory / "source", directory / "continuation"
    plan = V2.fixture(source)
    task = next(t for t in plan["tasks"] if t["arm"] == "full")
    client = pipeline._client(
        source,
        plan,
        task,
        0,
        "http://offline",
        lambda: True,
        V2.FIXTURE.FIXTURE.synthetic_transport([]),
    )
    pipeline.propose_one(source, plan, task, 0, client)
    first = pipeline.fit_one(source, plan, task, 0)
    assert first["selected"] is not None
    for item in plan["tasks"]:
        sealed_write(
            io.round_path(source, item, 2) / "result.json",
            {
                "task": item,
                "round": 2,
                "status": "closed_lineage",
                "closed": True,
                "selected": first["selected"] if item == task else None,
                "test_data_opened": False,
            },
        )
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    plan = continuation.prepare(source, root, 2, 2)
    assert not io.read_round(root, task, 0)["closed"]
    calls = []
    for index in (1, 2):

        def transport(url, body, timeout, visit=index):
            payload = json.loads(body["messages"][1]["content"])
            calls.append(payload)
            assert "validation" not in payload and "test" not in payload
            target = next(
                e for e in payload["model"]["equations"] if e["component"] == "v01"
            )
            raw = {
                "hypothesis": "A decaying memory could address the mismatch.",
                "evidence_refs": ["unavailable_samples"],
                "equations": [
                    {
                        "component": "extra_memory",
                        "kind": "dynamic",
                        "expression": "-extra_memory",
                    },
                    {
                        "component": "v01",
                        "expression": target["expression"] + "+extra_memory",
                    },
                ]
                if visit == 1
                else [],
                "initializers": [{"state": "extra_memory", "causal_map": None}]
                if visit == 1
                else [],
                "remove_variables": [] if visit == 1 else ["state_equations"],
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
        assert result["selected"] is not None and not result["closed"]
        assert result["trial"]["fit"]["training"]["available"]
        if index == 1:
            assert proposal["status"] == "committed"
            assert result["citation_audit"]["status"] == "warning"
            assert not result["citation_audit"]["valid_evidence_ids"]
        else:
            assert proposal["status"] == "revision_failed"
            assert len(proposal["attempts"]) == 3
            assert result["fit_trigger"] == "incumbent_fallback"
            assert result["proposal_status"] == "revision_failed"
        assert pipeline.fit_one(root, plan, task, index) == result
        assert pipeline.propose_one(root, plan, task, index, None) == proposal
    for item in plan["tasks"]:
        if item != task:
            for index in (1, 2):
                empty = pipeline._client(
                    root, plan, item, index, "http://offline", lambda: True
                )
                pipeline.propose_one(root, plan, item, index, empty)
                pipeline.fit_one(root, plan, item, index)
    summary = reporting.report(root)
    assert summary["fallback_fits"] == 1
    assert summary["proposal_status_counts"]["revision_failed"] == 1
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    return {
        "status": "passed",
        "live_llm_calls": 0,
        "test_data_opened": False,
        "real_new_state_fit": True,
        "real_fallback_refit": True,
        "exact_resume": True,
        "source_preserved": True,
        "recorded_calls": len(calls),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="review-continuation-smoke-") as tmp:
        print(json.dumps(run(Path(tmp))))
