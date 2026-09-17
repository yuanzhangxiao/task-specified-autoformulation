#!/usr/bin/env python3
"""Real-fitting smoke across two continuation phases with exact resume."""

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
    "v3_smoke", Path(__file__).with_name("smoke_review_continuation.py")
)
V3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V3)


def run(directory: Path) -> dict:
    V3.run(directory / "previous")
    source, root = directory / "previous/continuation", directory / "parameters"
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    plan = continuation.prepare(source, root, 4, 2, protocol=io.PARAMETER_PROTOCOL)
    assert plan["continuation"]["source_phase_round"] == 2
    task = next(t for t in plan["tasks"] if t["arm"] == "full")
    calls = []
    for index in (1, 2):

        def transport(url, body, timeout, visit=index):
            payload = json.loads(body["messages"][1]["content"])
            calls.append(payload)
            assert "validation" not in payload and "test" not in payload
            target = next(
                e for e in payload["model"]["equations"] if e["component"] == "v01"
            )
            inherited = payload["model"]["parameters"][0]["name"]
            raw = {
                "hypothesis": "Test a direct input contribution.",
                "new_parameters": (
                    [
                        {"name": inherited, "role": "shape"},
                        {"name": inherited, "role": "rate"},
                        {"name": "extra_gain"},
                    ]
                    if visit == 1
                    else [
                        {"name": "conflict", "role": "shape"},
                        {"name": "conflict", "role": "rate"},
                    ]
                ),
                "equations": [
                    {
                        "component": "v01",
                        "expression": target["expression"] + "+extra_gain*u01",
                    }
                ]
                if visit == 1
                else [],
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
        assert result["trial"]["origin_round"] == 4 + index
        if index == 1:
            assert proposal["status"] == "committed"
            assert (
                len(
                    proposal["decision"]["provenance"]["parameter_declaration_audit"][
                        "inherited_declarations_ignored"
                    ]
                )
                == 2
            )
        else:
            assert proposal["status"] == "revision_failed"
            assert result["fit_trigger"] == "incumbent_fallback"
        assert pipeline.propose_one(root, plan, task, index, None) == proposal
        assert pipeline.fit_one(root, plan, task, index) == result
    assert reporting.report(root)["fallback_fits"] == 1
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    return {
        "status": "passed",
        "real_global_parameter_fit": True,
        "real_fallback_fit": True,
        "chained_import": True,
        "source_preserved": True,
        "exact_resume": True,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "recorded_calls": len(calls),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="review-parameters-smoke-") as tmp:
        print(json.dumps(run(Path(tmp))))
