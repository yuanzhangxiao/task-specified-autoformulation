#!/usr/bin/env python3
"""Prescribed general construction -> frozen fit -> shared-law edit -> warm fit."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.search.shared_model_revision import relationships

_spec = importlib.util.spec_from_file_location(
    "integration_smoke_fixture",
    Path(__file__).with_name("smoke_shared_process_pilot.py"),
)
assert _spec is not None and _spec.loader is not None
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)


def fixture(root: Path) -> dict:
    plan = old.fixture(root)
    config = io.DeadlineConfig.model_validate(
        {**plan["config"], "protocol": io.INTEGRATION_PROTOCOL}
    )
    plan.pop("artifact_sha256")
    plan.update(
        protocol=config.protocol,
        config=config.model_dump(mode="json"),
        tasks=io.tasks(config),
        launcher_sha256=io.launcher_hash(config.protocol),
    )
    (root / "plan.json").unlink()  # This isolated toy fixture only.
    return sealed_write(root / "plan.json", plan)


def transport_for(calls, *, empty=False, invalid=False):
    """Use generic names and no domain physics, with optional process decisions."""
    previous = old.transport_for([])

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        p = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        calls.append(p)
        if p.get("protocol") == "general-shared-revision-1":
            equation = next(e for e in p["model"]["equations"] if e["component"] == "q")
            reply = {
                "hypothesis": "Change the common memory law coherently.",
                "equations": [{**equation, "expression": "m"}],
            }
        elif "eligible_equation_targets" in p:
            reply = {
                "processes": []
                if empty
                else [
                    {
                        "name": "q",
                        "depends_on": ["m"],
                        "kind": "transfer",
                        "scientific_meaning": "Common memory release with "
                        "declared opposite contributions.",
                        "uses": [
                            {"target": "m", "sign": "negative", "conversion": "1"},
                            {
                                "target": "m" if invalid else "v01",
                                "sign": "positive",
                                "conversion": "1",
                            },
                        ],
                    }
                ]
            }
        elif "requested_slots" in p:
            values = []
            for name, selected in p["requested_slots"].items():
                sources = selected["sources"]
                expression = (
                    "tanh(m)" if p["equation_context"]["name"] == "q" else sources[0]
                )
                values.append(
                    {"interaction_id": name, "expression": expression, "parameters": []}
                )
            reply = {"functions": values}
        else:
            return previous(url, body, timeout)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {
                "total_tokens": 100,
                "prompt_tokens": 70,
                "completion_tokens": 30,
            },
        }

    return transport


def run(root: Path) -> dict:
    plan = fixture(root)
    io.verify(root)
    task, calls = plan["tasks"][0], []
    for index in (0, 1):
        client = pipeline._client(
            root,
            plan,
            task,
            index,
            "http://offline",
            lambda: True,
            transport_for(calls),
        )
        proposal = pipeline.propose_one(root, plan, task, index, client)
        # Only the shared law changes; both consumer definitions stay intact.
        assert proposal["status"] == ("constructed" if index == 0 else "committed"), (
            proposal
        )
        result = pipeline.fit_one(root, plan, task, index)
        assert result["trial"] is not None, result
        assert result["trial"]["fit"]["training"]["available"], result
        count = len(calls)
        assert proposal == pipeline.propose_one(root, plan, task, index, client)
        assert result == pipeline.fit_one(root, plan, task, index)
        assert len(calls) == count
    report = reporting.report(root)
    assert len(report["rows"]) == 4
    assert (root / "integration_summary.json").exists()
    law = next(
        x
        for x in relationships(result["trial"]["bundle"]["candidate"])
        if x["process"] == "q"
    )
    assert set(law["direct_consumers"]) == {"m", "v01"}
    return {
        "status": "passed",
        "real_fit_rounds": 2,
        "prescribed_calls": len(calls),
        "live_llm_calls": 0,
        "resume_repeated_fits": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="shared-integration-smoke-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
