#!/usr/bin/env python3
"""Offline prescribed shared law -> real frozen fit -> whole-model revision."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
from pathlib import Path

from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.staged_topology import content_hash

_spec = importlib.util.spec_from_file_location(
    "shared_pilot_support", Path(__file__).with_name("smoke_review_deadline.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
original_fixture = _module.fixture


def fixture(root: Path) -> dict:
    """Synthetic public-data fixture with paired prompts, no stored benchmark data."""
    original = original_fixture(root)
    config = io.DeadlineConfig.model_validate(
        {
            **original["config"],
            "protocol": io.SHARED_PROTOCOL,
            "limits": {
                "generated_variables": 64,
                "terms_per_equation": 32,
                "total_terms": 256,
            },
        }
    )
    original.pop("artifact_sha256")
    for cell in original["cells"].values():
        cell["brief"]["limits"] = config.limits.model_dump(mode="json")
    original.update(
        protocol=io.SHARED_PROTOCOL,
        config=config.model_dump(mode="json"),
        tasks=io.tasks(config),
        launcher_sha256=io.launcher_hash(io.SHARED_PROTOCOL),
    )
    (root / "plan.json").unlink()  # Only replace the temporary fixture's protocol.
    return sealed_write(root / "plan.json", original)


def transport_for(calls, *, nonlinear=False):
    """Prescribe reuse without asserting physical conservation."""

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        calls.append({"system": body["messages"][0]["content"], "payload": payload})
        if payload.get("protocol") == "scientific-content-revision-6":
            equations = payload["model"]["equations"]
            consumers = [e for e in equations if re.search(r"\bq\b", e["expression"])]
            # One coherent factorization plus all consumers; zero new parameters.
            reply = {
                "hypothesis": "Preserve this prescribed response in a named law.",
                "equations": [
                    {"component": "shared_law", "kind": "algebraic", "expression": "q"}
                ]
                + [
                    {**e, "expression": re.sub(r"\bq\b", "shared_law", e["expression"])}
                    for e in consumers
                ],
            }
        elif "selected_state" in payload:
            reply = {
                "initial": {
                    "mode": "causal_map",
                    "expression": "a+b*v01",
                    "parameters": [
                        {"name": "a", "role": "coefficient"},
                        {"name": "b", "role": "coefficient"},
                    ],
                }
            }
        elif "selected_equation" in payload:
            lhs = payload["selected_equation"]["lhs"]
            functions = {
                "q": [
                    {
                        "expression": "rate*tanh(m)" if nonlinear else "rate*m",
                        "parameters": [{"name": "rate", "role": "coefficient"}],
                    }
                ],
                "m": [
                    {
                        "expression": "gain*u01",
                        "parameters": [{"name": "gain", "role": "coefficient"}],
                    },
                    {"expression": "q", "parameters": []},
                ],
                "v01": [
                    {"expression": "q", "parameters": []},
                    {
                        "expression": "decay*v01",
                        "parameters": [{"name": "decay", "role": "coefficient"}],
                    },
                ],
            }
            reply = {"functions": functions[lhs]}
        elif "selected_lhs" in payload:
            lhs = payload["selected_lhs"]["name"]
            sources = {
                "q": [("m", "positive")],
                "m": [("u01", "positive"), ("q", "negative")],
                "v01": [("q", "positive"), ("v01", "negative")],
            }[lhs]
            reply = {
                "terms": [
                    {
                        "sources": [source],
                        "outer_weight_sign": sign,
                        "scientific_role": (
                            "nonlinear response law"
                            if lhs == "q" and nonlinear
                            else "unit-compatible contribution"
                        ),
                    }
                    for source, sign in sources
                ],
                "inventory_revision": None,
            }
        else:
            reply = {
                "variables": [
                    {
                        "name": "v01",
                        "definition": "differential",
                        "scientific_role": "output response",
                    },
                    {
                        "name": "m",
                        "definition": "differential",
                        "scientific_role": "input memory",
                    },
                    {
                        "name": "q",
                        "definition": "algebraic",
                        "scientific_role": "shared memory release contribution",
                    },
                ]
            }
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
    calls = []
    task = next(
        t for t in plan["tasks"] if t["arm"] == "full" and t["process_guidance"] == "on"
    )
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
        assert proposal["status"] == ("constructed" if index == 0 else "committed"), (
            proposal
        )
        result = pipeline.fit_one(root, plan, task, index)
        assert result["selected"] is not None, result
        assert result["trial"]["fit"]["training"]["available"], result
        assert result == pipeline.fit_one(root, plan, task, index)
        assert content_hash(proposal) == content_hash(
            pipeline.propose_one(root, plan, task, index, client)
        )
    summary = reporting.report(root)
    assert len(summary["rows"]) == 8
    retained = next(
        r
        for r in summary["rows"]
        if r["task_id"] == task["task_id"] and r["round"] == 0
    )
    counts = retained["shared_process_evidence"]["counts"]
    assert counts["named_processes_shared_between_state_equations"] == 1, counts
    return {
        "status": "passed",
        "prescribed_provider_calls": len(calls),
        "real_fit_rounds": 2,
        "resume_repeated_fits": 0,
        "benchmark_data_opened": False,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="shared-process-smoke-") as temp:
        print(json.dumps(run(Path(temp)), indent=2))
