#!/usr/bin/env python3
"""Offline content revision, certificate retry, new-state fitting and exact resume."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.sibling_fit import EDIT_POLICY
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.staged_topology import content_hash

SPEC = importlib.util.spec_from_file_location(
    "legacy_review_smoke", Path(__file__).with_name("smoke_review_deadline.py")
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def fixture(root: Path) -> dict:
    """Only the isolated synthetic fixture is resealed as a three-visit v2 plan."""
    old = FIXTURE.fixture(root)
    config = io.DeadlineConfig.model_validate(
        {
            **old["config"],
            "protocol": io.CONTENT_PROTOCOL,
            "rounds": 3,
        }
    )
    (root / "plan.json").unlink()
    return sealed_write(
        root / "plan.json",
        {
            **{k: v for k, v in old.items() if k != "artifact_sha256"},
            "protocol": io.CONTENT_PROTOCOL,
            "config": config.model_dump(mode="json"),
            "launcher_sha256": io.launcher_hash(io.CONTENT_PROTOCOL),
        },
    )


def run(root: Path) -> dict:
    plan = fixture(root)
    io.verify(root)
    calls, revisions = [], []
    ordinary = FIXTURE.FIXTURE.synthetic_transport(calls)

    def transport(url, body, timeout):
        try:
            payload = json.loads(body["messages"][1]["content"])
        except json.JSONDecodeError:
            return ordinary(url, body, timeout)
        assert "training_evidence" in payload
        assert "validation" not in payload and "test" not in payload
        revisions.append(payload)
        model = payload["model"]
        equation = next(e for e in model["state_equations"] if e["state"] == "v01")
        if len(revisions) == 1:
            # Valid grammar and input path, but removes the required memory path.
            equations = [{"component": "v01", "expression": "-v01+u01"}]
            initials = []
        else:
            name = f"smoke_z{len(revisions)}"
            equations = [
                {"component": name, "kind": "dynamic", "expression": f"-{name}"},
                {"component": "v01", "expression": equation["rhs"] + f"+{name}"},
            ]
            initials = [
                {
                    "state": name,
                    "causal_map": {
                        "expression": "a*v01",
                        "parameters": [{"name": "a", "role": "coefficient"}],
                    },
                }
            ]
        reply = {
            "hypothesis": "A decaying initial response could explain a residual. "
            "This prescribed interface smoke makes no scientific claim.",
            "evidence_ids": [payload["evidence_catalog"][0]["row"]],
            "equations": equations,
            "initializers": initials,
        }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    task = next(t for t in plan["tasks"] if t["arm"] == "full")
    for index in range(3):
        client = pipeline._client(
            root, plan, task, index, "http://offline", lambda: True, transport
        )
        proposal = pipeline.propose_one(root, plan, task, index, client)
        assert proposal is not None
        if index:
            assert proposal["status"] == "committed", proposal
            routes = proposal["bundle"]["revision_provenance"]["routes"]
            assert {"state_inventory", "topology", "initialization"} <= set(routes)
        if index == 1:
            assert len(proposal["attempts"]) == 2
            feedback = proposal["attempts"][0]["feedback"]
            assert feedback["code"] == "PUBLIC_MODEL_REQUIREMENTS", feedback
            assert feedback["details"]["failed_mechanisms"], feedback
            assert revisions[-1]["retry_feedback"]["incumbent_unchanged"]
        result = pipeline.fit_one(root, plan, task, index)
        assert result["trial"]["fit"]["training"]["available"], result
        assert result["trial"]["fit"]["validation"]["available"], result
        assert result["selected"] is not None
        if index:
            frozen = public._read(io.round_path(root, task, index) / "fit/freeze.json")
            assert frozen["warm_start_policy"] == EDIT_POLICY
            assert frozen["seed"]["changed_boundaries"]
            assert frozen["seed"]["retained_initializer_parameters"]
            assert any(
                n.startswith("init_smoke_z") for n in frozen["seed"]["fresh_parameters"]
            )
        assert pipeline.fit_one(root, plan, task, index) == result
        resumed = pipeline._client(
            root,
            plan,
            task,
            index,
            "http://offline",
            lambda: True,
            lambda *args: (_ for _ in ()).throw(
                AssertionError("unexpected provider call")
            ),
        )
        assert content_hash(
            pipeline.propose_one(root, plan, task, index, resumed)
        ) == content_hash(proposal)
    return {
        "status": "passed",
        "construction_calls": len(calls),
        "revision_calls": len(revisions),
        "completed_visits": 3,
        "certificate_retry": True,
        "real_new_state_fits": 2,
        "causal_initializers": True,
        "exact_resume": True,
        "live_llm_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="review-deadline-v2-smoke-") as tmp:
        print(json.dumps(run(Path(tmp))))
