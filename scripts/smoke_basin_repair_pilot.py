#!/usr/bin/env python3
"""Prescribed reply smoke: full reconstruction, model review, warm fit and resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import basin_equation_audit as audit
from autoformalism.rebuttal import basin_repair_pilot as pilot
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import smoke_basin_equation_audit as previous
from scripts import smoke_process_revision_confirmation as construction


def fixture(root: Path):
    """Complete toy source with explicitly synthetic parameter/score records."""
    source = previous.fixture(root / "history")
    plan = construction.experiment.pilot.verify(source)
    common = {
        t["construction_task"]["task_id"]: t["construction_task"] for t in plan["tasks"]
    }
    for task in common.values():
        client = construction.experiment.pilot.make_client(
            source / "construction",
            plan,
            task,
            "offline",
            transport=construction.transport_for(
                [], independent=task["case"] == "independent"
            ),
        )
        proposal = construction.experiment.pilot.construct(
            source / "construction", plan, task, client
        )
        assert proposal["status"] == "constructed", proposal
    for task in plan["tasks"]:
        proposal = construction.experiment.pilot.gain_proposal(source, plan, task)
        assert proposal["status"] == "constructed"
        values = {p["name"]: 1.0 for p in proposal["bundle"]["candidate"]["parameters"]}
        sealed_write(
            source / "results" / task["task_id"] / "result.json",
            {
                "task": task,
                "proposal_sha256": proposal["artifact_sha256"],
                "status": "complete",
                "synthetic_fixture_only": True,
                "fit": {
                    "parameters": values,
                    "training": {"normalized_mse": 1.0},
                    "validation": {"normalized_mse": 1.0},
                },
                "replay": None,
            },
        )
    gate = root / "audit"
    assert audit.audit(source, gate)["status_counts"] == {"assessed": 16}
    return source, gate


def transport(calls, *, accept=True):
    """One explicit RHS edit, followed by a review of its assembled preview."""

    def send(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if payload["previous_transaction"] is None:
            reply = {
                "hypothesis": "Use the public surveyed crest and depth conversion.",
                "equations": [
                    {
                        "component": "h_down",
                        "expression": (
                            "inflow_down/area_down - "
                            "outlet_rate*max(0,h_down-crest_down)"
                        ),
                    }
                ],
                "new_parameters": [{"name": "outlet_rate", "role": "rate"}],
            }
        else:
            reply = {
                "hypothesis": "I reviewed the complete rebuilt model.",
                "accept_displayed": accept,
            }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return send


def client(root, plan, task, calls, **kwargs):
    """Use real durable request transport with prescribed replies."""
    translated = {
        **plan,
        "config": {**plan["config"], "protocol": pilot.prior.GAIN_PROTOCOL},
    }
    return pilot.prior.make_client(
        root, translated, task, "offline", transport=transport(calls, **kwargs)
    )


def run(root):
    source, gate = fixture(root / "fixture")
    original = previous.snapshot(source)
    output = root / "repair"
    plan = pilot.freeze(source, gate, output)
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    c = client(output, plan, task, calls)
    proposal = pilot.run_one(output, plan, task, c)
    assert proposal["status"] == "confirmed", proposal
    assert len(calls) == 2
    assert "outlet_rate" in str(calls[1]["assembled_model"])
    assert pilot.run_one(output, plan, task, c) == proposal and len(calls) == 2
    result = pilot.fit_task(output, task["index"])
    assert result["status"] == "complete", result
    assert result["replay"]["replay_agreement"], result
    assert pilot.fit_task(output, task["index"]) == result
    report = pilot.report(output)
    assert report["status_counts"] == {"missing": 15, "complete": 1}
    assert report["physical_calls"] == 2
    assert previous.snapshot(source) == original
    return {
        "status": "passed",
        "live_llm_calls": 0,
        "prescribed_calls": 2,
        "child_fits": 1,
        "resume_unchanged": True,
        "historical_bytes_unchanged": True,
        "reported_prediction_accuracy_not_asserted": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
