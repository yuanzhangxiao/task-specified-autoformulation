#!/usr/bin/env python3
"""Offline end-to-end shared-law smoke using temporary synthetic data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.search import shared_process_contract as shared
from scripts import smoke_detention_process_pilot as smoke


def transport_for(calls, mode="add"):
    """Answer the new process stages; exercise real topology/function constructors."""
    legacy = smoke.transport_for(calls, "empty" if mode == "empty" else "add")

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        if "selected_term" in payload:
            selected = payload["selected_term"]
            sources = selected["sources"]
            parameter = None
            if selected["lhs"] == "q":
                expression, parameter = "k*h_up", "k"
            elif "q" in sources:
                expression = "q/" + next(s for s in sources if s != "q")
            elif "inflow_up" in sources:
                expression = "inflow_up/area_up"
            elif "inflow_down" in sources:
                expression = "inflow_down/area_down"
            else:
                expression, parameter = "d*h_down/area_down", "d"
            calls.append({"body": body, "payload": payload})
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "expression": expression,
                                    "parameters": [
                                        {"name": parameter, "role": "coefficient"}
                                    ]
                                    if parameter
                                    else [],
                                }
                            )
                        },
                    }
                ],
                "usage": {"total_tokens": 100},
            }
        if payload.get("protocol") != shared.POLICY:
            return legacy(url, body, timeout)
        calls.append({"body": body, "payload": payload})
        if payload.get("stage") == "process_uses":
            reply = {
                "uses": [
                    {
                        "target": n,
                        "outer_weight_sign": sign,
                        "conversion_sources": [area],
                        "scientific_role": "area conversion",
                    }
                    for n, area, sign in (
                        ("h_up", "area_up", "negative"),
                        ("h_down", "area_down", "positive"),
                    )
                ]
            }
        else:
            p = {
                "name": "q",
                "depends_on": ["h_up"],
                "used_in_equations_for": ["h_up", "h_down"],
                "scientific_meaning": "volumetric transfer from upstream to downstream",
            }
            reply = {
                "processes": []
                if mode == "empty"
                else [p, {**p, "name": "bad", "used_in_equations_for": ["inflow_up"]}]
            }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return transport


def run(root: Path) -> dict:
    """Check saved-inventory pairing, one shared coefficient, fitting and replay."""
    pilot = smoke.pilot
    source, parent, output = (root / n for n in ("source", "parent", "bound"))
    old = smoke.fixture(source, parent)
    for task in old["tasks"]:
        if task["review"]:
            mode = "independent" if task["case"] == "independent" else "add"
            pilot.construct(
                parent,
                old,
                task,
                pilot.make_client(
                    parent,
                    old,
                    task,
                    "offline",
                    transport=smoke.transport_for([], mode),
                ),
            )
    plan = pilot.freeze(
        source, output, pilot.REPO / "configs/detention_process_pilot_v2.json", parent
    )
    task = next(t for t in plan["tasks"] if t["case"] == "coupled" and t["review"])
    calls = []
    client = pilot.make_client(
        output, plan, task, "offline", transport=transport_for(calls)
    )
    proposal = pilot.construct(output, plan, task, client)
    assert proposal["status"] == "constructed" and not proposal["fallback_used"], (
        proposal
    )
    assert len(proposal["bundle"]["candidate"]["parameters"]) == 2
    result = pilot.fit_task(output, task["index"])
    assert result["fit"]["validation"]["normalized_mse"] < 1e-4, result
    assert result["replay"]["replay_agreement"], result
    assert pilot.fit_task(output, task["index"]) == result
    count = len(calls)
    assert (
        pilot.construct(output, plan, task, client) == proposal and len(calls) == count
    )
    pilot.report(output)
    return {
        "status": "passed",
        "calls": count,
        "shared_function_count": 1,
        "shared_transfer_parameter_count": 1,
        "validation": result["fit"]["validation"],
        "resume_unchanged": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
