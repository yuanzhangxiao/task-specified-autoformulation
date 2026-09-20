#!/usr/bin/env python3
"""Offline signed construction, gain compilation, fit, replay and resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.search import signed_processes as signed
from scripts import smoke_detention_process_pilot as old


def transport_for(calls, mode="transfer"):
    """Prescribed public scientific choices, not LLM calls or benchmark labels."""
    legacy = old.transport_for(calls)

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        if payload.get("protocol") == signed.POLICY:
            p = {
                "name": "q",
                "depends_on": ["h_up"],
                "kind": "transfer",
                "scientific_meaning": "volumetric transfer",
                "uses": [
                    {"target": "h_up", "sign": "negative", "conversion": "1/area_up"},
                    {
                        "target": "h_down",
                        "sign": "positive",
                        "conversion": "1/area_down",
                    },
                ],
            }
            if mode == "unknown_conversion":
                p["uses"][1]["conversion"] = None
            if mode == "nonlinear_prose":
                p["scientific_meaning"] = "Nonlinear saturating sigmoid transfer."
            reply = {"processes": [] if mode == "empty" else [p]}
        elif "selected_equation" in payload:
            functions = []
            eq = payload["selected_equation"]
            for term in eq["terms"]:
                assert "shared_process_use" not in term
                sources = term["sources"]
                parameter = None
                if eq["lhs"] == "q":
                    expression, parameter = (
                        "-k*h_up" if mode == "signed_law" else "k*h_up",
                        "k",
                    )
                elif "inflow_up" in sources:
                    expression = "inflow_up/area_up"
                elif "inflow_down" in sources:
                    expression = "inflow_down/area_down"
                elif "h_up" in sources:
                    expression = "k*h_up/" + next(s for s in sources if s != "h_up")
                    parameter = "k"
                else:
                    expression, parameter = "d*h_down/area_down", "d"
                functions.append(
                    {
                        "expression": expression,
                        "parameters": [{"name": parameter, "role": "coefficient"}]
                        if parameter
                        else [],
                    }
                )
            reply = {"functions": functions}
        elif "selected_lhs" in payload:
            raw = legacy(url, body, timeout)
            reply = json.loads(raw["choices"][0]["message"]["content"])
            if any(v["name"] == "q" for v in payload["frozen_inventory"]):
                reply["terms"] = [t for t in reply["terms"] if "q" not in t["sources"]]
            raw["choices"][0]["message"]["content"] = json.dumps(reply)
            return raw
        else:
            return legacy(url, body, timeout)
        calls.append({"body": body, "payload": payload})
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return transport


def fixture(root):
    """Import identical inventories from a separately generated public fixture."""
    pilot = old.pilot
    source, parent, output = (root / n for n in ("source", "parent", "signed"))
    historical = old.fixture(source, parent)
    for task in historical["tasks"]:
        if task["review"]:
            pilot.construct(
                parent,
                historical,
                task,
                pilot.make_client(
                    parent,
                    historical,
                    task,
                    "offline",
                    transport=old.transport_for(
                        [], "independent" if task["case"] == "independent" else "add"
                    ),
                ),
            )
    plan = pilot.freeze(
        source, output, pilot.REPO / "configs/detention_process_pilot_v3.json", parent
    )
    return output, plan


def construct_pair(root, plan, mode="transfer"):
    pilot = old.pilot
    tasks = [
        t
        for t in plan["tasks"]
        if t["case"] == "coupled" and t["seed"] == 0 and t["arm"] == "full"
    ]
    task = tasks[0]["construction_task"]
    calls = []
    client = pilot.make_client(
        root / "construction",
        plan,
        task,
        "offline",
        transport=transport_for(calls, mode),
    )
    proposal = pilot.construct(root / "construction", plan, task, client)
    assert proposal["status"] == "constructed", proposal
    assert not proposal["fallback_used"], proposal
    derived = [pilot.gain_proposal(root, plan, t) for t in tasks]
    return tasks, proposal, derived, calls, client


def run(root):
    """Both policies recover outputs and resume without new calls or fits."""
    pilot = old.pilot
    output, plan = fixture(root)
    tasks, common, derived, calls, client = construct_pair(output, plan)
    results = []
    for task in tasks:
        fitted = pilot.fit_task(output, task["index"])
        assert fitted["status"] == "complete", fitted
        assert fitted["fit"]["validation"]["normalized_mse"] < 1e-4, fitted
        assert fitted["replay"]["replay_agreement"], fitted
        assert pilot.fit_task(output, task["index"]) == fitted
        results.append(
            {"policy": task["gain_policy"], "validation": fitted["fit"]["validation"]}
        )
    count = len(calls)
    assert (
        pilot.construct(
            output / "construction", plan, tasks[0]["construction_task"], client
        )
        == common
    )
    assert len(calls) == count
    for task, proposal in zip(tasks, derived, strict=True):
        assert pilot.gain_proposal(output, plan, task) == proposal
    pilot.report(output)
    return {
        "status": "passed",
        "resume_unchanged": True,
        "calls": count,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
