#!/usr/bin/env python3
"""Offline v7 construction, independent reconstruction, paired fitting and resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import process_revision_confirmation as experiment
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import smoke_process_assembly_confirmation as old
from scripts import smoke_signed_processes as signed


def historical_fixture(root):
    """Build genuine v6 toy history and audit; no benchmark data or live provider."""
    parent, audit, _ = old.historical_fixture(root / "parent")
    source = root / "v6"
    plan = experiment.parent.freeze(parent, audit, source)
    tasks = {
        t["construction_task"]["task_id"]: t["construction_task"] for t in plan["tasks"]
    }
    for task in tasks.values():
        transport = (
            old.independent_transport([])
            if task["case"] == "independent"
            else old.transport_for([])
        )
        client = experiment.pilot.make_client(
            source / "construction", plan, task, "offline", transport=transport
        )
        experiment.pilot.construct(source / "construction", plan, task, client)
    for task in plan["tasks"]:
        proposal = experiment.pilot.gain_proposal(source, plan, task)
        sealed_write(
            source / "results" / task["task_id"] / "result.json",
            {
                "task": task,
                "proposal_sha256": proposal["artifact_sha256"],
                "status": "fit_failed" if proposal["bundle"] else "construction_failed",
                "fit": None,
                "replay": None,
            },
        )
    audit = root / "revision-audit"
    experiment.offline.audit(source, audit)
    return source, audit, plan


def transport_for(calls, *, independent=False):
    """Prescribed scope error and conversion correction use the real cached client."""
    legacy = (
        old.independent_transport(calls) if independent else signed.transport_for(calls)
    )

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        if "selected_slot" in payload:
            reply = {
                "expression": "q = -k*h_up",
                "parameters": [{"name": "k", "role": "coefficient"}],
            }
        elif "requested_slots" in payload:
            functions = []
            for name, slot in payload["requested_slots"].items():
                parameter = None
                if slot["lhs"] == "q":
                    expression, parameter = "q = -k*h_up/area_up", "k"
                elif "inflow_up" in slot["sources"]:
                    expression = "inflow_up/area_up"
                elif "inflow_down" in slot["sources"]:
                    expression = "inflow_down/area_down"
                elif "h_up" in slot["sources"]:
                    expression, parameter = "k*h_up/area_down", "k"
                else:
                    expression, parameter = "d*h_down/area_down", "d"
                functions.append(
                    {
                        "interaction_id": name,
                        "expression": expression,
                        "parameters": [{"name": parameter, "role": "coefficient"}]
                        if parameter
                        else [],
                    }
                )
            functions.insert(
                0,
                {"interaction_id": "term_99_0", "expression": "typo", "parameters": []},
            )
            reply = {"functions": functions}
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


def construct_pair(root, plan):
    tasks = [
        t
        for t in plan["tasks"]
        if t["case"] == "coupled" and t["seed"] == 0 and t["arm"] == "full"
    ]
    task, calls = tasks[0]["construction_task"], []
    client = experiment.pilot.make_client(
        root / "construction", plan, task, "offline", transport=transport_for(calls)
    )
    proposal = experiment.pilot.construct(root / "construction", plan, task, client)
    assert proposal["status"] == "constructed", proposal
    assert not proposal["fallback_used"], proposal
    derived = [experiment.pilot.gain_proposal(root, plan, t) for t in tasks]
    return tasks, proposal, derived, calls, client


def run(root):
    source, audit, old_plan = historical_fixture(root / "historical")
    output = root / "fresh"
    plan = experiment.freeze(source, audit, output)
    assert experiment._matched(plan) == experiment._matched(old_plan)
    tasks, proposal, derived, calls, client = construct_pair(output, plan)
    count = len(calls)
    assert (
        experiment.pilot.construct(
            output / "construction", plan, tasks[0]["construction_task"], client
        )
        == proposal
    )
    assert len(calls) == count
    fits = []
    for task in tasks:
        result = experiment.pilot.fit_task(output, task["index"])
        assert result["status"] == "complete", result
        assert result["fit"]["validation"]["normalized_mse"] < 1e-4, result
        assert result["replay"]["replay_agreement"]
        assert experiment.pilot.fit_task(output, task["index"]) == result
        fits.append(
            {
                "gain_policy": task["gain_policy"],
                "validation": result["fit"]["validation"],
            }
        )
    summary = experiment.report(output)
    assert summary["current_status_counts"] == {"complete": 2, "missing": 14}
    assert summary["current_accounting"]["physical_calls"] == count
    assert derived[0]["common_proposal_sha256"] == derived[1]["common_proposal_sha256"]
    assert experiment.freeze(source, audit, output) == plan
    return {
        "status": "passed",
        "live_llm_calls": 0,
        "test_data_opened": False,
        "prescribed_calls": count,
        "resume_unchanged": True,
        "fits": fits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
