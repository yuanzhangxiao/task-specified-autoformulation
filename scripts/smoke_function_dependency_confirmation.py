#!/usr/bin/env python3
"""Offline v4-to-v5 freeze, real dependency repairs, paired fits and resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import function_dependency_confirmation as experiment
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.search.signed_processes import POLICY as SIGNED_POLICY
from scripts import smoke_process_handoff_confirmation as old
from scripts import smoke_signed_processes as signed


def historical_fixture(root: Path) -> tuple[Path, Path, dict]:
    """Build terminal toy v4 artifacts and an actual read-only dependency audit."""
    parent, audit, _ = old.historical_fixture(root / "parent")
    source = root / "v4"
    plan = experiment.previous.freeze(parent, audit, source)
    signed.construct_pair(source, plan)
    for task in plan["tasks"]:
        common = task["construction_task"]
        path = source / "construction/results" / common["task_id"] / "proposal.json"
        if not path.exists():
            sealed_write(
                path,
                {
                    "task": common,
                    "status": "construction_failed",
                    "bundle": None,
                    "attempts": [],
                    "fallback_used": False,
                    "structural": None,
                    "equation_inventory": None,
                    "usage": {
                        "physical_calls": 0,
                        "observed_total_tokens": 0,
                        "unmeasured_calls": 0,
                    },
                },
            )
        p = experiment.pilot.gain_proposal(source, plan, task)
        sealed_write(
            source / "results" / task["task_id"] / "result.json",
            {
                "task": task,
                "proposal_sha256": p["artifact_sha256"],
                "status": "fit_failed" if p["bundle"] else "construction_failed",
                "fit": None,
                "replay": None,
            },
        )
    audit = root / "dependency-audit"
    result = experiment.offline.audit(source, audit)
    assert not result["previously_accepted_now_blocked"]
    return source, audit, plan


def transport_for(calls: list):
    """Exercise both repair mechanisms with prescribed toy scientific decisions."""
    legacy = signed.transport_for(calls)

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        if "selected_term" in payload:
            assert payload["selected_term"]["lhs"] == "q"
            calls.append({"body": body, "payload": payload})
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "expression": "k*h_up",
                                    "parameters": [
                                        {"name": "k", "role": "coefficient"}
                                    ],
                                    "revise_dependencies": True,
                                }
                            )
                        },
                    }
                ],
                "usage": {"total_tokens": 100},
            }
        record = legacy(url, body, timeout)
        reply = json.loads(record["choices"][0]["message"]["content"])
        if payload.get("protocol") == SIGNED_POLICY:
            reply["processes"][0]["depends_on"].append("inflow_up")
        elif payload.get("selected_lhs", {}).get("name") == "h_up":
            for term in reply["terms"]:
                if "inflow_up" in term["sources"]:
                    term["sources"].remove("area_up")
        record["choices"][0]["message"]["content"] = json.dumps(reply)
        return record

    return transport


def construct_pair(root: Path, plan: dict) -> tuple:
    tasks = [
        t
        for t in plan["tasks"]
        if t["case"] == "coupled" and t["seed"] == 0 and t["arm"] == "full"
    ]
    task, calls = tasks[0]["construction_task"], []
    client = experiment.pilot.make_client(
        root / "construction", plan, task, "offline", transport=transport_for(calls)
    )
    p = experiment.pilot.construct(root / "construction", plan, task, client)
    assert p["status"] == "constructed", p
    assert not p["fallback_used"], p
    derived = [experiment.pilot.gain_proposal(root, plan, t) for t in tasks]
    return tasks, p, derived, calls, client


def run(root: Path) -> dict:
    """Confirm actual worker handoff, frozen numerical fits and idempotent resume."""
    source, audit, old_plan = historical_fixture(root)
    output = root / "confirmation"
    plan = experiment.freeze(source, audit, output)
    assert experiment.previous._matched(plan) == experiment.previous._matched(old_plan)
    tasks, common, derived, calls, client = construct_pair(output, plan)
    results = []
    for task in tasks:
        fitted = experiment.pilot.fit_task(output, task["index"])
        assert fitted["status"] == "complete", fitted
        assert fitted["fit"]["validation"]["normalized_mse"] < 1e-4, fitted
        assert fitted["replay"]["replay_agreement"], fitted
        assert experiment.pilot.fit_task(output, task["index"]) == fitted
        results.append(
            {"policy": task["gain_policy"], "fit": fitted["fit"]["validation"]}
        )
    count = len(calls)
    assert (
        experiment.pilot.construct(
            output / "construction", plan, tasks[0]["construction_task"], client
        )
        == common
    )
    assert len(calls) == count
    assert experiment.freeze(source, audit, output) == plan
    report = experiment.report(output)
    assert report["current_status_counts"] == {"complete": 2, "missing": 14}
    assert (
        report["current_accounting"]["atomic_function_repairs"]["physical_calls"] == 1
    )
    assert derived[0]["common_proposal_sha256"] == derived[1]["common_proposal_sha256"]
    return {
        "status": "passed",
        "resume_unchanged": True,
        "calls": count,
        "fits": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
