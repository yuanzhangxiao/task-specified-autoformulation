#!/usr/bin/env python3
"""Offline v5-to-v6 audit gate, fresh assembly repair, paired fits and resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import process_assembly_confirmation as experiment
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import smoke_function_dependency_confirmation as old
from scripts import smoke_signed_processes as signed


def independent_transport(calls: list):
    """Prescribed local-only topology for the independent toy inventory."""
    legacy = signed.transport_for(calls, "empty")

    def transport(url, body, timeout):
        record = legacy(url, body, timeout)
        if "selected_lhs" in calls[-1]["payload"]:
            reply = json.loads(record["choices"][0]["message"]["content"])
            reply["terms"] = [t for t in reply["terms"] if "h_up" not in t["sources"]]
            record["choices"][0]["message"]["content"] = json.dumps(reply)
        return record

    return transport


def historical_fixture(root: Path) -> tuple[Path, Path, dict]:
    """Prepare eight real toy function histories; retain terminal failed fits."""
    parent, audit, _ = old.historical_fixture(root / "parent")
    source = root / "v5"
    plan = experiment.previous.freeze(parent, audit, source)
    tasks = {
        t["construction_task"]["task_id"]: t["construction_task"] for t in plan["tasks"]
    }
    for task in tasks.values():
        transport = (
            independent_transport([])
            if task["case"] == "independent"
            else old.transport_for([])
        )
        client = experiment.pilot.make_client(
            source / "construction", plan, task, "offline", transport=transport
        )
        experiment.pilot.construct(source / "construction", plan, task, client)
    for task in plan["tasks"]:
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
    audit = root / "assembly-audit"
    result = experiment.offline.audit(source, audit)
    assert not result["missing_constructions"], result
    assert not result["unexpected_acceptance_regressions"], result
    return source, audit, plan


def transport_for(calls: list):
    """Deliberately duplicate a consumer conversion, then make a fresh correction."""
    legacy = old.transport_for(calls)

    def transport(url, body, timeout):
        record = legacy(url, body, timeout)
        payload = calls[-1]["payload"]
        reply = json.loads(record["choices"][0]["message"]["content"])
        if "selected_term" in payload:
            assert "CONSUMER_CONVERSION_OVERLAP" in json.dumps(payload)
            reply["expression"] = "-k*h_up"
            reply["conversion_factor_is_intrinsic"] = False
        elif payload.get("selected_equation", {}).get("lhs") == "q":
            reply["functions"][0]["expression"] = "-k*h_up/area_up"
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
    """Exercise the actual campaign handoff and frozen fitter on a separate toy."""
    source, audit, old_plan = historical_fixture(root)
    output = root / "confirmation"
    plan = experiment.freeze(source, audit, output)
    assert experiment._matched(plan) == experiment._matched(old_plan)
    tasks, common, derived, calls, client = construct_pair(output, plan)
    results = []
    for task in tasks:
        fitted = experiment.pilot.fit_task(output, task["index"])
        assert fitted["status"] == "complete", fitted
        assert fitted["fit"]["validation"]["normalized_mse"] < 1e-4, fitted
        assert fitted["replay"]["replay_agreement"], fitted
        assert experiment.pilot.fit_task(output, task["index"]) == fitted
        results.append(
            {"policy": task["gain_policy"], "validation": fitted["fit"]["validation"]}
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
    assert report["current_accounting"]["physical_calls"] == count
    assert report["current_accounting"]["assembly"]["outer_sign_normalized_slots"] == 1
    assert report["current_accounting"]["assembly"]["conversion_rejected_attempts"] == 1
    assert derived[0]["common_proposal_sha256"] == derived[1]["common_proposal_sha256"]
    return {
        "status": "passed",
        "resume_unchanged": True,
        "prescribed_calls": count,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "fits": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
