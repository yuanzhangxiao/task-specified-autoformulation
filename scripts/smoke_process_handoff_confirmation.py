#!/usr/bin/env python3
"""Offline confirmation gate, fresh construction, two fits and deterministic resume."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import process_handoff_confirmation as experiment
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import smoke_signed_processes as signed


def historical_fixture(root):
    """Temporary failed historical campaign; no scientific recovery is invented."""
    source, plan = signed.fixture(root)
    for task in plan["tasks"]:
        common = task["construction_task"]
        sealed_write(
            source / "construction/results" / common["task_id"] / "proposal.json",
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
        experiment.pilot.gain_proposal(source, plan, task)
        experiment.pilot.fit_task(source, task["index"])
    audit = root / "audit"
    experiment.offline.audit(source, audit)
    return source, audit, plan


def run(root):
    source, audit, old = historical_fixture(root)
    output = root / "confirmation"
    plan = experiment.freeze(source, audit, output)
    assert experiment._matched(plan) == experiment._matched(old)
    tasks, common, derived, calls, client = signed.construct_pair(output, plan)
    assert calls, "Fresh construction must not reuse historical requests."
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
    assert derived[0]["common_proposal_sha256"] == derived[1]["common_proposal_sha256"]
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
