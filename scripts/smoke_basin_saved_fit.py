#!/usr/bin/env python3
"""Real frozen-fitter smoke from cached synthetic replies, including exact resume."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import basin_repair_audit as audit
from autoformalism.rebuttal import basin_repair_pilot as old
from autoformalism.rebuttal import basin_saved_fit as io
from scripts import smoke_basin_repair_pilot as previous


def run(root: Path) -> dict:
    """Fit one released draft; assert replay and immutable source on resume."""
    root = root.resolve()
    source, gate = previous.fixture(root / "fixture")
    history = root / "history"
    plan = old.freeze(source, gate, history)
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    old.run_one(
        history, plan, task, previous.client(history, plan, task, calls, accept=False)
    )
    before = audit._files(history, plan)
    gate = root / "handoff-audit"
    audit.audit(history, gate)
    output = root / "saved-fit"
    frozen = io.freeze(history, gate, output)
    assert frozen["maximum_new_fits"] == 1
    result = io.fit_task(output, task["index"])
    assert result["status"] == "complete", result
    assert result["replay"]["replay_agreement"], result
    assert io.fit_task(output, task["index"]) == result
    assert io.freeze(history, gate, output) == frozen
    assert io.report(output)["new_fit_results"] == 1
    assert audit._files(history, plan) == before
    return {
        "status": "passed",
        "live_llm_calls": 0,
        "child_fits": 1,
        "prescribed_historical_calls": len(calls),
        "numerical_replay_agreement": True,
        "historical_bytes_unchanged": True,
        "resume_unchanged": True,
        "prediction_accuracy_not_asserted": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
