#!/usr/bin/env python3
"""Synthetic saved-response audit smoke; no live calls, fitting or rollouts."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import basin_repair_audit as audit
from autoformalism.rebuttal import basin_repair_pilot as old
from scripts import smoke_basin_repair_pilot as previous


def run(root):
    """Exercise immutable source verification, released draft and exact resume."""
    root = root.resolve()
    source, gate = previous.fixture(root / "fixture")
    repair_root = root / "historical_repair"
    plan = old.freeze(source, gate, repair_root)
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    proposal = old.run_one(
        repair_root,
        plan,
        task,
        previous.client(repair_root, plan, task, calls, accept=False),
    )
    assert proposal["status"] == "unconfirmed_trial"
    before = audit._files(repair_root, plan)
    result = audit.audit(repair_root, root / "audit")
    assert result["newly_eligible_saved_drafts"] == [task["task_id"]]
    assert not result["unexpected_acceptance_regressions"]
    assert result == audit.audit(repair_root, root / "audit")
    assert before == audit._files(repair_root, plan)
    return {
        "status": "passed",
        "live_llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "prescribed_historical_calls": len(calls),
        "historical_bytes_unchanged": True,
        "resume_unchanged": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
