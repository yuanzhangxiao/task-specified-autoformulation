#!/usr/bin/env python3
"""Check M18 execution/resume; report recovery separately, including stalled fits."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import larger_coupled_campaign as campaign
from autoformalism.fitting import public_fitting as public


def run(root: Path, inputs: Path) -> dict:
    policy = campaign.PolishingPolicy.model_validate_json(
        Path("configs/phase_c_larger_coupled_v1.json").read_text()
    )
    campaign.prepare(root, inputs, policy)
    campaign.qualify(root)
    plan, _ = campaign.verify(root)
    rows = []
    for index, task in enumerate(plan["tasks"]):
        if task["common"] not in {"linear3_s0", "linear6_s0"}:
            continue
        row = campaign.run_task(root, index)
        assert row["status"] == "complete", row
        folder = root / "results" / row["task_id"]
        backend = read_seal(folder / "backend.json")
        assert backend["actual_residual_calls"] <= policy.maximum_rollout_calls
        # Numerical recovery is the experiment's outcome, not an execution gate.
        # A terminal stalled fit with complete finite replay stays in the report.
        assert row["evaluation"]["status"] == "complete", row
        operations = campaign.coupled_polishing.timing_records(folder)
        assert operations and all(op["timing"] for op in operations)
        stamps = {p: p.stat().st_mtime_ns for p in folder.rglob("process.json")}
        assert campaign.run_task(root, index) == row
        assert stamps == {p: p.stat().st_mtime_ns for p in stamps}
        rows.append(
            {key: row[key] for key in ("task_id", "fit_seconds", "evaluation")}
            | {"strict_prediction_certified": backend["strict_prediction_certified"]}
        )
    result = {"status": "execution_pass", "resume_spent_no_budget": True, "rows": rows}
    public._write(root / "smoke.json", result)
    campaign.report(root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.inputs), indent=2))
