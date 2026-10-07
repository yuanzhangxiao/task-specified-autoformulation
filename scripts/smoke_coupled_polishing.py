#!/usr/bin/env python3
"""Exercise timed repeats and polishing on the small frozen coupled controls."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import coupled_polishing as campaign
from autoformalism.fitting import public_fitting as public


def run(root: Path, inputs: Path) -> dict:
    policy = campaign.PolishingPolicy.model_validate_json(
        Path("configs/phase_c_coupled_polishing_v1.json").read_text()
    )
    campaign.prepare(root, inputs, policy)
    plan, _ = campaign.verify(root)
    rows = []
    for index, task in enumerate(plan["tasks"]):
        if not (
            task["arm"].startswith("repeat_")
            or task["common"] == "coupled_fast_slow_s0"
        ):
            continue
        row = campaign.run_task(root, index)
        assert row["status"] == "complete" and row["training_prediction_certified"], row
        folder = root / "results" / row["task_id"]
        backend = read_seal(folder / "backend.json")
        operations = campaign.timing_records(folder)
        assert operations and all(op["timing"] for op in operations)
        assert all(op["status"] == "complete" for op in operations)
        if task["arm"].startswith("polish_"):
            assert backend["strict_prediction_certified"]
            assert backend["actual_residual_calls"] <= policy.maximum_rollout_calls
            assert row["evaluation"]["coefficients_recovered"]
            assert row["evaluation"]["initials_recovered"]
        receipts = list(folder.rglob("process.json"))
        stamps = {path: path.stat().st_mtime_ns for path in receipts}
        assert campaign.run_task(root, index) == row
        assert stamps == {path: path.stat().st_mtime_ns for path in receipts}
        rows.append({key: row[key] for key in ("task_id", "fit_seconds", "evaluation")})
    result = {"status": "pass", "resume_spent_no_budget": True, "rows": rows}
    public._write(root / "smoke.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.inputs), indent=2))
