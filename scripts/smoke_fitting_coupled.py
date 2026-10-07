#!/usr/bin/env python3
"""Small real coupled-ODE fit, original-equation replay and no-cost terminal resume."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import coupled_campaign as campaign
from autoformalism.fitting import public_fitting as public


def run(root: Path) -> dict:
    campaign.export_inputs(root / "source.json")
    campaign.prepare(
        root,
        root / "source.json",
        campaign.CoupledPolicy(
            seconds=90,
            certificate_seconds=15,
            replay_seconds=30,
            training_nmse=1e-8,
            trajectory_nmse=1e-7,
        ),
    )
    plan, _ = campaign.verify(root)
    rows = []
    for index, task in enumerate(plan["tasks"]):
        if task["common"] != "coupled_linear_s0":
            continue
        row = campaign.run_task(root, index)
        assert row["status"] == "complete" and row["training_prediction_certified"], row
        assert all(
            row["evaluation"][key]
            for key in (
                "accuracy_passed",
                "coefficients_recovered",
                "initials_recovered",
            )
        ), row
        paths = list((root / "results" / task["task_id"]).rglob("process.json"))
        assert paths and all(read_seal(p)["termination_confirmed"] for p in paths)
        stamps = {p: p.stat().st_mtime_ns for p in paths}
        assert campaign.run_task(root, index) == row
        assert stamps == {p: p.stat().st_mtime_ns for p in paths}
        rows.append({k: row[k] for k in ("task_id", "fit_seconds", "evaluation")})
    result = {"status": "pass", "rows": rows, "resume_spent_no_budget": True}
    public._write(root / "smoke.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().root), indent=2))
