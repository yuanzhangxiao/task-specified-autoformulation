#!/usr/bin/env python3
"""Exercise M10 on an exported, previously training-fitted small linear case."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import screening_replay as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args()
    assert read_seal(args.inputs)["case_name"] == "linear"
    campaign.prepare(
        args.root,
        args.inputs,
        campaign.ReplayPolicy(
            calibration_seconds=20,
            minimum_point_seconds=15,
            maximum_point_seconds=60,
            pool_seconds=60,
            assisted_seconds=180,
            node_seconds=20,
            fixed_seconds=30,
            replay_seconds=60,
        ),
    )
    calibration = [campaign.calibrate(args.root, i) for i in range(2)]
    assert all(c["ready"] for c in calibration), calibration
    plan, _ = campaign.verify(args.root)
    rows = [campaign.run_task(args.root, i) for i in range(len(plan["tasks"]))]
    assert all(r["accuracy_passed"] and r["coefficients_recovered"] for r in rows), rows
    for row in rows:
        if row["kind"] == "assisted":
            assert [p["phase"] for p in row["assisted"]["phases"]] == [
                "fixed",
                "released",
            ]
            assert row["released_final_screen_status"] == "complete"
    assert calibration == [campaign.calibrate(args.root, i) for i in range(2)]
    assert rows == [campaign.run_task(args.root, i) for i in range(len(rows))]
    report = campaign.report(args.root)
    assert report["status"] == "complete"
    print(json.dumps({"status": "pass", "rows": len(rows), "exact_resume": True}))


if __name__ == "__main__":
    main()
