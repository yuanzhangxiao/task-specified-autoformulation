#!/usr/bin/env python3
"""Small linear native test, using a previously training-fitted source export."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import mesh_refinement as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args()
    assert read_seal(args.inputs)["case_name"] == "linear"
    campaign.prepare(
        args.root,
        args.inputs,
        campaign.MeshPolicy(
            targets=(12, 400, None),
            minimum_intervals=1,
            node_seconds=20,
            fixed_seconds=40,
            released_seconds=40,
            point_seconds=20,
            replay_seconds=60,
        ),
    )
    plan, _ = campaign.verify(args.root)
    rows = [campaign.run_task(args.root, i) for i in range(len(plan["tasks"]))]
    for row in rows:
        assert row["status"] == "complete", row
        assert row["selected_evaluation"]["accuracy_passed"], row
        for stage in row["levels"]:
            assert stage["advance_qualified"], stage
            assert stage["endpoint_evaluation"]["status"] == "complete", stage
        if row["arm"] == "continuation":
            assert all(
                s["transfer"] == "old_radau_polynomial" for s in row["levels"][1:]
            )
    assert rows == [campaign.run_task(args.root, i) for i in range(len(rows))]
    assert campaign.report(args.root)["status"] == "complete"
    print(json.dumps({"status": "pass", "rows": len(rows), "exact_resume": True}))


if __name__ == "__main__":
    main()
