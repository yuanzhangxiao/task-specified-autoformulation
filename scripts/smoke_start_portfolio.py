#!/usr/bin/env python3
"""Small native M13 fitting/evaluation smoke, with deterministic terminal resume."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import generic_recovery as generic
from autoformalism.fitting import portfolio_campaign as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    assert read_seal(args.source)["case_name"] == "linear"
    args.root.mkdir(parents=True, exist_ok=True)
    inputs = args.root / "generic-inputs.json"
    generic.export_inputs(args.source, inputs)
    campaign.prepare(
        args.root,
        inputs,
        campaign.PortfolioPolicy(
            seconds=120,
            certificate_seconds=20,
            native_seconds=15,
            point_seconds=5,
            probe_seconds=15,
            continuation_seconds=25,
            probe_calls=20,
            continuation_calls=50,
            minimum_intervals=1,
            observation_anchors=2,
            targets=(12, 400, None),
            medium_target=400,
            replay_seconds=60,
        ),
    )
    rows = [campaign.run_task(args.root, i) for i in range(3)]
    for row in rows:
        assert row["status"] == "complete", row
        assert row["evaluation"]["accuracy_passed"], row
        assert row["training_prediction_certified"], row
        assert row["fit_seconds"] < 121, row
    assert rows == [campaign.run_task(args.root, i) for i in range(3)]
    assert campaign.report(args.root)["recorded"] == 3
    print(
        json.dumps(
            {
                "status": "pass",
                "tasks": 3,
                "exact_resume": True,
                "trials": [
                    sum(len(c["trials"]) for c in r.get("portfolio", [])) for r in rows
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
