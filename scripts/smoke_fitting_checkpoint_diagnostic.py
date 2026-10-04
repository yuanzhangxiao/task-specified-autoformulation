#!/usr/bin/env python3
"""Exercise all M8 branches, independent replay and exact completed resume."""

import argparse
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting.numerical_diagnostic import NumericalDiagnosticPolicy
from autoformalism.fitting.transcription_fit import StrategyPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="fitting-checkpoint-"))
    c.prepare(
        root,
        c.CampaignConfig(
            starts=1,
            include_cstr=False,
            cases=("linear",),
            strategy=StrategyPolicy(seconds=120, mesh_passes=1),
            numerical_diagnostic=NumericalDiagnosticPolicy(
                target_variables=1000, minimum_intervals=24
            ),
            checkpoint_diagnostic=True,
            replay_seconds=60,
        ),
    )
    c.qualify(root)
    rows = [c.run_task(root, i) for i in range(6)]
    assert all(r["accuracy_passed"] and r["coefficients_recovered"] for r in rows), rows
    assert rows == [c.run_task(root, i) for i in range(6)]
    summary = c.report(root)
    assert summary["recorded"] == 6
    print(
        json.dumps(
            {
                "status": "pass",
                "root": str(root),
                "groups": summary["groups"],
                "exact_resume": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
