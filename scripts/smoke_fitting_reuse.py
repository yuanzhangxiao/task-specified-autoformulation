#!/usr/bin/env python3
"""Four native reuse arms, independent rollout checks and exact completed resume."""

import argparse
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting.reuse_diagnostic import ReuseDiagnosticPolicy
from autoformalism.fitting.transcription_fit import StrategyPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="fitting-reuse-"))
    c.prepare(
        root,
        c.CampaignConfig(
            starts=1,
            include_cstr=False,
            cases=("linear",),
            strategy=StrategyPolicy(seconds=120),
            reuse_diagnostic=ReuseDiagnosticPolicy(native_seconds=30, point_seconds=2),
            replay_seconds=60,
        ),
    )
    c.qualify(root)
    rows = [c.run_task(root, i) for i in range(4)]
    assert all(r["accuracy_passed"] and r["coefficients_recovered"] for r in rows), rows
    assert rows == [c.run_task(root, i) for i in range(4)]
    summary = c.report(root)
    assert summary["recorded"] == 4
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
