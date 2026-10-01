#!/usr/bin/env python3
"""Native four-arm smoke on the small linear control, with independent replay."""

import argparse
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting.transcription_fit import StrategyPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="fitting-strategies-"))
    config = c.CampaignConfig(
        starts=1,
        include_cstr=False,
        strategy=StrategyPolicy(seconds=60),
        replay_seconds=60,
    )
    c.prepare(root, config)
    c.qualify(root)
    rows = [c.run_task(root, i) for i in range(4)]
    c.report(root)
    assert all(r["accuracy_passed"] for r in rows), rows
    before = json.dumps(rows, sort_keys=True)
    assert json.dumps([c.run_task(root, i) for i in range(4)], sort_keys=True) == before
    print(
        json.dumps(
            {
                "status": "pass",
                "root": str(root),
                "arms": [
                    {
                        k: r[k]
                        for k in (
                            "arm",
                            "training_nmse",
                            "validation_nmse",
                            "seconds",
                            "recovery_passed",
                        )
                    }
                    for r in rows
                ],
                "exact_resume": True,
                "test_data_opened": False,
                "gpus": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
