#!/usr/bin/env python3
"""Small M9 end-to-end check with a genuinely training-fitted assisted source."""

import argparse
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting.screening_diagnostic import export_starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="fitting-screening-"))
    config = {
        "starts": 1,
        "include_cstr": False,
        "cases": ("linear",),
        "strategy": {"seconds": 120, "mesh_passes": 1},
        "numerical_diagnostic": {"target_variables": 1000},
        "replay_seconds": 60,
    }
    source = root / "source"
    c.prepare(source, c.CampaignConfig(**config))
    c.qualify(source)
    plan, _ = c.verify(source)
    index = next(
        i for i, t in enumerate(plan["tasks"]) if t["arm"] == "rollout_continue"
    )
    row = c.run_task(source, index)
    assert row["arm"] == "rollout_continue" and row["accuracy_passed"], row
    export_starts(source, root / "assisted-inputs.json")
    campaign = root / "campaign"
    c.prepare(
        campaign,
        c.CampaignConfig(
            **config,
            screening_diagnostic={
                "point_seconds": 10,
                "node_seconds": 20,
                "fixed_seconds": 30,
            },
        ),
        root / "assisted-inputs.json",
    )
    c.qualify(campaign)
    rows = [c.run_task(campaign, i) for i in range(8)]
    assert all(r["accuracy_passed"] and r["coefficients_recovered"] for r in rows), rows
    for row in rows[-2:]:
        assert [s["phase"] for s in row["assisted"]["phases"]] == ["fixed", "released"]
        assert all(s["native"]["native_success"] for s in row["assisted"]["phases"])
        assert row["assisted"]["source"]["source_seconds"] > 0
    assert rows == [c.run_task(campaign, i) for i in range(8)]
    summary = c.report(campaign)
    print(
        json.dumps(
            {
                "status": "pass",
                "root": str(root),
                "exact_resume": True,
                "groups": summary["groups"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
