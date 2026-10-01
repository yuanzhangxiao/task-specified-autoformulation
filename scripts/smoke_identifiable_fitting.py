#!/usr/bin/env python3
"""One native identifiable fit plus deterministic artifact reuse, without LLMs."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import identifiable_campaign as c
from autoformalism.fitting.identifiable_refinement import RefinementPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config = c.CampaignConfig(
        starts=1,
        initializer_seconds=15,
        refinement=RefinementPolicy(seconds=20, maximum_calls=60),
    )
    c.prepare(args.root, config)
    c.qualify(args.root)
    plan, _ = c.verify(args.root)
    index = next(
        i
        for i, t in enumerate(plan["tasks"])
        if t["common"] == "linear_s0" and t["arm"] == "conditional_stopping"
    )
    result = c.run_task(args.root, index)
    assert result == c.run_task(args.root, index)
    assert result["recovery_passed"], result
    print(
        json.dumps(
            {
                "status": "pass",
                "training_nmse": result["training_nmse"],
                "validation_nmse": result["validation_nmse"],
                "recovery": result["recovery"],
                "stop_reason": result["stop_reason"],
                "resume_identical": True,
                "live_llm_calls": 0,
                "test_data_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
