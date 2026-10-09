#!/usr/bin/env python3
"""Submit M22 nonlinear fitting comparison to Delta CPUs with journaled submission."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import nonlinear_comparison_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler


def submit(root, config, *, account, concurrency, inputs):
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=inputs,
        campaign=campaign,
        policy_type=campaign.ComparisonPolicy,
        worker_name="run_phase_c_nonlinear_comparison_delta.sh",
        job_prefix="nonlinear-conditional",
        run_time="01:30:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(submit(**vars(args)), indent=2))
