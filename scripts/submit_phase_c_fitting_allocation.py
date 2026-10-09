#!/usr/bin/env python3
"""Submit M23 evaluator diagnostics, then six matched fitting tasks on Delta CPUs."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import nonlinear_comparison_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler


def submit(root, config, *, account, concurrency, inputs):
    policy = campaign.ComparisonPolicy.model_validate_json(config.read_text())
    if policy.allocation != "measured":
        raise ValueError("M23 requires measured allocation")
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=inputs,
        campaign=campaign,
        policy_type=campaign.ComparisonPolicy,
        worker_name="run_phase_c_fitting_allocation_delta.sh",
        job_prefix="fit-allocation",
        run_time="01:30:00",
        prepare_time="00:30:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=3)
    print(json.dumps(submit(**vars(parser.parse_args())), indent=2))
