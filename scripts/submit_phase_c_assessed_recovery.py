#!/usr/bin/env python3
"""Submit M21 matched assessed recovery to Delta CPUs with journaled submission."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import assessed_recovery_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler


def submit(root, config, *, account, concurrency, inputs):
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=inputs,
        campaign=campaign,
        policy_type=campaign.RecoveryPolicy,
        worker_name="run_phase_c_assessed_recovery_delta.sh",
        job_prefix="assessed-recovery",
        run_time="01:00:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(submit(**vars(args)), indent=2))
