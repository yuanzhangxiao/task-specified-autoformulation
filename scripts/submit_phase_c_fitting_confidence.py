#!/usr/bin/env python3
"""Submit M19 confidence diagnostics to Delta CPUs with journaled submission."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import confidence_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler


def submit(root, config, *, account, concurrency, inputs):
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=inputs,
        campaign=campaign,
        policy_type=campaign.ConfidencePolicy,
        worker_name="run_phase_c_fitting_confidence_delta.sh",
        job_prefix="fit-confidence",
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
