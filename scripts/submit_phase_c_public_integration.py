#!/usr/bin/env python3
"""Submit M29 archived construction fits and known controls to Delta CPUs."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import public_integration_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler


def submit(root, config, *, account, concurrency, inputs):
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=inputs,
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name="run_phase_c_public_integration_delta.sh",
        job_prefix="fit-public",
        run_time="02:30:00",
        prepare_time="00:10:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=3)
    print(json.dumps(submit(**vars(parser.parse_args())), indent=2))
