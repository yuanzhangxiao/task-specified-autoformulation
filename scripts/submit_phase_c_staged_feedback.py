#!/usr/bin/env python3
"""Submit the separate M32 staged feedback qualification to Delta CPUs."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import staged_feedback_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler


def submit(root, config, *, account, concurrency, inputs):
    """Reuse durable scheduler intents and fail closed on uncertain receipts."""
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=inputs,
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name="run_phase_c_staged_feedback_delta.sh",
        job_prefix="staged-feedback",
        run_time="03:00:00",
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
