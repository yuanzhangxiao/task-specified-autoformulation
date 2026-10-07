#!/usr/bin/env python3
"""Submit twenty-four M18 CPU tasks using the existing journaled Delta launcher."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import larger_coupled_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler


def submit(root, config, *, account, concurrency, inputs):
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=inputs,
        campaign=campaign,
        policy_type=campaign.PolishingPolicy,
        worker_name="run_phase_c_larger_coupled_delta.sh",
        job_prefix="larger-coupled",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--inputs", type=Path, default=scheduler.REPO / "larger-coupled-inputs.json"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=scheduler.REPO / "configs/phase_c_larger_coupled_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                args.root,
                args.config,
                account=args.account,
                concurrency=args.concurrency,
                inputs=args.inputs,
            ),
            indent=2,
        )
    )
