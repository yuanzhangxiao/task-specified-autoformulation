#!/usr/bin/env python3
"""Submit M33 portable verification and the CSTR supplement to Delta CPUs."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import diagnostic_supplement_campaign as campaign
from scripts import submit_phase_c_generic_recovery as scheduler

REPO = Path(__file__).resolve().parents[1]


def submit(root: Path, source: Path, config: Path, *, account: str) -> dict:
    """Reuse durable scheduler intents; one CPU task, no fitting or GPU work."""
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=1,
        inputs=source,
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name="run_phase_c_diagnostic_supplement_delta.sh",
        job_prefix="diagnostic-supplement",
        run_time="00:45:00",
        prepare_time="00:10:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/phase_c_diagnostic_supplement_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            submit(args.root, args.source, args.config, account=args.account), indent=2
        )
    )
