#!/usr/bin/env python3
"""Submit four frozen M43 qualification tasks to existing Delta CPU resources."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import initialization_qualification as campaign

if __package__:
    from scripts import submit_phase_c_generic_recovery as scheduler
else:
    import submit_phase_c_generic_recovery as scheduler

REPO = Path(__file__).resolve().parents[1]


def submit(
    root: Path,
    config: Path,
    *,
    account: str = "bibo-delta-cpu",
    concurrency: int = 2,
    inputs: Path | None = None,
) -> dict:
    """Freeze actual synthetic arrays once and preserve all Slurm intent receipts."""
    root = root.resolve()
    print(f"M43: preparing four fixed qualification tasks at {root}", flush=True)
    campaign.prepare(
        root, inputs, campaign.Policy.model_validate_json(config.read_text())
    )
    print("M43: submitting bounded CPU stages", flush=True)
    return scheduler.submit(
        root,
        config,
        inputs=root / "inputs.json",
        account=account,
        concurrency=concurrency,
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name="run_phase_c_initialization_qualification_delta.sh",
        job_prefix="initialization-qualification",
        run_time="00:10:00",
        prepare_time="00:10:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/phase_c_initialization_qualification_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    print(json.dumps(submit(**vars(parser.parse_args())), indent=2))
