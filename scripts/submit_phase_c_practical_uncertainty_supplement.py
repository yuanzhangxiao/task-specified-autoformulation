#!/usr/bin/env python3
"""Submit a separately accounted M40 profile supplement on Delta or ACES CPUs."""

import argparse
import json
import sys
from pathlib import Path

from autoformalism.fitting import practical_uncertainty_supplement as campaign

if __package__:
    from scripts import submit_phase_c_generic_recovery as scheduler
else:
    import submit_phase_c_generic_recovery as scheduler

REPO = Path(__file__).resolve().parents[1]


def submit(
    root: Path,
    source: Path,
    config: Path,
    *,
    account: str,
    concurrency: int = 2,
    platform: str = "delta",
) -> dict:
    """Freeze pending diagnostics and retain idempotent Slurm intent receipts."""
    if platform not in {"delta", "aces"}:
        raise ValueError("platform must be delta or aces")
    print(
        f"M40: auditing saved fits and freezing source into {root}",
        file=sys.stderr,
        flush=True,
    )
    root = root.resolve()
    campaign.prepare(
        root, source, campaign.Policy.model_validate_json(config.read_text())
    )
    print("M40: snapshot prepared; verifying frozen plan", file=sys.stderr, flush=True)
    plan, _ = campaign.verify(root)
    if not plan["tasks"]:
        return {
            "status": "no_pending_diagnostics",
            "jobs": {},
            "array_tasks": 0,
            "gpus": 0,
            "parameter_refitting_performed": False,
        }
    print(
        f"M40: {len(plan['tasks'])} diagnostics selected; checking submission receipts",
        file=sys.stderr,
        flush=True,
    )
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=source,
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name=f"run_phase_c_practical_uncertainty_supplement_{platform}.sh",
        job_prefix="practical-profile-supplement",
        run_time="00:10:00",
        prepare_time="00:10:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/phase_c_practical_uncertainty_supplement_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--platform", choices=("delta", "aces"), default="delta")
    args = parser.parse_args()
    print(json.dumps(submit(**vars(args)), indent=2))
