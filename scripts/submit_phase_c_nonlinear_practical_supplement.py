#!/usr/bin/env python3
"""Submit a separately accounted M42 nonlinear diagnostic supplement to Delta."""

import argparse
import json
import os
import sys
from pathlib import Path

from autoformalism.fitting import nonlinear_practical_supplement as campaign

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
) -> dict:
    """Freeze source fits once and never repeat an uncertain scheduler submission."""
    python = os.environ.get("AF_PYTHON", "")
    if not Path(python).is_file() or not 1 <= concurrency <= 8:
        raise ValueError("require an existing Python and concurrency 1..8")
    print(
        f"M42: reading saved M41 diagnostics at {source}; output={root}",
        file=sys.stderr, flush=True,
    )
    root = root.resolve()
    campaign.prepare(
        root, source, campaign.Policy.model_validate_json(config.read_text())
    )
    print(
        "M42: source frozen; verifying immutable endpoints", file=sys.stderr, flush=True
    )
    plan, _ = campaign.verify(root)
    if not plan["tasks"]:
        return {
            "status": "no_pending_diagnostics",
            "jobs": {},
            "array_tasks": 0,
            "gpus": 0,
            "live_llm_calls": 0,
            "test_data_opened": False,
            "parameter_refitting_performed": False,
        }
    print(
        f"M42: {len(plan['tasks'])} diagnostics selected; checking submission receipts",
        file=sys.stderr, flush=True,
    )
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=source,
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name="run_phase_c_nonlinear_practical_supplement_delta.sh",
        job_prefix="nonlinear-profile-supplement",
        run_time="00:10:00",
        prepare_time="00:10:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/phase_c_nonlinear_practical_supplement_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(submit(**vars(args)), indent=2))
