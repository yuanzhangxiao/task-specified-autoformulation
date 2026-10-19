#!/usr/bin/env python3
"""Submit frozen M44 prefix-window tasks to the existing Delta CPU environment."""

import argparse
import json
import sys
from pathlib import Path

from autoformalism.fitting import prefix_window_selection as campaign

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
    """Freeze actual inputs once and retain the scheduler's no-duplicate receipts."""
    root = root.resolve()
    print(f"M44: preparing frozen tasks at {root}", file=sys.stderr, flush=True)
    campaign.prepare(
        root, inputs, campaign.Policy.model_validate_json(config.read_text())
    )
    plan, _ = campaign.verify(root)
    if not plan["tasks"]:
        return {"status": "no_tasks", "jobs": {}, "array_tasks": 0, "gpus": 0}
    print(
        f"M44: {len(plan['tasks'])} tasks; checking CPU submission receipts",
        file=sys.stderr,
        flush=True,
    )
    return scheduler.submit(
        root,
        config,
        inputs=root / "inputs.json",
        account=account,
        concurrency=concurrency,
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name="run_phase_c_prefix_window_selection_delta.sh",
        job_prefix="prefix-window-selection",
        run_time="00:30:00",
        prepare_time="00:10:00",
    )


def main() -> None:
    """Print only scheduler status and the location of complete submission receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/phase_c_prefix_window_selection_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    result = submit(**vars(args))
    fields = (
        "status",
        "jobs",
        "array_tasks",
        "gpus",
        "live_llm_calls",
        "test_data_opened",
    )
    receipt = {name: result[name] for name in fields if name in result}
    receipt["artifact"] = str((args.root / "submission_manifest.json").resolve())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
