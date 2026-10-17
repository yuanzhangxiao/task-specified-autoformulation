#!/usr/bin/env python3
"""Journal M41 prepare -> 36 CPU tasks -> afterany report on Delta."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import nonlinear_practical_campaign as campaign

if __package__:
    from scripts import submit_phase_c_generic_recovery as scheduler
else:
    import submit_phase_c_generic_recovery as scheduler


def submit(
    root: Path,
    config: Path,
    *,
    account: str,
    concurrency: int = 2,
    inputs: Path | None = None,
) -> dict:
    """Reuse durable submission intents; never resubmit an uncertain receipt."""
    root = root.resolve()
    policy = campaign.Policy.model_validate_json(config.read_text())
    if inputs is None and (root / "inputs.json").exists():
        inputs = root / "inputs.json"
    print(f"Preparing sealed M41 inputs and plan: {root}", flush=True)
    campaign.prepare(root, inputs, policy)
    print("Plan ready; verifying identity and submitting CPU stages.", flush=True)
    return scheduler.submit(
        root,
        config,
        account=account,
        concurrency=concurrency,
        inputs=root / "inputs.json",
        campaign=campaign,
        policy_type=campaign.Policy,
        worker_name="run_phase_c_nonlinear_practical_delta.sh",
        job_prefix="nonlinear-practical-fit",
        run_time="00:15:00",
        prepare_time="00:15:00",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(submit(**vars(args)), indent=2))
