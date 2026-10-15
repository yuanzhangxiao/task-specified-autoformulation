#!/usr/bin/env python3
"""Prepare, verify, run and report the isolated M36 sensor-drift qualification."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import sensor_drift_campaign as campaign


def main() -> None:
    """Keep all fitting and reports runnable without a notebook or LLM."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--inputs", type=Path)
            command.add_argument(
                "--config",
                type=Path,
                default=Path(__file__).resolve().parents[1]
                / "configs/phase_c_sensor_drift_v1.json",
            )
        if name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.inputs,
            campaign.Policy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(args.root)
        result = {
            "verified": True,
            "plan_sha256": campaign.public.content_sha256(plan),
            "tasks": len(plan["tasks"]),
            "test_data_opened": False,
            "live_llm_calls": 0,
        }
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
