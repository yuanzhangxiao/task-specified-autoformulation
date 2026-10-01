#!/usr/bin/env python3
"""Freeze, qualify, run or report the Phase C fitting-strategy comparison."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import transcription_campaign as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "qualify", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/phase_c_fitting_strategies_v1.json",
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = campaign.prepare(
            args.root,
            campaign.CampaignConfig.model_validate_json(args.config.read_text()),
            args.inputs,
        )
    elif args.command == "qualify":
        result = campaign.qualify(args.root)
    elif args.command == "run":
        if args.index is None:
            parser.error("run requires --index")
        try:
            result = campaign.run_task(args.root, args.index)
        finally:
            campaign.report(args.root)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
