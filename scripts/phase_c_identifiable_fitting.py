#!/usr/bin/env python3
"""Freeze, qualify, run and summarize identifiable joint latent fitting controls."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.benchmarks.fitting_qualification_inputs import experiment_request
from autoformalism.fitting import identifiable_campaign as campaign
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicSplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "qualify", "run", "report", "audit-cstr")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/phase_c_identifiable_fitting_v1.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = campaign.prepare(
            args.root,
            campaign.CampaignConfig.model_validate_json(args.config.read_text()),
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
    elif args.command == "audit-cstr":
        if args.inputs is None:
            parser.error("audit-cstr requires milestone-1 --inputs")
        # Diagnostic only: no optimizer, parameter updates, or public benchmark edits.
        inputs = read_seal(args.inputs)
        if inputs["protocol"] != "phase-c-fitting-inputs-1":
            parser.error("expected sealed milestone-1 qualification inputs")
        case = inputs["cases"]["cstr_hard"]
        request, truth = experiment_request(case, "joint", 0, False)
        result = campaign.sensitivity_audit(
            request, PublicSplit.model_validate(case["training"]), truth, args.root
        )
        public._write(args.root / "audit.json", result)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
