#!/usr/bin/env python3
"""Run frozen nine-case functional mechanism tests with no fitting or LLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assess_results_package_mechanisms import adapt
from audit_experiments_results_package import load_package, read_roster, summarize

from autoformalism.rebuttal import mechanism_functional_campaign as campaign
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.prefit_replay import sealed_read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    inputs = prepare.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--package", type=Path)
    inputs.add_argument("--bundle", type=Path)
    prepare.add_argument("--public-root", type=Path, required=True)
    prepare.add_argument(
        "--config",
        type=Path,
        default=campaign.REPO / "configs/mechanism_functional_v1.json",
    )
    run = commands.add_parser("run")
    run.add_argument("--index", type=int, required=True)
    report = commands.add_parser("report")
    for command in (prepare, run, report):
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        config = json.loads(args.config.read_text())
        if args.package:
            rows, manifest, identity = load_package(args.package)
            if manifest["schema_version"] != "phase-b-results-package-2":
                raise ValueError("require the v2 fitted-model package")
            roster = read_roster(
                campaign.REPO / "scripts/build_experiments_table_results.py",
                campaign.REPO / "configs/final_component_campaign_v1.json",
            )
            summarize(rows, roster)  # Roster verification; scores never enter probes.
            sources = [adapt(r, config) for r in rows if r["benchmark_id"] in roster]
        else:
            bundle = sealed_read(args.bundle)
            if bundle["protocol"] != BUNDLE_PROTOCOL:
                raise ValueError("unknown model bundle protocol")
            sources, identity = (
                bundle["rows"],
                {"bundle_sha256": bundle["artifact_sha256"]},
            )
        value = campaign.prepare(sources, config, args.root, args.public_root, identity)
    elif args.command == "run":
        value = campaign.run(args.root, args.index)
    else:
        value = campaign.report(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in value.items()
                if k
                in {
                    "artifact_sha256",
                    "protocol",
                    "status",
                    "status_counts",
                    "methods",
                    "llm_calls",
                    "optimizer_calls",
                    "test_data_opened",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
