#!/usr/bin/env python3
"""Nine-benchmark critic/verifier matrix, persistent workers and sealed evaluation."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from autoformalism.rebuttal import component_campaign as campaign
from autoformalism.rebuttal import component_critic, component_workers
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_reporting as reporting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "matrix",
            "stage-public",
            "prepare",
            "verify",
            "authorize-critic",
            "work",
            "run",
            "report",
            "export",
            "evaluate",
            "evaluation-report",
            "merge",
        ),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=io.REPO / "configs/final_component_campaign_v1.json",
    )
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument(
        "--platform", choices=("aces-h100x1", "delta-a40x1", "koa-h100x1")
    )
    parser.add_argument(
        "--blocks", help="whole cell/seed/prompt blocks, comma-separated 0..35"
    )
    parser.add_argument("--calibration-plan", type=Path)
    parser.add_argument("--calibration-summary", type=Path)
    parser.add_argument("--stage", choices=component_workers.STAGES)
    parser.add_argument("--base-url")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    # Compatibility with the existing pinned local vLLM launcher.
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = (args.root or (args.plan.parent if args.plan else Path("."))).resolve()
    if args.command not in {"matrix", "run"} and args.root is None:
        parser.error("--root is required")
    if args.command == "matrix":
        config = io.DeadlineConfig.model_validate_json(args.config.read_text())
        if args.blocks:
            config = io.DeadlineConfig.model_validate(
                {
                    **config.model_dump(mode="json"),
                    "campaign_blocks": [int(i) for i in args.blocks.split(",")],
                }
            )
        tasks = campaign.tasks(config)
        value = {
            "benchmarks": campaign.CELLS,
            "lineages": len(tasks),
            "primary": sum(not t["secondary"] for t in tasks),
            "secondary": sum(t["secondary"] for t in tasks),
            "rounds": config.rounds,
            "blocks": sorted({t["block"] for t in tasks}),
            "tasks": tasks,
        }
    elif args.command == "stage-public":
        if not args.source:
            parser.error("one or more --source public data roots required")
        value = campaign.stage_public(args.source, root)
    elif args.command == "prepare":
        if args.public_root is None:
            parser.error("--public-root required")
        config_path = args.config
        if args.blocks or args.platform:
            # Freeze an explicit shard; never alter a running plan's task list.
            data = json.loads(config_path.read_text())
            if args.blocks:
                data["campaign_blocks"] = [int(i) for i in args.blocks.split(",")]
            if args.platform:
                data["platform"] = args.platform
            config = io.DeadlineConfig.model_validate(data)
            root.mkdir(parents=True, exist_ok=True)
            config_path = root / "effective_config.json"
            payload = config.model_dump_json(indent=2)
            if config_path.exists() and config_path.read_text() != payload:
                raise ValueError("existing effective configuration differs")
            config_path.write_text(payload)
        plan = campaign.freeze(config_path, args.public_root.resolve(), root)
        value = {
            "identity": plan["artifact_sha256"],
            "lineages": len(plan["tasks"]),
            "rounds": plan["config"]["rounds"],
        }
    elif args.command == "merge":
        value = campaign.merge(args.source, root)
    elif args.command == "verify":
        value = {"identity": campaign.verify(root)["artifact_sha256"]}
    elif args.command == "authorize-critic":
        if not args.calibration_plan or not args.calibration_summary:
            parser.error("--calibration-plan and --calibration-summary required")
        value = component_critic.authorize(
            root, args.calibration_plan, args.calibration_summary
        )
    elif args.command in {"run", "work"}:
        stage = "propose" if args.command == "run" else args.stage
        if not stage:
            parser.error("--stage required")
        key = None
        if stage == "critic":
            key = os.environ.get("AF_JETSTREAM_API_KEY") or getpass.getpass(
                "Jetstream API key (not saved): "
            )
            if not key:
                raise ValueError("API key required")
        value = component_workers.run(
            root,
            stage,
            seconds=args.wall_seconds,
            base_url=args.base_url,
            once=args.once,
            key_supplier=(lambda: key) if key else None,
        )
    elif args.command == "report":
        value = campaign.report(root)
        value = {k: v for k, v in value.items() if k != "rows"}
    elif args.command == "export":
        value = campaign.export(root)
        value = {k: v for k, v in value.items() if k != "roster"}
    elif args.command == "evaluate":
        if args.public_root is None:
            parser.error(
                "--public-root with matching public files and held-out split required"
            )
        campaign.verify(root, execution=False)
        value = reporting.evaluate(root, args.public_root, args.shard, args.shards)
    else:
        value = campaign.evaluation_report(root)
        value = {k: v for k, v in value.items() if k != "rows"}
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
