#!/usr/bin/env python3
"""Prepare, run and report the isolated M45 prefix-window recovery supplement."""

import argparse
import json
import sys
from pathlib import Path

from autoformalism.fitting import prefix_window_supplement as campaign

REPO = Path(__file__).resolve().parents[1]


def _scalars(value: dict) -> dict:
    """Keep receipt values while leaving arrays and detailed ledgers on disk."""
    return {
        key: item
        for key, item in value.items()
        if item is None or isinstance(item, (str, int, float, bool))
    }


def compact(result: dict, command: str, root: Path) -> dict:
    """Return bounded terminal output and the full durable artifact location."""
    value = _scalars(result)
    for name in (
        "status_counts",
        "selection_counts",
        "selection",
        "cost",
        "source_cost",
        "supplemental_cost",
        "combined_cost",
    ):
        if isinstance(result.get(name), dict):
            value[name] = _scalars(result[name])
    if isinstance(result.get("selections"), list):
        value["selections"] = [
            _scalars(item) for item in result["selections"] if isinstance(item, dict)
        ]
    task_id = result.get("task_id") or result.get("identity", {}).get("task", {}).get(
        "task_id"
    )
    if command == "report":
        artifact = root / "summary.json"
    elif command == "run" and task_id:
        artifact = root / "results" / task_id / "result.json"
    else:
        artifact = root / "plan.json"
    return value | {"artifact": str(artifact.resolve())}


def main() -> None:
    """Expose standalone preparation, verification, tasks and compact reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            command.add_argument(
                "--source",
                type=Path,
                help="Saved M44 root; omit only when resuming a prepared supplement.",
            )
            command.add_argument(
                "--config",
                type=Path,
                default=REPO / "configs/phase_c_prefix_window_supplement_v1.json",
            )
        elif name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(f"M45: {args.command} root={args.root}", file=sys.stderr, flush=True)
    if args.command == "prepare":
        print(
            f"M45: auditing saved M44 artifacts; source={args.source or 'snapshot'}",
            file=sys.stderr,
            flush=True,
        )
        result = campaign.prepare(
            args.root,
            args.source,
            campaign.Policy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(args.root)
        result = {"verified": True, "tasks": len(plan["tasks"])}
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(compact(result, args.command, args.root), indent=2))


if __name__ == "__main__":
    main()
