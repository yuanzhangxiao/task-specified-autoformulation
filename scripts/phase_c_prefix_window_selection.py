#!/usr/bin/env python3
"""Prepare, run and report the isolated M44 prefix-window selection campaign."""

import argparse
import json
import sys
from pathlib import Path

from autoformalism.fitting import prefix_window_selection as campaign

REPO = Path(__file__).resolve().parents[1]


def _scalars(value: dict) -> dict:
    """Keep high-level values while omitting numerical arrays and nested records."""
    return {
        key: item
        for key, item in value.items()
        if item is None or isinstance(item, (str, int, float, bool))
    }


def compact(result: dict, command: str, root: Path) -> dict:
    """Print a small receipt; complete campaign artifacts remain on disk."""
    value = _scalars(result)
    for name in (
        "status_counts",
        "selection_counts",
        "selection_status_counts",
        "selected_window_counts",
        "selected_duration_counts",
        "cost",
        "selection",
    ):
        if isinstance(result.get(name), dict):
            value[name] = _scalars(result[name])
    if isinstance(result.get("selections"), list):
        value["selections"] = [
            _scalars(item) for item in result["selections"] if isinstance(item, dict)
        ]
    if command == "report":
        artifact = root / "summary.json"
    elif command == "run" and result.get("task_id"):
        artifact = root / "results" / result["task_id"] / "result.json"
    else:
        artifact = root / "plan.json"
    return value | {"artifact": str(artifact.resolve())}


def main() -> None:
    """Keep preparation, verification, tasks and compact reports standalone."""
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
                default=REPO / "configs/phase_c_prefix_window_selection_v1.json",
            )
        elif name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(f"M44: {args.command} root={args.root}", file=sys.stderr, flush=True)
    if args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.inputs,
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
