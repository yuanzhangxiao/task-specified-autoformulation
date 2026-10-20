#!/usr/bin/env python3
"""Prepare, run and report the isolated M47 prefix noise-reliability campaign."""

import argparse
import json
import sys
from pathlib import Path

from autoformalism.fitting import prefix_noise_reliability as campaign

REPO = Path(__file__).resolve().parents[1]


def _scalars(value: dict) -> dict:
    """Keep receipt values while leaving arrays and detailed ledgers on disk."""
    return {
        key: item
        for key, item in value.items()
        if item is None or isinstance(item, (str, int, float, bool))
    }


def _summary(value: dict) -> dict:
    """Preserve nested summary counts without copying detailed result arrays."""
    return _scalars(value) | {
        key: _summary(item)
        for key, item in value.items()
        if isinstance(item, dict)
    }


def compact(result: dict, command: str, root: Path) -> dict:
    """Return a concise receipt and the complete durable artifact location."""
    value = _scalars(result)
    for name in (
        "status_counts",
        "selection_counts",
        "selection_status_counts",
        "selected_duration_counts",
        "cost",
        "selection",
    ):
        if isinstance(result.get(name), dict):
            value[name] = _summary(result[name])
    if isinstance(result.get("policies"), list):
        value["policies"] = [
            item for item in result["policies"] if isinstance(item, str)
        ]
    for name in ("groups", "selections"):
        if isinstance(result.get(name), list):
            value[name] = [
                _summary(item) for item in result[name] if isinstance(item, dict)
            ]
    identity = result.get("identity")
    task = identity.get("task") if isinstance(identity, dict) else None
    task_id = result.get("task_id") or (
        task.get("task_id") if isinstance(task, dict) else None
    )
    if command == "report":
        artifact = root / "summary.json"
    elif command == "run" and task_id:
        artifact = root / "results" / task_id / "result.json"
    else:
        artifact = root / "plan.json"
    return value | {"artifact": str(artifact.resolve())}


def main() -> None:
    """Expose self-contained preparation, verification, tasks and reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            command.add_argument(
                "--config",
                type=Path,
                default=REPO / "configs/phase_c_prefix_noise_reliability_v1.json",
            )
        elif name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(f"M47: {args.command} root={args.root}", file=sys.stderr, flush=True)
    if args.command == "prepare":
        result = campaign.prepare(
            args.root, campaign.Policy.model_validate_json(args.config.read_text())
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
