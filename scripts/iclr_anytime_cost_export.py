"""Read-only, standard-library export of recorded Full search checkpoint costs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def checked(path: Path) -> dict:
    """Authenticate an existing campaign receipt without importing its runtime."""
    value = json.loads(path.read_text())
    digest = value.get("artifact_sha256")
    content = {k: v for k, v in value.items() if k != "artifact_sha256"}
    if (
        hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        != digest
    ):
        raise ValueError(f"receipt checksum differs: {path}")
    return value


def export(root: Path) -> dict:
    """Accumulate actual recorded costs, retaining missing-receipt counts."""
    plan = checked(root / "plan.json")
    rows = []
    for block in plan["blocks"]:
        for task in block["tasks"]:
            if task["treatment"] != "full":
                continue
            totals = {
                "proposer_tokens": 0,
                "proposer_calls": 0,
                "unknown_token_calls": 0,
                "fit_seconds": 0.0,
                "total_seconds": 0.0,
                "unknown_visit_costs": 0,
            }
            for index in range(plan["visits"]):
                directory = root / "results" / task["task_id"] / f"round_{index:02d}"
                result = checked(directory / "result.json")
                cost = result["cost"]
                totals["proposer_tokens"] += cost.get("observed_tokens", 0)
                totals["proposer_calls"] += cost.get("physical_requests", 0)
                totals["unknown_token_calls"] += cost.get(
                    "requests_with_unknown_usage", 0
                )
                if (directory / "cost.json").exists():
                    clock = checked(directory / "cost.json")
                    totals["fit_seconds"] += clock["fit_seconds"]
                    totals["total_seconds"] += clock["visit_seconds"]
                else:
                    totals["unknown_visit_costs"] += 1
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "condition": task["cell"],
                        "seed": task["seed"],
                        "checkpoint": str(index + 1),
                        "result_sha256": result["artifact_sha256"],
                        **totals,
                    }
                )
            path = root / "pruning" / task["task_id"] / "cost.json"
            pruning = checked(path).get("pruning_seconds") if path.exists() else None
            rows.append(
                {
                    "task_id": task["task_id"],
                    "condition": task["cell"],
                    "seed": task["seed"],
                    "checkpoint": "pruned",
                    **totals,
                    "total_seconds": totals["total_seconds"] + (pruning or 0),
                    "unknown_pruning_cost": pruning is None,
                }
            )
    value = {
        "protocol": "iclr-anytime-cost-export-1",
        "root": str(root),
        "plan_sha256": plan["artifact_sha256"],
        "rows": rows,
        "test_data_opened": False,
        "cost_scope": (
            "Recorded visits plus pruning; no queue, startup, or evaluation time"
        ),
    }
    return value | {
        "artifact_sha256": hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
