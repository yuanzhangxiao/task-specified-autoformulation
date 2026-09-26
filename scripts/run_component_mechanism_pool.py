#!/usr/bin/env python3
"""Assess a deterministic shard of frozen models in one CPU allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import mechanism_functional_campaign as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read


def run(root: Path, worker: int, workers: int) -> dict:
    """Reuse per-model locks, numerical limits and completed-result checkpoints."""
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("require positive workers and 0 <= worker < workers")
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != campaign.PROTOCOL:
        raise ValueError("unsupported assessment protocol")
    if plan["code_identity"] != campaign.code_identity():
        raise ValueError("assessment source changed; use the pinned checkout")
    indices = list(range(worker, len(plan["rows"]), workers))
    for position, index in enumerate(indices):
        print(
            json.dumps({"worker": worker, "index": index, "status": "starting"}),
            flush=True,
        )
        result = campaign.run(root, index)
        print(
            json.dumps(
                {
                    "worker": worker,
                    "index": index,
                    "status": result["status"],
                    "finished": position + 1,
                    "assigned": len(indices),
                }
            ),
            flush=True,
        )
    return {"worker": worker, "workers": workers, "indices": indices}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.worker, args.workers), indent=2))
