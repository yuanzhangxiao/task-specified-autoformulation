#!/usr/bin/env python3
"""Prepare, run, resume and freeze the ICLR paired development searches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import iclr_controlled_search as study
from autoformalism.rebuttal.iclr_search_controls import ARMS
from autoformalism.rebuttal.prefit_replay import sealed_read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare")
    prep.add_argument("--public", type=Path, required=True)
    prep.add_argument("--cells", nargs="+", required=True)
    prep.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    prep.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    prep.add_argument(
        "--max-parameters", type=int, help="Predeclared cap shared across every arm"
    )
    prep.add_argument("--visits", type=int, default=15)
    prep.add_argument(
        "--without-critic",
        action="store_true",
        help="Explicit critic-off engineering study",
    )
    prep.add_argument(
        "--without-pruning",
        action="store_true",
        help="Freeze the pre-pruning endpoint in a separate study",
    )
    auth = stages.add_parser("bind-critic")
    auth.add_argument("--calibration-plan", required=True, type=Path)
    auth.add_argument("--calibration-summary", required=True, type=Path)
    worker = stages.add_parser("run")
    worker.add_argument("--block", required=True, type=int)
    worker.add_argument("--base-url", required=True)
    worker.add_argument("--seconds", type=int, default=21000)
    for sub in (
        prep,
        auth,
        worker,
        stages.add_parser("report"),
        stages.add_parser("export"),
    ):
        sub.add_argument("--root", type=Path, required=True)
    a = parser.parse_args()
    if a.stage == "prepare":
        p = study.prepare(
            a.root,
            a.public,
            a.cells,
            a.arms,
            a.seeds,
            a.visits,
            not a.without_critic,
            not a.without_pruning,
            max_parameters=a.max_parameters,
        )
        result = {
            "plan_sha256": p["artifact_sha256"],
            "blocks": len(p["blocks"]),
            "arms": p["arms"],
        }
    elif a.stage == "bind-critic":
        p = sealed_read(a.root / "plan.json")
        if p["protocol"] != study.PROTOCOL or not p["critic_enabled"]:
            raise ValueError("requires a critic-enabled controlled search")
        result = study.critic.bind_calibration(
            a.root, p, a.calibration_plan, a.calibration_summary
        )
    elif a.stage == "run":
        result = study.run_block(a.root, a.block, a.base_url, a.seconds)
    else:
        result = study.report(a.root, a.stage == "export")
        if "rows" in result:
            result = {
                "roster_sha256": result["artifact_sha256"],
                "subjects": len(result["rows"]),
            }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
