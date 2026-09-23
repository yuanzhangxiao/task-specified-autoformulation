#!/usr/bin/env python3
"""Run the staged Milestone 4 pilot independently of the HPC launcher."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import general_critic as campaign
from autoformalism.rebuttal import general_critic_io as io


def main():
    """Each stage consumes or resumes only its own bounded work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "freeze",
            "verify",
            "evidence",
            "parent-review",
            "propose",
            "child-review",
            "fit",
            "report",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--judge-revision")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None or args.judge_revision is None:
            parser.error("freeze requires --source and --judge-revision")
        plan = io.freeze(args.source, args.root, args.judge_revision)
        print(
            json.dumps(
                {"identity": plan["artifact_sha256"], "tasks": len(plan["rows"])}
            )
        )
        return
    plan = io.verify(args.root)
    if args.stage == "verify":
        print(json.dumps({"identity": plan["artifact_sha256"]}))
        return
    if args.stage != "report":
        index = args.index
        if index is None and args.stage == "fit":
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        indices = [index] if index is not None else range(len(plan["rows"]))
        for i in indices:
            if i not in range(len(plan["rows"])):
                raise ValueError("task index outside frozen matrix")
            if args.stage == "evidence":
                value = campaign.prepare_evidence(args.root, i)
            elif args.stage.endswith("review"):
                value = campaign.review_one(
                    args.root, i, args.stage.split("-")[0], args.base_url
                )
            elif args.stage == "propose":
                value = campaign.propose_one(args.root, i, args.base_url)
            else:
                value = campaign.fit_one(args.root, i)
            print(
                json.dumps(
                    {
                        "index": i,
                        "status": value.get(
                            "status", value.get("review", {}).get("status")
                        ),
                    }
                ),
                flush=True,
            )
    # Parallel array tasks do not race to publish summaries.
    if args.stage != "fit":
        summary = campaign.report(args.root)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "status_counts",
                        "selection_counts",
                        "judge_cost",
                        "proposer_cost",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
