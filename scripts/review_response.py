#!/usr/bin/env python3
"""Prepare and inspect bounded, training-only response feedback."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.llm.response_revision import ResponseRevisionClient
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search.response_evidence import presentation


def latest_source(root: Path) -> int:
    """Choose by completion order, never by score or scientific outcome."""
    plan = io.verify(root, execution=False)
    for index in reversed(range(plan["config"]["rounds"])):
        if all(io.read_round(root, task, index) is not None for task in plan["tasks"]):
            return index + plan.get("continuation", {}).get("source_round", 0)
    raise ValueError("No common completed source checkpoint")


def prepare(root: Path) -> dict:
    """Compute imported response features on CPU before allocating a proposer GPU."""
    plan = io.verify(root)
    if plan["protocol"] not in io.RESPONSE_PROTOCOLS:
        raise ValueError("response preparation requires review-deadline-7 or -8")
    rows = []
    for task in plan["tasks"]:
        selected = io.read_round(root, task, 0)["selected"]
        response = pipeline.response_for_selected(root, plan, task, selected)
        if response is None:
            raise ValueError("Fixed training replay unavailable: " + task["task_id"])
        shown, refs = presentation(response, selected["packet"])
        rows.append({"task_id": task["task_id"], "evidence": shown, "references": refs})
    return sealed_write(
        root / "response-preview.json",
        {
            "protocol": "training-response-preview-1",
            "plan_sha256": plan["artifact_sha256"],
            "rows": rows,
            "parameter_fitting_performed": False,
            "validation_used_for_response_evidence": False,
            "test_data_opened": False,
        },
    )


def check_server(root: Path, base_url: str) -> dict:
    """Fail before generation if the serving tokenizer or context differs."""
    plan = io.verify(root)
    client = ResponseRevisionClient(
        settings=io.StagedModelSettings.model_validate(
            plan["config"]["model_settings"]
        ),
        seed=0,
        base_url=base_url,
        directory=root
        / "runtime"
        / ("tokenizer-probe-" + os.environ.get("SLURM_JOB_ID", "local")),
        namespace=plan["artifact_sha256"],
        can_start=lambda: True,
    )
    count = client._count([{"role": "user", "content": "Tokenizer preflight."}])
    return {"tokenizer_ready": True, "probe_tokens": count, "live_llm_calls": 0}


def audit(root: Path) -> dict:
    """Separate physical generation, packing checks and delivery failures."""
    plan = io.verify(root, execution=False)
    rows = []
    for path in sorted((root / "results").glob("*/round_*/calls/*.json")):
        record = json.loads(path.read_text())
        rows.append(
            {
                "task_id": path.parents[2].name,
                "phase_round": path.parents[1].name,
                "attempt": record["attempt"],
                "status": record["status"],
                "preflight": record.get("prompt_preflight"),
                "error": record.get("error"),
            }
        )
    failures = []
    for path in sorted((root / "results").glob("*/round_*/proposal.json")):
        value = sealed_read(path)
        if value.get("status") in {
            "request_preflight_failed",
            "provider_request_failed",
        }:
            failures.append(
                {
                    "task_id": value["task"]["task_id"],
                    "round": value["round"],
                    "status": value["status"],
                    "error": value.get("error"),
                    "attempts": value.get("attempts"),
                }
            )
    return {
        "plan_sha256": plan["artifact_sha256"],
        "physical_generation_requests": len(rows),
        "delivery_failures": failures,
        "requests": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("latest-source", "prepare", "check-server", "audit")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url")
    args = parser.parse_args()
    if args.command == "latest-source":
        print(latest_source(args.root))
    elif args.command == "check-server":
        if not args.base_url:
            parser.error("check-server requires --base-url")
        print(json.dumps(check_server(args.root, args.base_url), indent=2))
    elif args.command == "prepare":
        value = prepare(args.root)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "tasks": len(value["rows"]),
                    "preview": str(args.root / "response-preview.json"),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(audit(args.root), indent=2))
