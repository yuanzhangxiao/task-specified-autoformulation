#!/usr/bin/env python3
"""Freeze, run, and summarize LLM-SR on Phase C public cells.

The search comes from the pinned upstream checkout, bound through
AF_LLM_SR_ROOT. AF_ENDPOINT_KIND declares where the model is served and must
match the frozen plan; AF_VLLM_BASE_URL is that endpoint's address, and for
the Jetstream2 hosted kind it defaults to the service's fixed route. `prepare`
and `report` never contact a provider, and `run` contacts one only when its
task still has a search to do.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal.llm_sr_campaign import prepare_phase_c, report, run_phase_c
from autoformalism.rebuttal.llm_sr_driver import build_searcher
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    check_served_model,
    resolve_endpoint,
    served_model_ids,
)
from autoformalism.rebuttal.prefit_replay import sealed_read


def _endpoint() -> tuple[str, str]:
    """The endpoint kind and address this run actually uses."""
    return resolve_endpoint(
        os.environ.get("AF_ENDPOINT_KIND", ""), os.environ.get("AF_VLLM_BASE_URL")
    )


def _searcher(root: Path, base_url: str):
    """Bind the pinned checkout; confirm the served model only before a search."""
    checkout = os.environ.get("AF_LLM_SR_ROOT")
    if not checkout:
        raise ValueError("AF_LLM_SR_ROOT must point at the pinned LLM-SR checkout")
    sealed = sealed_read(root / "plan.json")
    budget = sealed["plan"]["budget"]
    if budget["unit"] != "llm_samples":
        raise ValueError(f"unexpected budget unit {budget['unit']!r}")
    model = sealed["plan"]["model"]
    search = build_searcher(
        upstream_root=Path(checkout),
        base_url=base_url,
        model=model,
        samples=int(budget["declared"]),
    )

    def checked(**kwargs) -> dict:
        # LLM-SR names its model in every request, so the endpoint must serve it.
        check_served_model(served_model_ids(base_url), model, first=False)
        return search(**kwargs)

    return checked


def main() -> None:
    """Discovery is explicit; prepare and report make no provider call."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--release", type=Path, required=True)
    freeze.add_argument("--root", type=Path, required=True)
    worker = sub.add_parser("run")
    worker.add_argument("--root", type=Path, required=True)
    worker.add_argument("--index", type=int, required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_phase_c(args.config, args.release, args.root)
        value = {
            "plan_sha256": result["artifact_sha256"],
            "release_summary_sha256": result["release_summary_sha256"],
            "endpoint": result["plan"]["endpoint"],
            "model": result["plan"]["model"],
            "expected": len(result["rows"]),
            "maximum_logical_samples": result["maximum_logical_samples"],
            "reporting_qualifications": result["reporting_qualifications"],
        }
    elif args.command == "run":
        endpoint, base_url = _endpoint()
        result = run_phase_c(
            args.root,
            args.index,
            endpoint=endpoint,
            search=_searcher(args.root, base_url),
        )
        value = {key: item for key, item in result.items() if key != "rows"}
    else:
        result = report(args.root)
        value = {key: item for key, item in result.items() if key != "rows"}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
