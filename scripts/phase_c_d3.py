#!/usr/bin/env python3
"""Freeze, run, and summarize D3 on Phase C public cells.

AF_ENDPOINT_KIND declares where the model is served and must match the frozen
plan; AF_VLLM_BASE_URL is that endpoint's address, and for the Jetstream2
hosted kind it defaults to the service's fixed route. `prepare` and `report`
never contact a provider, and `run` contacts one only when its task still has
generations to run. A task stopped by an outage longer than the endpoint's
patience exits with status 3 and no result; running it again resumes it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal.phase_c_d3 import (
    EndpointUnavailable,
    prepare_phase_c,
    report,
    run_phase_c,
)
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    CACHING_ENDPOINTS,
    OUTAGE_PATIENCE_SECONDS,
    resolve_endpoint,
)

#: The exit status of a task an outage stopped; it has no result yet.
ENDPOINT_UNAVAILABLE_STATUS = 3


def _endpoint() -> tuple[str, str]:
    """The endpoint kind and address this run actually uses."""
    return resolve_endpoint(
        os.environ.get("AF_ENDPOINT_KIND", ""), os.environ.get("AF_VLLM_BASE_URL")
    )


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
            "model": result["model"],
            "expected": len(result["rows"]),
            "maximum_logical_calls": result["maximum_logical_calls"],
            "reporting_qualifications": result["reporting_qualifications"],
        }
    elif args.command == "run":
        endpoint, base_url = _endpoint()
        try:
            result = run_phase_c(
                args.root,
                args.index,
                endpoint=endpoint,
                base_url=base_url,
                patience_seconds=OUTAGE_PATIENCE_SECONDS[endpoint],
                bypass_cache=endpoint in CACHING_ENDPOINTS,
            )
        except EndpointUnavailable as exc:
            print(json.dumps({"status": "endpoint_unavailable", "error": str(exc)}))
            raise SystemExit(ENDPOINT_UNAVAILABLE_STATUS) from None
        value = {key: item for key, item in result.items() if key != "rows"}
    else:
        result = report(args.root)
        value = {key: item for key, item in result.items() if key != "rows"}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
