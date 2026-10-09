#!/usr/bin/env python3
"""Freeze, run, and summarize LLM-SR on Phase C public cells.

The search comes from the pinned upstream checkout, bound through
AF_LLM_SR_ROOT. AF_ENDPOINT_KIND declares where the model is served and must
match the frozen plan; AF_VLLM_BASE_URL is that endpoint's address, and for
the Jetstream2 hosted kind it defaults to the service's fixed route. `prepare`
and `report` never contact a provider, and `run` contacts one only when its
task still has a search to do. `run --target` searches one target of a task,
so separate processes can search a task's targets at once; a later `run`
without it assembles and seals the task's model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal.llm_sr_campaign import prepare_phase_c, report, run_phase_c
from autoformalism.rebuttal.llm_sr_driver import build_searcher
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    CACHING_ENDPOINTS,
    OUTAGE_PATIENCE_SECONDS,
    llm_sr_search_settings,
    llm_sr_transport_settings,
    resolve_endpoint,
    served_model_ids,
    wait_for_served_model,
)
from autoformalism.rebuttal.prefit_replay import sealed_read


def _endpoint() -> tuple[str, str]:
    """The endpoint kind and address this run actually uses."""
    return resolve_endpoint(
        os.environ.get("AF_ENDPOINT_KIND", ""), os.environ.get("AF_VLLM_BASE_URL")
    )


def _searcher(
    root: Path,
    base_url: str,
    *,
    patience_seconds: float,
    bypass_cache: bool = False,
):
    """Bind the pinned checkout; confirm the served model only before a search.

    `patience_seconds` is how long a search waits out an endpoint that keeps
    failing, and `bypass_cache` whether it replays stored answers; both follow
    the endpoint kind, not the plan. The generation limit and header reading
    follow the plan's declared adaptation, and the temperature and search
    settings follow the paper's values it declares, if any. LLM-SR names its
    model in every request, so the endpoint must serve it; that is checked
    before each target search that starts, and not when a run only assembles.
    An endpoint that cannot list its models is asked again for as long as a
    search would wait for it.
    """
    checkout = os.environ.get("AF_LLM_SR_ROOT")
    if not checkout:
        raise ValueError("AF_LLM_SR_ROOT must point at the pinned LLM-SR checkout")
    sealed = sealed_read(root / "plan.json")
    budget = sealed["plan"]["budget"]
    if budget["unit"] != "llm_samples":
        raise ValueError(f"unexpected budget unit {budget['unit']!r}")
    model = sealed["plan"]["model"]
    return build_searcher(
        upstream_root=Path(checkout),
        base_url=base_url,
        model=model,
        samples=int(budget["declared"]),
        patience_seconds=patience_seconds,
        bypass_cache=bypass_cache,
        before_search=lambda: wait_for_served_model(
            base_url,
            model,
            patience_seconds=patience_seconds,
            list_models=served_model_ids,
        ),
        **llm_sr_transport_settings(sealed["plan"]),
        **llm_sr_search_settings(sealed["plan"]),
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
    worker.add_argument("--target", help="search only this target of the task")
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
            search=_searcher(
                args.root,
                base_url,
                patience_seconds=OUTAGE_PATIENCE_SECONDS[endpoint],
                bypass_cache=endpoint in CACHING_ENDPOINTS,
            ),
            target=args.target,
        )
        value = {key: item for key, item in result.items() if key != "rows"}
    else:
        result = report(args.root)
        value = {key: item for key, item in result.items() if key != "rows"}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
