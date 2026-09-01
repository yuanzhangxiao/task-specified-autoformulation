#!/usr/bin/env python3
"""Summarize separate deterministic endpoints for the full GPT-5.6 baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from autoformalism.rebuttal.final_evaluation import FinalEvaluationRecord
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome
from scripts.summarize_phase_b_final_evaluation_pilot import (
    build_report,
    render_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--source-outcomes", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--resource-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.freeze_manifest.read_text(encoding="utf-8")),
        tuple(
            SourceAdapterOutcome.model_validate_json(line)
            for line in args.source_outcomes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        tuple(
            FinalEvaluationRecord.model_validate_json(line)
            for line in args.records.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
    )
    resources = tuple(
        json.loads(line)
        for line in args.resource_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected = {str(item["request_id"]) for item in report["subjects"]}
    actual = {str(item["request_id"]) for item in resources}
    if actual != expected:
        raise ValueError("resource ledger differs from frozen evaluation rows")
    available = [item for item in resources if item["resource_status"] == "available"]
    report["resource_usage"] = {
        "coverage": len(available) / len(resources) if resources else None,
        "agent_latency_seconds_sum": _sum(available, "agent_latency_seconds"),
        "agent_latency_seconds_mean": _mean(available, "agent_latency_seconds"),
        "tool_call_count_sum": _sum(available, "tool_call_count"),
        "input_tokens_sum": _sum(available, "input_tokens"),
        "output_tokens_sum": _sum(available, "output_tokens"),
        "total_tokens_sum": _sum(available, "total_tokens"),
        "monetary_cost_usd": None,
        "monetary_cost_status": "not_provider_reported",
    }
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "raw_agent_endpoint_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report).replace(
        "# Phase-B two-cell final-evaluation pilot",
        "# GPT-5.6 raw-data-agent full deterministic evaluation",
        1,
    )
    resource = report["resource_usage"]
    markdown += (
        "\n## Provider-agent resource usage\n\n"
        "These values come from frozen provider artifacts; monetary cost was not "
        "reported by the provider.\n\n"
        "| Coverage | Latency sum (s) | Tool calls | Input tokens | Output tokens | "
        "Total tokens | Monetary cost |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {resource['coverage']} | {resource['agent_latency_seconds_sum']} | "
        f"{resource['tool_call_count_sum']} | {resource['input_tokens_sum']} | "
        f"{resource['output_tokens_sum']} | {resource['total_tokens_sum']} | N/A |\n"
    )
    (output_root / "raw_agent_endpoint_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        f"wrote raw-agent endpoint report for {report['requested_source_count']} "
        f"planned subjects"
    )


def _values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(item[key]) for item in rows if item.get(key) is not None]


def _sum(rows: list[dict[str, object]], key: str) -> float | None:
    values = _values(rows, key)
    return None if not values else sum(values)


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = _values(rows, key)
    return None if not values else statistics.fmean(values)


if __name__ == "__main__":
    main()
