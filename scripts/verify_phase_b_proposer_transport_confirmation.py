#!/usr/bin/env python3
"""Verify a selected GPT-OSS proposer operating point across two clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.proposer_transport_calibration import (
    verify_proposer_cross_cluster_confirmation,
)


def main() -> None:
    """Write the cross-cluster confirmation report and fail closed on a miss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-analysis", type=Path, required=True)
    parser.add_argument("--confirmation-analysis", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify_proposer_cross_cluster_confirmation(
        args.source_analysis,
        args.confirmation_analysis,
        args.handoff,
    )
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "cross_cluster_confirmation.json"
    markdown_path = output / "cross_cluster_confirmation.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    if report["status"] != "pass":
        raise SystemExit(1)


def _markdown(report: dict[str, object]) -> str:
    """Render the concise operating-point confirmation report."""
    metrics = report["confirmation_metrics"]
    assert isinstance(metrics, dict)
    lines = [
        "# GPT-OSS proposer cross-cluster confirmation",
        "",
        f"Overall result: **{str(report['status']).upper()}**.",
        "",
        f"Primary platform: `{report['primary_platform']}`.",
        f"Confirmation platform: `{report['confirmation_platform']}`.",
        f"Selected output budget: `{report['selected_max_output_tokens']}` tokens.",
        "",
        "| Metric | Confirmation |",
        "|---|---:|",
    ]
    for name in (
        "response_success",
        "first_attempt_response_success",
        "deterministic_validity",
        "public_target_pass_rate",
        "length_exhausted_attempt_rate",
        "mean_successful_budget_utilization",
    ):
        value = metrics.get(name)
        rendered = "N/A" if value is None else f"{float(value):.3f}"
        lines.append(f"| {name} | {rendered} |")
    lines.extend(
        [
            "",
            "Candidate text equality is intentionally not required across "
            "accelerator platforms; both runs must independently pass the same "
            "public transport and deterministic-validity gates.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
