#!/usr/bin/env python3
"""Select a GPT-OSS proposer output budget from frozen calibration results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.proposer_transport_calibration import (
    ProposerCalibrationResult,
    analyze_proposer_calibration,
    load_proposer_calibration_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    plan = load_proposer_calibration_plan(args.plan)
    result_paths = sorted(args.results_root.glob("task_*/budget_*.json"))
    results = tuple(
        ProposerCalibrationResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in result_paths
    )
    analysis = analyze_proposer_calibration(plan, results)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "proposer_transport_calibration.json"
    markdown_path = output_root / "proposer_transport_calibration.md"
    json_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(analysis), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    if analysis["status"] != "pass":
        raise SystemExit(1)


def _markdown(analysis: dict[str, object]) -> str:
    selected = analysis["selected_max_output_tokens"]
    lines = [
        "# GPT-OSS-120B proposer transport calibration",
        "",
        f"Overall result: **{str(analysis['status']).upper()}**.",
        "",
        "Only round-zero proposer transport and deterministic candidate validity "
        "are evaluated. No fitting, scientific judge, test data, or private "
        "reference is used.",
        "",
        "| Max output tokens | Response success | First-attempt success | "
        "Deterministic validity | Public-target pass | Length exhaustion | "
        "Mean utilization | Mean latency (s) | Result |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in analysis["operating_points"]:
        utilization = row["mean_successful_budget_utilization"]
        latency = row["mean_latency_ms"]
        lines.append(
            f"| {row['max_output_tokens']} | {row['response_success']:.3f} | "
            f"{row['first_attempt_response_success']:.3f} | "
            f"{row['deterministic_validity']:.3f} | "
            f"{row['public_target_pass_rate']:.3f} | "
            f"{row['length_exhausted_attempt_rate']:.3f} | "
            f"{_format_optional(utilization)} | "
            f"{_format_optional(None if latency is None else latency / 1000)} | "
            f"{'pass' if row['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            (
                "Selected operating point: **none**."
                if selected is None
                else "Selected operating point: "
                f"**high reasoning with {selected} max output tokens**."
            ),
            "",
            "The GPT-5.6 resource figures are descriptive context only and do "
            "not enter selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_optional(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
