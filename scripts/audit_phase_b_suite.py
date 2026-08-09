#!/usr/bin/env python3
"""Audit all 20 numerical task/tier/dynamics cases in Phase-B v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks import audit_task_gates, mechanism_gate_definition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for task in ("T1", "T2", "T3", "T4"):
        for dynamics in ("canonical", "perturbed"):
            for tier in ("easy", "hard"):
                definition = mechanism_gate_definition(
                    "dalla_man", tier, task=task, data_root=args.data_root
                )
                report = audit_task_gates(
                    definition,
                    dynamics=dynamics,
                    data_root=args.data_root,
                )
                case_id = f"dalla_man_{task}_{dynamics}_{tier}"
                _record(reports, rows, case_id, report)
    for family in ("cstr", "alien_device"):
        for tier in ("easy", "hard"):
            definition = mechanism_gate_definition(
                family, tier, data_root=args.data_root
            )
            report = audit_task_gates(definition, data_root=args.data_root)
            _record(reports, rows, f"{family}_{tier}", report)

    payload = {
        "schema_version": "phase_b_release_audit_v1",
        "uses_discovery_method_outputs": False,
        "number_of_cases": len(rows),
        "release_ready_cases": sum(bool(row["release_ready"]) for row in rows),
        "cases": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(_markdown(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _record(reports: Path, rows: list[dict[str, object]], case_id: str, report) -> None:
    (reports / f"{case_id}.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    rows.append(
        {
            "case_id": case_id,
            "release_ready": report.release_ready,
            "rank_pass": report.rank_pass,
            "condition_pass": report.condition_pass,
            "stable_rank_pass": report.stable_rank_pass,
            "ablation_pass": report.ablation_pass,
            "basic_pass": (
                report.basic.finite_rollouts_pass
                and report.basic.input_design_pass
                and report.basic.persistence_pass
            ),
            "rank_at_1e3": report.rank_at_1e3,
            "claimed_dimension": report.claimed_dimension,
            "condition_number": report.claimed_subspace_condition_number,
            "stable_rank": report.stable_rank,
            "minimum_ablation_discrepancy": min(
                item.normalized_discrepancy for item in report.ablations
            ),
        }
    )


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Phase-B pre-release gate summary",
        "",
        "No discovery-method output or test trajectory was used.",
        "",
        "| Case | Ready | Rank | Condition | Stable rank | Ablation | Basic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        marks = [
            "yes" if row[key] else "no"
            for key in (
                "release_ready",
                "rank_pass",
                "condition_pass",
                "stable_rank_pass",
                "ablation_pass",
                "basic_pass",
            )
        ]
        lines.append(f"| {row['case_id']} | " + " | ".join(marks) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
