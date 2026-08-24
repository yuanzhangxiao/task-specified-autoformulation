"""Apply frozen pass/fail gates to an unseen-structure judge confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_confirmation(
    hybrid: dict[str, object],
    atomic: dict[str, object],
    gate: dict[str, float],
) -> dict[str, object]:
    """Evaluate predeclared thresholds without fitting or tuning parameters."""
    observed = {
        "minimum_response_success": hybrid["structured_response_success_rate"],
        "minimum_pair_aggregate_accuracy": hybrid["pair_aggregated_accuracy"],
        "minimum_order_consistency": hybrid["order_consistency_rate"],
        "minimum_wrong_sink_atomic_accuracy": atomic[
            "wrong_sink_expected_direction_accuracy"
        ],
        "minimum_duplicate_atomic_accuracy": atomic[
            "duplicate_relation_accuracy"
        ],
        "minimum_targeted_comparative_accuracy": hybrid[
            "comparative_question_accuracy"
        ],
    }
    checks = {
        key: {
            "observed": observed[key],
            "minimum": minimum,
            "passed": observed[key] is not None and observed[key] >= minimum,
        }
        for key, minimum in gate.items()
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "decision_rule": "all_predeclared_gates_must_pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-metrics", type=Path, required=True)
    parser.add_argument("--atomic-metrics", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hybrid_by_model = json.loads(args.hybrid_metrics.read_text(encoding="utf-8"))
    atomic_by_model = json.loads(args.atomic_metrics.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    model = protocol["judge_model"]
    result = {
        "schema_version": "hybrid-judge-structure-confirmation-result-1",
        "judge_model": model,
        "protocol_status": protocol["status"],
        **evaluate_confirmation(
            hybrid_by_model[model],
            atomic_by_model[model],
            protocol["confirmation_gate"],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = args.output.with_suffix(".md")
    lines = [
        "# Frozen unseen-structure confirmation",
        "",
        f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**.",
        "",
        "| Predeclared gate | Observed | Minimum | Result |",
        "|---|---:|---:|:---:|",
    ]
    for name, item in result["checks"].items():
        observed = item["observed"]
        observed_text = "N/A" if observed is None else f"{observed:.3f}"
        lines.append(
            f"| {name} | {observed_text} | {item['minimum']:.3f} | "
            f"{'pass' if item['passed'] else 'fail'} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
