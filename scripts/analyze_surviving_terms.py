"""List frozen final state equations and surviving terms across seeds."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="full")
    parser.add_argument("--tier", default="hard")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with args.authoritative_runs.open(encoding="utf-8", newline="") as handle:
        for run in csv.DictReader(handle):
            if (
                run["method"] != args.method
                or run["tier"] != args.tier
                or run["status"] != "complete"
            ):
                continue
            source = Path(run["source"])
            payload = json.loads(source.read_text(encoding="utf-8"))
            frozen = payload.get("frozen")
            if not isinstance(frozen, dict) or not isinstance(
                frozen.get("candidate"), dict
            ):
                raise ValueError(f"missing frozen candidate: {source}")
            candidate = CandidateModel.model_validate(frozen["candidate"])
            summary = summarize_candidate(candidate)
            rows.append(
                {
                    "benchmark_id": run["benchmark"],
                    "tier": run["tier"],
                    "method": run["method"],
                    "seed": int(run["seed"]),
                    **summary,
                    "source": str(source),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0]) if rows else (
        "benchmark_id",
        "tier",
        "method",
        "seed",
        "target_equations",
        "state_equations",
        "dynamic_terms",
        "source",
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["benchmark_id"], row["seed"]))
        )


def summarize_candidate(candidate: CandidateModel) -> dict[str, object]:
    """Return readable frozen equations and additive-term counts."""
    parser = RestrictedParser()
    target_states = {
        symbol
        for mapping in candidate.observation_mappings
        for symbol in parser.parse(
            mapping.expression, location=f"observation:{mapping.channel}"
        ).symbols
    }
    equations = [
        (equation.state, equation.rhs) for equation in candidate.state_equations
    ]
    target_equations = [
        f"d({state})/dt = {rhs}" for state, rhs in equations if state in target_states
    ]
    return {
        "target_equations": " ; ".join(target_equations),
        "state_equations": " ; ".join(
            f"d({state})/dt = {rhs}" for state, rhs in equations
        ),
        "dynamic_terms": sum(_term_count(rhs) for _, rhs in equations),
    }


def _term_count(expression: str) -> int:
    return len(_additive_terms(ast.parse(expression, mode="eval").body))


def _additive_terms(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _additive_terms(node.left) + _additive_terms(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return (*_additive_terms(node.left), node.right)
    return (node,)


if __name__ == "__main__":
    main()
