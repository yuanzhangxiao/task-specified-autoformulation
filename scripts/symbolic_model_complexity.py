#!/usr/bin/env python3
"""Measure the complexity of the symbolic models that were actually scored.

Reads the source artifacts named by a frozen evaluation's adapter requests, so
the models measured are exactly the models evaluated -- not whatever else is
lying in a development directory. It opens no data of any kind; it parses the
saved expression strings.

Complexity is reported three ways because no single count is neutral. Node
count is the expression-tree size, comparable to the traversal count upstream
symbolic-regression tools report. Term count is the number of additive terms,
which is what a reader means by "how many terms is this model". Operator count
excludes the leaves, separating structure from how many channels are named.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path

#: Method labels whose saved artifacts carry a symbolic equation set.
SYMBOLIC_KINDS = ("pysr", "sindy", "llm_sr", "llm_ode")


def expression_complexity(expression: str) -> dict[str, int] | None:
    """Tree size, additive terms and operator count for one right-hand side."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    nodes = [node for node in ast.walk(tree) if not isinstance(node, ast.Load)]
    operators = [
        node
        for node in nodes
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call))
    ]

    def additive_terms(node: ast.AST) -> int:
        """Count top-level terms, so a + b + c is three and a * b is one."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return additive_terms(node.left) + additive_terms(node.right)
        return 1

    return {
        "nodes": len(nodes) - 1,  # discard the Expression wrapper
        "terms": additive_terms(tree.body),
        "operators": len(operators),
    }


def model_complexity(equations: dict[str, str]) -> dict[str, int] | None:
    """Sum over a model's state equations; a model is all of its equations."""
    totals = {"nodes": 0, "terms": 0, "operators": 0, "equations": 0}
    for expression in equations.values():
        measured = expression_complexity(expression)
        if measured is None:
            return None
        for key, value in measured.items():
            totals[key] += value
        totals["equations"] += 1
    return totals if totals["equations"] else None


def quartiles(values: list[float]) -> dict[str, float | None]:
    """Median and interquartile range, reported with the count behind them."""
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    ordered = sorted(values)
    if len(ordered) == 1:
        only = float(ordered[0])
        return {"n": 1, "median": only, "q1": only, "q3": only, "iqr": 0.0}
    # Inclusive quartiles: with small per-method counts the exclusive method
    # discards too much, and the halves must include the median for an odd n.
    middle = len(ordered) // 2
    lower = ordered[: middle + (len(ordered) % 2)]
    upper = ordered[middle:]
    q1, q3 = statistics.median(lower), statistics.median(upper)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def collect(requests_path: Path) -> dict:
    """Measure every symbolic source the frozen evaluation named."""
    per_method: dict[str, list[dict]] = {}
    unreadable: dict[str, int] = {}
    for line in requests_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        request = json.loads(line)
        kind = request.get("source_kind")
        if kind not in SYMBOLIC_KINDS:
            continue
        path = Path(request["source_path"])
        if not path.is_file():
            unreadable[f"{kind}:missing_source"] = (
                unreadable.get(f"{kind}:missing_source", 0) + 1
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        equations = payload.get("equations")
        if not isinstance(equations, dict) or not equations:
            unreadable[f"{kind}:no_equations"] = (
                unreadable.get(f"{kind}:no_equations", 0) + 1
            )
            continue
        measured = model_complexity(equations)
        if measured is None:
            unreadable[f"{kind}:unparsable"] = (
                unreadable.get(f"{kind}:unparsable", 0) + 1
            )
            continue
        per_method.setdefault(kind, []).append(
            {
                "benchmark_id": payload.get("benchmark_id"),
                "tier": payload.get("tier"),
                "repetition": payload.get("seed"),
                **measured,
            }
        )
    summary = {
        kind: {
            measure: quartiles([float(row[measure]) for row in rows])
            for measure in ("nodes", "terms", "operators", "equations")
        }
        for kind, rows in sorted(per_method.items())
    }
    return {
        "schema_version": "phase-b-symbolic-complexity-1",
        "requests": str(requests_path),
        "test_data_opened": False,
        "summary": summary,
        "unreadable": unreadable,
        "models": per_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    value = collect(args.requests)
    if args.out:
        args.out.write_text(json.dumps(value, indent=2) + "\n")

    def fmt(value: object) -> str:
        return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"

    header = (
        f"{'method':<10}{'measure':<12}{'n':>5}{'median':>10}"
        f"{'Q1':>9}{'Q3':>9}{'IQR':>9}"
    )
    print(header)
    print("-" * len(header))
    for kind, measures in value["summary"].items():
        for measure, stats in measures.items():
            print(
                f"{kind:<10}{measure:<12}{stats['n']:>5}{fmt(stats['median']):>10}"
                f"{fmt(stats['q1']):>9}{fmt(stats['q3']):>9}{fmt(stats['iqr']):>9}"
            )
    if value["unreadable"]:
        print("\nnot measured:", value["unreadable"])


if __name__ == "__main__":
    main()
