#!/usr/bin/env python3
"""Measure the complexity of the symbolic models that were actually scored.

Reads the source artifacts named by a frozen evaluation's adapter requests, so
the models measured are exactly the models evaluated -- not whatever else is
lying in a development directory. It opens no data of any kind; it parses the
saved expression strings.

The counts are the project's own, from `autoformalism.pruning.process_aware`,
which is what the pruner already uses to decide whether one model is smaller
than another. Using a second definition here would put two different numbers
called "complexity" in one paper. `complexity()` itself takes a fit request, so
this applies its constituent counts to the candidate the evaluation adapters
build from the same saved equations.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.pruning.process_aware import definitions, terms
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate

#: Method labels whose saved artifacts carry a symbolic equation set.
SYMBOLIC_KINDS = ("pysr", "sindy", "llm_sr", "llm_ode")


def model_complexity(equations: dict[str, str]) -> dict[str, int] | None:
    """The project's own counts, applied to one saved symbolic model.

    Mirrors ``pruning.process_aware.complexity``: the same signed additive
    term decomposition, and the same node count over every definition,
    observation mapping and initial-condition expression.
    """
    if not equations:
        return None
    try:
        candidate = equation_candidate(
            "complexity", equations, ValidationContext(targets=tuple(equations))
        )
    except Exception:
        return None
    expressions = (
        list(definitions(candidate).values())
        + [item.expression for item in candidate.observation_mappings]
        + [item.expression for item in candidate.initial_conditions if item.expression]
    )
    try:
        nodes = sum(
            sum(1 for _ in ast.walk(ast.parse(item, mode="eval")))
            for item in expressions
        )
        decomposed = len(terms(candidate))
    except SyntaxError:
        return None
    return {
        "states": len(candidate.states),
        "processes": len(candidate.processes),
        "parameters": len(candidate.parameters),
        "terms": decomposed,
        "expression_nodes": nodes,
    }


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
            for measure in (
                "states", "processes", "parameters", "terms", "expression_nodes"
            )
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
