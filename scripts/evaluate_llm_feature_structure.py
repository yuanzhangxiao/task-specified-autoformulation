"""Evaluate selected LLM-feature-SINDy equations with deterministic graph checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from autoformalism.expressions import RestrictedParser
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.schemas import CandidateModel


def _proposal(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        parsed = event.get("parsed_response")
        if event.get("event") == "llm_response" and isinstance(parsed, dict):
            return parsed
    raise ValueError(f"no structured feature proposal in {path}")


def _candidate(result_path: Path) -> CandidateModel:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    equations = result.get("equations") or {}
    if len(equations) != 1:
        raise ValueError("LLM-feature structural audit requires exactly one target")
    target, rhs = next(iter(equations.items()))
    proposal = _proposal(result_path.parent / "llm_events.jsonl")
    proposed = {
        item["name"]: item
        for item in proposal.get("algebraics", [])
    }
    parser = RestrictedParser()
    pending = list(parser.parse(rhs, location="selected_equation").symbols)
    selected_names: set[str] = set()
    while pending:
        name = pending.pop()
        if name in selected_names or name not in proposed:
            continue
        selected_names.add(name)
        pending.extend(
            parser.parse(
                proposed[name]["expression"], location=f"feature:{name}"
            ).symbols
        )
    processes = [
        {
            "name": item["name"],
            "expression": item["expression"],
            "mechanisms": item.get("mechanisms", []),
        }
        for name, item in proposed.items()
        if name in selected_names
    ]
    return CandidateModel.model_validate(
        {
            "candidate_id": f"llm_feature_audit_{result_path.parent.name}",
            "parent_candidate_id": None,
            "states": [{"name": target, "kind": "observed"}],
            "processes": processes,
            "state_equations": [{"state": target, "rhs": rhs}],
            "observation_mappings": [
                {"channel": target, "expression": target}
            ],
            "parameters": [],
            "initial_conditions": [
                {"state": target, "scope": "global", "expression": target}
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specifications = {
        spec.benchmark_id: spec
        for path in sorted(args.config_root.glob("*.json"))
        for spec in (
            MechanismEvaluationSpec.model_validate_json(
                path.read_text(encoding="utf-8")
            ),
        )
    }
    runs = pd.read_csv(args.runs)
    runs = runs[runs.method == "llm_feature_sindy"]
    rows = []
    for row in runs.itertuples(index=False):
        spec = specifications.get(row.benchmark)
        if spec is None or not Path(row.source).is_file():
            continue
        evaluation = evaluate_mechanisms(_candidate(Path(row.source)), spec)
        rows.append(
            {
                "method": row.method,
                "benchmark": row.benchmark,
                "tier": row.tier,
                "seed": row.seed,
                "structural_validity": evaluation.structural_validity,
                "source": row.source,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
