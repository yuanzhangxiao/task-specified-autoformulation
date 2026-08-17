"""Build public-task scientific-degradation pairs from completed Phase-B runs."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel


def _component(payload: dict, name: str) -> tuple[dict, str]:
    for equation in payload["state_equations"]:
        if equation["state"] == name:
            return equation, "rhs"
    for process in payload["processes"]:
        if process["name"] == name:
            return process, "expression"
    raise ValueError(f"candidate has no component named {name}")


def _names(expression: str) -> set[str]:
    return {
        node.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name)
    }


def _negate_symbol(expression: str, symbol: str) -> str:
    class Negate(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id != symbol:
                return node
            return ast.copy_location(
                ast.UnaryOp(op=ast.USub(), operand=ast.Name(id=symbol, ctx=node.ctx)),
                node,
            )

    changed = Negate().visit(ast.parse(expression, mode="eval"))
    return ast.unparse(ast.fix_missing_locations(changed))


def _unique_name(payload: dict, stem: str) -> str:
    occupied = {
        *(item["name"] for item in payload["states"]),
        *(item["name"] for item in payload["processes"]),
        *(item["name"] for item in payload["parameters"]),
    }
    if stem not in occupied:
        return stem
    suffix = 2
    while f"{stem}_{suffix}" in occupied:
        suffix += 1
    return f"{stem}_{suffix}"


def _mutations(candidate: CandidateModel) -> list[tuple[str, CandidateModel]]:
    base = candidate.model_dump(mode="json")
    glucose, expression_key = _component(base, "Gp")
    glucose_expression = glucose[expression_key]
    if "meal_event_g" not in _names(glucose_expression):
        raise ValueError("selected candidate has no meal_event_g pathway to perturb")
    mutations: list[tuple[str, dict]] = []

    wrong_sign = copy.deepcopy(base)
    row, key = _component(wrong_sign, "Gp")
    row[key] = _negate_symbol(row[key], "meal_event_g")
    mutations.append(("wrong_meal_source_sign", wrong_sign))

    duplicated_source = copy.deepcopy(base)
    row, key = _component(duplicated_source, "Gp")
    row[key] = f"({row[key]}) + meal_event_g"
    mutations.append(("duplicated_meal_source", duplicated_source))

    disconnected = copy.deepcopy(base)
    disconnected["processes"].append(
        {
            "name": _unique_name(disconnected, "claimed_meal_pathway"),
            "expression": "meal_event_g",
            "mechanisms": ["ClaimedMealPathway"],
            "description": "Claimed scientific mechanism disconnected from outputs.",
            "unit": "unspecified",
        }
    )
    mutations.append(("disconnected_claimed_mechanism", disconnected))

    accumulator = copy.deepcopy(base)
    state_name = _unique_name(accumulator, "unjustified_accumulator")
    accumulator["states"].append(
        {
            "name": state_name,
            "kind": "latent",
            "mechanisms": ["UnjustifiedAccumulator"],
            "description": "Additional one-signed accumulator without task role.",
            "unit": "unspecified",
        }
    )
    accumulator["state_equations"].append(
        {"state": state_name, "rhs": "meal_event_g"}
    )
    accumulator["initial_conditions"].append(
        {"state": state_name, "scope": "global", "fixed_value": 0.0}
    )
    mutations.append(("unjustified_one_sided_accumulator", accumulator))

    output = []
    for mutation_type, payload in mutations:
        payload["candidate_id"] = f"{candidate.candidate_id}_{mutation_type}"
        payload["parent_candidate_id"] = candidate.candidate_id
        payload["change_summary"] = "Controlled calibration mutation."
        output.append((mutation_type, CandidateModel.model_validate(payload)))
    return output


def _validation_context(data_root: Path, benchmark_id: str, tier: str):
    dataset = BenchmarkLoader(BenchmarkRegistry()).load_development(
        DataConfig(root=data_root, benchmark_id=benchmark_id, tier=tier)
    )
    arguments = ExecutionArguments(
        data_root=data_root,
        benchmark_id=benchmark_id,
        tier=tier,
        seed=0,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=Path("artifacts/v2-judge-calibration"),
        resume=False,
        dry_run=True,
        mock_llm=True,
        use_clean_observations=False,
    )
    return _context(arguments, dataset)


def build_pairs(runs_root: Path, data_root: Path) -> tuple[AdversarialPair, ...]:
    """Build four deterministic-valid degradations of every selected candidate."""
    pairs: list[AdversarialPair] = []
    contexts = {}
    for summary_path in sorted(runs_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_config = json.loads(
            (summary_path.parent / "run_config.json").read_text(encoding="utf-8")
        )
        benchmark_id = run_config["benchmark_id"]
        tier = run_config["tier"]
        context = contexts.setdefault(
            (benchmark_id, tier),
            _validation_context(data_root, benchmark_id, tier),
        )
        baseline = CandidateModel.model_validate(summary["selected_candidate"])
        compile_candidate(baseline, context)
        for mutation_type, mutated in _mutations(baseline):
            compile_candidate(mutated, context)
            digest = hashlib.sha256(
                f"{summary_path.parent.name}:{mutation_type}".encode()
            ).hexdigest()[:16]
            pairs.append(
                AdversarialPair(
                    pair_id=f"v2_{digest}",
                    benchmark_id=benchmark_id,
                    tier=tier,
                    mutation_type=mutation_type,
                    valid_candidate=baseline,
                    adversarial_candidate=mutated,
                )
            )
    return tuple(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = build_pairs(args.runs_root.resolve(), args.data_root.resolve())
    if not pairs:
        raise SystemExit("no completed run summaries found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} calibration pairs to {args.output}")


if __name__ == "__main__":
    main()
