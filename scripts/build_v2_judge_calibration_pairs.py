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


def _additive_terms(expression: str) -> tuple[str, ...]:
    """Return signed top-level additive terms without evaluating text."""
    def visit(node: ast.expr, sign: int = 1) -> list[ast.expr]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return [*visit(node.left, sign), *visit(node.right, sign)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return [*visit(node.left, sign), *visit(node.right, -sign)]
        if sign < 0:
            node = ast.UnaryOp(op=ast.USub(), operand=node)
        return [node]

    parsed = ast.parse(expression, mode="eval")
    return tuple(
        ast.unparse(ast.fix_missing_locations(node))
        for node in visit(parsed.body)
    )


def _blind_candidate(candidate: CandidateModel, token: str) -> CandidateModel:
    """Remove identifiers and summaries that could reveal pair membership."""
    payload = candidate.model_dump(mode="json")
    payload["candidate_id"] = f"calibration_{token}"
    payload["parent_candidate_id"] = None
    payload["change_summary"] = "Candidate submitted for scientific assessment."
    return CandidateModel.model_validate(payload)


def _mutations(candidate: CandidateModel) -> list[tuple[str, CandidateModel]]:
    base = candidate.model_dump(mode="json")
    glucose, expression_key = _component(base, "Gp")
    glucose_expression = glucose[expression_key]
    mutations: list[tuple[str, dict]] = []

    wrong_sink = copy.deepcopy(base)
    row, key = _component(wrong_sink, "Gp")
    row[key] = f"({row[key]}) - abs(meal_event_g)"
    mutations.append(("wrong_meal_sink", wrong_sink))

    duplicated_flux = copy.deepcopy(base)
    row, key = _component(duplicated_flux, "Gp")
    duplicated_term = _additive_terms(glucose_expression)[0]
    row[key] = f"({row[key]}) + ({duplicated_term})"
    mutations.append(("duplicated_gp_flux", duplicated_flux))

    disconnected = copy.deepcopy(base)
    disconnected["processes"].append(
        {
            "name": _unique_name(disconnected, "claimed_meal_pathway"),
            "expression": "meal_event_g",
            "mechanisms": ["MealPathway"],
            "description": "Additional candidate meal mechanism.",
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
            "mechanisms": ["AuxiliaryAccumulation"],
            "description": "Additional candidate accumulation state.",
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
        token = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        output.append(
            (
                mutation_type,
                _blind_candidate(CandidateModel.model_validate(payload), token),
            )
        )
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
        raw_baseline = CandidateModel.model_validate(summary["selected_candidate"])
        baseline_token = hashlib.sha256(
            json.dumps(raw_baseline.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()[:16]
        baseline = _blind_candidate(raw_baseline, baseline_token)
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
