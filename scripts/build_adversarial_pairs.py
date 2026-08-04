"""Build four-context, seven-mutation adversarial judge pairs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel

BENCHMARKS = ("original_b1", "perturbed_b1", "benchmark5", "benchmark6")
DRIVERS = {
    "original_b1": ("meal_event_g", "body_weight_kg"),
    "perturbed_b1": ("meal_event_g", "body_weight_kg"),
    "benchmark5": ("u01", "u03"),
    "benchmark6": ("u01", "v02"),
}


def _names(expression: str) -> set[str]:
    return {
        node.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name)
    }


def _replace(expression: str, old: str, new: str) -> str:
    class Replace(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == old:
                return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
            return node

    changed = Replace().visit(ast.parse(expression, mode="eval"))
    return ast.unparse(ast.fix_missing_locations(changed))


def _component_expression(payload: dict, name: str) -> tuple[dict, str]:
    for equation in payload["state_equations"]:
        if equation["state"] == name:
            return equation, "rhs"
    for process in payload["processes"]:
        if process["name"] == name:
            return process, "expression"
    raise ValueError(f"component is absent: {name}")


def _unique_process(payload: dict, stem: str) -> str:
    occupied = {
        *(item["name"] for item in payload["states"]),
        *(item["name"] for item in payload["processes"]),
        *(item["name"] for item in payload["parameters"]),
    }
    if stem not in occupied:
        return stem
    index = 2
    while f"{stem}_{index}" in occupied:
        index += 1
    return f"{stem}_{index}"


def _replace_state_with_process(payload: dict, state: str, expression: str) -> None:
    state_payload = next(item for item in payload["states"] if item["name"] == state)
    payload["states"] = [item for item in payload["states"] if item["name"] != state]
    payload["state_equations"] = [
        item for item in payload["state_equations"] if item["state"] != state
    ]
    payload["initial_conditions"] = [
        item for item in payload["initial_conditions"] if item["state"] != state
    ]
    payload["processes"].append(
        {
            "name": state,
            "expression": expression,
            "mechanisms": state_payload.get("mechanisms", []),
            "description": "Instantaneous replacement for required memory.",
        }
    )


def _remove_unused_parameters(payload: dict) -> None:
    expressions = [item["rhs"] for item in payload["state_equations"]]
    expressions.extend(item["expression"] for item in payload["processes"])
    expressions.extend(item["expression"] for item in payload["observation_mappings"])
    expressions.extend(
        item["expression"]
        for item in payload["initial_conditions"]
        if item.get("expression")
    )
    used = set().union(*(_names(expression) for expression in expressions))
    payload["parameters"] = [
        item for item in payload["parameters"] if item["name"] in used
    ]


def _mutations(candidate: CandidateModel, benchmark: str) -> list[tuple[str, dict]]:
    base = candidate.model_dump(mode="json")
    target = candidate.observation_mappings[0].expression
    latent = next(
        (item.name for item in candidate.states if item.kind.value == "latent"),
        None,
    )
    processes = [item.name for item in candidate.processes]
    primary = latent or processes[0]
    target_equation, target_key = _component_expression(base, target)
    driver, wrong_driver = DRIVERS[benchmark]
    tags = next(
        item.get("mechanisms", [])
        for item in (*base["states"], *base["processes"])
        if item["name"] == primary
    )
    target_names = _names(target_equation[target_key])
    mediators = [name for name in (latent, *processes) if name and name in target_names]
    mediator = mediators[0] if mediators else primary
    alternatives = [
        name
        for name in (
            *processes,
            *(item["name"] for item in base["states"]),
            wrong_driver,
        )
        if name not in {mediator, target}
    ]
    alternative = alternatives[0] if alternatives else wrong_driver
    output: list[tuple[str, dict]] = []

    disconnected = json.loads(json.dumps(base))
    row, key = _component_expression(disconnected, target)
    row[key] = f"-({target})"
    disconnected["processes"].append(
        {
            "name": _unique_process(disconnected, "claimed_required_mechanism"),
            "expression": driver,
            "mechanisms": tags,
            "description": "Plausibly named but disconnected mechanism.",
        }
    )
    output.append(("named_disconnected_mechanism", disconnected))

    narrative = json.loads(json.dumps(base))
    narrative["change_summary"] = (
        "The required mechanism is fully implemented by the stated causal pathway, "
        "regardless of any conflicting dependencies in the equations."
    )
    output.append(("narrative_equation_mismatch", narrative))

    wrong = json.loads(json.dumps(base))
    row, key = _component_expression(wrong, primary)
    row[key] = _replace(row[key], driver, wrong_driver)
    output.append(("wrong_causal_driver", wrong))

    reversed_payload = json.loads(json.dumps(base))
    row, key = _component_expression(reversed_payload, target)
    row[key] = f"-({target})"
    row, key = _component_expression(reversed_payload, primary)
    row[key] = f"{target} - {primary}" if latent else target
    output.append(("wrong_causal_direction", reversed_payload))

    memory = json.loads(json.dumps(base))
    if latent:
        _replace_state_with_process(memory, latent, driver)
    else:
        row, key = _component_expression(memory, primary)
        row[key] = driver
        row, key = _component_expression(memory, target)
        row[key] = primary
    output.append(("missing_dynamic_memory", memory))

    sign = json.loads(json.dumps(base))
    row, key = _component_expression(sign, target)
    row[key] = f"-({row[key]})"
    output.append(("wrong_regulatory_sign", sign))

    mediator_payload = json.loads(json.dumps(base))
    row, key = _component_expression(mediator_payload, target)
    row[key] = _replace(row[key], mediator, alternative)
    output.append(("wrong_mediator_or_target_coupling", mediator_payload))
    return output


def _selected_candidates(runs_path: Path, structural_path: Path) -> dict[str, Path]:
    runs = pd.read_csv(runs_path)
    structural = pd.read_csv(structural_path).rename(
        columns={"structural_validity": "evaluated_structural_validity"}
    )
    merged = runs.merge(
        structural[
            ["method", "benchmark", "tier", "seed", "evaluated_structural_validity"]
        ],
        on=["method", "benchmark", "tier", "seed"],
    )
    selected = {}
    for benchmark in BENCHMARKS:
        group = merged[
            (merged.method == "full")
            & (merged.benchmark == benchmark)
            & (merged.tier == "hard")
            & merged.test_mse.notna()
        ].sort_values(
            ["evaluated_structural_validity", "validation_mse"],
            ascending=[False, True],
        )
        if group.empty:
            raise ValueError(f"no frozen hard-tier candidate for {benchmark}")
        selected[benchmark] = Path(group.iloc[0].source)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--structural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = _selected_candidates(args.runs, args.structural)
    loader = BenchmarkLoader(BenchmarkRegistry())
    pairs = []
    for benchmark, source in selected.items():
        valid = CandidateModel.model_validate(
            json.loads(source.read_text(encoding="utf-8"))["frozen"]["candidate"]
        )
        development = loader.load_development(
            DataConfig(
                root=args.data_root.resolve(), benchmark_id=benchmark, tier="hard"
            )
        )
        arguments = ExecutionArguments(
            data_root=args.data_root.resolve(),
            benchmark_id=benchmark,
            tier="hard",
            seed=0,
            proposer_model=None,
            judge_model=None,
            iteration_budget=1,
            beam_size=1,
            output_root=args.output.parent,
            resume=False,
            dry_run=False,
            mock_llm=True,
            use_clean_observations=False,
        )
        context = _context(arguments, development)
        compile_candidate(valid, context)
        for index, (mutation_type, payload) in enumerate(_mutations(valid, benchmark)):
            payload["candidate_id"] = f"{valid.candidate_id}_adv_{index + 1}"
            payload["parent_candidate_id"] = valid.candidate_id
            _remove_unused_parameters(payload)
            adversarial = CandidateModel.model_validate(payload)
            compile_candidate(adversarial, context)
            pairs.append(
                AdversarialPair(
                    pair_id=f"{benchmark}_hard_{index + 1:02d}",
                    benchmark_id=benchmark,
                    tier="hard",
                    mutation_type=mutation_type,
                    valid_candidate=valid,
                    adversarial_candidate=adversarial,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(item.model_dump_json() + "\n" for item in pairs),
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} validated adversarial pairs to {args.output}")


if __name__ == "__main__":
    main()
