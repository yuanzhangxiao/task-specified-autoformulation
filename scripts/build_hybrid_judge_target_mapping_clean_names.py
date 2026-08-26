"""Build clean-name target-mapping pairs from the frozen protocol-v2 pairs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from autoformalism.data import BenchmarkRegistry
from autoformalism.expressions import (
    RestrictedParser,
    ValidationContext,
    compile_candidate,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel

if __package__:
    from scripts.build_hybrid_judge_consensus_validation_pairs import _read_pairs
    from scripts.build_v2_judge_calibration_pairs import _validation_context
else:
    from build_hybrid_judge_consensus_validation_pairs import _read_pairs
    from build_v2_judge_calibration_pairs import _validation_context

MANIFEST_SCHEMA_VERSION = "hybrid-judge-target-mapping-clean-names-pairs-4"
PAIR_TYPE = "omitted_target_component"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def _rename_identifier(expression: str, old: str, new: str) -> str:
    """Rename one expression identifier without substring replacement."""

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == old:
                return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
            return node

    parsed = ast.parse(expression, mode="eval")
    changed = Rename().visit(parsed)
    return ast.unparse(ast.fix_missing_locations(changed))


def _rename_component(
    candidate: CandidateModel,
    *,
    old: str,
    new: str,
) -> CandidateModel:
    """Rename one algebraic component and every typed reference to it."""
    payload = candidate.model_dump(mode="json")
    occupied = {
        *(item["name"] for item in payload["states"]),
        *(item["name"] for item in payload["processes"]),
        *(item["name"] for item in payload["parameters"]),
    }
    if new in occupied:
        raise ValueError(f"clean component name is already declared: {new}")
    matching = [item for item in payload["processes"] if item["name"] == old]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one process named {old}")
    matching[0]["name"] = new

    for equation in payload["state_equations"]:
        equation["rhs"] = _rename_identifier(equation["rhs"], old, new)
    for process in payload["processes"]:
        process["expression"] = _rename_identifier(
            process["expression"], old, new
        )
    for mapping in payload["observation_mappings"]:
        mapping["expression"] = _rename_identifier(
            mapping["expression"], old, new
        )
    for initial in payload["initial_conditions"]:
        if initial["expression"] is not None:
            initial["expression"] = _rename_identifier(
                initial["expression"], old, new
            )
    for constraint in payload["constraints"]:
        if constraint["subject"] == old:
            constraint["subject"] = new
    return CandidateModel.model_validate(payload)


def _with_total_process(
    candidate: CandidateModel,
    *,
    target_channel: str,
    target_process: str,
    expression: str,
    candidate_id: str,
) -> CandidateModel:
    payload = candidate.model_dump(mode="json")
    mappings = [
        item
        for item in payload["observation_mappings"]
        if item["channel"] == target_channel
    ]
    if len(mappings) != 1:
        raise ValueError(
            f"expected exactly one observation mapping for {target_channel}"
        )
    mappings[0]["expression"] = target_process
    payload["processes"].append(
        {
            "name": target_process,
            "expression": expression,
            "unit": "unspecified",
            "description": "Total glucose utilization/disposal target.",
            "mechanisms": [],
        }
    )
    payload["candidate_id"] = candidate_id
    payload["parent_candidate_id"] = candidate.candidate_id
    payload["change_summary"] = (
        "Clean separation of total disposal from its insulin-dependent component."
    )
    return CandidateModel.model_validate(payload)


def _without_controlled_total_expression(
    candidate: CandidateModel,
    *,
    target_process: str,
) -> dict[str, object]:
    payload = candidate.model_dump(
        mode="json",
        exclude={"candidate_id", "parent_candidate_id", "change_summary"},
    )
    for process in payload["processes"]:
        if process["name"] == target_process:
            process["expression"] = "__CONTROLLED_TOTAL_EXPRESSION__"
    return payload


def certify_clean_pair(
    baseline: CandidateModel,
    mutated: CandidateModel,
    *,
    target_channel: str,
    target_process: str,
    dependent_process: str,
    supplied_component: str,
) -> dict[str, object]:
    """Fail closed unless only the total-process expression differs."""
    parser = RestrictedParser()

    def process(candidate: CandidateModel, name: str):
        matches = [item for item in candidate.processes if item.name == name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one process named {name}")
        return matches[0]

    def mapping(candidate: CandidateModel) -> str:
        matches = [
            item
            for item in candidate.observation_mappings
            if item.channel == target_channel
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one observation mapping for {target_channel}"
            )
        return matches[0].expression

    baseline_dependent = process(baseline, dependent_process)
    mutated_dependent = process(mutated, dependent_process)
    if baseline_dependent != mutated_dependent:
        raise ValueError("insulin-dependent process differs across the pair")
    dependent_symbols = set(
        parser.parse(
            baseline_dependent.expression,
            location=f"process:{dependent_process}",
        ).symbols
    )
    if supplied_component in dependent_symbols:
        raise ValueError(
            f"{dependent_process} must exclude supplied {supplied_component}"
        )
    if not any(
        "insulin" in mechanism.lower()
        for mechanism in baseline_dependent.mechanisms
    ):
        raise ValueError(
            f"{dependent_process} lacks an insulin-related mechanism claim"
        )

    baseline_total = process(baseline, target_process)
    mutated_total = process(mutated, target_process)
    baseline_symbols = set(
        parser.parse(
            baseline_total.expression,
            location=f"process:{target_process}:baseline",
        ).symbols
    )
    mutated_symbols = set(
        parser.parse(
            mutated_total.expression,
            location=f"process:{target_process}:mutated",
        ).symbols
    )
    if baseline_symbols != {supplied_component, dependent_process}:
        raise ValueError(
            "complete total-process symbols differ from the clean contract: "
            f"{sorted(baseline_symbols)}"
        )
    if mutated_symbols != {dependent_process}:
        raise ValueError(
            "omitted total-process symbols differ from the clean contract: "
            f"{sorted(mutated_symbols)}"
        )
    if _compact(baseline_total.expression) != (
        f"{supplied_component}+{dependent_process}"
    ):
        raise ValueError("complete total-process expression is not the frozen sum")
    if _compact(mutated_total.expression) != dependent_process:
        raise ValueError("omitted total-process expression is not the frozen component")
    if mapping(baseline).strip() != target_process:
        raise ValueError("baseline target mapping is not the total process")
    if mapping(mutated).strip() != target_process:
        raise ValueError("mutated target mapping is not the total process")
    if _without_controlled_total_expression(
        baseline, target_process=target_process
    ) != _without_controlled_total_expression(
        mutated, target_process=target_process
    ):
        raise ValueError("pair differs outside the controlled total expression")
    return {
        "target_mapping_is_same_named_total_process": True,
        "dependent_process_excludes_supplied_component": True,
        "dependent_process_has_insulin_claim": True,
        "baseline_total_symbols": sorted(baseline_symbols),
        "mutated_total_symbols": sorted(mutated_symbols),
        "pair_diff_isolated_to_total_process_expression": True,
    }


def build_clean_target_mapping_pairs(
    source_pairs: Sequence[AdversarialPair],
    *,
    contexts: Mapping[tuple[str, str], ValidationContext],
    target_channel: str = "U",
    target_process: str = "U",
    dependent_process: str = "Uid",
    supplied_component: str = "Uii",
) -> tuple[tuple[AdversarialPair, ...], dict[str, dict[str, object]]]:
    """Convert each frozen v2 baseline into one clean-name omission pair."""
    output = []
    certifications = {}
    for source_pair in source_pairs:
        if source_pair.mutation_type != PAIR_TYPE:
            raise ValueError(
                f"unexpected source mutation type: {source_pair.mutation_type}"
            )
        renamed = _rename_component(
            source_pair.valid_candidate,
            old=target_process,
            new=dependent_process,
        )
        token = hashlib.sha256(
            f"target-mapping-clean-names-v4:{source_pair.pair_id}".encode()
        ).hexdigest()[:16]
        baseline = _with_total_process(
            renamed,
            target_channel=target_channel,
            target_process=target_process,
            expression=f"{supplied_component} + {dependent_process}",
            candidate_id=f"clean_total_baseline_{token}",
        )
        mutated = _with_total_process(
            renamed,
            target_channel=target_channel,
            target_process=target_process,
            expression=dependent_process,
            candidate_id=f"clean_total_omitted_{token}",
        )
        context = contexts[(source_pair.benchmark_id, source_pair.tier)]
        compile_candidate(baseline, context)
        compile_candidate(mutated, context)
        pair_id = f"cleanmap_{token}"
        certifications[pair_id] = certify_clean_pair(
            baseline,
            mutated,
            target_channel=target_channel,
            target_process=target_process,
            dependent_process=dependent_process,
            supplied_component=supplied_component,
        )
        output.append(
            AdversarialPair(
                pair_id=pair_id,
                benchmark_id=source_pair.benchmark_id,
                tier=source_pair.tier,
                mutation_type=PAIR_TYPE,
                valid_candidate=baseline,
                adversarial_candidate=mutated,
            )
        )
    return tuple(output), certifications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pairs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    expected_pair_hash = config["matched_source"]["source_pairs_sha256"]
    source_pair_hash = _sha256(args.input_pairs)
    if source_pair_hash != expected_pair_hash:
        raise ValueError(
            f"source pair SHA-256 differs from frozen value: {source_pair_hash}"
        )
    source_pairs = _read_pairs(args.input_pairs)
    expected_pair_count = int(config["planned"]["pairs"])
    if len(source_pairs) != expected_pair_count:
        raise ValueError(
            f"source pair count differs from frozen value: {len(source_pairs)}"
        )
    tasks = {(pair.benchmark_id, pair.tier) for pair in source_pairs}
    contexts = {
        task: _validation_context(args.data_root.resolve(), *task)
        for task in tasks
    }

    prompt_contract = config["public_prompt_contract"]
    benchmark_id = prompt_contract["benchmark_id"]
    if {pair.benchmark_id for pair in source_pairs} != {benchmark_id}:
        raise ValueError("source pair benchmarks differ from the public prompt")
    spec = BenchmarkRegistry().get(benchmark_id)
    prompt = args.data_root.resolve() / spec.relative_root / "proposer_prompt.txt"
    prompt_hash = _sha256(prompt)
    if prompt_hash != prompt_contract["proposer_prompt_sha256"]:
        raise ValueError(
            f"public prompt SHA-256 differs from frozen value: {prompt_hash}"
        )

    construction = config["pair_construction"]
    pairs, certifications = build_clean_target_mapping_pairs(
        source_pairs,
        contexts=contexts,
        target_channel=construction["target_channel"],
        target_process=construction["total_process"],
        dependent_process=construction["dependent_process"],
        supplied_component=construction["supplied_component"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_before_judge_calls",
        "source_pairs": str(args.input_pairs.resolve()),
        "source_pairs_sha256": source_pair_hash,
        "public_prompt_sha256": prompt_hash,
        "pair_count": len(pairs),
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "mutation_types": sorted({pair.mutation_type for pair in pairs}),
        "pair_construction": construction,
        "certifications": certifications,
        "mutation_labels_visible_to_judge": False,
        "hidden_generator_visible_to_judge": False,
        "interpretation_boundary": (
            "U is explicitly the observed total process in both candidates; "
            "Uid is the insulin-dependent component. Only inclusion of the "
            "supplied Uii component in U differs across each pair."
        ),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(pairs)} clean-name target-mapping pairs to {args.output}"
    )


if __name__ == "__main__":
    main()
