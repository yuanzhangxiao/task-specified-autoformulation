"""Build certified mutation-contract labels for hybrid judge calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import (
    ValidationContext,
    repair_protected_declarations,
)
from autoformalism.judging import (
    deterministic_pair_assessments,
    extract_public_requirements,
    semantic_absolute_units,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedAbsoluteLabel,
    ExpectedComparativeLabel,
    ExpectedPairPreference,
    ExpectedVerdict,
    HybridCalibrationLabels,
    expected_from_runtime,
    mutation_label_contract,
)
from autoformalism.schemas import RelativeCriterion


def _public_prompt_and_context(
    data_root: Path,
    pair: AdversarialPair,
) -> tuple[str, ValidationContext]:
    registry = BenchmarkRegistry()
    development = BenchmarkLoader(registry).load_development(
        DataConfig(root=data_root, benchmark_id=pair.benchmark_id, tier=pair.tier)
    )
    arguments = ExecutionArguments(
        data_root=data_root,
        benchmark_id=pair.benchmark_id,
        tier=pair.tier,
        seed=0,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=Path("artifacts/rebuttal/hybrid-labels"),
        resume=False,
        dry_run=True,
        mock_llm=True,
        use_clean_observations=False,
    )
    context = _context(arguments, development)
    spec = registry.get(pair.benchmark_id)
    root = data_root / spec.relative_root
    if spec.data_layout == "legacy_split_files":
        root /= spec.tier_directory_template.format(tier=pair.tier)
    prompt = (root / "proposer_prompt.txt").read_text(encoding="utf-8")
    return prompt, context


def build_label_template(
    pair: AdversarialPair,
    *,
    public_prompt: str,
    task_inputs: tuple[str, ...],
    validation_context: ValidationContext | None = None,
    include_model_semantics: bool = False,
    include_target_mapping_semantics: bool = False,
) -> HybridCalibrationLabels:
    """Create runtime and mutation-contract labels without expert inference."""
    requirements = extract_public_requirements(public_prompt)
    baseline = pair.valid_candidate
    mutated = pair.adversarial_candidate
    if validation_context is not None:
        baseline, _ = repair_protected_declarations(baseline, validation_context)
        mutated, _ = repair_protected_declarations(mutated, validation_context)
    deterministic = deterministic_pair_assessments(
        baseline,
        mutated,
        task_inputs=task_inputs,
    )
    absolute = [
        ExpectedAbsoluteLabel(
            criterion=item.criterion,
            subject_id=item.subject_id,
            baseline=expected_from_runtime(item.candidate_a.verdict),
            mutated=expected_from_runtime(item.candidate_b.verdict),
            rationale="Certified from the canonical dependency graph.",
            label_source="deterministic_runtime",
        )
        for item in deterministic
    ]
    semantic = {
        (criterion, subject): ExpectedAbsoluteLabel(
            criterion=criterion,
            subject_id=subject,
            baseline=ExpectedVerdict.UNLABELED,
            mutated=ExpectedVerdict.UNLABELED,
            rationale=(
                "Not targeted by this controlled mutation; excluded from "
                "question-level accuracy."
            ),
            label_source="not_scored_by_mutation_contract",
        )
        for criterion, subject in semantic_absolute_units(
            requirements,
            include_model_semantics=include_model_semantics,
            include_target_mapping_semantics=(
                include_target_mapping_semantics
            ),
        )
    }
    comparative = {
        criterion: ExpectedComparativeLabel(
            criterion=criterion,
            preference=ExpectedPairPreference.UNLABELED,
            rationale=(
                "Not implied by this controlled mutation; excluded from "
                "question-level accuracy."
            ),
            label_source="not_scored_by_mutation_contract",
        )
        for criterion in RelativeCriterion
    }
    baseline_structure = baseline.model_dump(
        mode="json",
        exclude={"candidate_id", "parent_candidate_id", "change_summary"},
    )
    mutated_structure = mutated.model_dump(
        mode="json",
        exclude={"candidate_id", "parent_candidate_id", "change_summary"},
    )
    structures_equal = baseline_structure == mutated_structure
    overall = ExpectedPairPreference.TIE
    if structures_equal:
        comparative = {
            criterion: ExpectedComparativeLabel(
                criterion=criterion,
                preference=ExpectedPairPreference.TIE,
                rationale=(
                    "Behavior-preserving canonicalization makes the submitted "
                    "scientific structures identical."
                ),
                label_source="canonical_structure_identity",
            )
            for criterion in RelativeCriterion
        }
    else:
        contract = mutation_label_contract(pair.mutation_type)
        overall = contract.overall_preference
        for item in contract.absolute:
            key = (item.criterion, item.subject_id)
            if key not in semantic:
                raise ValueError(
                    "mutation contract requests an unavailable semantic unit: "
                    f"{item.criterion.value}/{item.subject_id}"
                )
            semantic[key] = ExpectedAbsoluteLabel(
                criterion=item.criterion,
                subject_id=item.subject_id,
                baseline=item.baseline,
                mutated=item.mutated,
                rationale=item.rationale,
                label_source=f"mutation_contract:{pair.mutation_type}",
            )
        for item in contract.comparative:
            comparative[item.criterion] = ExpectedComparativeLabel(
                criterion=item.criterion,
                preference=item.preference,
                rationale=item.rationale,
                label_source=f"mutation_contract:{pair.mutation_type}",
            )
    absolute.extend(semantic.values())
    return HybridCalibrationLabels(
        pair_id=pair.pair_id,
        overall_preference=overall,
        absolute_labels=tuple(absolute),
        comparative_labels=tuple(comparative.values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-semantic-contract",
        action="store_true",
        help="include target-mapping and initialization semantic labels",
    )
    parser.add_argument(
        "--target-mapping-semantic-contract",
        action="store_true",
        help="include only the target-mapping semantic label",
    )
    args = parser.parse_args()
    if args.model_semantic_contract and args.target_mapping_semantic_contract:
        raise SystemExit("semantic contract flags are mutually exclusive")
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    contexts: dict[tuple[str, str], tuple[str, ValidationContext]] = {}
    labels = []
    for pair in pairs:
        key = (pair.benchmark_id, pair.tier)
        if key not in contexts:
            contexts[key] = _public_prompt_and_context(
                args.data_root.resolve(), pair
            )
        public_prompt, context = contexts[key]
        labels.append(
            build_label_template(
                pair,
                public_prompt=public_prompt,
                task_inputs=tuple(context.external_inputs),
                validation_context=context,
                include_model_semantics=args.model_semantic_contract,
                include_target_mapping_semantics=(
                    args.target_mapping_semantic_contract
                ),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{item.model_dump_json()}\n" for item in labels),
        encoding="utf-8",
    )
    scored_absolute_sides = sum(
        item.baseline is not ExpectedVerdict.UNLABELED
        for label in labels
        for item in label.absolute_labels
    ) + sum(
        item.mutated is not ExpectedVerdict.UNLABELED
        for label in labels
        for item in label.absolute_labels
    )
    scored_comparative = sum(
        item.preference is not ExpectedPairPreference.UNLABELED
        for label in labels
        for item in label.comparative_labels
    )
    print(
        f"wrote {len(labels)} label templates to {args.output}; "
        f"scored_absolute_sides={scored_absolute_sides}; "
        f"scored_comparative={scored_comparative}; expert_review_items=0"
    )


if __name__ == "__main__":
    main()
