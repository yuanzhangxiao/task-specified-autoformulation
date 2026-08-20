"""Build human-review templates for hybrid judge question-level labels."""

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
) -> HybridCalibrationLabels:
    """Create certified labels and explicit placeholders for expert review."""
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
    absolute.extend(
        ExpectedAbsoluteLabel(
            criterion=criterion,
            subject_id=subject,
            baseline=ExpectedVerdict.UNLABELED,
            mutated=ExpectedVerdict.UNLABELED,
            rationale="REVIEW REQUIRED against the public task and equations.",
            label_source="domain_expert_pending",
        )
        for criterion, subject in semantic_absolute_units(requirements)
    )
    comparative = tuple(
        ExpectedComparativeLabel(
            criterion=criterion,
            preference=ExpectedPairPreference.UNLABELED,
            rationale="REVIEW REQUIRED using only the public task and candidates.",
            label_source="domain_expert_pending",
        )
        for criterion in RelativeCriterion
    )
    baseline_structure = baseline.model_dump(
        mode="json",
        exclude={"candidate_id", "parent_candidate_id", "change_summary"},
    )
    mutated_structure = mutated.model_dump(
        mode="json",
        exclude={"candidate_id", "parent_candidate_id", "change_summary"},
    )
    return HybridCalibrationLabels(
        pair_id=pair.pair_id,
        overall_preference=(
            ExpectedPairPreference.TIE
            if baseline_structure == mutated_structure
            else ExpectedPairPreference.BASELINE
        ),
        absolute_labels=tuple(absolute),
        comparative_labels=comparative,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{item.model_dump_json()}\n" for item in labels),
        encoding="utf-8",
    )
    pending = sum(
        item.baseline is ExpectedVerdict.UNLABELED
        or item.mutated is ExpectedVerdict.UNLABELED
        for label in labels
        for item in label.absolute_labels
    ) + sum(
        item.preference is ExpectedPairPreference.UNLABELED
        for label in labels
        for item in label.comparative_labels
    )
    print(
        f"wrote {len(labels)} label templates to {args.output}; "
        f"expert_review_items={pending}"
    )


if __name__ == "__main__":
    main()
