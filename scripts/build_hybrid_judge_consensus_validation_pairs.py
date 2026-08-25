"""Build fresh-structure validation pairs for symmetric judge aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from autoformalism.expressions import (
    ValidationContext,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel

if __package__:
    from scripts.build_hybrid_judge_equivalence_tradeoff_pairs import (
        _equivalent_additive_reordering,
    )
    from scripts.build_hybrid_judge_heldout_pairs import (
        candidate_structure_fingerprint,
    )
    from scripts.build_v2_judge_calibration_pairs import (
        _mutations,
        _validation_context,
        build_pairs,
    )
else:
    from build_hybrid_judge_equivalence_tradeoff_pairs import (
        _equivalent_additive_reordering,
    )
    from build_hybrid_judge_heldout_pairs import candidate_structure_fingerprint
    from build_v2_judge_calibration_pairs import (
        _mutations,
        _validation_context,
        build_pairs,
    )

PAIR_TYPES = (
    "algebraic_reordering_equivalent",
    "wrong_meal_sink",
    "duplicated_gp_flux",
    "unjustified_one_sided_accumulator",
    "additional_accumulator_on_wrong_sink",
    "additional_accumulator_on_duplicate",
    "tradeoff_wrong_sink_vs_duplicate",
)
KNOWN_TIE_TYPES = ("algebraic_reordering_equivalent",)
KNOWN_DOMINANCE_TYPES = (
    "wrong_meal_sink",
    "duplicated_gp_flux",
    "unjustified_one_sided_accumulator",
    "additional_accumulator_on_wrong_sink",
    "additional_accumulator_on_duplicate",
)
UNLABELED_TYPES = ("tradeoff_wrong_sink_vs_duplicate",)
MANIFEST_SCHEMA_VERSION = "hybrid-judge-consensus-validation-pairs-1"


def _read_pairs(path: Path) -> tuple[AdversarialPair, ...]:
    return tuple(
        AdversarialPair.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def select_unseen_baselines(
    candidate_pairs: Sequence[AdversarialPair],
    exclusion_pairs: Sequence[AdversarialPair],
    *,
    baseline_count: int,
    contexts: Mapping[tuple[str, str], ValidationContext],
) -> tuple[tuple[tuple[str, CandidateModel, str, str], ...], int]:
    """Select canonical baseline structures absent from every opened pair set."""
    if baseline_count < 1:
        raise ValueError("baseline_count must be positive")
    excluded = {
        candidate_structure_fingerprint(
            pair.valid_candidate,
            contexts[(pair.benchmark_id, pair.tier)],
        )
        for pair in exclusion_pairs
    }
    selected = []
    seen = set()
    for pair in candidate_pairs:
        task = (pair.benchmark_id, pair.tier)
        baseline, _ = repair_protected_declarations(
            pair.valid_candidate,
            contexts[task],
        )
        fingerprint = candidate_structure_fingerprint(baseline, contexts[task])
        if fingerprint in excluded or fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append((fingerprint, baseline, *task))
        if len(selected) == baseline_count:
            break
    if len(selected) != baseline_count:
        raise ValueError(
            f"requested {baseline_count} unseen baseline structures but found "
            f"{len(selected)}"
        )
    return tuple(selected), len(excluded)


def _pair(
    *,
    fingerprint: str,
    pair_type: str,
    first: CandidateModel,
    second: CandidateModel,
    benchmark_id: str,
    tier: str,
) -> AdversarialPair:
    digest = hashlib.sha256(
        f"consensus-validation-v1:{fingerprint}:{pair_type}".encode()
    ).hexdigest()[:16]
    return AdversarialPair(
        pair_id=f"consensusval_{digest}",
        benchmark_id=benchmark_id,
        tier=tier,
        mutation_type=pair_type,
        valid_candidate=first,
        adversarial_candidate=second,
    )


def build_consensus_validation_pairs(
    baselines: Sequence[tuple[str, CandidateModel, str, str]],
    *,
    contexts: Mapping[tuple[str, str], ValidationContext],
) -> tuple[AdversarialPair, ...]:
    """Create tie, dominance, monotonic-defect, and tradeoff pairs."""
    output = []
    for fingerprint, baseline, benchmark_id, tier in baselines:
        context = contexts[(benchmark_id, tier)]
        mutations = dict(_mutations(baseline))
        wrong_sink = mutations["wrong_meal_sink"]
        duplicate = mutations["duplicated_gp_flux"]
        accumulator = mutations["unjustified_one_sided_accumulator"]
        wrong_plus_accumulator = dict(_mutations(wrong_sink))[
            "unjustified_one_sided_accumulator"
        ]
        duplicate_plus_accumulator = dict(_mutations(duplicate))[
            "unjustified_one_sided_accumulator"
        ]
        equivalent = _equivalent_additive_reordering(baseline)
        for candidate in (
            baseline,
            equivalent,
            wrong_sink,
            duplicate,
            accumulator,
            wrong_plus_accumulator,
            duplicate_plus_accumulator,
        ):
            compile_candidate(candidate, context)
        definitions = (
            ("algebraic_reordering_equivalent", baseline, equivalent),
            ("wrong_meal_sink", baseline, wrong_sink),
            ("duplicated_gp_flux", baseline, duplicate),
            ("unjustified_one_sided_accumulator", baseline, accumulator),
            (
                "additional_accumulator_on_wrong_sink",
                wrong_sink,
                wrong_plus_accumulator,
            ),
            (
                "additional_accumulator_on_duplicate",
                duplicate,
                duplicate_plus_accumulator,
            ),
            ("tradeoff_wrong_sink_vs_duplicate", wrong_sink, duplicate),
        )
        output.extend(
            _pair(
                fingerprint=fingerprint,
                pair_type=pair_type,
                first=first,
                second=second,
                benchmark_id=benchmark_id,
                tier=tier,
            )
            for pair_type, first, second in definitions
        )
    pair_ids = [pair.pair_id for pair in output]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("consensus-validation pair identifiers must be unique")
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-pairs", type=Path, action="append", required=True
    )
    parser.add_argument("--baseline-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    runs_roots = tuple(path.resolve() for path in args.runs_root)
    exclusion_paths = tuple(path.resolve() for path in args.exclude_pairs)
    if len(runs_roots) != len(set(runs_roots)):
        raise ValueError("runs roots must be unique")
    if len(exclusion_paths) != len(set(exclusion_paths)):
        raise ValueError("exclusion pair files must be unique")
    exclusions_by_path = {path: _read_pairs(path) for path in exclusion_paths}
    exclusions = tuple(
        pair for pairs in exclusions_by_path.values() for pair in pairs
    )
    candidates = tuple(
        pair
        for root in runs_roots
        for pair in build_pairs(root, args.data_root.resolve())
    )
    if not candidates:
        raise SystemExit("no completed run summaries found")
    tasks = {
        (pair.benchmark_id, pair.tier) for pair in (*candidates, *exclusions)
    }
    contexts = {
        task: _validation_context(args.data_root.resolve(), *task) for task in tasks
    }
    baselines, excluded_count = select_unseen_baselines(
        candidates,
        exclusions,
        baseline_count=args.baseline_count,
        contexts=contexts,
    )
    pairs = build_consensus_validation_pairs(baselines, contexts=contexts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "consensus_validation_pairs_manifest.json"
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_before_judge_calls",
        "baseline_holdout_unit": "canonical_candidate_structure",
        "pair_identifier_scheme": "consensus_validation_structure_v1",
        "source_runs_roots": [str(path) for path in runs_roots],
        "exclusion_pair_files": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pair_count": len(exclusions_by_path[path]),
            }
            for path in exclusion_paths
        ],
        "excluded_baseline_fingerprint_count": excluded_count,
        "selected_baseline_count": len(baselines),
        "selected_baseline_fingerprints": [item[0] for item in baselines],
        "pair_count": len(pairs),
        "pair_types": list(PAIR_TYPES),
        "known_tie_types": list(KNOWN_TIE_TYPES),
        "known_dominance_types": list(KNOWN_DOMINANCE_TYPES),
        "unlabeled_types": list(UNLABELED_TYPES),
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "overall_truth_policy": {
            "equivalence": "tie",
            "single_and_additional_defects": "first_member_dominates",
            "defect_tradeoffs": "unlabeled",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(pairs)} frozen consensus-validation pairs from "
        f"{len(baselines)} unseen baseline structures to {args.output}; "
        f"excluded_structures={excluded_count}"
    )


if __name__ == "__main__":
    main()
