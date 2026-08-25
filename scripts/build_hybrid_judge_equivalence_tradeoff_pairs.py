"""Build equivalence and balanced-defect tradeoff judge-development pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.expressions import (
    ValidationContext,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel

if __package__:
    from scripts.build_hybrid_judge_heldout_pairs import (
        candidate_structure_fingerprint,
    )
    from scripts.build_v2_judge_calibration_pairs import (
        _additive_terms,
        _blind_candidate,
        _component,
        _mutations,
        _validation_context,
    )
else:
    from build_hybrid_judge_heldout_pairs import candidate_structure_fingerprint
    from build_v2_judge_calibration_pairs import (
        _additive_terms,
        _blind_candidate,
        _component,
        _mutations,
        _validation_context,
    )

PAIR_TYPES = (
    "algebraic_reordering_equivalent",
    "tradeoff_wrong_sink_vs_unjustified_accumulator",
    "tradeoff_duplicate_vs_unjustified_accumulator",
    "tradeoff_wrong_sink_vs_duplicate",
)
MANIFEST_SCHEMA_VERSION = "hybrid-judge-equivalence-tradeoff-pairs-1"


def _equivalent_additive_reordering(candidate: CandidateModel) -> CandidateModel:
    """Reverse Gp top-level additive terms without changing its mathematics."""
    payload = candidate.model_dump(mode="json")
    component, key = _component(payload, "Gp")
    terms = _additive_terms(component[key])
    if len(terms) < 2:
        raise ValueError("Gp requires at least two additive terms for reordering")
    reordered = " + ".join(reversed(terms))
    if reordered == component[key]:
        raise ValueError("Gp additive reordering did not change the expression")
    component[key] = reordered
    token = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    return _blind_candidate(CandidateModel.model_validate(payload), token)


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
        f"equivalence-tradeoff-v1:{fingerprint}:{pair_type}".encode()
    ).hexdigest()[:16]
    return AdversarialPair(
        pair_id=f"scorecal_{digest}",
        benchmark_id=benchmark_id,
        tier=tier,
        mutation_type=pair_type,
        valid_candidate=first,
        adversarial_candidate=second,
    )


def build_equivalence_tradeoff_pairs(
    source_pairs: tuple[AdversarialPair, ...],
    *,
    contexts: dict[tuple[str, str], ValidationContext],
    baseline_count: int,
) -> tuple[tuple[AdversarialPair, ...], tuple[str, ...]]:
    """Create one equivalence and three non-ordered tradeoffs per structure."""
    if baseline_count < 1:
        raise ValueError("baseline_count must be positive")
    baselines: dict[str, tuple[CandidateModel, str, str]] = {}
    for pair in source_pairs:
        task = (pair.benchmark_id, pair.tier)
        baseline, _ = repair_protected_declarations(
            pair.valid_candidate,
            contexts[task],
        )
        fingerprint = candidate_structure_fingerprint(baseline, contexts[task])
        baselines.setdefault(fingerprint, (baseline, *task))
    if len(baselines) < baseline_count:
        raise ValueError(
            f"requested {baseline_count} baseline structures but found "
            f"{len(baselines)}"
        )

    output = []
    selected_fingerprints = tuple(list(baselines)[:baseline_count])
    for fingerprint in selected_fingerprints:
        baseline, benchmark_id, tier = baselines[fingerprint]
        context = contexts[(benchmark_id, tier)]
        mutations = dict(_mutations(baseline))
        wrong_sink = mutations["wrong_meal_sink"]
        duplicate = mutations["duplicated_gp_flux"]
        accumulator = mutations["unjustified_one_sided_accumulator"]
        equivalent = _equivalent_additive_reordering(baseline)
        for candidate in (baseline, equivalent, wrong_sink, duplicate, accumulator):
            compile_candidate(candidate, context)
        output.extend(
            (
                _pair(
                    fingerprint=fingerprint,
                    pair_type="algebraic_reordering_equivalent",
                    first=baseline,
                    second=equivalent,
                    benchmark_id=benchmark_id,
                    tier=tier,
                ),
                _pair(
                    fingerprint=fingerprint,
                    pair_type=(
                        "tradeoff_wrong_sink_vs_unjustified_accumulator"
                    ),
                    first=wrong_sink,
                    second=accumulator,
                    benchmark_id=benchmark_id,
                    tier=tier,
                ),
                _pair(
                    fingerprint=fingerprint,
                    pair_type=(
                        "tradeoff_duplicate_vs_unjustified_accumulator"
                    ),
                    first=duplicate,
                    second=accumulator,
                    benchmark_id=benchmark_id,
                    tier=tier,
                ),
                _pair(
                    fingerprint=fingerprint,
                    pair_type="tradeoff_wrong_sink_vs_duplicate",
                    first=wrong_sink,
                    second=duplicate,
                    benchmark_id=benchmark_id,
                    tier=tier,
                ),
            )
        )
    pair_ids = [pair.pair_id for pair in output]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("equivalence/tradeoff pair identifiers must be unique")
    return tuple(output), selected_fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    source_bytes = args.source_pairs.read_bytes()
    source_pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    )
    tasks = {(pair.benchmark_id, pair.tier) for pair in source_pairs}
    contexts = {
        task: _validation_context(args.data_root.resolve(), *task) for task in tasks
    }
    pairs, fingerprints = build_equivalence_tradeoff_pairs(
        source_pairs,
        contexts=contexts,
        baseline_count=args.baseline_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "equivalence_tradeoff_pairs_manifest.json"
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_before_judge_calls",
        "source_pairs": str(args.source_pairs.resolve()),
        "source_pairs_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "baseline_count": len(fingerprints),
        "baseline_fingerprints": list(fingerprints),
        "pair_count": len(pairs),
        "pair_types": list(PAIR_TYPES),
        "pair_ids": [pair.pair_id for pair in pairs],
        "overall_truth_policy": {
            "algebraic_reordering_equivalent": "tie",
            "tradeoff_pairs": "unlabeled",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(pairs)} score-calibration development pairs from "
        f"{len(fingerprints)} opened confirmation structures to {args.output}"
    )


if __name__ == "__main__":
    main()
