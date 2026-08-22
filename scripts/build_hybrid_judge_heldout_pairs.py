"""Build baseline-structure-held-out hybrid judge calibration pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from autoformalism.expressions import ValidationContext, repair_protected_declarations
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel
from autoformalism.search.controller import _structural_hash

if __package__:
    from scripts.build_hybrid_judge_pairs import augment_pairs
    from scripts.build_v2_judge_calibration_pairs import (
        _validation_context,
        build_pairs,
    )
else:
    from build_hybrid_judge_pairs import augment_pairs
    from build_v2_judge_calibration_pairs import _validation_context, build_pairs

BASE_MUTATIONS = frozenset(
    {
        "wrong_meal_sink",
        "duplicated_gp_flux",
        "disconnected_claimed_mechanism",
        "unjustified_one_sided_accumulator",
    }
)
MANIFEST_SCHEMA_VERSION = "hybrid-judge-heldout-pairs-1"


def candidate_structure_fingerprint(
    candidate: CandidateModel,
    context: ValidationContext | None = None,
) -> str:
    """Hash executable structure while excluding blinded bookkeeping fields."""
    if context is not None:
        candidate, _ = repair_protected_declarations(candidate, context)
    return _structural_hash(candidate)


def select_heldout_pairs(
    candidate_pairs: tuple[AdversarialPair, ...],
    calibration_pairs: tuple[AdversarialPair, ...],
    *,
    baseline_count: int,
    contexts: Mapping[tuple[str, str], ValidationContext] | None = None,
) -> tuple[tuple[AdversarialPair, ...], tuple[str, ...]]:
    """Select complete mutation groups with unseen baseline structures."""
    if baseline_count < 1:
        raise ValueError("baseline_count must be positive")
    calibration_fingerprints = {
        candidate_structure_fingerprint(
            pair.valid_candidate,
            None if contexts is None else contexts[(pair.benchmark_id, pair.tier)],
        )
        for pair in calibration_pairs
    }
    groups: dict[str, list[AdversarialPair]] = defaultdict(list)
    group_order = []
    for pair in candidate_pairs:
        fingerprint = candidate_structure_fingerprint(
            pair.valid_candidate,
            None if contexts is None else contexts[(pair.benchmark_id, pair.tier)],
        )
        if fingerprint not in groups:
            group_order.append(fingerprint)
        groups[fingerprint].append(pair)
    selected_groups = []
    selected_fingerprints = []
    for fingerprint in group_order:
        if fingerprint in calibration_fingerprints:
            continue
        group = groups[fingerprint]
        mutations = {pair.mutation_type for pair in group}
        if mutations != BASE_MUTATIONS or len(group) != len(BASE_MUTATIONS):
            raise ValueError(
                "candidate mutation group is incomplete or duplicated: "
                f"fingerprint={fingerprint} mutations={sorted(mutations)} "
                f"pairs={len(group)}"
            )
        selected_groups.extend(group)
        selected_fingerprints.append(fingerprint)
        if len(selected_fingerprints) == baseline_count:
            break
    if len(selected_fingerprints) != baseline_count:
        raise ValueError(
            f"requested {baseline_count} unseen baseline structures but found "
            f"{len(selected_fingerprints)}"
        )
    augmented = augment_pairs(tuple(selected_groups))
    pair_ids = [pair.pair_id for pair in augmented]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("held-out pair identifiers must be unique")
    calibration_ids = {pair.pair_id for pair in calibration_pairs}
    overlap = calibration_ids & set(pair_ids)
    if overlap:
        raise ValueError(f"held-out pair identifiers overlap calibration: {overlap}")
    return augmented, tuple(selected_fingerprints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration-pairs", type=Path, required=True)
    parser.add_argument("--baseline-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    calibration_bytes = args.calibration_pairs.read_bytes()
    calibration_pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in calibration_bytes.decode("utf-8").splitlines()
        if line.strip()
    )
    candidates = build_pairs(args.runs_root.resolve(), args.data_root.resolve())
    tasks = {
        (pair.benchmark_id, pair.tier)
        for pair in (*calibration_pairs, *candidates)
    }
    contexts = {
        task: _validation_context(args.data_root.resolve(), *task) for task in tasks
    }
    heldout, fingerprints = select_heldout_pairs(
        candidates,
        calibration_pairs,
        baseline_count=args.baseline_count,
        contexts=contexts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in heldout),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "heldout_pairs_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "calibration_pairs_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "baseline_holdout_unit": "canonical_candidate_structure",
        "selected_baseline_count": len(fingerprints),
        "selected_baseline_fingerprints": list(fingerprints),
        "pair_count": len(heldout),
        "selected_pair_ids": [pair.pair_id for pair in heldout],
        "mutation_types": sorted({pair.mutation_type for pair in heldout}),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(heldout)} held-out pairs from {len(fingerprints)} unseen "
        f"baseline structures to {args.output}"
    )


if __name__ == "__main__":
    main()
