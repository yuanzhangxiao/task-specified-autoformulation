"""Build frozen judge-confirmation pairs from never-evaluated model structures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.adversarial import AdversarialPair

if __package__:
    from scripts.build_hybrid_judge_heldout_pairs import (
        candidate_structure_fingerprint,
        select_heldout_pairs,
    )
    from scripts.build_v2_judge_calibration_pairs import (
        _validation_context,
        build_pairs,
    )
else:
    from build_hybrid_judge_heldout_pairs import (
        candidate_structure_fingerprint,
        select_heldout_pairs,
    )
    from build_v2_judge_calibration_pairs import _validation_context, build_pairs

CONFIRMATION_MUTATIONS = (
    "duplicated_gp_flux",
    "wrong_meal_sink",
)
MANIFEST_SCHEMA_VERSION = "hybrid-judge-structure-confirmation-pairs-1"


def _read_pairs(path: Path) -> tuple[AdversarialPair, ...]:
    """Read a newline-delimited adversarial-pair artifact."""
    return tuple(
        AdversarialPair.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def deduplicate_candidate_pairs(
    pairs: Sequence[AdversarialPair],
    *,
    contexts: Mapping[tuple[str, str], ValidationContext | None],
) -> tuple[AdversarialPair, ...]:
    """Keep one deterministic mutation per canonical baseline structure."""
    output: list[AdversarialPair] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        context = contexts[(pair.benchmark_id, pair.tier)]
        fingerprint = candidate_structure_fingerprint(pair.valid_candidate, context)
        key = (fingerprint, pair.mutation_type)
        if key in seen:
            continue
        seen.add(key)
        output.append(pair)
    return tuple(output)


def _confirmation_pair(
    pair: AdversarialPair,
    *,
    baseline_fingerprint: str,
) -> AdversarialPair:
    """Assign an ID in a namespace reserved for the frozen confirmation."""
    digest = hashlib.sha256(
        (
            "hybrid-structure-confirmation-v1:"
            f"{baseline_fingerprint}:{pair.mutation_type}:{pair.pair_id}"
        ).encode()
    ).hexdigest()[:16]
    return pair.model_copy(update={"pair_id": f"confirmation_{digest}"})


def build_confirmation_pairs(
    candidate_pairs: Sequence[AdversarialPair],
    exclusion_pairs: Sequence[AdversarialPair],
    *,
    baseline_count: int,
    contexts: Mapping[tuple[str, str], ValidationContext | None],
) -> tuple[tuple[AdversarialPair, ...], tuple[str, ...]]:
    """Select targeted mutations over structures absent from every exclusion set."""
    candidates = deduplicate_candidate_pairs(candidate_pairs, contexts=contexts)
    heldout, fingerprints = select_heldout_pairs(
        candidates,
        tuple(exclusion_pairs),
        baseline_count=baseline_count,
        contexts=contexts,
    )
    heldout_by_key: dict[tuple[str, str], AdversarialPair] = {}
    for pair in heldout:
        fingerprint = candidate_structure_fingerprint(
            pair.valid_candidate,
            contexts[(pair.benchmark_id, pair.tier)],
        )
        heldout_by_key[(fingerprint, pair.mutation_type)] = pair
    selected = tuple(
        _confirmation_pair(
            heldout_by_key[(fingerprint, mutation)],
            baseline_fingerprint=fingerprint,
        )
        for fingerprint in fingerprints
        for mutation in CONFIRMATION_MUTATIONS
    )
    expected = baseline_count * len(CONFIRMATION_MUTATIONS)
    if len(selected) != expected:
        raise ValueError(
            f"expected {expected} targeted confirmation pairs but found "
            f"{len(selected)}"
        )
    pair_ids = [pair.pair_id for pair in selected]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("confirmation pair identifiers must be unique")
    excluded_fingerprints = {
        candidate_structure_fingerprint(
            pair.valid_candidate,
            contexts[(pair.benchmark_id, pair.tier)],
        )
        for pair in exclusion_pairs
    }
    overlap = excluded_fingerprints & set(fingerprints)
    if overlap:
        raise AssertionError(
            f"confirmation structures were previously opened: {overlap}"
        )
    return selected, fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        required=True,
        help="Completed run root; repeat to inventory multiple experiment roots.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-pairs",
        type=Path,
        action="append",
        required=True,
        help=(
            "Previously opened pair file; repeat for every "
            "development/evaluation set."
        ),
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
    exclusion_pairs = tuple(
        pair for pairs in exclusions_by_path.values() for pair in pairs
    )
    candidate_pairs = tuple(
        pair
        for root in runs_roots
        for pair in build_pairs(root, args.data_root.resolve())
    )
    if not candidate_pairs:
        raise SystemExit("no completed run summaries found")
    tasks = {
        (pair.benchmark_id, pair.tier)
        for pair in (*candidate_pairs, *exclusion_pairs)
    }
    contexts = {
        task: _validation_context(args.data_root.resolve(), *task) for task in tasks
    }
    confirmation, fingerprints = build_confirmation_pairs(
        candidate_pairs,
        exclusion_pairs,
        baseline_count=args.baseline_count,
        contexts=contexts,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in confirmation),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "confirmation_pairs_manifest.json"
    )
    exclusion_manifest = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pair_count": len(exclusions_by_path[path]),
        }
        for path in exclusion_paths
    ]
    excluded_fingerprints = {
        candidate_structure_fingerprint(
            pair.valid_candidate,
            contexts[(pair.benchmark_id, pair.tier)],
        )
        for pair in exclusion_pairs
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_before_judge_calls",
        "baseline_holdout_unit": "canonical_candidate_structure",
        "pair_identifier_scheme": "structure_confirmation_mutation_source_v1",
        "source_runs_roots": [str(path) for path in runs_roots],
        "exclusion_pair_files": exclusion_manifest,
        "excluded_baseline_fingerprint_count": len(excluded_fingerprints),
        "selected_baseline_count": len(fingerprints),
        "selected_baseline_fingerprints": list(fingerprints),
        "pair_count": len(confirmation),
        "selected_pair_ids": [pair.pair_id for pair in confirmation],
        "mutation_types": list(CONFIRMATION_MUTATIONS),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(confirmation)} frozen confirmation pairs from "
        f"{len(fingerprints)} unseen baseline structures to {args.output}; "
        f"excluded_structures={len(excluded_fingerprints)}"
    )


if __name__ == "__main__":
    main()
