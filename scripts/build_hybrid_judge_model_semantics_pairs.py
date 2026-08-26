"""Build controlled pairs for target-mapping and initialization judging."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    from scripts.build_hybrid_judge_consensus_validation_pairs import _read_pairs
    from scripts.build_hybrid_judge_heldout_pairs import (
        candidate_structure_fingerprint,
    )
    from scripts.build_v2_judge_calibration_pairs import (
        _validation_context,
        build_pairs,
    )
else:
    from build_hybrid_judge_consensus_validation_pairs import _read_pairs
    from build_hybrid_judge_heldout_pairs import candidate_structure_fingerprint
    from build_v2_judge_calibration_pairs import _validation_context, build_pairs

PAIR_TYPES = (
    "omitted_target_component",
    "unjustified_zero_observed_initialization",
)
MANIFEST_SCHEMA_VERSION = "hybrid-judge-model-semantics-pairs-1"


def _compact(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def _eligible(
    candidate: CandidateModel,
    *,
    target_channel: str,
    target_component: str,
    observed_state: str,
) -> bool:
    """Return whether the controlled edits have an unambiguous local target."""
    target_mappings = [
        item
        for item in candidate.observation_mappings
        if item.channel == target_channel
    ]
    observed_mappings = [
        item
        for item in candidate.observation_mappings
        if item.channel == observed_state
    ]
    initials = [
        item for item in candidate.initial_conditions if item.state == observed_state
    ]
    mechanism_symbols = [
        item
        for item in (*candidate.states, *candidate.processes)
        if item.name == target_component
    ]
    mechanism_claims = {
        mechanism.lower()
        for item in mechanism_symbols
        for mechanism in item.mechanisms
    }
    return (
        len(target_mappings) == 1
        and _compact(target_mappings[0].expression) == target_component
        and len(observed_mappings) == 1
        and _compact(observed_mappings[0].expression) == observed_state
        and len(initials) == 1
        and bool(mechanism_symbols)
        and any("insulin" in claim for claim in mechanism_claims)
    )


def select_unseen_semantic_baselines(
    candidate_pairs: Sequence[AdversarialPair],
    exclusion_pairs: Sequence[AdversarialPair],
    *,
    baseline_count: int,
    contexts: Mapping[tuple[str, str], ValidationContext],
    target_channel: str,
    target_component: str,
    observed_state: str,
    allow_opened_baselines: bool = False,
) -> tuple[tuple[tuple[str, CandidateModel, str, str], ...], int]:
    """Select eligible structures, optionally permitting prior structure use."""
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
        if fingerprint in seen:
            continue
        if not _eligible(
            baseline,
            target_channel=target_channel,
            target_component=target_component,
            observed_state=observed_state,
        ):
            continue
        if fingerprint in excluded and not allow_opened_baselines:
            continue
        seen.add(fingerprint)
        selected.append((fingerprint, baseline, *task))
        if len(selected) == baseline_count:
            break
    if len(selected) != baseline_count:
        raise ValueError(
            f"requested {baseline_count} eligible structures but found "
            f"{len(selected)}"
        )
    return tuple(selected), len(excluded)


def _with_mapping(
    candidate: CandidateModel,
    *,
    target_channel: str,
    expression: str,
    candidate_id: str,
) -> CandidateModel:
    mappings = tuple(
        item.model_copy(update={"expression": expression})
        if item.channel == target_channel
        else item
        for item in candidate.observation_mappings
    )
    return candidate.model_copy(
        update={
            "candidate_id": candidate_id,
            "parent_candidate_id": candidate.candidate_id,
            "change_summary": "controlled target-mapping contract edit",
            "observation_mappings": mappings,
        }
    )


def _with_initialization(
    candidate: CandidateModel,
    *,
    observed_state: str,
    expression: str | None,
    fixed_value: float | None,
    candidate_id: str,
) -> CandidateModel:
    initials = tuple(
        item.model_copy(
            update={
                "expression": expression,
                "fixed_value": fixed_value,
                "initialization_range": None,
            }
        )
        if item.state == observed_state
        else item
        for item in candidate.initial_conditions
    )
    return candidate.model_copy(
        update={
            "candidate_id": candidate_id,
            "parent_candidate_id": candidate.candidate_id,
            "change_summary": "controlled initialization contract edit",
            "initial_conditions": initials,
        }
    )


def build_model_semantics_pairs(
    baselines: Sequence[tuple[str, CandidateModel, str, str]],
    *,
    contexts: Mapping[tuple[str, str], ValidationContext],
    target_channel: str,
    target_component: str,
    complete_target_expression: str,
    observed_state: str,
) -> tuple[AdversarialPair, ...]:
    """Construct isolated mapping-omission and zero-initialization mutations."""
    output = []
    for fingerprint, source, benchmark_id, tier in baselines:
        token = fingerprint[:10]
        baseline = _with_mapping(
            source,
            target_channel=target_channel,
            expression=complete_target_expression,
            candidate_id=f"semantic_baseline_{token}",
        )
        baseline = _with_initialization(
            baseline,
            observed_state=observed_state,
            expression=observed_state,
            fixed_value=None,
            candidate_id=f"semantic_baseline_ready_{token}",
        )
        omitted = _with_mapping(
            baseline,
            target_channel=target_channel,
            expression=target_component,
            candidate_id=f"semantic_omitted_target_{token}",
        )
        zero_initialized = _with_initialization(
            baseline,
            observed_state=observed_state,
            expression=None,
            fixed_value=0.0,
            candidate_id=f"semantic_zero_initial_{token}",
        )
        context = contexts[(benchmark_id, tier)]
        for candidate in (baseline, omitted, zero_initialized):
            compile_candidate(candidate, context)
        for pair_type, mutated in (
            ("omitted_target_component", omitted),
            ("unjustified_zero_observed_initialization", zero_initialized),
        ):
            digest = hashlib.sha256(
                f"model-semantics-v1:{fingerprint}:{pair_type}".encode()
            ).hexdigest()[:16]
            output.append(
                AdversarialPair(
                    pair_id=f"modelsem_{digest}",
                    benchmark_id=benchmark_id,
                    tier=tier,
                    mutation_type=pair_type,
                    valid_candidate=baseline,
                    adversarial_candidate=mutated,
                )
            )
    pair_ids = [pair.pair_id for pair in output]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("model-semantics pair identifiers must be unique")
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exclude-pairs", type=Path, action="append", required=True)
    parser.add_argument("--baseline-count", type=int, default=2)
    parser.add_argument("--target-channel", default="U")
    parser.add_argument("--target-component", default="U")
    parser.add_argument("--complete-target-expression", default="Uii + U")
    parser.add_argument("--observed-state", default="I")
    parser.add_argument(
        "--allow-opened-baselines",
        action="store_true",
        help=(
            "permit structures used by earlier judge studies; the manifest "
            "records the overlap and the result is development calibration, "
            "not fresh-structure confirmation"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    runs_roots = tuple(path.resolve() for path in args.runs_root)
    exclusion_paths = tuple(path.resolve() for path in args.exclude_pairs)
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
    baselines, excluded_count = select_unseen_semantic_baselines(
        candidates,
        exclusions,
        baseline_count=args.baseline_count,
        contexts=contexts,
        target_channel=args.target_channel,
        target_component=args.target_component,
        observed_state=args.observed_state,
        allow_opened_baselines=args.allow_opened_baselines,
    )
    pairs = build_model_semantics_pairs(
        baselines,
        contexts=contexts,
        target_channel=args.target_channel,
        target_component=args.target_component,
        complete_target_expression=args.complete_target_expression,
        observed_state=args.observed_state,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "model_semantics_pairs_manifest.json"
    )
    selected_fingerprints = [item[0] for item in baselines]
    excluded_fingerprints = {
        candidate_structure_fingerprint(
            pair.valid_candidate,
            contexts[(pair.benchmark_id, pair.tier)],
        )
        for pair in exclusions
    }
    selected_overlap = sorted(set(selected_fingerprints) & excluded_fingerprints)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_before_judge_calls",
        "baseline_holdout_unit": "canonical_candidate_structure",
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
        "selected_baseline_fingerprints": selected_fingerprints,
        "allow_opened_baselines": args.allow_opened_baselines,
        "selected_previously_opened_fingerprint_count": len(selected_overlap),
        "selected_previously_opened_fingerprints": selected_overlap,
        "pair_count": len(pairs),
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "mutation_types": list(PAIR_TYPES),
        "public_contract": {
            "target_channel": args.target_channel,
            "target_component": args.target_component,
            "complete_target_expression": args.complete_target_expression,
            "observed_state": args.observed_state,
        },
        "mutation_labels_visible_to_judge": False,
        "interpretation_boundary": (
            "Prior structure exposure is recorded explicitly. The two semantic "
            "criteria and controlled mutations are new, so this artifact is "
            "protocol development calibration rather than unseen-structure "
            "confirmation."
            if args.allow_opened_baselines
            else "Selected canonical structures are absent from supplied opened pairs."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(pairs)} frozen model-semantics pairs from "
        f"{len(baselines)} eligible structures to {args.output}; "
        f"excluded_structures={excluded_count}"
    )


if __name__ == "__main__":
    main()
