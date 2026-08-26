"""Build certified target-mapping pairs for judge protocol v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from autoformalism.expressions import (
    RestrictedParser,
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

MANIFEST_SCHEMA_VERSION = "hybrid-judge-target-mapping-pairs-2"
PAIR_TYPE = "omitted_target_component"


def _compact(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def _target_process_symbols(
    candidate: CandidateModel,
    *,
    target_component: str,
) -> set[str] | None:
    processes = [
        item for item in candidate.processes if item.name == target_component
    ]
    if len(processes) != 1:
        return None
    return set(
        RestrictedParser().parse(
            processes[0].expression,
            location=f"process:{target_component}",
        ).symbols
    )


def _eligible(
    candidate: CandidateModel,
    *,
    target_channel: str,
    target_component: str,
    supplied_component: str,
) -> bool:
    mappings = [
        item
        for item in candidate.observation_mappings
        if item.channel == target_channel
    ]
    symbols = _target_process_symbols(
        candidate, target_component=target_component
    )
    claims = {
        mechanism.lower()
        for item in candidate.processes
        if item.name == target_component
        for mechanism in item.mechanisms
    }
    return (
        len(mappings) == 1
        and _compact(mappings[0].expression) == target_component
        and symbols is not None
        and supplied_component not in symbols
        and any("insulin" in claim for claim in claims)
    )


def select_baselines(
    candidate_pairs: Sequence[AdversarialPair],
    exclusion_pairs: Sequence[AdversarialPair],
    *,
    baseline_count: int,
    contexts: Mapping[tuple[str, str], ValidationContext],
    target_channel: str,
    target_component: str,
    supplied_component: str,
    allow_opened_baselines: bool,
) -> tuple[tuple[tuple[str, CandidateModel, str, str], ...], set[str]]:
    """Select distinct eligible structures and return excluded fingerprints."""
    excluded = {
        candidate_structure_fingerprint(
            pair.valid_candidate,
            contexts[(pair.benchmark_id, pair.tier)],
        )
        for pair in exclusion_pairs
    }
    selected = []
    seen: set[str] = set()
    for pair in candidate_pairs:
        task = (pair.benchmark_id, pair.tier)
        baseline, _ = repair_protected_declarations(
            pair.valid_candidate, contexts[task]
        )
        fingerprint = candidate_structure_fingerprint(
            baseline, contexts[task]
        )
        if fingerprint in seen:
            continue
        if not _eligible(
            baseline,
            target_channel=target_channel,
            target_component=target_component,
            supplied_component=supplied_component,
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
    return tuple(selected), excluded


def _with_mapping(
    candidate: CandidateModel,
    *,
    target_channel: str,
    expression: str,
    candidate_id: str,
) -> CandidateModel:
    return candidate.model_copy(
        update={
            "candidate_id": candidate_id,
            "parent_candidate_id": candidate.candidate_id,
            "change_summary": "certified controlled target-mapping edit",
            "observation_mappings": tuple(
                item.model_copy(update={"expression": expression})
                if item.channel == target_channel
                else item
                for item in candidate.observation_mappings
            ),
        }
    )


def _scientific_structure_without_target_mapping(
    candidate: CandidateModel,
    *,
    target_channel: str,
) -> dict[str, object]:
    payload = candidate.model_dump(
        mode="json",
        exclude={"candidate_id", "parent_candidate_id", "change_summary"},
    )
    for mapping in payload["observation_mappings"]:
        if mapping["channel"] == target_channel:
            mapping["expression"] = "__CONTROLLED_TARGET_MAPPING__"
    return payload


def certify_pair(
    baseline: CandidateModel,
    mutated: CandidateModel,
    *,
    target_channel: str,
    target_component: str,
    supplied_component: str,
) -> dict[str, object]:
    """Fail closed unless the intended target-mapping contrast is isolated."""
    parser = RestrictedParser()
    process_symbols = _target_process_symbols(
        baseline, target_component=target_component
    )
    if process_symbols is None or supplied_component in process_symbols:
        raise ValueError(
            f"{target_component} must be one process that excludes "
            f"{supplied_component}"
        )
    baseline_mapping = next(
        item
        for item in baseline.observation_mappings
        if item.channel == target_channel
    )
    mutated_mapping = next(
        item
        for item in mutated.observation_mappings
        if item.channel == target_channel
    )
    baseline_symbols = set(
        parser.parse(
            baseline_mapping.expression,
            location=f"observation:{target_channel}:baseline",
        ).symbols
    )
    mutated_symbols = set(
        parser.parse(
            mutated_mapping.expression,
            location=f"observation:{target_channel}:mutated",
        ).symbols
    )
    expected_baseline = {supplied_component, target_component}
    expected_mutated = {target_component}
    if baseline_symbols != expected_baseline:
        raise ValueError(
            "complete target mapping symbols differ from certified contract: "
            f"{sorted(baseline_symbols)}"
        )
    if mutated_symbols != expected_mutated:
        raise ValueError(
            "omitted target mapping symbols differ from certified contract: "
            f"{sorted(mutated_symbols)}"
        )
    if _scientific_structure_without_target_mapping(
        baseline, target_channel=target_channel
    ) != _scientific_structure_without_target_mapping(
        mutated, target_channel=target_channel
    ):
        raise ValueError("pair differs outside the controlled target mapping")
    return {
        "target_process_excludes_supplied_component": True,
        "baseline_mapping_symbols": sorted(baseline_symbols),
        "mutated_mapping_symbols": sorted(mutated_symbols),
        "pair_diff_isolated_to_target_mapping": True,
    }


def build_target_mapping_pairs(
    baselines: Sequence[tuple[str, CandidateModel, str, str]],
    *,
    contexts: Mapping[tuple[str, str], ValidationContext],
    target_channel: str,
    target_component: str,
    supplied_component: str,
) -> tuple[tuple[AdversarialPair, ...], dict[str, dict[str, object]]]:
    """Build one certified omission pair per baseline structure."""
    pairs = []
    certifications = {}
    for fingerprint, source, benchmark_id, tier in baselines:
        token = fingerprint[:10]
        baseline = _with_mapping(
            source,
            target_channel=target_channel,
            expression=f"{supplied_component} + {target_component}",
            candidate_id=f"target_mapping_baseline_{token}",
        )
        mutated = _with_mapping(
            baseline,
            target_channel=target_channel,
            expression=target_component,
            candidate_id=f"target_mapping_omitted_{token}",
        )
        context = contexts[(benchmark_id, tier)]
        compile_candidate(baseline, context)
        compile_candidate(mutated, context)
        digest = hashlib.sha256(
            f"target-mapping-v2:{fingerprint}:{PAIR_TYPE}".encode()
        ).hexdigest()[:16]
        pair_id = f"targetmap_{digest}"
        certifications[pair_id] = certify_pair(
            baseline,
            mutated,
            target_channel=target_channel,
            target_component=target_component,
            supplied_component=supplied_component,
        )
        pairs.append(
            AdversarialPair(
                pair_id=pair_id,
                benchmark_id=benchmark_id,
                tier=tier,
                mutation_type=PAIR_TYPE,
                valid_candidate=baseline,
                adversarial_candidate=mutated,
            )
        )
    return tuple(pairs), certifications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exclude-pairs", type=Path, action="append", required=True)
    parser.add_argument("--baseline-count", type=int, default=2)
    parser.add_argument("--target-channel", default="U")
    parser.add_argument("--target-component", default="U")
    parser.add_argument("--supplied-component", default="Uii")
    parser.add_argument("--allow-opened-baselines", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    runs_roots = tuple(path.resolve() for path in args.runs_root)
    exclusion_paths = tuple(path.resolve() for path in args.exclude_pairs)
    exclusions_by_path = {path: _read_pairs(path) for path in exclusion_paths}
    exclusions = tuple(
        pair for values in exclusions_by_path.values() for pair in values
    )
    candidates = tuple(
        pair
        for root in runs_roots
        for pair in build_pairs(root, args.data_root.resolve())
    )
    if not candidates:
        raise SystemExit("no completed run summaries found")
    tasks = {(pair.benchmark_id, pair.tier) for pair in (*candidates, *exclusions)}
    contexts = {
        task: _validation_context(args.data_root.resolve(), *task)
        for task in tasks
    }
    baselines, excluded = select_baselines(
        candidates,
        exclusions,
        baseline_count=args.baseline_count,
        contexts=contexts,
        target_channel=args.target_channel,
        target_component=args.target_component,
        supplied_component=args.supplied_component,
        allow_opened_baselines=args.allow_opened_baselines,
    )
    pairs, certifications = build_target_mapping_pairs(
        baselines,
        contexts=contexts,
        target_channel=args.target_channel,
        target_component=args.target_component,
        supplied_component=args.supplied_component,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "target_mapping_pairs_manifest.json"
    )
    selected_fingerprints = [item[0] for item in baselines]
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
        "excluded_baseline_fingerprint_count": len(excluded),
        "selected_baseline_count": len(baselines),
        "selected_baseline_fingerprints": selected_fingerprints,
        "allow_opened_baselines": args.allow_opened_baselines,
        "selected_previously_opened_fingerprint_count": len(
            set(selected_fingerprints) & excluded
        ),
        "pair_count": len(pairs),
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "mutation_types": [PAIR_TYPE],
        "certifications": certifications,
        "public_contract": {
            "target_channel": args.target_channel,
            "target_component": args.target_component,
            "supplied_component": args.supplied_component,
            "complete_target_expression": (
                f"{args.supplied_component} + {args.target_component}"
            ),
        },
        "mutation_labels_visible_to_judge": False,
        "hidden_generator_visible_to_judge": False,
        "interpretation_boundary": (
            "This development test validates only the target-mapping question. "
            "Every pair is certified to exclude the supplied component from "
            "the target process before the complete mapping is constructed."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(pairs)} certified target-mapping pairs from "
        f"{len(baselines)} structures to {args.output}"
    )


if __name__ == "__main__":
    main()
