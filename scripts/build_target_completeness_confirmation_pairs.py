"""Build frozen target-completeness pairs from fresh proposer structures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from autoformalism.data import BenchmarkRegistry
from autoformalism.expressions import ValidationContext, repair_protected_declarations
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel

if __package__:
    from scripts.build_hybrid_judge_consensus_validation_pairs import _read_pairs
    from scripts.build_hybrid_judge_heldout_pairs import (
        candidate_structure_fingerprint,
    )
    from scripts.build_hybrid_judge_target_mapping_clean_names import (
        _rename_component,
        build_clean_target_mapping_pairs,
    )
    from scripts.build_hybrid_judge_target_mapping_pairs import (
        _eligible,
        build_target_mapping_pairs,
    )
    from scripts.build_v2_judge_calibration_pairs import (
        _validation_context,
        build_pairs,
    )
else:
    from build_hybrid_judge_consensus_validation_pairs import _read_pairs
    from build_hybrid_judge_heldout_pairs import candidate_structure_fingerprint
    from build_hybrid_judge_target_mapping_clean_names import (
        _rename_component,
        build_clean_target_mapping_pairs,
    )
    from build_hybrid_judge_target_mapping_pairs import (
        _eligible,
        build_target_mapping_pairs,
    )
    from build_v2_judge_calibration_pairs import _validation_context, build_pairs

MANIFEST_SCHEMA_VERSION = "target-completeness-fresh-confirmation-pairs-1"
ANALYSIS_SCHEMA_VERSION = "target-completeness-judge-analysis-1"
RUN_MANIFEST_SCHEMA_VERSION = "target-completeness-judge-run-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_structure(candidate: CandidateModel) -> CandidateModel:
    """Remove the clean-name total wrapper before structure fingerprinting."""
    process_names = [item.name for item in candidate.processes]
    if process_names.count("U") != 1 or process_names.count("Uid") != 1:
        return candidate
    total = next(item for item in candidate.processes if item.name == "U")
    if "Uid" not in total.expression:
        return candidate
    payload = candidate.model_dump(mode="json")
    payload["processes"] = [
        item for item in payload["processes"] if item["name"] != "U"
    ]
    without_total = CandidateModel.model_validate(payload)
    return _rename_component(without_total, old="Uid", new="U")


def source_structure_fingerprint(
    candidate: CandidateModel,
    context: ValidationContext,
) -> str:
    """Hash the underlying proposer structure across raw and clean wrappers."""
    source, _ = repair_protected_declarations(_source_structure(candidate), context)
    return candidate_structure_fingerprint(source, context)


def validate_development_prerequisite(
    *,
    analysis_path: Path,
    run_manifest_path: Path,
    pairs_path: Path,
    config_path: Path,
    expected_config_sha256: str,
) -> dict[str, object]:
    """Fail unless the exact frozen V7 development run passed."""
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unexpected V7 development analysis schema")
    if analysis.get("passed") is not True:
        raise ValueError("V7 development prerequisite did not pass")
    config_sha256 = _sha256(config_path)
    if config_sha256 != expected_config_sha256:
        raise ValueError(
            "V7 protocol config SHA-256 differs from the frozen prerequisite: "
            f"{config_sha256}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected V7 run-manifest schema")
    if run_manifest.get("protocol_version") != config["protocol"]["protocol_version"]:
        raise ValueError("V7 run and protocol versions differ")
    pairs_sha256 = _sha256(pairs_path)
    if run_manifest.get("pairs_sha256") != pairs_sha256:
        raise ValueError("V7 run manifest does not certify the supplied pair bytes")
    return {
        "analysis_path": str(analysis_path.resolve()),
        "analysis_sha256": _sha256(analysis_path),
        "analysis_passed": True,
        "run_manifest_path": str(run_manifest_path.resolve()),
        "run_manifest_sha256": _sha256(run_manifest_path),
        "pairs_path": str(pairs_path.resolve()),
        "pairs_sha256": pairs_sha256,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
    }


def select_fresh_baselines(
    candidate_pairs: Sequence[AdversarialPair],
    exclusion_pairs: Sequence[AdversarialPair],
    *,
    baseline_count: int,
    benchmark_id: str,
    tier: str,
    context: ValidationContext,
    target_channel: str,
    target_component: str,
    supplied_component: str,
) -> tuple[tuple[tuple[str, CandidateModel, str, str], ...], set[str]]:
    """Select eligible structures absent after clean-wrapper normalization."""
    if baseline_count < 1:
        raise ValueError("baseline_count must be positive")
    relevant_exclusions = [
        pair
        for pair in exclusion_pairs
        if pair.benchmark_id == benchmark_id and pair.tier == tier
    ]
    excluded = {
        source_structure_fingerprint(pair.valid_candidate, context)
        for pair in relevant_exclusions
    }
    selected: list[tuple[str, CandidateModel, str, str]] = []
    seen: set[str] = set()
    for pair in candidate_pairs:
        if pair.benchmark_id != benchmark_id or pair.tier != tier:
            continue
        baseline, _ = repair_protected_declarations(pair.valid_candidate, context)
        fingerprint = source_structure_fingerprint(baseline, context)
        if fingerprint in excluded or fingerprint in seen:
            continue
        if not _eligible(
            baseline,
            target_channel=target_channel,
            target_component=target_component,
            supplied_component=supplied_component,
        ):
            continue
        selected.append((fingerprint, baseline, benchmark_id, tier))
        seen.add(fingerprint)
        if len(selected) == baseline_count:
            break
    if len(selected) != baseline_count:
        raise ValueError(
            f"requested {baseline_count} fresh eligible structures but found "
            f"{len(selected)}"
        )
    return tuple(selected), excluded


def build_confirmation_pairs(
    baselines: Sequence[tuple[str, CandidateModel, str, str]],
    *,
    contexts: Mapping[tuple[str, str], ValidationContext],
    target_channel: str,
    total_process: str,
    dependent_process: str,
    supplied_component: str,
) -> tuple[tuple[AdversarialPair, ...], dict[str, dict[str, object]]]:
    """Create clean-name omission pairs and assign confirmation-local IDs."""
    source_pairs, _ = build_target_mapping_pairs(
        baselines,
        contexts=contexts,
        target_channel=target_channel,
        target_component=total_process,
        supplied_component=supplied_component,
    )
    clean_pairs, source_certifications = build_clean_target_mapping_pairs(
        source_pairs,
        contexts=contexts,
        target_channel=target_channel,
        target_process=total_process,
        dependent_process=dependent_process,
        supplied_component=supplied_component,
    )
    output = []
    certifications = {}
    for baseline, pair in zip(baselines, clean_pairs, strict=True):
        fingerprint = baseline[0]
        token = hashlib.sha256(
            f"target-completeness-confirmation-v1:{fingerprint}".encode()
        ).hexdigest()[:16]
        pair_id = f"targetconfirm_{token}"
        output.append(pair.model_copy(update={"pair_id": pair_id}))
        certifications[pair_id] = source_certifications[pair.pair_id]
    return tuple(output), certifications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exclude-pairs", type=Path, action="append", required=True)
    parser.add_argument("--development-analysis", type=Path, required=True)
    parser.add_argument("--development-run-manifest", type=Path, required=True)
    parser.add_argument("--development-pairs", type=Path, required=True)
    parser.add_argument("--development-config", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    prerequisite = validate_development_prerequisite(
        analysis_path=args.development_analysis,
        run_manifest_path=args.development_run_manifest,
        pairs_path=args.development_pairs,
        config_path=args.development_config,
        expected_config_sha256=config["development_prerequisite"][
            "protocol_config_sha256"
        ],
    )
    runs_roots = tuple(path.resolve() for path in args.runs_root)
    exclusion_paths = tuple(path.resolve() for path in args.exclude_pairs)
    if len(runs_roots) != len(set(runs_roots)):
        raise ValueError("runs roots must be unique")
    if len(exclusion_paths) != len(set(exclusion_paths)):
        raise ValueError("exclusion pair files must be unique")
    if args.development_pairs.resolve() not in exclusion_paths:
        raise ValueError("the V7 development pairs must be an explicit exclusion")
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

    prompt_contract = config["public_prompt_contract"]
    benchmark_id = str(prompt_contract["benchmark_id"])
    tier = str(prompt_contract["tier"])
    context = _validation_context(args.data_root.resolve(), benchmark_id, tier)
    spec = BenchmarkRegistry().get(benchmark_id)
    prompt_path = args.data_root.resolve() / spec.relative_root / "proposer_prompt.txt"
    prompt_sha256 = _sha256(prompt_path)
    if prompt_sha256 != prompt_contract["proposer_prompt_sha256"]:
        raise ValueError(
            "public prompt SHA-256 differs from the frozen confirmation: "
            f"{prompt_sha256}"
        )
    construction = config["pair_construction"]
    baselines, excluded = select_fresh_baselines(
        candidates,
        exclusions,
        baseline_count=int(construction["baseline_count"]),
        benchmark_id=benchmark_id,
        tier=tier,
        context=context,
        target_channel=str(construction["target_channel"]),
        target_component=str(construction["total_process"]),
        supplied_component=str(construction["supplied_component"]),
    )
    contexts = {(benchmark_id, tier): context}
    pairs, certifications = build_confirmation_pairs(
        baselines,
        contexts=contexts,
        target_channel=str(construction["target_channel"]),
        total_process=str(construction["total_process"]),
        dependent_process=str(construction["dependent_process"]),
        supplied_component=str(construction["supplied_component"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs_text = "".join(f"{pair.model_dump_json()}\n" for pair in pairs)
    args.output.write_text(pairs_text, encoding="utf-8")
    selected_fingerprints = [item[0] for item in baselines]
    if set(selected_fingerprints) & excluded:
        raise AssertionError("fresh confirmation reused an opened structure")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_before_fresh_structure_calls",
        "protocol_config_path": str(args.protocol_config.resolve()),
        "protocol_config_sha256": _sha256(args.protocol_config),
        "development_prerequisite": prerequisite,
        "public_prompt_path": str(prompt_path),
        "public_prompt_sha256": prompt_sha256,
        "source_runs_roots": [str(path) for path in runs_roots],
        "exclusion_pair_files": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "pair_count": len(exclusions_by_path[path]),
            }
            for path in exclusion_paths
        ],
        "baseline_holdout_unit": "canonical_unwrapped_proposer_structure",
        "excluded_baseline_fingerprint_count": len(excluded),
        "selected_baseline_count": len(baselines),
        "selected_baseline_fingerprints": selected_fingerprints,
        "selected_fingerprints_overlap_exclusions": False,
        "pair_count": len(pairs),
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "pairs_sha256": hashlib.sha256(pairs_text.encode()).hexdigest(),
        "mutation_types": sorted({pair.mutation_type for pair in pairs}),
        "certifications": certifications,
        "test_data_opened": False,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(pairs)} frozen target-completeness confirmation pairs "
        f"from {len(baselines)} fresh structures to {args.output}; "
        f"excluded_structures={len(excluded)}"
    )


if __name__ == "__main__":
    main()
