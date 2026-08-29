"""Verify frozen target-completeness confirmation inputs before provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.rebuttal.adversarial import AdversarialPair

MANIFEST_SCHEMA_VERSION = "target-completeness-fresh-confirmation-pairs-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(
    *,
    pairs_path: Path,
    manifest_path: Path,
    config_path: Path,
    development_analysis_path: Path,
) -> dict[str, object]:
    """Validate hashes, counts, freshness claims, and the V7 prerequisite."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    analysis = json.loads(development_analysis_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected confirmation manifest schema")
    if manifest.get("status") != "frozen_before_fresh_structure_calls":
        raise ValueError("confirmation manifest is not frozen before calls")
    if manifest.get("pairs_sha256") != _sha256(pairs_path):
        raise ValueError("confirmation pair SHA-256 differs from the manifest")
    if manifest.get("protocol_config_sha256") != _sha256(config_path):
        raise ValueError("confirmation config SHA-256 differs from the manifest")
    prerequisite = manifest.get("development_prerequisite")
    if not isinstance(prerequisite, dict):
        raise ValueError("confirmation manifest lacks the V7 prerequisite")
    if prerequisite.get("analysis_sha256") != _sha256(development_analysis_path):
        raise ValueError("V7 analysis SHA-256 differs from the manifest")
    if analysis.get("passed") is not True:
        raise ValueError("V7 development prerequisite did not pass")
    if manifest.get("selected_fingerprints_overlap_exclusions") is not False:
        raise ValueError("confirmation structures overlap an opened structure set")
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected_pairs = int(config["planned"]["pairs"])
    if len(pairs) != expected_pairs or manifest.get("pair_count") != expected_pairs:
        raise ValueError("confirmation pair count differs from the frozen plan")
    pair_ids = [pair.pair_id for pair in pairs]
    if pair_ids != manifest.get("selected_pair_ids"):
        raise ValueError("confirmation pair identifiers differ from the manifest")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("confirmation pair identifiers are not unique")
    return {
        "status": "verified",
        "pair_count": len(pairs),
        "pairs_sha256": _sha256(pairs_path),
        "protocol_config_sha256": _sha256(config_path),
        "development_analysis_sha256": _sha256(development_analysis_path),
        "development_passed": True,
        "fresh_structure_overlap": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-analysis", type=Path, required=True)
    args = parser.parse_args()
    result = verify_inputs(
        pairs_path=args.pairs,
        manifest_path=args.manifest,
        config_path=args.config,
        development_analysis_path=args.development_analysis,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
