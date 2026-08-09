"""Freeze prespecified selector choices before post-selection evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

POLICIES = {
    "validation_only": "validation_only",
    "weighted_j0.5_s0.1": "normalized_weighted_sum__j0.5__s0.1",
    "epsilon_d0.2": "epsilon_constrained__d0.2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows, changed = build_confirmation(pd.read_csv(args.selections))
    rows.to_csv(args.output_root / "frozen_confirmation_selections.csv", index=False)
    changed_payload = changed.to_dict(orient="records")
    serialized = json.dumps(
        changed_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    selection_hash = hashlib.sha256(serialized).hexdigest()
    (args.output_root / "frozen_changed_selection_manifest.json").write_text(
        json.dumps(changed_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1",
        "source_selections": str(args.selections.resolve()),
        "policies": POLICIES,
        "uses_test_metrics": False,
        "uses_private_mechanism_references": False,
        "run_policy_selections": len(rows),
        "changed_policy_selections": len(changed),
        "distinct_changed_artifacts": int(changed.artifact_id.nunique()),
        "changed_selection_sha256": selection_hash,
    }
    (args.output_root / "frozen_confirmation_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def build_confirmation(
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all frozen choices and alternative choices that need refitting."""
    parts = []
    for policy, config_id in POLICIES.items():
        subset = selections[selections.config_id == config_id].copy()
        if subset.empty:
            raise ValueError(f"selection grid is missing {config_id}")
        subset["confirmation_policy"] = policy
        parts.append(subset)
    frozen = pd.concat(parts, ignore_index=True)
    run_counts = frozen.groupby("confirmation_policy").run_directory.nunique()
    if run_counts.nunique() != 1:
        raise ValueError(f"policy run counts differ: {run_counts.to_dict()}")
    baseline = frozen[frozen.confirmation_policy == "validation_only"][[
        "run_directory", "artifact_id"
    ]].rename(columns={"artifact_id": "validation_artifact_id"})
    alternatives = frozen[
        frozen.confirmation_policy != "validation_only"
    ].merge(baseline, on="run_directory", how="inner")
    changed = alternatives[
        alternatives.artifact_id != alternatives.validation_artifact_id
    ].copy()
    manifest_columns = [
        "run_directory",
        "benchmark",
        "tier",
        "seed",
        "confirmation_policy",
        "config_id",
        "artifact_id",
        "validation_artifact_id",
    ]
    return frozen, changed.loc[:, manifest_columns]


if __name__ == "__main__":
    main()
