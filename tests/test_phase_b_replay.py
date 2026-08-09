"""Unit tests for the development-only Phase-B replay driver."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from autoformalism.rebuttal.artifacts import candidate_warm_start_parameters
from scripts.merge_phase_b_replay_shards import _read_complete_shards
from scripts.replay_phase_b_frozen_candidates import (
    _better_result,
)


def _artifact(path: Path):
    return SimpleNamespace(
        source_checkpoint=str(path),
        fitted_global_parameters={},
        candidate=SimpleNamespace(
            parameters=(SimpleNamespace(name="kept"), SimpleNamespace(name="other"))
        ),
    )


def test_source_parameters_reads_nested_pruned_fit_and_filters_names(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "round.json"
    checkpoint.write_text(
        json.dumps(
            {
                "record": {
                    "pruned_fit": {
                        "global_parameters": {
                            "kept": 1.25,
                            "unknown": 9.0,
                            "other": "not-numeric",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert candidate_warm_start_parameters(_artifact(checkpoint)) == {"kept": 1.25}


def test_source_parameters_fails_closed_when_checkpoint_is_missing(
    tmp_path: Path,
) -> None:
    assert candidate_warm_start_parameters(_artifact(tmp_path / "missing.json")) == {}


def test_source_parameters_prefers_portable_embedded_values(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "missing.json")
    artifact.fitted_global_parameters = {"kept": 2.5, "unknown": 7.0}

    assert candidate_warm_start_parameters(artifact) == {"kept": 2.5}


def test_refinement_cannot_replace_a_better_valid_screening_result() -> None:
    screening = {"success": True, "validation_nmse": 0.2}
    failed = {"success": False, "validation_nmse": 1e12}
    worse = {"success": True, "validation_nmse": 0.3}
    better = {"success": True, "validation_nmse": 0.1}

    assert _better_result(screening, failed) is screening
    assert _better_result(screening, worse) is screening
    assert _better_result(screening, better) is better


def test_complete_shards_merge_in_deterministic_artifact_order(tmp_path: Path) -> None:
    selected = [SimpleNamespace(artifact_id="b"), SimpleNamespace(artifact_id="a")]
    for index, artifact_id in enumerate(("a", "b")):
        root = tmp_path / f"shard-{index:03d}"
        (root / "results").mkdir(parents=True)
        (root / "results" / f"{artifact_id}.json").write_text(
            json.dumps({"artifact_id": artifact_id, "success": True}),
            encoding="utf-8",
        )
        (root / "shard_manifest.json").write_text(
            json.dumps(
                {
                    "stage": "screen_complete",
                    "shard_count": 2,
                    "shard_index": index,
                    "assigned_artifact_ids": [artifact_id],
                    "completed_artifact_ids": [artifact_id],
                }
            ),
            encoding="utf-8",
        )

    artifact_ids = [
        row["artifact_id"] for row in _read_complete_shards(tmp_path, selected)
    ]
    assert artifact_ids == [
        "a",
        "b",
    ]


def test_shard_merge_rejects_incomplete_declared_coverage(tmp_path: Path) -> None:
    root = tmp_path / "shard-000"
    (root / "results").mkdir(parents=True)
    (root / "shard_manifest.json").write_text(
        json.dumps(
            {
                "stage": "screen_complete",
                "shard_count": 1,
                "shard_index": 0,
                "assigned_artifact_ids": ["a"],
                "completed_artifact_ids": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        _read_complete_shards(tmp_path, [SimpleNamespace(artifact_id="a")])
    except SystemExit as exc:
        assert "missing results" in str(exc)
    else:
        raise AssertionError("incomplete shard unexpectedly merged")
