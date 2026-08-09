"""Unit tests for the development-only Phase-B replay driver."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from autoformalism.rebuttal.artifacts import candidate_warm_start_parameters
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
