"""Frozen baseline replay must not use test scores or reinterpret D3 equations."""

import copy
import tarfile

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.schemas import CandidateModel
from scripts import replay_t1_external_interventions as replay


def subject(discrete=False):
    """Make a synthetic complete saved subject with irrelevant private fields."""
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "frozen",
            "parent_candidate_id": None,
            "states": [{"name": "Gp", "kind": "observed"}],
            "state_equations": [{"state": "Gp", "rhs": "Gp" if discrete else "0"}],
            "observation_mappings": [{"channel": "Gp", "expression": "Gp"}],
            "initial_conditions": [
                {"state": "Gp", "expression": "Gp", "scope": "global"}
            ],
        }
    )
    return {
        "schema_version": "frozen-evaluation-subject-1",
        "subject_id": "synthetic",
        "method": "d3_native_no_tools" if discrete else "sindy",
        "benchmark_id": replay.CELLS[0],
        "tier": "easy",
        "repetition": 0,
        "selection_frozen": True,
        "source_provenance": {
            "adapter": "d3_result" if discrete else "sindy_result",
            "request_id": "synthetic",
            "source_path": "synthetic",
            "source_sha256": "a" * 64,
            "candidate_sha256": replay.public.content_sha256(candidate),
        },
        "candidate": candidate.model_dump(mode="json"),
        "parameterization": {"status": "not_required"},
        "validation_context": ValidationContext(targets=("Gp",)).model_dump(
            mode="json"
        ),
        "execution_semantics": "discrete_increment_recursive_rollout"
        if discrete
        else "continuous_ode_free_rollout",
        "target_prediction": {"arbitrary_unread_test_score": "do not parse this"},
        "hidden_mechanisms": "not used",
        "interventions": "not used",
    }


def test_test_scores_are_discarded_and_candidate_identity_checked():
    raw = subject()
    sanitized = replay.sanitize_subject(raw)
    assert sanitized.target_prediction.status == "missing"
    assert not sanitized.private_metrics_opened_after_freeze
    assert not sanitized.interventions and not sanitized.hidden_mechanisms
    raw["candidate"]["state_equations"][0]["rhs"] = "Gp"
    with pytest.raises(ValueError, match="digest"):
        replay.sanitize_subject(raw)


@pytest.mark.parametrize("discrete", [False, True])
def test_end_to_end_native_semantics_and_cached_resume(tmp_path, monkeypatch, discrete):
    raw = replay.sanitize_subject(subject(discrete)).model_dump(mode="json")
    row = {
        "trajectory_id": "synthetic",
        "time": [0, 1, 2],
        "targets": {"Gp": [1, 2, 4]},
        "auxiliaries": {},
        "external_inputs": {},
    }
    reference = copy.deepcopy(row)
    reference["targets"] = {"v01": [1, 2, 4]}
    plan = {
        "cells": {
            replay.CELLS[0]: {
                "context": raw["validation_context"],
                "training": {
                    "name": "train",
                    "fingerprint": "synthetic",
                    "rows": [row],
                },
                "validation": {
                    "name": "val",
                    "fingerprint": "synthetic",
                    "rows": [row],
                },
            }
        },
        "cases": [{"id": "control", "paired_control": None}],
        "references": {"original": {"control": {"row": reference}}},
    }
    # Exact minimal synthetic public contract has no auxiliary channels.
    monkeypatch.setattr(
        replay,
        "CHANNELS",
        {
            "targets": {"v01": "Gp"},
            "auxiliaries": {},
            "external_inputs": {},
        },
    )
    result = replay.run_subject((raw, plan, tmp_path))
    assert result["train"]["complete"]
    assert (result["train"]["nmse"] == 0) == discrete
    curve = replay.sealed_read(
        tmp_path / "replays" / result["model_id"] / "probes/control.json"
    )
    np.testing.assert_allclose(curve["predicted"], [1, 2, 4] if discrete else [1, 1, 1])
    monkeypatch.setattr(replay, "compile_candidate", lambda *a: pytest.fail("cached"))
    monkeypatch.setattr(replay, "predict", lambda *a, **kw: pytest.fail("cached"))
    assert replay.run_subject((raw, plan, tmp_path)) == result


def test_native_failure_is_saved_without_a_partial_score(tmp_path):
    raw = subject(True)
    raw["candidate"]["state_equations"][0]["rhs"] = "1 / (Gp - Gp)"
    candidate = CandidateModel.model_validate(raw["candidate"])
    model = replay.NativeMap.build(candidate, {}, ("Gp",), ())
    row = {
        "trajectory_id": "bad",
        "time": [0, 1, 2],
        "targets": {"Gp": [1, 2, 4]},
        "auxiliaries": {},
        "external_inputs": {},
    }
    (trajectory,) = replay.public.unpack_split(
        replay.PublicSplit.model_validate(
            {
                "name": "val",
                "fingerprint": "synthetic",
                "rows": [row],
            }
        )
    ).trajectories
    record = replay.discrete_record(
        model, trajectory, "Gp", 1, tmp_path / "record.json"
    )
    assert not record["success"] and record["nmse"] is None
    assert record["predicted"] is None


def test_unknown_archive_member_rejected_before_reading(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.addfile(tarfile.TarInfo("../escape"))
    with pytest.raises(ValueError, match="inventory"):
        replay.freeze(archive, tmp_path, tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reference_contract_rejects_missing_auxiliary():
    raw = {
        "trajectory_id": "x",
        "time": [0, 1],
        "targets": {"v01": [1, 2]},
        "auxiliaries": {},
        "external_inputs": {"u01": [0, 1]},
    }
    context = {
        "targets": ["Gp"],
        "auxiliaries": ["Gt"],
        "external_inputs": ["meal_event_g"],
    }
    with pytest.raises(ValueError, match="auxiliaries"):
        replay.translate_reference(raw, replay.CELLS[0], context)
