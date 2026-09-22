"""Diagnostic exports must retain frozen boundaries and free-rollout semantics."""

from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = load_script("replay_t1_curves")
SMOKE = load_script("smoke_public_fitting")


def test_analytic_causal_initialization_and_exact_resume(tmp_path, monkeypatch):
    request, _, validation = SMOKE.control("general-rollout-v1")
    model, _, _ = REPLAY.public._lower(request)
    trajectory = REPLAY.public.unpack_split(validation).trajectories[0]
    parameters = {"a": 0.7, "b": 1.2, "init_z_scale": 0.5}
    path = tmp_path / "curve.json"
    first = REPLAY.replay_one(model, trajectory, parameters, {}, "v01", 1, path)
    assert first["success"] and first["nmse"] < 1e-12
    assert first["initial_state"]["z"] == pytest.approx(0.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed replay must not simulate again")

    monkeypatch.setattr(REPLAY, "simulate_trajectory", forbidden)
    assert REPLAY.replay_one(model, trajectory, parameters, {}, "v01", 1, path) == first


def test_zero_derivative_is_persistence_not_teacher_forcing(tmp_path):
    _, _, validation = SMOKE.control("general-rollout-v1")
    trajectory = REPLAY.public.unpack_split(validation).trajectories[0]
    candidate = REPLAY.CandidateModel.model_validate(
        {
            "candidate_id": "persistence",
            "parent_candidate_id": None,
            "states": [{"name": "v01", "kind": "observed"}],
            "state_equations": [{"state": "v01", "rhs": "0"}],
            "observation_mappings": [{"channel": "v01", "expression": "v01"}],
            "initial_conditions": [
                {"state": "v01", "expression": "v01", "scope": "global"}
            ],
        }
    )
    model = REPLAY.compile_candidate(
        candidate, REPLAY.ValidationContext(targets=("v01",))
    )
    result = REPLAY.replay_one(
        model, trajectory, {}, {}, "v01", 1, tmp_path / "curve.json"
    )
    assert result["success"]
    assert np.allclose(result["predicted"], trajectory.targets["v01"][0])
    assert result["nmse"] > 0.01


def test_partial_results_are_not_an_aggregate_success():
    records = [
        {"success": True, "nmse": 1, "time": [0, 1]},
        {"success": True, "nmse": 4, "time": [0, 1, 2, 3]},
    ]
    assert REPLAY.pooled_error(records) == 3
    assert REPLAY.pooled_error([*records, {"success": False}]) is None


def test_baseline_must_match_recorded_data():
    with pytest.raises(ValueError, match="fingerprint"):
        REPLAY.baseline_job(
            {"data_identity": {"train": "old"}},
            {"training": {"fingerprint": "changed"}},
            {},
        )


@pytest.mark.parametrize("name,is_link", [("../escape", False), ("plan.json", True)])
def test_archive_bounds_before_writes(tmp_path, name, is_link):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        if is_link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
        stream.addfile(member)
    with pytest.raises(ValueError):
        REPLAY.archive_inputs(archive, tmp_path / "inputs", ("plan.json",))
    assert not (tmp_path / "inputs").exists()
