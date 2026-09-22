"""Frozen-parameter plotting must preserve causality, scoring, and archive bounds."""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = load_script("replay_cstr_curves")
SMOKE = load_script("smoke_public_fitting")


def test_causal_replay_and_exact_resume(tmp_path, monkeypatch):
    request, _, validation = SMOKE.control("general-rollout-v1")
    # Reuse the analytic control with the CSTR exporter's single target name.
    request = PublicFitRequest.model_validate(
        json.loads(request.model_dump_json().replace("v01", "T"))
    )
    validation = PublicSplit.model_validate(
        json.loads(validation.model_dump_json().replace("v01", "T"))
    )
    model, _, _ = public._lower(request)
    trajectory = public.unpack_split(validation).trajectories[0]
    parameters = {"a": 0.7, "b": 1.2, "init_z_scale": 0.5}
    path = tmp_path / "curve.json"
    first = REPLAY.replay_one(model, trajectory, parameters, 1.0, path)
    assert first["success"]
    assert first["nmse"] < 1e-12
    assert first["initial_state"]["z"] == pytest.approx(0.3)
    assert np.allclose(first["predicted"], trajectory.targets["T"], atol=1e-7)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed replay must not simulate again")

    monkeypatch.setattr(REPLAY, "simulate_trajectory", forbidden)
    assert REPLAY.replay_one(model, trajectory, parameters, 1.0, path) == first


def test_failed_rollout_prevents_partial_success_score():
    complete = [
        {"success": True, "nmse": 1.0, "time": [0, 1]},
        {"success": True, "nmse": 4.0, "time": [0, 1, 2, 3]},
    ]
    assert REPLAY.pooled_error(complete) == 3.0
    assert REPLAY.pooled_error([*complete, {"success": False}]) is None
    with pytest.raises(ValueError, match="nonfinite"):
        REPLAY.normalized_error(np.zeros(2), np.array([0, np.inf]), 1)


def test_archive_rejects_unexpected_path_before_writing(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.addfile(tarfile.TarInfo("../outside.json"))
    with pytest.raises(ValueError, match="exactly"):
        REPLAY.read_archive(archive, tmp_path / "inputs")
    assert not (tmp_path / "inputs").exists()


def test_archive_rejects_links_before_writing(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in REPLAY.MEMBERS:
            member = tarfile.TarInfo(name)
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/unused"
            stream.addfile(member)
    with pytest.raises(ValueError, match="regular"):
        REPLAY.read_archive(archive, tmp_path / "inputs")
    assert not (tmp_path / "inputs").exists()
