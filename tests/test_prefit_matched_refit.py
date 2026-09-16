"""Matched compute, inherited starts, immutable history and consumed allocations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import prefit_parameter_replay as replay
from scripts import prefit_matched_refit as matched
from scripts.smoke_prefit_parameter_replay import saved_history


@pytest.fixture
def campaign(tmp_path):
    historical, *_ = saved_history(tmp_path)
    source, root = tmp_path / "replay", tmp_path / "matched"
    replay.prepare(historical, source)
    replay.fit(source)
    return source, root


def test_matched_fit_and_cli_report_resume_without_touching_history(
    campaign, monkeypatch
):
    source, root = campaign
    before = {p: p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    prepared = matched.prepare(source, root)
    assert prepared["status"] == "ready_for_fit"
    assert prepared["comparison"]["validation"]["child_minus_refit"] is None
    assert not (root / "refit").exists()
    _, contract, frozen, child, _ = matched._verify(root)
    assert contract["parent"] == contract["child"]
    assert frozen["seed"]["parameters"] == contract["parameters"]
    assert not frozen["seed"]["fresh_parameters"]
    assert frozen["seed"]["retained_initializer_parameters"]
    assert all(
        child["seed"]["parameters"][k] == v for k, v in contract["parameters"].items()
    )
    original = public._run_backend
    starts = []

    def backend(*args):
        starts.append(args[4])
        return original(*args)

    monkeypatch.setattr(public, "_run_backend", backend)
    result = matched.fit(root)
    assert result["status"] == "complete"
    assert matched.fit(root) == matched.report(root) == result
    assert starts == [contract["parameters"]]
    assert before == {p: p.read_bytes() for p in before}
    assert not result["automatic_branch_selection"]
    cli = subprocess.run(
        [sys.executable, str(Path(matched.__file__)), "report", "--root", str(root)],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(public.__file__).resolve().parents[2]),
        },
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout) == result


def test_another_output_cannot_renew_allocation(campaign):
    source, root = campaign
    first = matched.prepare(source, root)
    assert matched.prepare(source, root) == first
    with pytest.raises(ValueError, match="frozen artifact differs"):
        matched.prepare(source, root.with_name("duplicate"))


def test_started_interruption_is_consumed(campaign, monkeypatch):
    source, root = campaign
    matched.prepare(source, root)
    _, contract, frozen, *_ = matched._verify(root)
    sibling_fit.prepare_child_fit(directory=root / "refit", **contract)
    public._write(root / "refit/started.json", {"identity": frozen["identity"]})
    monkeypatch.setattr(public, "_run_backend", lambda *a: pytest.fail("renewed fit"))
    result = matched.fit(root)
    assert result["status"] == "refit_failed"
    assert result["refit"]["status"] == "interrupted"
    assert matched.fit(root) == result


def test_wrong_executor_is_rejected_before_allocation(campaign, monkeypatch):
    source, root = campaign
    monkeypatch.setattr(public, "_source_identity", lambda: "different-fitter")
    with pytest.raises(ValueError, match="source, runtime or policy changed"):
        matched.prepare(source, root)
    assert not root.exists()


@pytest.mark.parametrize("target", ["result.json", "backend_result.json"])
def test_changed_child_is_rejected_before_allocation(campaign, target):
    source, root = campaign
    path = source / "child_fit" / target
    path.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        matched.prepare(source, root)
    assert not root.exists()


def test_control_cannot_be_replaced_with_child(campaign):
    source, root = campaign
    matched.prepare(source, root)
    (root / "refit").mkdir()
    (root / "refit/freeze.json").write_bytes(
        (source / "child_fit/freeze.json").read_bytes()
    )
    with pytest.raises(ValueError, match="different matched comparison"):
        matched.report(root)


def test_source_overlap_rejected_without_side_effects(campaign):
    source, _ = campaign
    root = source / "nested-control"
    with pytest.raises(ValueError, match="separate"):
        matched.prepare(source, root)
    assert not root.exists()


def test_score_differences_preserve_zero_and_unavailable():
    parent = {"training": {"normalized_mse": 1}, "validation": {"normalized_mse": 1}}
    child = {"training": {"normalized_mse": 0}, "validation": {"normalized_mse": 0.4}}
    control = {
        "training": {"normalized_mse": 0.2},
        "validation": {"normalized_mse": 0.3},
    }
    result = matched._comparison(parent, child, control)
    assert result["training"]["child_minus_refit"] == -0.2
    assert result["validation"]["child_minus_refit"] == pytest.approx(0.1)
    assert (
        matched._comparison(parent, child, None)["training"]["child_minus_refit"]
        is None
    )
