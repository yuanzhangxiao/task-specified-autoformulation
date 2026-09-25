"""Historical initialization diagnostics cannot refresh fit budgets or seals."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.initialization_audit import audit_saved_initialization
from autoformalism.schemas.public_fitting import PublicFitRequest
from tests.test_fitted_initialization import initial_channel_alias_problem


def saved_fit(directory):
    model, row, plan = initial_channel_alias_problem()
    request = PublicFitRequest.model_validate(
        {
            "base_candidate": model.validated.candidate,
            "context": model.validated.context,
            "initialization_plan": plan,
            "parameter_guesses": {"rate": 0.3, "gain": 2},
            "profile": "collocation-multi-target-v1",
            "source": {
                "stage": "synthetic_control",
                "task_id": "initial-channel-alias",
                "artifact_sha256": public.content_sha256(model.validated.candidate),
            },
        }
    )
    public.prepare_fit(
        request,
        DatasetSplit(SplitName.TRAIN, (row,), "training"),
        DatasetSplit(SplitName.VALIDATION, (replace(row, trajectory_id="val"),), "val"),
        directory,
    )
    # A genuinely different historical source is allowed only by this audit.
    frozen = json.loads((directory / "freeze.json").read_text())
    frozen["source_sha256"] = "a" * 64
    reseal(directory, frozen)


def reseal(directory, frozen):
    frozen["identity"] = public.content_sha256(
        {k: v for k, v in frozen.items() if k != "identity"}
    )
    (directory / "freeze.json").write_text(json.dumps(frozen))


def test_saved_audit_is_read_only_and_does_not_enable_execution(tmp_path, monkeypatch):
    saved_fit(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(
        public, "_run_backend", lambda *a: pytest.fail("optimizer called")
    )
    report = audit_saved_initialization(tmp_path)
    assert report["initial_observation_process_name_collisions"] == {"m": ["U"]}
    assert report["capability_supported"]
    assert not report["source_matches"]
    assert not report["fit_started_marker_exists"]
    assert report["boundary_comparison"]["status"] == "agree"
    assert len(report["boundary_comparison"]["rows"]) == 2
    with pytest.raises(ValueError, match="source changed"):
        public.execute_fit(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_audit_rejects_tampering_and_resealed_contract_changes(tmp_path):
    saved_fit(tmp_path)
    frozen = json.loads((tmp_path / "freeze.json").read_text())
    frozen["lowered_candidate"]["state_equations"][0]["rhs"] = "0"
    (tmp_path / "freeze.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="digest differs"):
        audit_saved_initialization(tmp_path)
    reseal(tmp_path, frozen)
    with pytest.raises(ValueError, match="beyond source"):
        audit_saved_initialization(tmp_path)


def test_audit_cli_reports_available_and_missing_fits(tmp_path):
    saved_fit(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/audit_fitted_initialization.py"
            ),
            "--fit",
            str(tmp_path),
            "--fit",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert [r["status"] for r in report["rows"]] == ["audited", "audit_error"]
    assert report["optimizer_calls"] == report["solver_rollouts"] == 0
