"""Verification retry is bounded, preserves failures, and cannot alter parameters."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import verified_replay as replay
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from tests.profiled_fixture import make_base


@pytest.fixture
def args():
    base, validation = make_base()
    return (
        PublicFitRequest.model_validate(base["request"]),
        base["start"],
        PublicSplit.model_validate(base["training"]),
        PublicSplit.model_validate(validation),
        replay.ReplayPolicy(),
    )


def score(difference, *, complete=True, loss=1e-15):
    return {
        "complete": complete,
        "maximum_solver_difference": difference,
        "metrics": {"train": loss, "val": loss},
        "rows": [],
    }


def test_disagreement_retries_same_parameters_and_charges_both(
    args, monkeypatch, tmp_path
):
    seen = []
    before = deepcopy(args[1])

    def run(req, params, train, val, seconds, **kw):
        seen.append((deepcopy(params), seconds, kw))
        return score(3e-4 if len(seen) == 1 else 1e-9)

    monkeypatch.setattr(replay.qualification, "replay", run)
    result = replay.run(*args, tmp_path)
    assert result["solver_agreement_passed"] and result["retry_used"]
    assert result["attempts"][0]["value"]["maximum_solver_difference"] == 3e-4
    assert result["selected_attempt"] == "tight"
    assert [v[2]["relative_tolerance"] for v in seen] == [1e-9, 1e-12]
    assert [v[2]["absolute_tolerance"] for v in seen] == [1e-11, 1e-14]
    assert all(v[0] == before and v[1] <= 300 for v in seen)
    assert result["budget_charge_seconds"] == sum(
        v["budget_charge_seconds"] for v in result["attempts"]
    )
    assert not result["parameter_fitting_performed"]
    monkeypatch.setattr(
        replay.qualification, "replay", lambda *a, **kw: pytest.fail("redo")
    )
    assert replay.run(*args, tmp_path) == result
    changed = (args[0], before | {"a": 123}, *args[2:])
    with pytest.raises(ValueError, match="identity"):
        replay.run(*changed, tmp_path)


@pytest.mark.parametrize("mode", ["disagree", "missing", "nan", "exception"])
def test_persistent_failures_never_become_success(args, monkeypatch, tmp_path, mode):
    seen = []

    def work(*a, **kw):
        seen.append(kw)
        if mode == "exception":
            raise RuntimeError("solver failed")
        return score(
            float("nan") if mode == "nan" else 0.01, complete=mode != "missing"
        )

    monkeypatch.setattr(replay.qualification, "replay", work)
    result = replay.run(*args, tmp_path)
    assert not result["solver_agreement_passed"]
    assert result["selected_attempt"] is None and len(seen) == 2
    assert result["status"] == "unverified"
    assert replay.run(*args, tmp_path) == result


def test_poor_but_agreeing_predictions_do_not_trigger_numerical_retry(
    args, monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        replay.qualification,
        "replay",
        lambda *a, **kw: calls.append(kw) or score(1e-9, loss=100),
    )
    result = replay.run(*args, tmp_path)
    assert result["solver_agreement_passed"] and not result["retry_used"]
    assert result["metrics"]["val"] == 100 and len(calls) == 1


def test_interrupted_initial_replay_is_charged_and_only_reserved_retry_runs(
    args, monkeypatch, tmp_path
):
    monkeypatch.setattr(replay.qualification, "replay", lambda *a, **kw: score(1e-9))
    replay.run(*args, tmp_path / "source")
    relative = "initial/started.json"
    seal(tmp_path / "interrupted" / relative, read_seal(tmp_path / "source" / relative))
    calls = []
    monkeypatch.setattr(
        replay.qualification, "replay", lambda *a, **kw: calls.append(kw) or score(1e-9)
    )
    result = replay.run(*args, tmp_path / "interrupted")
    assert len(calls) == 1 and calls[0]["relative_tolerance"] == 1e-12
    assert result["budget_charge_seconds"] >= args[-1].initial_seconds
    assert result["attempts"][0]["status"] == "interrupted"
    assert not result["accounting_complete"]


def test_qualification_journal_binds_tolerances(args, tmp_path):
    # Zero replay allowance permits a real journal check without integration.
    path = tmp_path / "replay.json"
    replay.qualification.replay(*args[:4], 0, journal=path)
    with pytest.raises(ValueError, match="journal identity"):
        replay.qualification.replay(
            *args[:4],
            0,
            journal=path,
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
        )
