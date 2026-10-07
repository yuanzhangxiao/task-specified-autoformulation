"""Shared budgets, independent certificates and preservation during polishing."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import polishing_campaign as campaign
from autoformalism.fitting import polishing_fit as fitting
from autoformalism.fitting import public_fitting as public
from tests.test_generic_recovery import saved as source_fixture


@pytest.fixture
def base(tmp_path, monkeypatch):
    path = source_fixture.__wrapped__(tmp_path, monkeypatch)
    return common.bases(read_seal(path))["linear_s0"]


def policy():
    return campaign.PolishingPolicy(seconds=40, certificate_seconds=10).model_dump(
        mode="json"
    )


def backend(parameters, *, error=5e-7, calls=100, stop="training_prediction_certified"):
    value = {
        "parameters": parameters,
        "complete": True,
        "training_nmse": error,
        "maximum_trajectory_nmse": error * 2,
        "maximum_solver_difference": 1e-10,
    }
    return {
        "selected": {"parameters": parameters, "training_nmse": error},
        "levels": [],
        "fit_seconds": 10,
        "stop_reason": stop,
        "certificates": [
            {
                "parameters_sha256": public.content_sha256(parameters),
                "value": value,
                "process": {"status": "complete", "termination_confirmed": True},
            }
        ],
        "rollout": {"value": {"actual_residual_calls": calls}},
    }


@pytest.mark.parametrize("arm", campaign.ARMS)
def test_two_phases_share_time_calls_and_use_training_only(
    base, tmp_path, monkeypatch, arm
):
    now = [0.0]
    calls = []
    first = {k: v + 0.01 for k, v in base["start"].items()}
    final = {k: v + 0.02 for k, v in base["start"].items()}

    def fit(b, a, p, folder):
        calls.append(deepcopy((b, a, p)))
        assert a == arm and set(b) == set(base)
        assert b["training"]["name"] == "train"
        now[0] += 10
        if len(calls) == 1:
            assert b["start"] == base["start"]
            return backend(first)
        assert (
            read_seal(folder.parent / "first-prediction.json")["value"]["selected"][
                "parameters"
            ]
            == first
        )
        assert b["start"] == first
        assert p["seconds"] == 30 and p["maximum_rollout_calls"] == 800
        assert p["training_nmse"] == 1e-8 and p["trajectory_nmse"] == 1e-7
        return backend(final, error=1e-9, calls=50)

    monkeypatch.setattr(fitting, "monotonic", lambda: now[0])
    monkeypatch.setattr(fitting.profiled, "fit", fit)
    folder = tmp_path / "fit"
    result = fitting.fit(base, arm, policy(), folder)
    assert result["first_prediction"]["selected"]["parameters"] == first
    assert result["selected"]["parameters"] == final
    assert result["strict_prediction_certified"]
    assert result["retained_stage"] == "polish" and result["polishing_seconds"] == 10
    assert result["actual_residual_calls"] == 150 and result["fit_seconds"] == 20
    assert fitting.fit(base, arm, policy(), folder) == result and len(calls) == 2
    with pytest.raises(ValueError, match="identity"):
        fitting.fit(base, arm, policy() | {"seconds": 41}, folder)


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("already", "training_prediction_certified"),
        ("uncertified", "prediction_not_certified"),
        ("unknown", "unknown_call_usage_no_polish"),
        ("calls", "call_budget_no_polish"),
        ("time", "time_budget_no_polish"),
        ("cleanup", "cleanup_unconfirmed"),
    ],
)
def test_no_restart_when_polishing_ineligible(
    base, tmp_path, monkeypatch, scenario, reason
):
    now = [0.0]
    calls = []

    def fit(*args):
        calls.append(args)
        now[0] += 30 if scenario == "time" else 10
        value = backend(base["start"])
        if scenario == "already":
            value = backend(base["start"], error=1e-10)
        if scenario == "uncertified":
            value["certificates"][0]["value"]["complete"] = False
        if scenario == "cleanup":
            value["stop_reason"] = "cleanup_unconfirmed"
        if scenario in {"unknown", "calls"}:
            value["rollout"]["value"]["actual_residual_calls"] = (
                None if scenario == "unknown" else 897
            )
        return value

    monkeypatch.setattr(fitting, "monotonic", lambda: now[0])
    monkeypatch.setattr(fitting.profiled, "fit", fit)
    result = fitting.fit(base, "rollout_only", policy(), tmp_path / "fit")
    assert len(calls) == 1 and not result["polishing_attempted"]
    assert result["stop_reason"] == reason
    assert result["selected"]["parameters"] == base["start"]
    if scenario == "unknown":
        assert (
            result["actual_residual_calls"] is None
            and not result["accounting_complete"]
        )


@pytest.mark.parametrize(
    "failure", ["worse", "partial", "mismatch", "disagreement", "cleanup"]
)
def test_bad_polish_preserves_first(base, tmp_path, monkeypatch, failure):
    count = []
    changed = {k: v + 1 for k, v in base["start"].items()}

    def fit(*args):
        count.append(1)
        if len(count) == 1:
            return backend(base["start"])
        v = backend(changed, error=1e-9)
        check = v["certificates"][0]
        if failure == "worse":
            check["value"]["training_nmse"] = 9e-7
        elif failure == "partial":
            check["value"]["complete"] = False
        elif failure == "mismatch":
            check["value"]["parameters"] = base["start"]
        elif failure == "disagreement":
            check["value"]["maximum_solver_difference"] = 1
        else:
            v["stop_reason"] = "cleanup_unconfirmed"
            check["process"]["termination_confirmed"] = False
        return v

    monkeypatch.setattr(fitting.profiled, "fit", fit)
    result = fitting.fit(base, "profiled_rollout", policy(), tmp_path / "fit")
    assert result["selected"]["parameters"] == base["start"]
    assert (
        result["retained_stage"] == "prediction"
        and not result["strict_prediction_certified"]
    )
    if failure == "cleanup":
        assert result["stop_reason"] == "cleanup_unconfirmed"


def test_coordinator_interruption_keeps_sealed_first_and_never_restarts(
    base, tmp_path, monkeypatch
):
    calls = []

    def fit(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return backend(base["start"])

    monkeypatch.setattr(fitting.profiled, "fit", fit)
    folder = tmp_path / "fit"
    with pytest.raises(KeyboardInterrupt):
        fitting.fit(base, "profiled_rollout", policy(), folder)
    resumed = fitting.fit(base, "profiled_rollout", policy(), folder)
    assert len(calls) == 2 and resumed["stop_reason"] == "interrupted_no_fit_restart"
    assert resumed["selected"]["parameters"] == base["start"]
    assert resumed["first_prediction"] and not resumed["accounting_complete"]


@pytest.mark.parametrize("number", [None, float("nan"), float("inf"), -1, 2])
def test_certificate_rejects_invalid_or_large_error(number):
    check = backend({"a": 1})["certificates"][0]["value"]
    check["training_nmse"] = number
    assert not fitting.passes(check, policy())


def test_reject_extra_fields_or_nontrain_split(base, tmp_path):
    with pytest.raises(ValueError, match="training-only"):
        fitting.fit(
            base | {"reference_parameters": {}}, "rollout_only", policy(), tmp_path
        )
    base["training"]["name"] = "validation"
    with pytest.raises(ValueError):
        fitting.fit(base, "rollout_only", policy(), tmp_path)
