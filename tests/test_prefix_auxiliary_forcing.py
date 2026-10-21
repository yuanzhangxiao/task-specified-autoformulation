"""Explicit auxiliary permissions, causal isolation and legacy prefix identities."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as p
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


def problem():
    """A hidden constant root and a supplied auxiliary jointly drive one target."""
    return p.PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "auxiliary_prefix_control",
                "parent_candidate_id": None,
                "states": [{"name": name, "kind": "latent"} for name in ("x", "r")],
                "state_equations": [
                    {"state": "x", "rhs": "r+a"},
                    {"state": "r", "rhs": "0"},
                ],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "parameters": [],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "r")
                ],
            },
            "context": {
                "targets": ["y"],
                "auxiliaries": ["a", "unused"],
                "external_inputs": ["u"],
                "fixed_covariates": ["c"],
            },
            "graph": {
                "rules": {
                    "r": {
                        "initial": {
                            "mode": "value",
                            "scope": "trajectory",
                            "guess": 0.8,
                        }
                    }
                }
            },
        }
    )


def row(name="heldout", root=2.3):
    """Exact synthetic observations for a(t)=1+t, with a separate unused channel."""
    time = np.linspace(0, 2, 9)
    return PublicTrajectory(
        trajectory_id=name,
        time=tuple(time),
        targets={"y": tuple((root + 1) * time + time**2 / 2)},
        auxiliaries={"a": tuple(1 + time), "unused": tuple(20 + time)},
        external_inputs={"u": tuple(time)},
        fixed_covariates={"c": 1},
    )


@pytest.fixture(scope="module")
def fitted():
    model = problem()
    numerical = p.NumericalPolicy(maximum_calls=40)
    fit = p.fit_training(
        model,
        PublicSplit(
            name="train",
            fingerprint="synthetic-auxiliary-only",
            rows=(row("train_a", 0.4), row("train_b", 1.7)),
        ),
        numerical,
    )
    assert fit["fit_success"], fit
    return model, fit, numerical


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_empty_auxiliary_opt_in_preserves_historical_payload_hashes(explicit_empty):
    legacy = PublicTrajectory(
        trajectory_id="legacy",
        time=(0, 0.5, 1),
        targets={"y": (0, 1, 2)},
        auxiliaries={"a": (1, 2, 3)},
        external_inputs={"u": (3, 4, 5)},
        fixed_covariates={"c": 4},
    )
    policy = p.PrefixPolicy(
        duration=0.5,
        future_inputs=("u",),
        **({"future_auxiliaries": ()} if explicit_empty else {}),
    )
    prefix = p.make_prefix(legacy, policy)
    schedule = p.make_forecast_schedule(legacy, policy)
    # Captured from the unchanged historical adapter before adding the opt-in.
    receipts = [
        (policy, "0729a15f553056d4bb159da84a0b41fa305be042c1f3b7a25aea9ae3eff98f5e"),
        (prefix, "c234b0c5f55e351a15718e160c144af0af36f5afa4f324ed336dc00bbce322d1"),
        (schedule, "d1700c23e4d29e6ba2a9938821b374647df1516a2a967c5259f902a2bc3a033c"),
        (
            p._boundary(prefix),
            "caabf48a00958f5e7aa1f3c6bbf29089091391428fb7d47909ca0d02c5928d32",
        ),
        (
            p._prefix_forcing(prefix, 0.5),
            "943be2a1d9e524cf4ab17ce7e57e587b62995f19960cdc7df86c8fa4aaee3338",
        ),
    ]
    for value, expected in receipts:
        assert content_sha256(value) == expected
    assert p._boundary(schedule) == p._boundary(prefix)
    assert p._prefix_forcing(schedule, 0.5) == p._prefix_forcing(prefix, 0.5)
    for view in (prefix, schedule):
        for mode in ("python", "json"):
            payload = view.model_dump(mode=mode)
            assert "auxiliaries" not in payload
            assert "future_auxiliaries" not in payload["policy"]
        assert type(view).model_validate_json(view.model_dump_json()) == view


def test_auxiliary_allowlist_is_explicit_separate_and_unique():
    policy = p.PrefixPolicy(duration=0.5, future_auxiliaries=("a",))
    assert policy.model_dump(mode="json")["future_auxiliaries"] == ["a"]
    assert p.PrefixPolicy.model_validate_json(policy.model_dump_json()) == policy
    assert content_sha256(policy) != content_sha256(p.PrefixPolicy(duration=0.5))
    with pytest.raises(ValidationError, match="duplicate future auxiliary"):
        p.PrefixPolicy(duration=0.5, future_auxiliaries=("a", "a"))
    with pytest.raises(ValidationError, match="roles overlap"):
        p.PrefixPolicy(duration=0.5, future_inputs=("a",), future_auxiliaries=("a",))


def test_opt_in_numeric_training_prefix_and_forecast(fitted):
    model, fit, numerical = fitted
    heldout = row()
    policy = p.PrefixPolicy(duration=0.5, future_auxiliaries=("a",))
    before = content_sha256(fit)
    prefix = p.make_prefix(heldout, policy)
    schedule = p.make_forecast_schedule(heldout, policy)
    assert prefix.auxiliaries == {"a": heldout.auxiliaries["a"][:3]}
    assert schedule.auxiliaries == {"a": heldout.auxiliaries["a"]}
    assert len(p._trajectory(prefix).auxiliaries["a"]) == 3
    assert np.array_equal(
        p._trajectory(schedule).auxiliaries["a"], 1 + np.array(row().time)
    )
    assert p.PrefixView.model_validate_json(prefix.model_dump_json()) == prefix
    assert (
        p.ForecastSchedule.model_validate_json(schedule.model_dump_json()) == schedule
    )
    inferred = p.infer_prefix(model, fit, prefix, numerical)
    assert inferred["fit_success"], inferred
    assert inferred["roots"] == pytest.approx({"init_r_value": 2.3}, abs=1e-7)
    predicted = p.forecast(model, fit, inferred, schedule, numerical)
    assert predicted["forecast_success"], predicted
    score = p.score_forecast(predicted, heldout, 0.5, fit["training_scales"])
    assert score["available"] and score["normalized_mse"] < 1e-14
    assert content_sha256(fit) == before


def test_auxiliary_forcing_is_still_denied_by_default(fitted):
    model, fit, numerical = fitted
    prefix = p.make_prefix(row(), p.PrefixPolicy(duration=0.5))
    assert not prefix.auxiliaries
    with pytest.raises(ValueError, match="target/auxiliary"):
        p.infer_prefix(model, fit, prefix, numerical)


@pytest.mark.parametrize("builder", [p.make_prefix, p.make_forecast_schedule])
@pytest.mark.parametrize("channel", ["y", "u", "c", "missing"])
def test_auxiliary_permission_requires_a_supplied_auxiliary_role(builder, channel):
    with pytest.raises(ValueError, match="not supplied auxiliary"):
        builder(row(), p.PrefixPolicy(duration=0.5, future_auxiliaries=(channel,)))


@pytest.mark.parametrize("builder", [p.make_prefix, p.make_forecast_schedule])
@pytest.mark.parametrize("channel", ["y", "u", "c"])
def test_duplicating_another_row_role_does_not_authorize_auxiliary(builder, channel):
    payload = row().model_dump(mode="json")
    payload["auxiliaries"][channel] = list(row().auxiliaries["a"])
    duplicated = PublicTrajectory.model_validate(payload)
    with pytest.raises(ValueError, match="overlaps a target, input or covariate"):
        builder(duplicated, p.PrefixPolicy(duration=0.5, future_auxiliaries=(channel,)))


@pytest.mark.parametrize("channel", ["y", "u", "c", "undeclared"])
def test_auxiliary_permission_must_match_context_role(fitted, channel):
    model, _, numerical = fitted
    system = p._compile(model, numerical)[0]
    mislabeled = PublicTrajectory(
        trajectory_id="wrong_context",
        time=(0, 1),
        targets={"other": (0, 1)},
        auxiliaries={channel: (1, 2)},
    )
    prefix = p.make_prefix(
        mislabeled, p.PrefixPolicy(duration=1, future_auxiliaries=(channel,))
    )
    with pytest.raises(ValueError, match="explicitly declared auxiliary"):
        p._validate_future(system, prefix)


@pytest.mark.parametrize("builder", [p.make_prefix, p.make_forecast_schedule])
@pytest.mark.parametrize("defect", ["missing", "extra", "initial", "length", "target"])
def test_auxiliary_schedule_consistency_fails_closed(builder, defect):
    view = builder(row(), p.PrefixPolicy(duration=0.5, future_auxiliaries=("a",)))
    payload = view.model_dump(mode="json")
    if defect == "missing":
        payload["auxiliaries"] = {}
    elif defect == "extra":
        payload["auxiliaries"]["unused"] = payload["auxiliaries"]["a"]
    elif defect == "initial":
        payload["initial_auxiliaries"]["a"] += 1
    elif defect == "length":
        payload["auxiliaries"]["a"].pop()
    else:
        payload["policy"]["future_auxiliaries"] = ["y"]
        payload["auxiliaries"]["y"] = payload["auxiliaries"].pop("a")
        payload["initial_auxiliaries"]["y"] = payload["auxiliaries"]["y"][0]
    with pytest.raises(ValidationError):
        type(view).model_validate(payload)


def test_required_auxiliary_schedule_cannot_be_replaced_by_initial_constant(fitted):
    model, fit, numerical = fitted
    prefix = p.make_prefix(
        row(), p.PrefixPolicy(duration=0.5, future_auxiliaries=("a",))
    )
    invalid = prefix.model_copy(update={"auxiliaries": {}})
    with pytest.raises(ValueError, match="auxiliary forcing schedule is incomplete"):
        p.infer_prefix(model, fit, invalid, numerical)


def test_suffix_targets_are_never_exposed_to_auxiliary_inference_or_forecast(fitted):
    model, fit, numerical = fitted
    original = row()
    payload = original.model_dump(mode="json")
    payload["targets"]["y"][3:] = [1e12] * 6
    poisoned = PublicTrajectory.model_validate(payload)
    policy = p.PrefixPolicy(duration=0.5, future_auxiliaries=("a",))
    prefixes = [p.make_prefix(item, policy) for item in (original, poisoned)]
    schedules = [
        p.make_forecast_schedule(item, policy) for item in (original, poisoned)
    ]
    assert prefixes[0] == prefixes[1]
    assert schedules[0] == schedules[1]
    results = [p.infer_prefix(model, fit, prefix, numerical) for prefix in prefixes]
    assert results[0]["roots"] == results[1]["roots"]
    forecasts = [
        p.forecast(model, fit, results[0], schedule, numerical)
        for schedule in schedules
    ]
    assert forecasts[0]["forecast_success"] and forecasts[1]["forecast_success"]
    assert forecasts[0]["predictions"] == forecasts[1]["predictions"]


def test_suffix_auxiliaries_change_forecast_without_changing_prefix_inference(fitted):
    model, fit, numerical = fitted
    original = row()
    payload = original.model_dump(mode="json")
    payload["auxiliaries"]["a"][3:] = [
        value + 2 * (time - 0.5)
        for time, value in zip(
            original.time[3:], original.auxiliaries["a"][3:], strict=True
        )
    ]
    changed = PublicTrajectory.model_validate(payload)
    policy = p.PrefixPolicy(duration=0.5, future_auxiliaries=("a",))
    prefixes = [p.make_prefix(item, policy) for item in (original, changed)]
    assert prefixes[0] == prefixes[1]
    results = [p.infer_prefix(model, fit, prefix, numerical) for prefix in prefixes]
    assert results[0]["roots"] == results[1]["roots"]
    assert results[0]["prefix_sha256"] == results[1]["prefix_sha256"]
    schedules = [p.make_forecast_schedule(item, policy) for item in (original, changed)]
    assert p._boundary(schedules[0]) == p._boundary(schedules[1])
    assert p._prefix_forcing(schedules[0], 0.5) == p._prefix_forcing(schedules[1], 0.5)
    forecasts = [
        p.forecast(model, fit, results[0], schedule, numerical)
        for schedule in schedules
    ]
    assert all(forecast["forecast_success"] for forecast in forecasts)
    assert forecasts[0]["schedule_sha256"] != forecasts[1]["schedule_sha256"]
    assert forecasts[1]["predictions"]["y"][-1] - forecasts[0]["predictions"]["y"][
        -1
    ] == pytest.approx(2.25)


def test_forecast_binds_auxiliary_prefix_history_and_initial_boundary(fitted):
    model, fit, numerical = fitted
    policy = p.PrefixPolicy(duration=0.5, future_auxiliaries=("a",))
    prefix = p.make_prefix(row(), policy)
    inference = p.infer_prefix(model, fit, prefix, numerical)
    schedule = p.make_forecast_schedule(row(), policy)
    payload = schedule.model_dump(mode="json")
    payload["auxiliaries"]["a"][1] += 1
    changed = p.ForecastSchedule.model_validate(payload)
    assert (
        content_sha256(p._prefix_forcing(changed, 0.5))
        != inference["prefix_forcing_sha256"]
    )
    with pytest.raises(ValueError, match="time/input history"):
        p.forecast(model, fit, inference, changed, numerical)
    payload = deepcopy(schedule.model_dump(mode="json"))
    payload["auxiliaries"]["a"][0] += 1
    payload["initial_auxiliaries"]["a"] += 1
    with pytest.raises(ValueError, match="initial boundary or policy"):
        p.forecast(
            model, fit, inference, p.ForecastSchedule.model_validate(payload), numerical
        )
