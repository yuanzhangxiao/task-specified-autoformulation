"""Public bridge: unchanged equations, typed boundaries, starts and durable resume."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_rollout as bridge
from tests.test_adapter_campaign import fake_backend
from tests.test_public_fitting import _control


def inputs(multiple=False):
    request, train, _ = _control(multiple_targets=multiple)
    return bridge.Inputs(request=request, training=train.model_dump(mode="json"))


def test_initial_map_multiple_outputs_and_deterministic_domain_only_starts():
    data = inputs(True)
    before = data.model_dump(mode="json")
    a = bridge.prepare(data, bridge.Policy())
    assert a == bridge.prepare(data, bridge.Policy())
    assert data.model_dump(mode="json") == before
    assert a["target_count"] == 2 and a["initial_parameter_names"] == ["init_z_scale"]
    assert a["parameter_count"] == 3
    assert a["problem"]["request"] == before["request"]
    assert len({bridge.public.content_sha256(p) for p in a["starts"]}) == 4
    oracle, _, _ = bridge.nonlinear_rollout.build(
        bridge.checks.TrainingProblem.model_validate(a["problem"])
    )
    for p in a["starts"]:
        oracle.vector(p)


def test_missing_guess_uses_runtime_interval_not_truth_or_clipping():
    payload = inputs().model_dump(mode="json")
    payload["request"]["parameter_guesses"].pop("a")
    p = bridge.prepare(bridge.Inputs.model_validate(payload), bridge.Policy())
    assert p["defaulted_guesses"] == {"a": 1.5}
    assert p["problem"]["start"]["b"] == 1.2
    payload["request"]["parameter_guesses"]["a"] = -1
    with pytest.raises(ValueError, match="outside declared domain"):
        bridge.prepare(bridge.Inputs.model_validate(payload), bridge.Policy())


def test_validation_and_extra_oracle_information_rejected():
    payload = inputs().model_dump(mode="json")
    for key in ("validation", "reference_parameters", "hidden_trajectories"):
        with pytest.raises(ValidationError):
            bridge.Inputs.model_validate(payload | {key: {}})
    payload["training"]["name"] = "val"
    with pytest.raises(ValidationError):
        bridge.Inputs.model_validate(payload)


def test_incumbent_preserved_and_terminal_resume_checks_identity(tmp_path, monkeypatch):
    data = inputs()
    point = {"a": 0.4, "b": 0.8, "init_z_scale": 0.7}
    data = data.model_copy(update={"incumbent": point})
    seen = []

    def backend(problem, starts, *args):
        seen.append(problem.incumbent)
        return fake_backend(problem, starts, *args)

    monkeypatch.setattr(bridge.adapter.recovery, "run", backend)
    result = bridge.run(data, bridge.Policy(), tmp_path)
    assert seen == [point]
    assert not result.validation_used_for_selection
    monkeypatch.setattr(bridge, "prepare", lambda *a: pytest.fail("repeat setup"))
    assert bridge.run(data, bridge.Policy(), tmp_path) == result
    with pytest.raises(ValueError, match="sealed artifact differs"):
        bridge.run(
            data.model_copy(update={"incumbent": None}), bridge.Policy(), tmp_path
        )
    backend = read_seal(tmp_path / "fit/backend.json")
    (tmp_path / "fit/backend.json").unlink()
    seal(tmp_path / "fit/backend.json", backend | {"tampered": True})
    with pytest.raises(ValueError, match="backend differs"):
        bridge.completed(tmp_path)


def test_unsupported_and_bad_incumbent_reported_without_search(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge.adapter, "run", lambda *a: pytest.fail("must not fit"))
    data = inputs().model_copy(update={"incumbent": {"a": 1.0}})
    row = bridge.run(data, bridge.Policy(), tmp_path / "initial")
    assert row.status == "setup_unavailable" and row.fit is None
    assert "incomplete parameter vector" in row.preparation["value"]["error"]
    payload = inputs().model_dump(mode="json")
    payload["request"]["context"]["auxiliaries"] = ["not_supplied"]
    result = bridge.run(
        bridge.Inputs.model_validate(payload), bridge.Policy(), tmp_path / "data"
    )
    assert "auxiliaries differ" in result.preparation["value"]["error"]


def test_interrupted_preparation_does_not_reset_budget(tmp_path, monkeypatch):
    data, policy = inputs(), bridge.Policy()

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(bridge, "prepare", interrupt)
    with pytest.raises(KeyboardInterrupt):
        bridge.run(data, policy, tmp_path)
    monkeypatch.setattr(
        bridge, "prepare", lambda *a: pytest.fail("reset interrupted setup")
    )
    row = bridge.run(data, policy, tmp_path)
    assert row.status == "setup_unavailable"
    assert row.preparation["status"] == "interrupted"
    assert row.preparation["budget_charge_seconds"] == 180
    assert not row.preparation["accounting_complete"]


def test_piecewise_opt_in_preserves_expression():
    payload = deepcopy(inputs().model_dump(mode="json"))
    payload["request"]["base_candidate"]["state_equations"][1]["rhs"] = "-b*max(z,0)"
    data = bridge.Inputs.model_validate(payload)
    with pytest.raises(ValueError):
        bridge.prepare(data, bridge.Policy())
    p = bridge.prepare(data, bridge.Policy(allow_piecewise=True))
    assert p["problem"]["allow_piecewise"] is True
    assert p["problem"]["request"] == payload["request"]
