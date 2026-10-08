"""Truth-free evidence checks, compensating parameters, limits and exact resume."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as c
from autoformalism.fitting import confidence_fit


class LinearOracle:
    names = ("a", "init_z")
    units = np.ones(2)
    lower = np.array([-20.0, -20.0])
    upper = np.array([20.0, 20.0])
    system = SimpleNamespace(initial_parameter_names=("init_z",))

    def __init__(self, weak=False):
        self.j = np.array([[1.0, 1.0], [1.0, 1.0 if weak else -1.0]])

    def vector(self, p):
        return np.array([p[n] for n in self.names])

    def parameters(self, x):
        return dict(zip(self.names, x.tolist(), strict=True))

    def evaluate(self, x, deadline, **kwargs):
        return self.j @ (x - [2, 3]), self.j


def test_joint_initial_sensitivity_detects_compensation():
    oracle = LinearOracle(weak=True)
    audit = c.sensitivity(oracle.j, oracle.units, oracle.names)
    assert audit["rank"] == 1 and audit["initials_included"]
    d = np.array(list(audit["weak_direction"].values()))
    assert np.linalg.norm(oracle.j @ d) < 1e-12


def test_profile_refits_nuisance_initial_and_preserves_fixed():
    from time import monotonic

    oracle = LinearOracle(weak=True)
    p = c.ConfidencePolicy()
    result = c.optimize(
        oracle,
        [{"a": 0.0, "init_z": 0.0}],
        p,
        monotonic() + 3,
        lambda _: None,
        fixed={"a": 4.0},
    )
    assert result["best"]["parameters"]["a"] == 4.0
    assert result["best"]["parameters"]["init_z"] == pytest.approx(1.0, abs=1e-8)
    assert result["best"]["training_nmse"] < 1e-18


def test_unavailable_rollout_not_zero_residual():
    from time import monotonic

    oracle = LinearOracle()
    oracle.evaluate = lambda *a, **k: (_ for _ in ()).throw(ValueError("bad solve"))
    result = c.optimize(
        oracle,
        [{"a": 0.0, "init_z": 0.0}],
        c.ConfidencePolicy(),
        monotonic() + 3,
        lambda _: None,
    )
    assert result["best"] is None and not result["all_attempts_converged"]


def test_profile_domain_edges_are_not_clipped():
    oracle = LinearOracle()
    oracle.lower = np.array([0.0, -20.0])
    points = list(c.profile_grid(oracle, {"a": 0.0, "init_z": 1.0}, (0.01, 0.1)))
    assert len(points) == 8
    assert sum(not p["in_domain"] for p in points) == 2
    assert points[0]["value"] < 0


def test_operation_resume_and_interruption(tmp_path):
    calls = []

    def work(deadline, checkpoint):
        calls.append(1)
        checkpoint({"calls": 2})
        return {"v": 1}

    a = c.operation(tmp_path / "ok", {"v": 1}, 10, work)
    assert a == c.operation(tmp_path / "ok", {"v": 1}, 10, work)
    assert len(calls) == 1 and a["calls"] == 2
    with pytest.raises(ValueError, match="identity"):
        c.operation(tmp_path / "ok", {"v": 2}, 10, work)
    seal(tmp_path / "broken" / "started.json", {"v": 1, "allowance_seconds": 10})
    b = c.operation(tmp_path / "broken", {"v": 1}, 10, work)
    assert b["status"] == "interrupted" and not b["accounting_complete"]
    assert len(calls) == 1 and b["budget_charge_seconds"] == 10


def test_confidence_is_not_probability_or_global_exclusion():
    policy = c.ConfidencePolicy()
    selected = {"a": 1.0}
    inspected = {"training_nmse": 0.0, "solver_loss_discrepancy": 0.0}
    grid = [
        {
            "parameter": "a",
            "value": 2.0,
            "scale": 1.0,
            "checked": inspected,
            "optimizer_converged": False,
        }
    ]
    result = c.confidence_report(selected, inspected, grid, selected, policy, False)
    assert result["level"] == "weakly_constrained" and result["probability"] is None
    grid[0]["checked"] = None
    assert (
        c.confidence_report(selected, inspected, grid, selected, policy, False)["level"]
        == "search_incomplete"
    )
    inspected["solver_loss_discrepancy"] = 1e-9
    assert (
        c.confidence_report(selected, inspected, grid, selected, policy, True)["level"]
        == "numerically_unresolved"
    )
    assert (
        c.confidence_report(selected, None, grid, selected, policy, False)["level"]
        == "unavailable"
    )


def test_changed_profile_center_never_claims_local_support():
    s = {"a": 1.0}
    check = {"training_nmse": 0.0, "solver_loss_discrepancy": 0.0}
    profiles = [{"optimizer_converged": True}]
    assert (
        c.confidence_report(s, check, profiles, {"a": 2.0}, c.ConfidencePolicy(), True)[
            "level"
        ]
        == "search_incomplete"
    )


def test_policy_rejects_invalid_tolerance():
    with pytest.raises(ValidationError):
        c.ConfidencePolicy(offsets=(0.1, -0.1))
    with pytest.raises(ValidationError):
        c.ConfidencePolicy(loss_tolerances=(1e-8, 1e-10, 1e-12))


def test_training_payload_excludes_reference_and_validation():
    with pytest.raises(ValidationError):
        c.TrainingProblem.model_validate({"reference_parameters": {}, "validation": {}})


def test_controller_stages_checkpoint_and_truth_free(monkeypatch, tmp_path):
    oracle = LinearOracle()
    monkeypatch.setattr(c, "Rollouts", lambda *_: oracle)
    problem = SimpleNamespace(
        incumbent={"a": 1.0, "init_z": 2.0},
        start={"a": 0.0, "init_z": 0.0},
        model_dump=lambda **_: {"incumbent": [1, 2]},
    )
    policy = c.ConfidencePolicy(restart_seconds=2, profile_seconds=2, check_seconds=2)
    result = confidence_fit.run(problem, policy, tmp_path)
    assert (
        result["status"] == "complete" and result["selected"]["training_nmse"] < 1e-18
    )
    assert len(result["profiles"]) == 8
    assert (
        not result["reference_values_used"]
        and not result["validation_used_for_selection"]
    )
    assert confidence_fit.run(problem, policy, tmp_path) == read_seal(
        tmp_path / "finished.json"
    )
