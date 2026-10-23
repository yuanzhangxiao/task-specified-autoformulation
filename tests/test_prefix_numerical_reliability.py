"""Opt-in numerical rejection and fixed-threshold independent replay controls."""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import recovery_numerics as recovery
from tests.test_prefix_initialization import problem, training


class TrialOracle:
    """A smooth reachable optimum beyond a deliberately unusable large trial."""

    names = ("a",)
    units = np.ones(1)
    lower, upper = np.array([-np.inf]), np.array([np.inf])
    start: ClassVar = {"a": 0.1}

    def __init__(self, failure="typed"):
        self.failure, self.points = failure, []

    def vector(self, parameters):
        return np.array([parameters["a"]])

    def parameters(self, vector):
        return {"a": float(vector[0])}

    def evaluate(self, vector, deadline):
        self.points.append(float(vector[0]))
        if vector[0] > 0.8:
            if self.failure == "residual":
                return np.array([np.inf]), np.ones((1, 1))
            if self.failure == "sensitivity":
                return vector**2 - 0.25, np.full((1, 1), np.nan)
            if self.failure == "timeout":
                raise TimeoutError("actual deadline")
            if self.failure == "programming":
                raise ValueError("invalid array contract")
            if self.failure == "shape":
                return np.empty(0), np.empty((0, 1))
            raise recovery.NumericalInvalidTrial("transition overflow")
        return vector**2 - 0.25, np.array([[2 * vector[0]]])


def run_search(oracle, **kwargs):
    """Exercise the installed SciPy TRF, not an emulated optimizer."""
    saved = []
    result = recovery.search(
        oracle,
        oracle.start,
        monotonic() + 10,
        lambda value: saved.append(deepcopy(value)),
        calls=40,
        target=1e-20,
        objective_multiplier=1.0,
        telemetry=True,
        record_steps=True,
        **kwargs,
    )
    return result, saved


@pytest.mark.parametrize("failure", ["typed", "residual", "sensitivity"])
def test_real_trf_rejects_nonfinite_trials_and_reaches_finite_solution(failure):
    oracle = TrialOracle(failure)
    result, saved = run_search(oracle, reject_invalid_trials=True)
    assert result["stop_reason"] == "numerical_target_reached"
    assert result["best"]["parameters"]["a"] == pytest.approx(0.5)
    assert result["rejected_calls"] >= 1
    assert result["failed_calls"] == result["rejected_calls"]
    assert result["calls_started"] == len(oracle.points) == result["calls"]
    assert result["completed_calls"] + result["failed_calls"] == result["calls"]
    rejected_index = next(i for i, value in enumerate(oracle.points) if value > 0.8)
    assert 0.1 < oracle.points[rejected_index + 1] < oracle.points[rejected_index]
    assert all(np.isfinite(row["training_nmse"]) for row in result["evaluations"])
    assert all(np.isfinite(row["training_nmse"]) for row in result["accepted_steps"])
    assert saved[-1]["rejected_calls"] == result["rejected_calls"]


def test_rejection_remains_opt_in_and_preserves_finite_incumbent():
    result, _ = run_search(TrialOracle())
    assert result["stop_reason"] == "numerical_failure"
    assert result["best"]["parameters"] == TrialOracle.start
    assert "rejected_calls" not in result


@pytest.mark.parametrize("failure", ["timeout", "programming", "shape"])
def test_rejection_never_hides_deadlines_or_genuine_errors(failure):
    oracle = TrialOracle(failure)
    result, _ = run_search(oracle, reject_invalid_trials=True)
    assert result["stop_reason"] == (
        "budget_limited" if failure == "timeout" else "numerical_failure"
    )
    assert result["failed_calls"] == 1
    assert result["rejected_calls"] == 0
    assert result["best"]["parameters"] == TrialOracle.start


def test_invalid_initial_point_cannot_supply_shape_or_curvature():
    oracle = TrialOracle()
    oracle.start = {"a": 1.0}
    result, _ = run_search(oracle, reject_invalid_trials=True)
    assert result["stop_reason"] == "numerical_failure"
    assert result["best"] is None
    assert result["calls_started"] == result["failed_calls"] == 1
    assert result["rejected_calls"] == 0


def test_rejected_point_never_gets_a_zero_fabricated_jacobian(monkeypatch):
    def optimizer(fun, start, *, jac, **kwargs):
        assert np.isfinite(fun(start)).all()
        assert np.isfinite(jac(start)).all()
        rejected = np.array([1.0])
        assert np.isinf(fun(rejected)).all()
        with pytest.raises(recovery.NumericalInvalidTrial, match="Jacobian requested"):
            jac(rejected)
        raise RuntimeError("stop after checking rejected curvature")

    monkeypatch.setattr(recovery, "least_squares", optimizer)
    result, _ = run_search(TrialOracle(), reject_invalid_trials=True)
    assert result["rejected_calls"] == 1
    assert result["best"]["parameters"] == TrialOracle.start


def test_affine_nonfinite_is_typed_but_domain_and_deadline_are_not(monkeypatch):
    oracle = affine.TrainingOracle(problem(), training(), kernel.NumericalPolicy())
    vector = oracle.vector(oracle.start)
    monkeypatch.setattr(affine, "expm", lambda matrix: np.full_like(matrix, np.inf))
    with pytest.raises(recovery.NumericalInvalidTrial, match="transition"):
        oracle.evaluate(vector, monotonic() + 10)
    with pytest.raises(TimeoutError):
        oracle.evaluate(vector, monotonic() - 1)
    bad = vector.copy()
    bad[oracle.names.index("theta")] = -1
    with pytest.raises(ValueError, match="outside declared domain") as raised:
        oracle.evaluate(bad, monotonic() + 10)
    assert not isinstance(raised.value, recovery.NumericalInvalidTrial)


def test_continuation_exposes_rejected_and_failed_trial_counts():
    result = continuation.search(
        TrialOracle("sensitivity"),
        TrialOracle.start,
        10,
        40,
        lambda _: None,
        reject_invalid_trials=True,
    )
    assert result["best"]["parameters"]["a"] == pytest.approx(0.5, abs=1e-6)
    assert result["search_calls_failed"] == result["rejected_calls"] > 0
    assert result["search_calls_started"] == (
        result["search_calls_failed"] + result["search_calls_completed"]
    )


@pytest.mark.parametrize(
    ("residual", "jacobian"),
    [
        (np.empty(0), np.empty((0, 1))),
        (np.array(np.nan), np.ones((1, 1))),
        (np.array([[np.nan]]), np.ones((1, 1))),
        (np.ones(1), np.array(np.nan)),
        (np.ones(1), np.full(1, np.nan)),
        (np.ones(1), np.full((2, 1), np.nan)),
        (np.ones(1), np.full((1, 2), np.nan)),
        (np.full(2, np.nan), np.ones((2, 1))),
    ],
    ids=[
        "empty",
        "scalar_residual",
        "matrix_residual",
        "scalar_jacobian",
        "vector_jacobian",
        "wrong_jacobian_rows",
        "wrong_jacobian_columns",
        "changed_residual_length",
    ],
)
def test_continuation_never_rejects_oracle_shape_contract_errors(residual, jacobian):
    class MalformedOracle(TrialOracle):
        def evaluate(self, vector, deadline):
            if vector[0] > 0.8:
                self.points.append(float(vector[0]))
                return residual, jacobian
            return super().evaluate(vector, deadline)

    oracle = MalformedOracle()
    result = continuation.search(
        oracle, oracle.start, 10, 40, lambda _: None, reject_invalid_trials=True
    )
    assert result["stop_reason"] == "numerical_failure"
    assert result["message"] == "invalid fitted residual or derivative shape"
    assert result["rejected_calls"] == 0
    assert result["search_calls_started"] == 2
    assert result["search_calls_completed"] == result["search_calls_failed"] == 1
    assert result["best"]["parameters"] == oracle.start
    assert len(oracle.points) == 2


class ReplayOracle:
    """A replay spy with tolerance-controlled error and deterministic timing."""

    system = SimpleNamespace(channels=("y",))
    numerical_policy = kernel.NumericalPolicy()
    start: ClassVar = {"a": 0.1}

    def __init__(self, errors, clock=None, failure=None):
        self.errors, self.clock, self.failure = errors, clock, failure
        self.used_policies, self.evaluations = [], []

    def vector(self, parameters):
        return np.array([parameters["a"]])

    def parameters(self, vector):
        return {"a": float(vector[0])}

    def with_numerical_policy(self, policy):
        self.used_policies.append(policy)
        replay = ReplayOracle(self.errors, self.clock, self.failure)
        replay.numerical_policy = policy
        replay.evaluations = self.evaluations
        return replay

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        assert jacobian is False
        self.evaluations.append((method, self.numerical_policy.rtol, deadline))
        if self.clock is not None:
            self.clock[0] += 0.3
        if method == "Radau" and self.failure is not None:
            raise self.failure
        error = self.errors[self.numerical_policy.rtol] if method == "Radau" else 0.0
        return np.array([1.0, -1.0]) + error, None


def test_ladder_advances_tolerances_without_relaxing_acceptance():
    oracle = ReplayOracle({1e-9: 2e-6, 1e-11: 2e-7, 1e-12: 5e-8})
    saved = []
    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 10, lambda r: saved.append(deepcopy(r))
    )
    assert result["original_equations_verified"]
    assert result["solver_prediction_discrepancy"] == pytest.approx(5e-8)
    assert result["no_search_performed"]
    assert result["verification_calls_started"] == 4
    assert result["verification_calls_completed"] == 4
    assert result["verification_calls_failed"] == 0
    assert [(p.rtol, p.atol) for p in oracle.used_policies] == [
        (1e-9, 1e-11),
        (1e-11, 1e-13),
        (1e-12, 1e-14),
    ]
    assert [r["original_equations_verified"] for r in result["replays"][1:]] == [
        False,
        False,
        True,
    ]
    assert oracle.numerical_policy == kernel.NumericalPolicy()
    assert all(
        any(
            len(snapshot["replays"]) == count
            and snapshot["replays"][-1]["status"] == "reserved"
            for snapshot in saved
        )
        for count in range(1, 5)
    )


def test_ladder_stops_after_first_agreement_and_preserves_old_verifier():
    oracle = ReplayOracle({1e-9: 1e-8})
    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 10, lambda _: None
    )
    assert result["original_equations_verified"]
    assert len(oracle.used_policies) == 1
    assert result["verification_calls_completed"] == 2
    oracle.used_policies.clear()
    old = continuation.verify_endpoint(oracle, oracle, oracle.start, 10, lambda _: None)
    assert old["original_equations_verified"]
    assert not oracle.used_policies
    assert "tolerance_ladder" not in old


def test_ladder_refuses_persistent_disagreement_and_loss_only_disagreement():
    oracle = ReplayOracle({1e-9: 2e-7, 1e-11: 2e-7, 1e-12: 2e-7})
    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 10, lambda _: None
    )
    assert result["status"] == "complete"
    assert not result["original_equations_verified"]
    assert result["verification_calls_completed"] == 4

    class LossOracle(ReplayOracle):
        def with_numerical_policy(self, policy):
            return self

        def evaluate(self, vector, deadline, *, jacobian=True, method=None):
            return np.array([1.0 + (5e-8 if method else 0.0)]), None

    oracle = LossOracle({})
    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 10, lambda _: None
    )
    assert result["solver_prediction_discrepancy"] <= 1e-7
    assert result["solver_loss_discrepancy"] > 1e-8
    assert not result["original_equations_verified"]


def test_ladder_uses_one_budget_for_all_replays(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(verification, "monotonic", lambda: clock[0])
    oracle = ReplayOracle({1e-9: 2e-6, 1e-11: 2e-7, 1e-12: 0.0}, clock)
    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 0.8, lambda _: None
    )
    assert result["status"] == "incomplete"
    assert not result["original_equations_verified"]
    assert result["calls_started"] == 3
    assert len(oracle.used_policies) == 2
    assert result["timing"]["numerical_active_wall_seconds"] == pytest.approx(0.9)
    assert result["timing"]["timeout_reason"] == "numerical_active_wall_limit"


@pytest.mark.parametrize(
    "failure", [ValueError("actual bug"), TimeoutError("deadline")]
)
def test_ladder_counts_failed_replay_and_aborts(failure):
    oracle = ReplayOracle({}, failure=failure)
    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 10, lambda _: None
    )
    assert result["status"] == "incomplete"
    assert result["verification_calls_started"] == 2
    assert result["verification_calls_completed"] == 1
    assert result["verification_calls_failed"] == 1
    assert result["replays"][-1]["status"] == "failed"
    assert len(oracle.used_policies) == 1


def test_recompiled_oracle_applies_actual_stricter_solver_settings(monkeypatch):
    oracle = affine.TrainingOracle(problem(), training(), kernel.NumericalPolicy())
    tighter = oracle.with_numerical_policy(
        kernel.NumericalPolicy(rtol=1e-12, atol=1e-14)
    )
    seen = []
    original = kernel.symbolic_rollout

    def observe(*args, **kwargs):
        seen.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "symbolic_rollout", observe)
    tighter.evaluate(tighter.vector(tighter.start), monotonic() + 20, method="Radau")
    assert tighter is not oracle
    assert all(s.integration_method == "Radau" for s in seen)
    assert all(s.relative_tolerance == 1e-12 for s in seen)
    assert all(s.absolute_tolerance == 1e-14 for s in seen)
    assert oracle.settings.relative_tolerance == 1e-9
    assert oracle.settings.absolute_tolerance == 1e-11


def test_real_affine_original_equation_ladder():
    oracle = affine.TrainingOracle(problem(), training(), kernel.NumericalPolicy())
    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 20, lambda _: None
    )
    assert result["original_equations_verified"]
    assert result["verification_calls_completed"] == 2
    assert result["parameters"] == oracle.start


@pytest.mark.parametrize("failure", [None, "timeout", "write_error", "interrupt"])
def test_ladder_terminal_write_is_measured_and_revokes_failed_pass(
    monkeypatch, failure
):
    clock = [100.0]
    monkeypatch.setattr(verification, "monotonic", lambda: clock[0])
    oracle = ReplayOracle({1e-9: 0.0}, clock)
    final_calls = []

    def checkpoint(snapshot):
        if "verification_calls_completed" in snapshot:
            final_calls.append(snapshot)
            clock[0] += 200 if failure == "timeout" else 2
            if failure == "write_error":
                raise OSError("write failed")
            if failure == "interrupt":
                raise KeyboardInterrupt

    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt) as raised:
            continuation.verify_endpoint_ladder(
                oracle, oracle, oracle.start, 1, checkpoint
            )
        result = raised.value.execution_receipt
    else:
        result = continuation.verify_endpoint_ladder(
            oracle, oracle, oracle.start, 1, checkpoint
        )
    assert len(final_calls) == 1
    assert result["original_equations_verified"] is (failure is None)
    assert result["status"] == ("complete" if failure is None else "incomplete")
    assert result["timing"]["callback_wall_seconds"] == pytest.approx(
        200 if failure == "timeout" else 2
    )
    assert result["timing"]["numerical_active_wall_seconds"] == pytest.approx(0.6)


def test_ladder_expired_reservation_does_not_claim_a_started_replay(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(verification, "monotonic", lambda: clock[0])
    oracle = ReplayOracle({}, clock)

    def checkpoint(_):
        clock[0] += 200

    result = continuation.verify_endpoint_ladder(
        oracle, oracle, oracle.start, 1, checkpoint
    )
    assert result["verification_calls_attempted"] == 1
    assert result["verification_calls_started"] == 0
    assert result["verification_calls_failed"] == 0
    assert not result["original_equations_verified"]
    assert not oracle.evaluations


def test_precision_policy_flows_through_qualification_and_diagnosis():
    class TinyOracle(TrialOracle):
        affine_audit: ClassVar = {"eligible": True}

        def evaluate(self, vector, deadline, *, jacobian=True, method=None):
            residual = 1 + vector * 1e-20
            return residual, np.array([[1e-20]]) if jacobian else None

    oracle = TinyOracle()
    policy = continuation.adequacy.Policy(precision_aware=True)
    qualified = continuation.qualification(
        oracle, oracle, oracle.start, 10, lambda _: None, adequacy_policy=policy
    )
    assert qualified["passed"]
    assert qualified["derivative_check"]["status"] == "inconclusive"
    assert qualified["derivative_check"]["search_qualified"]
    assert not continuation.qualification(
        oracle, oracle, oracle.start, 10, lambda _: None
    )["passed"]
    verified = {
        "reduced_vector": [0.1],
        "training_nmse": 1.0,
        "alternate_training_nmse": 1.0,
        "solver_prediction_discrepancy": 0.0,
        "solver_loss_discrepancy": 0.0,
        "original_equations_verified": True,
    }
    diagnosed = continuation.diagnose(
        oracle,
        oracle.start,
        verified,
        {"stop_reason": "step_or_gradient_converged", "budget_exhausted": False},
        10,
        lambda _: None,
        adequacy_policy=policy,
    )
    assert diagnosed["assessment"]["policy"]["precision_aware"]
    assert diagnosed["assessment"]["parameter_information"] == (
        "derivative_precision_unresolved"
    )


def test_fixed_strict_replay_uses_only_declared_tolerance():
    oracle = ReplayOracle({1e-11: 2e-7})
    result = continuation.verify_endpoint_ladder(
        oracle,
        oracle,
        oracle.start,
        10,
        lambda _: None,
        tolerances=((1e-11, 1e-13),),
    )
    assert not result["original_equations_verified"]
    assert result["verification_calls_started"] == 2
    assert result["tolerance_ladder"] == [{"rtol": 1e-11, "atol": 1e-13}]
    assert [(p.rtol, p.atol) for p in oracle.used_policies] == [(1e-11, 1e-13)]


@pytest.mark.parametrize(
    "tolerances",
    [
        (),
        ((0, 1e-13),),
        ((1e-11, float("nan")),),
        ((1e-11, 1e-13), (1e-9, 1e-11)),
        ((True, 1e-13),),
    ],
)
def test_bad_tolerance_policy_rejected_before_any_rollout(tolerances):
    oracle = ReplayOracle({})
    with pytest.raises(ValueError, match="tolerances required"):
        continuation.verify_endpoint_ladder(
            oracle, oracle, oracle.start, 10, lambda _: None, tolerances=tolerances
        )
    assert not oracle.evaluations
