"""Exact domains, inactive definitions and nonlinear private-coefficient lifts."""

from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting.prefix_initialization import NumericalPolicy, TrainingOracle
from autoformalism.fitting.prefix_structural_reduction import (
    certify_structural_reduction,
)
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.candidate import ConstraintSpec, ValueRange
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory
from tests.test_prefix_gain_reduction import inspect_candidate, oracle_for


def structural_case(*, rhs="-x/tauA-g*q+b", extra=(), observation="x"):
    """A private decay with a separately offset nonnegative source coefficient."""
    return oracle_for(
        law="x/tauB",
        rhs=f"({rhs})+a+b",
        observation=observation,
        extra_parameters=[
            {"name": n, "scope": "global", "role": "time_constant"}
            for n in ("tauA", "tauB")
        ]
        + [{"name": "b", "scope": "global", "role": "nonnegative_coefficient"}],
        extra_processes=extra,
    )


def wrapped(source, retained=None):
    retained = source.start if retained is None else retained
    return recovery.make_oracle(
        source.problem,
        training(source),
        NumericalPolicy(),
        retained,
        structural_reduction=True,
    )


def training(source):
    return PublicSplit(
        name="train",
        fingerprint="structural-control",
        rows=tuple(
            PublicTrajectory(
                trajectory_id=r.trajectory_id,
                time=tuple(r.time),
                targets={k: tuple(v) for k, v in r.targets.items()},
            )
            for r in source.rows
        ),
    )


def inspected_definition(source, expression, retained):
    """Check boundary guards independently of the current compiler's rejection."""
    original, base = recovery.make_oracle(
        source.problem, training(source), NumericalPolicy(), retained
    )
    candidate = original.system.model.validated.candidate
    candidate = candidate.model_copy(
        update={
            "processes": tuple(
                p.model_copy(update={"expression": expression})
                if p.name == "unused"
                else p
                for p in candidate.processes
            )
        }
    )
    return certify_structural_reduction(
        inspect_candidate(original, candidate),
        base.reduction,
        original.vector(retained),
    )


def test_inactive_process_tau_is_anchored_without_deleting_definition():
    source = oracle_for(
        law="z",
        rhs="-x+g*q+a",
        extra_parameters=[{"name": "tau", "scope": "global", "role": "time_constant"}],
        extra_processes=[{"name": "unused", "expression": "x/tau"}],
    )
    retained = source.start | {"tau": 2.75}
    before = source.problem.model_dump_json()
    original, oracle = wrapped(source, retained)
    assert oracle.reduction.inactive == ("tau",)
    assert "tau" not in oracle.names
    effective = continuation.to_effective(original, oracle, retained)
    lifted = continuation.to_original(original, oracle, effective)
    assert lifted["tau"] == 2.75
    assert source.problem.model_dump_json() == before
    assert any(p.name == "unused" for p in original.problem.base_candidate.processes)
    actual, derivative = oracle.evaluate(oracle.vector(effective), monotonic() + 10)
    expected, full = original.evaluate(original.vector(lifted), monotonic() + 10)
    assert actual == pytest.approx(expected, abs=1e-14)
    assert derivative == pytest.approx(
        full @ oracle.reduction.lift_jacobian(oracle.vector(effective)), abs=1e-14
    )


@pytest.mark.parametrize("expression", ["sqrt(tau*x)", "1/(tau-1)", "log(tau)"])
def test_unsafe_dead_definitions_are_not_frozen(expression):
    source = oracle_for(
        law="z",
        rhs="-x+g*q+a",
        extra_parameters=[{"name": "tau", "scope": "global", "role": "time_constant"}],
        extra_processes=[{"name": "unused", "expression": "tau"}],
    )
    reduction = inspected_definition(source, expression, source.start | {"tau": 2.0})
    assert "tau" in reduction.names
    assert not reduction.inactive


@pytest.mark.parametrize("rate", [1e-12, 0.25, 4.0, 1e20])
def test_private_decay_has_full_open_domain_and_smooth_admissible_lift(rate):
    source = structural_case(rhs="(1+b)-x/tauA-g*q")
    original, oracle = wrapped(source)
    assert len(oracle.reduction.decays) == 1
    decay = oracle.reduction.decays[0]
    assert oracle.lower[oracle.names.index(decay.coordinate)] == 0
    assert "g" not in oracle.names and "tauB" not in oracle.names
    assert "b" in oracle.names and oracle.lower[oracle.names.index("b")] == 0
    assert decay.coordinate not in oracle.sensitive_names
    parameters = continuation.to_effective(original, oracle, source.start)
    point = oracle.vector(parameters | {decay.coordinate: rate})
    restored = continuation.to_original(original, oracle, oracle.parameters(point))
    assert restored["tauA"] >= original.lower[original.names.index("tauA")]
    assert restored["tauB"] == 1
    assert restored["g"] >= 0
    assert 1 / restored["tauA"] + restored["g"] / restored["tauB"] == pytest.approx(
        rate
    )
    assert oracle.reduction.reduce(oracle.reduction.lift(point)) == pytest.approx(point)
    jacobian = oracle.reduction.lift_jacobian(point)
    step = rate * 1e-5
    direction = np.eye(len(point))[oracle.names.index(decay.coordinate)] * step
    numeric = (
        oracle.reduction.lift(point + direction)
        - oracle.reduction.lift(point - direction)
    ) / (2 * step)
    assert jacobian[:, oracle.names.index(decay.coordinate)] == pytest.approx(
        numeric, rel=1e-8, abs=1e-24
    )
    with pytest.raises(ValueError, match="strictly positive"):
        oracle.vector(parameters | {decay.coordinate: 0})


def test_private_decay_rollout_and_chain_rule_match_original_and_finite_difference():
    source = structural_case()
    retained = source.start | {"tauA": 2.0, "tauB": 4.0, "g": 3.0}
    original, oracle = wrapped(source, retained)
    point = oracle.reduction.reduce(original.vector(retained))
    actual, derivative = oracle.evaluate(point, monotonic() + 15)
    expected = original.evaluate(original.vector(retained), monotonic() + 15)[0]
    assert actual == pytest.approx(expected, abs=1e-9)
    for column in range(len(point)):
        if point[column] <= 1e-5:
            continue
        step = np.eye(len(point))[column] * 1e-5
        plus = oracle.evaluate(point + step, monotonic() + 15, jacobian=False)[0]
        minus = oracle.evaluate(point - step, monotonic() + 15, jacobian=False)[0]
        assert derivative[:, column] == pytest.approx((plus - minus) / 2e-5, abs=2e-7)


@pytest.mark.parametrize(
    "rhs,observation",
    [
        ("-x/tauA-g*q+tauA*z", "x"),
        ("-x/tauA-g*q", "x+tauA"),
        ("-x/tauA-g*q+g", "x"),
    ],
)
def test_extra_consumers_block_private_decay(rhs, observation):
    _, oracle = wrapped(structural_case(rhs=rhs, observation=observation))
    assert not oracle.reduction.decays


@pytest.mark.parametrize("expression", ["sqrt(g-1)", "1/(g-1)"])
def test_eager_dead_definition_can_block_private_decay(expression):
    source = structural_case(extra=[{"name": "unused", "expression": "g"}])
    reduction = inspected_definition(source, expression, source.start | {"g": 2.0})
    assert not reduction.decays
    assert any(
        r["reason"] == "retained_definition_domain_not_certified"
        for r in reduction.proof["retained"]
    )


@pytest.mark.parametrize("boundary", ["initializer", "constraint", "bounded"])
def test_initializer_constraints_and_explicit_bounds_fail_closed(boundary):
    source = structural_case()
    original, base = recovery.make_oracle(
        source.problem, training(source), NumericalPolicy(), source.start
    )
    candidate = original.system.model.validated.candidate
    if boundary == "initializer":
        initial = candidate.initial_conditions[1].model_copy(
            update={"expression": "tauA"}
        )
        candidate = candidate.model_copy(
            update={"initial_conditions": (candidate.initial_conditions[0], initial)}
        )
    elif boundary == "constraint":
        candidate = candidate.model_copy(
            update={"constraints": (ConstraintSpec(subject="g", kind="nonnegative"),)}
        )
    else:
        parameters = tuple(
            p.model_copy(update={"bounds": ValueRange(lower=0.5, upper=3)})
            if p.name == "tauA"
            else p
            for p in candidate.parameters
        )
        candidate = candidate.model_copy(update={"parameters": parameters})
    inspected = inspect_candidate(original, candidate)
    reduction = certify_structural_reduction(
        inspected, base.reduction, original.vector(source.start)
    )
    assert not reduction.decays


def test_time_constant_shared_with_another_state_preserves_the_tie():
    source = structural_case()
    payload = source.problem.model_dump(mode="json")
    payload["base_candidate"]["state_equations"][1]["rhs"] = "-z/tauA"
    shared = TrainingOracle(
        PrefixFitProblem.model_validate(payload), training(source), NumericalPolicy()
    )
    _, oracle = wrapped(shared)
    assert not oracle.reduction.decays
    assert {"tauA", "tauB", "g"} <= set(oracle.names)


def test_invalid_original_anchor_remains_a_domain_error():
    source = structural_case()
    with pytest.raises(ValueError) as error:
        wrapped(source, source.start | {"tauA": 0})
    assert not isinstance(error.value, NumericalInvalidTrial)


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_nonpositive_effective_rate_remains_a_domain_error(rate):
    source = structural_case()
    original, oracle = wrapped(source)
    effective = continuation.to_effective(original, oracle, source.start)
    name = oracle.reduction.decays[0].coordinate
    with pytest.raises(ValueError) as error:
        oracle.vector(effective | {name: rate})
    assert not isinstance(error.value, NumericalInvalidTrial)


@pytest.mark.parametrize(
    "rate,detail",
    [(np.nextafter(0.0, 1.0), "lift$"), (1e-200, "lift derivative$")],
)
def test_positive_unrepresentable_lift_is_a_hashable_rejected_trial(rate, detail):
    source = structural_case()
    original, oracle = wrapped(source)
    effective = continuation.to_effective(original, oracle, source.start)
    name = oracle.reduction.decays[0].coordinate
    point = oracle.vector(effective | {name: rate})
    expected_hash = content_sha256(oracle.parameters(point))
    assert oracle.parameters(point)[name] == rate
    for operation in (oracle.reduction.lift, oracle.reduction.lift_jacobian):
        with pytest.raises(NumericalInvalidTrial, match=detail):
            operation(point)
    with pytest.raises(NumericalInvalidTrial, match=detail):
        oracle.evaluate(point, monotonic() + 10)
    # This is the same serialization used by the optimizer's rejected-trial
    # ledger. It must not execute the failed original-domain lift again.
    assert content_sha256(oracle.parameters(point)) == expected_hash


def test_optional_policy_does_not_change_historical_coordinate_chart():
    source = structural_case()
    _, old = recovery.make_oracle(
        source.problem, training(source), NumericalPolicy(), source.start
    )
    _, new = wrapped(source)
    assert "tauA" in old.names and "tauA" not in new.names
    assert "structural_reduction" not in recovery.policy_receipt()
    assert (
        recovery.policy_receipt(structural_reduction=True)["structural_reduction"][
            "protocol"
        ]
        == "prefix-certified-structural-reduction-1"
    )
