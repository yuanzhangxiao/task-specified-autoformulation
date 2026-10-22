"""Exact gain quotients preserve scoped roots, outputs, derivatives and domains."""

from copy import copy
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting.prefix_gain_reduction import (
    ReducedTrainingOracle,
    certify_gain_reduction,
)
from autoformalism.fitting.prefix_initialization import (
    NumericalPolicy,
    TrainingOracle,
)
from autoformalism.schemas.candidate import ConstraintSpec
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


def oracle_for(
    *,
    law="a*z",
    rhs="-x+g*q",
    observation="x",
    amplitude=None,
    gain=None,
    extra_parameters=(),
    extra_processes=(),
    constraints=(),
    root_scope="trajectory",
):
    """Build a small stable original model with independent or shared latent roots."""
    parameters = [
        {"name": "a", "scope": "global", "role": "coefficient", **(amplitude or {})},
        {
            "name": "g",
            "scope": "global",
            "role": "nonnegative_coefficient",
            **(gain or {}),
        },
        *extra_parameters,
    ]
    graph = {"z": {"initial": {"mode": "value", "scope": root_scope, "guess": 0.3}}}
    if observation != "x":
        graph["x"] = {
            "initial": {"mode": "known", "value": 1, "justification": "preparation"}
        }
    start = {
        parameter["name"]: min(
            max(1.0, parameter.get("bounds", {}).get("lower", -np.inf)),
            parameter.get("bounds", {}).get("upper", np.inf),
        )
        for parameter in parameters
    }
    problem = PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "gain_quotient_control",
                "parent_candidate_id": None,
                "states": [{"name": name, "kind": "latent"} for name in ("x", "z")],
                "parameters": parameters,
                "processes": [{"name": "q", "expression": law}, *extra_processes],
                "state_equations": [
                    {"state": "x", "rhs": rhs},
                    {"state": "z", "rhs": "-z"},
                ],
                "observation_mappings": [{"channel": "y", "expression": observation}],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "z")
                ],
                "constraints": list(constraints),
            },
            "context": {"targets": ["y"]},
            "graph": {"rules": graph},
            "parameter_guesses": start,
        }
    )
    time = np.linspace(0, 1, 7)
    training = PublicSplit(
        name="train",
        fingerprint="isolated-gain-control",
        rows=tuple(
            PublicTrajectory(
                trajectory_id=name,
                time=tuple(time),
                targets={"y": tuple(np.exp(-time) * (1 + root * time))},
            )
            for name, root in (("first", 0.3), ("second", 0.8))
        ),
    )
    return TrainingOracle(problem, training, NumericalPolicy())


@pytest.mark.parametrize("scope", ["trajectory", "shared_training"])
@pytest.mark.parametrize("a,g", [(-0.7, 2), (0, 0), (3.0, 0), (1.2, 0.4)])
def test_exact_lift_preserves_original_rollouts_and_root_scopes(scope, a, g):
    original = oracle_for(root_scope=scope)
    before = original.problem.model_dump_json()
    parameters = original.start | {"a": a, "g": g}
    quotient = ReducedTrainingOracle(original, original_start=parameters)
    assert quotient.names == tuple(name for name in original.names if name != "g")
    assert quotient.start["a"] == a * g
    assert quotient.units[quotient.names.index("a")] == max(abs(a * g), 1)
    point = quotient.vector(quotient.start)
    restored = quotient.lift_parameters(quotient.parameters(point))
    assert restored["g"] == 1
    assert restored["a"] == a * g
    assert all(restored[name] == parameters[name] for name in original.names[2:])
    r0 = original.evaluate(original.vector(parameters), monotonic() + 10)[0]
    r1, derivative = quotient.evaluate(point, monotonic() + 10)
    assert r1 == pytest.approx(r0, abs=1e-8)
    assert derivative.shape == (len(r1), len(point))
    assert quotient.reduction.reduce(quotient.reduction.lift(point)) == pytest.approx(
        point
    )
    assert original.problem.model_dump_json() == before
    receipt = quotient.reduction.receipt
    assert receipt["products"][0]["meaning"] == "a * g"
    assert receipt["initializer_scopes_unchanged"]
    assert receipt == certify_gain_reduction(original).receipt


def test_chain_rule_matches_original_jacobian_and_effective_finite_differences():
    original = oracle_for()
    quotient = ReducedTrainingOracle(original, original_start=original.start | {"g": 3})
    point = quotient.vector(quotient.start)
    _, original_jac = original.evaluate(
        quotient.reduction.lift(point), monotonic() + 10
    )
    residual, derivative = quotient.evaluate(point, monotonic() + 10, method="Radau")
    assert derivative == pytest.approx(
        original_jac @ quotient.reduction.lift_jacobian(), abs=2e-7
    )
    for column in range(len(point)):
        step = np.eye(len(point))[column] * 1e-5
        plus = quotient.evaluate(point + step, monotonic() + 10, jacobian=False)[0]
        minus, empty = quotient.evaluate(point - step, monotonic() + 10, jacobian=False)
        assert empty is None
        assert derivative[:, column] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)
    assert np.isfinite(residual).all()


def test_direct_product_inside_signed_rhs_is_certified():
    original = oracle_for(law="z", rhs="-x-g*a*q")
    quotient = ReducedTrainingOracle(original)
    assert len(quotient.reduction.products) == 1
    assert quotient.reduction.products[0].process is None
    assert quotient.start["a"] == 1


@pytest.mark.parametrize(
    "amplitude,gain,expected",
    [
        ({}, {}, True),
        ({"role": "nonnegative_coefficient"}, {}, True),
        ({"role": "rate"}, {}, False),
        ({"bounds": {"lower": -2, "upper": 3}}, {}, False),
        (
            {"bounds": {"lower": -2, "upper": 3}},
            {"bounds": {"lower": 0, "upper": 1}},
            True,
        ),
        (
            {"bounds": {"lower": 1, "upper": 3}},
            {"bounds": {"lower": 0, "upper": 1}},
            False,
        ),
        ({}, {"bounds": {"lower": 0, "upper": 0.5}}, False),
    ],
)
def test_domains_require_an_exact_gain_one_section(amplitude, gain, expected):
    original = oracle_for(amplitude=amplitude, gain=gain)
    reduction = certify_gain_reduction(original)
    assert bool(reduction.products) is expected
    if expected:
        for a in (reduction.lower[0], 0, reduction.upper[0]):
            if np.isfinite(a):
                original_vector = original.vector(original.start | {"a": a, "g": 0})
                effective = reduction.reduce(original_vector)
                original.vector(
                    reduction.lift_parameters(
                        dict(zip(reduction.names, effective, strict=True))
                    )
                )
    else:
        assert reduction.names == original.names
        assert reduction.retained


@pytest.mark.parametrize("observation", ["x+a", "x+g", "x+q"])
def test_observation_uses_block_reduction(observation):
    original = oracle_for(observation=observation)
    assert not certify_gain_reduction(original).products


@pytest.mark.parametrize("subject", ["a", "g", "q"])
def test_explicit_factor_or_law_constraints_block_reduction(subject):
    original = oracle_for()
    candidate = original.system.model.validated.candidate.model_copy(
        update={
            "constraints": (
                ConstraintSpec(subject=subject, kind="nonnegative", source="proposer"),
            )
        }
    )
    # Today's sensitivity backend rejects all constraints before execution. The
    # quotient's independent structural audit must remain conservative as well.
    assert not certify_gain_reduction(inspect_candidate(original, candidate)).products


def test_shared_amplitude_cannot_be_replaced_by_independent_signed_coefficients():
    original = oracle_for(
        rhs="-x+g*q-h*q",
        extra_parameters=[
            {"name": "h", "scope": "global", "role": "nonnegative_coefficient"}
        ],
    )
    reduction = certify_gain_reduction(original)
    assert not reduction.products
    assert all("consumers" in reason for _, reason in reduction.retained)
    assert reduction.names == original.names


def test_reused_amplitude_and_nonlinear_law_are_retained():
    for original in (oracle_for(rhs="-x+g*q+a*z"), oracle_for(law="exp(a*z)")):
        assert not certify_gain_reduction(original).products


@pytest.mark.parametrize("symbol", ["a", "g", "q"])
def test_expanded_initializer_occurrences_block_reduction(symbol):
    original = oracle_for()
    # Exercise the boundary audit independently of today's graph namespace rules;
    # a future compiler must not silently make cross-boundary gain removal safe.
    candidate = original.system.model.validated.candidate
    initial = candidate.initial_conditions[1].model_copy(update={"expression": symbol})
    candidate = candidate.model_copy(
        update={"initial_conditions": (candidate.initial_conditions[0], initial)}
    )
    assert not certify_gain_reduction(inspect_candidate(original, candidate)).products


def inspect_candidate(original, candidate):
    """Expose a candidate to structural certification without numerical execution."""
    inspected = copy(original)
    inspected.system = SimpleNamespace(
        model=SimpleNamespace(validated=SimpleNamespace(candidate=candidate))
    )
    return inspected


def test_vectors_and_cross_problem_certificate_fail_closed():
    original = oracle_for()
    quotient = ReducedTrainingOracle(original)
    for bad in (
        np.zeros(len(quotient.names) + 1),
        np.full(len(quotient.names), np.nan),
    ):
        with pytest.raises(ValueError):
            quotient.reduction.lift(bad)
    with pytest.raises(ValueError, match="incomplete"):
        quotient.vector({"a": 1})
    with pytest.raises(ValueError, match="outside"):
        quotient.reduction.reduce(original.vector(original.start) * [1, -1, 1, 1])
    with pytest.raises(ValueError, match="another oracle"):
        ReducedTrainingOracle(oracle_for(law="a*x"), quotient.reduction)
    with pytest.raises(ValueError, match="certificate differs"):
        ReducedTrainingOracle(original, replace(quotient.reduction, products=()))


def test_retained_parameters_and_empty_quotient_have_identity_chain_rule():
    original = oracle_for(law="z", rhs="-x+g*q+a")
    quotient = ReducedTrainingOracle(original)
    assert quotient.names == original.names
    assert quotient.reduction.lift_jacobian() == pytest.approx(
        np.eye(len(original.names))
    )
    assert quotient.lift_parameters(quotient.start) == original.start
