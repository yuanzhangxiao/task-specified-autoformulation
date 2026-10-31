"""Exact additive quotients preserve whole-model consumers and feasible lifts."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting.prefix_additive_reduction import certify_additive_reduction
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from tests.test_prefix_gain_reduction import inspect_candidate, oracle_for
from tests.test_prefix_structural_reduction import training


def source(*, sign="+", amplitude=None, b=None, observation="x", rhs=None, extra=()):
    return oracle_for(
        rhs=rhs or f"-x+g*q{sign}b*z",
        observation=observation,
        amplitude=amplitude,
        extra_parameters=[
            {"name": "b", "scope": "global", "role": "coefficient", **(b or {})}
        ],
        extra_processes=[
            *extra,
            *(
                [{"name": "unused_gain", "expression": "g"}]
                if rhs is not None and "g" not in rhs
                else []
            ),
        ],
    )


def wrapped(original, retained=None):
    return polish.make_oracle(
        original.problem,
        training(original),
        kernel.NumericalPolicy(),
        original.start if retained is None else retained,
    )


@pytest.mark.parametrize("sign,weight", [("+", 1), ("-", -1), ("-2*", -2)])
def test_real_pivot_preserves_product_chart_signed_sum_and_lift(sign, weight):
    src = source(sign=sign, amplitude={"role": "nonnegative_coefficient"})
    retained = src.start | {"a": 3.0, "g": 2.0, "b": -1.5}
    original, oracle = wrapped(src, retained)
    assert len(oracle.reduction.groups) == 1
    group = oracle.reduction.groups[0]
    assert group.pivot == "b"
    assert "g" not in oracle.names and "a" not in oracle.names
    assert oracle.start[group.coordinate] == pytest.approx(6 - 1.5 * weight)
    for value in (-100, 0, 200):
        point = oracle.vector(oracle.start | {group.coordinate: value})
        lifted = oracle.reduction.lift(point)
        params = original.parameters(lifted)
        assert params["a"] == 0 and params["g"] == 1
        assert params["a"] * params["g"] + weight * params["b"] == pytest.approx(value)
        assert oracle.reduction.reduce(lifted) == pytest.approx(point)
    actual, jac = oracle.evaluate(oracle.vector(oracle.start), monotonic() + 10)
    expected, _ = original.evaluate(original.vector(retained), monotonic() + 10)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert jac.shape == (len(actual), len(oracle.names))
    assert oracle.reduction.receipt["base_reduction"]["base_reduction"][
        "gain_reduction"
    ]["products"]


def test_exact_chain_rule_and_finite_difference_include_composite_outputs():
    src = source(rhs="-x+(a+b)*z", observation="x+(a+b)*z")
    original, oracle = wrapped(src, src.start | {"a": 2.0, "b": -0.5})
    group = oracle.reduction.groups[0]
    assert "observation:y" in group.consumers
    point = oracle.vector(oracle.start)
    residual, jac = oracle.evaluate(point, monotonic() + 10)
    expected, full = original.evaluate(oracle.reduction.lift(point), monotonic() + 10)
    assert residual == pytest.approx(expected, abs=1e-12)
    assert jac == pytest.approx(full @ oracle.reduction.lift_jacobian(point), abs=1e-12)
    for i in range(len(point)):
        step = np.eye(len(point))[i] * 1e-5
        plus = oracle.evaluate(point + step, monotonic() + 10, jacobian=False)[0]
        minus = oracle.evaluate(point - step, monotonic() + 10, jacobian=False)[0]
        assert jac[:, i] == pytest.approx((plus - minus) / 2e-5, abs=1e-8)


def test_initializer_sum_and_root_chain_rule_are_certified_together():
    src = source(rhs="-x+(a+b)*z", observation="x+z")
    payload = src.problem.model_dump(mode="json")
    payload["graph"]["rules"]["x"]["initial"] = {
        "mode": "map",
        "expression": "(phi+psi)*root",
        "parameters": [{"name": "phi", "guess": 0.5}, {"name": "psi", "guess": 0.8}],
        "state_initials": {"root": "z"},
    }
    model = PrefixFitProblem.model_validate(payload)
    base = kernel.TrainingOracle(model, training(src), kernel.NumericalPolicy())
    original, oracle = wrapped(base)
    groups = oracle.reduction.groups
    assert any("initial:x" in group.consumers for group in groups)
    point = oracle.vector(oracle.start)
    residual, jac = oracle.evaluate(point, monotonic() + 10)
    expected, full = original.evaluate(oracle.reduction.lift(point), monotonic() + 10)
    assert residual == pytest.approx(expected, abs=1e-12)
    assert jac == pytest.approx(full @ oracle.reduction.lift_jacobian(point), abs=1e-12)


@pytest.mark.parametrize("which", ["bounded", "nonnegative"])
def test_group_without_real_pivot_retains_original_feasible_domain(which):
    spec = (
        {"bounds": {"lower": -1, "upper": 2}}
        if which == "bounded"
        else {"role": "nonnegative_coefficient"}
    )
    _, oracle = wrapped(source(rhs="-x+(a+b)*z", amplitude=spec, b=spec))
    assert not oracle.reduction.groups
    assert any(
        r["reason"] == "no_unrestricted_real_pivot"
        for r in oracle.reduction.proof["retained"]
    )
    assert "a" in oracle.names and "b" in oracle.names


@pytest.mark.parametrize("observation", ["x+a*z", "x+(a-b)*z", "x+a*b*z"])
def test_unmatched_or_nonlinear_shared_readout_blocks_collapse(observation):
    _, oracle = wrapped(source(rhs="-x+(a+b)*z", observation=observation))
    assert not any(set(g.members) == {"a", "b"} for g in oracle.reduction.groups)


def test_nonzero_lower_bound_member_is_anchored_inside_declared_domain():
    src = source(rhs="-x+(a+b)*z", amplitude={"bounds": {"lower": 2, "upper": 5}})
    original, oracle = wrapped(src, src.start | {"a": 4})
    group = oracle.reduction.groups[0]
    assert dict(group.constants) == {"a": 4.0}
    point = oracle.vector(oracle.start | {group.coordinate: -200})
    lift = original.parameters(oracle.reduction.lift(point))
    assert lift["a"] == 4 and lift["b"] == -204
    assert oracle.reduction.reduce(oracle.reduction.lift(point)) == pytest.approx(point)


def test_unsafe_retained_definition_and_constraints_block_sum():
    src = source(rhs="-x+(a+b)*z", extra=[{"name": "unused", "expression": "a+1"}])
    original, oracle = wrapped(src)
    candidate = original.system.model.validated.candidate
    changed = candidate.model_copy(
        update={
            "processes": tuple(
                p.model_copy(update={"expression": "1/a"}) if p.name == "unused" else p
                for p in candidate.processes
            )
        }
    )
    reduction = certify_additive_reduction(
        inspect_candidate(original, changed),
        oracle.reduction.base,
        original.vector(src.start),
    )
    assert not reduction.groups
    assert (
        reduction.proof["retained"][0]["reason"]
        == "retained_definition_domain_not_certified"
    )
    from autoformalism.schemas.candidate import ConstraintSpec

    changed = candidate.model_copy(
        update={"constraints": (ConstraintSpec(kind="nonnegative", subject="a"),)}
    )
    reduction = certify_additive_reduction(
        inspect_candidate(original, changed),
        oracle.reduction.base,
        original.vector(src.start),
    )
    assert not reduction.groups
    assert reduction.proof["retained"][0]["reason"] == "explicit_constraints"


def test_receipt_and_input_are_immutable_and_nonfinite_coordinates_fail():
    src = source()
    before = deepcopy(src.problem.model_dump(mode="json"))
    original, oracle = wrapped(src)
    receipt = oracle.reduction.receipt
    receipt["groups"][0]["constants"].clear()
    assert oracle.reduction.receipt["groups"][0]["constants"]
    assert src.problem.model_dump(mode="json") == before
    name = oracle.reduction.groups[0].coordinate
    with pytest.raises(ValueError, match="outside declared domain"):
        oracle.vector(oracle.start | {name: np.nan})
    lifted = continuation.to_original(original, oracle, oracle.start)
    assert continuation.to_effective(original, oracle, lifted) == pytest.approx(
        oracle.start
    )
