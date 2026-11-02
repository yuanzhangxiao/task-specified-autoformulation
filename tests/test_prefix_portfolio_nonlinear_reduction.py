"""Exact nonlinear reductions preserve domains, roots, values and derivatives."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_nonlinear_reduction as p
from tests.test_prefix_gain_reduction import oracle_for


def case(**kwargs):
    return oracle_for(
        law="a*z/(1+z*z)",
        rhs="-x-x*x+g*q+b*z+c*z",
        observation="x+z",
        extra_parameters=(
            {"name": "b", "scope": "global", "role": "coefficient"},
            {"name": "c", "scope": "global", "role": "coefficient"},
        ),
        **kwargs,
    )


def build(source, parameters=None):
    return p.make_oracle(source.problem, source.training, parameters or source.start)


@pytest.mark.parametrize("a,g", [(-0.7, 2), (0, 0), (1.3, 0), (1.2, 0.4)])
def test_product_and_sum_lift_preserve_full_family_and_zero_branch(a, g):
    source = case()
    parameters = source.start | {"a": a, "g": g, "b": 0.7, "c": -0.2}
    original, reduced = build(source, parameters)
    assert len(reduced.names) == len(original.names) - 2
    receipt = reduced.reduction.receipt
    assert len(receipt["base_reduction"]["numerator_products"]["products"]) == 1
    assert len(receipt["groups"]) == 1
    point = reduced.vector(reduced.start)
    lifted = reduced.reduction.lift(point)
    assert lifted[original.names.index("g")] == 1
    assert lifted[original.names.index("a")] == pytest.approx(a * g)
    assert reduced.reduction.reduce(lifted) == pytest.approx(point)
    r0 = original.evaluate(
        original.vector(parameters), monotonic() + 10, jacobian=False, method="Radau"
    )[0]
    r1 = reduced.evaluate(point, monotonic() + 10, jacobian=False)[0]
    assert r0 == pytest.approx(r1, abs=2e-7)
    assert receipt["proof"]["zero_product_branch_preserved"]


def test_reduced_local_sensitivity_matches_original_chain_and_finite_differences():
    original, reduced = build(case())
    point = reduced.vector(reduced.start)
    residual, derivative = reduced.evaluate(point, monotonic() + 10)
    lifted = reduced.reduction.lift(point)
    raw, raw_j = original.evaluate(lifted, monotonic() + 10, method="BDF")
    assert residual == pytest.approx(raw, abs=1e-7)
    assert derivative == pytest.approx(
        raw_j @ reduced.reduction.lift_jacobian(point), abs=2e-7
    )
    for j in range(len(point)):
        step = np.eye(len(point))[j] * 1e-5
        plus = reduced.evaluate(point + step, monotonic() + 10, jacobian=False)[0]
        minus = reduced.evaluate(point - step, monotonic() + 10, jacobian=False)[0]
        assert derivative[:, j] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)
    # Local AD systems have no irrelevant roots from another training trajectory.
    local_sizes = list(reduced.portfolio_audit["sensitivity_columns"].values())
    assert local_sizes == [len(reduced.names) - 1] * 2
    assert all(v < len(original.system.names) for v in local_sizes)


@pytest.mark.parametrize(
    "law,observation",
    [
        ("a*z/(1+a*z*z)", "x"),  # denominator consumer is not an outer amplitude
        ("a*z/(1+z*z)", "x+a"),  # extra output consumer forbids gain quotient
        ("a*z/(1+z*z)", "x+g"),
    ],
)
def test_nonexclusive_or_denominator_parameters_are_retained(law, observation):
    source = oracle_for(law=law, rhs="-x-x*x+g*q", observation=observation)
    original, reduced = build(source)
    assert "a" in reduced.names and "g" in reduced.names
    assert not reduced.reduction.receipt["base_reduction"]["numerator_products"][
        "products"
    ]
    assert len(reduced.names) == len(original.names)


def test_nonclosed_product_domains_are_not_changed():
    source = case(
        amplitude={"bounds": {"lower": -2, "upper": 2}},
        gain={"bounds": {"lower": 0, "upper": 3}},
    )
    original, reduced = build(source)
    assert "a" in reduced.names and "g" in reduced.names
    assert len(reduced.names) == len(original.names) - 1
    with pytest.raises(ValueError, match="domain"):
        reduced.vector(reduced.start | {"a": 4})


def test_finite_interval_additive_group_without_real_pivot_is_retained():
    source = case()
    payload = source.problem.model_dump(mode="json")
    for parameter in payload["base_candidate"]["parameters"]:
        if parameter["name"] in ("b", "c"):
            parameter["bounds"] = {"lower": -1, "upper": 2}
    problem = source.problem.model_validate(payload)
    original, reduced = p.make_oracle(problem, source.training, source.start)
    assert "b" in reduced.names and "c" in reduced.names
    assert not reduced.reduction.groups
    assert len(reduced.names) == len(original.names) - 1


def test_no_source_mutation_and_training_only_and_deadline():
    source = case()
    before = source.problem.model_dump_json(), source.training.model_dump_json()
    _, reduced = build(source)
    reduced.evaluate(reduced.vector(reduced.start), monotonic() + 10, jacobian=False)
    assert before == (
        source.problem.model_dump_json(),
        source.training.model_dump_json(),
    )
    bad = source.training.model_copy(update={"name": "validation"})
    with pytest.raises(ValueError, match="training only"):
        p.make_oracle(source.problem, bad, source.start)
    with pytest.raises(TimeoutError):
        reduced.evaluate(reduced.vector(reduced.start), monotonic() - 1)


def test_prior_chart_identity_and_algebraic_certificate_are_deterministic():
    source = case()
    original, prior = portfolio.make_oracle(
        source.problem, source.training, kernel.NumericalPolicy()
    )
    original_copy = deepcopy(original.problem.model_dump(mode="json"))
    first = p.certify(original, prior.reduction, original.vector(original.start))
    second = p.certify(original, prior.reduction, original.vector(original.start))
    assert first.receipt == second.receipt
    assert original.problem.model_dump(mode="json") == original_copy
    with pytest.raises(ValueError, match="constant base"):
        p.certify(original, object(), original.vector(original.start))


def test_three_way_equivalence_is_independent_and_accounted():
    source = case()
    original, reduced = build(source, source.start | {"g": 0, "a": 3, "b": 1, "c": 2})
    assert original.settings.integration_method == "BDF"
    saved = []
    value = p.chart_equivalence(original, reduced, 10, saved.append)
    assert value["passed"] and value["original_equations_verified"]
    assert value["verification_calls_started"] == 3
    assert value["verification_calls_completed"] == 3
    assert value["verification_calls_failed"] == 0
    assert not value["jacobian_qualification_claimed"]
    assert saved[-1]["verification_calls_completed"] == 3


def test_unused_definition_domain_still_blocks_reduction():
    source = oracle_for(
        law="a*z/(1+z*z)",
        rhs="-x-x*x+g*q",
        extra_processes=({"name": "unused", "expression": "z/a"},),
    )
    _, reduced = build(source)
    assert "a" in reduced.names and "g" in reduced.names


def test_matching_opaque_nonlinear_atoms_can_be_summed():
    source = case()
    payload = source.problem.model_dump(mode="json")
    payload["base_candidate"]["state_equations"][0]["rhs"] = (
        "-x-x*x+g*q+b*z/(1+z*z)-c*z/(1+z*z)"
    )
    problem = source.problem.model_validate(payload)
    original, reduced = p.make_oracle(problem, source.training, source.start)
    assert len(reduced.reduction.groups) == 1
    assert len(reduced.names) == len(original.names) - 3
    assert set(reduced.reduction.groups[0].members) == {"a", "b", "c"}
    verified = p.chart_equivalence(original, reduced, 10, lambda _: None)
    assert verified["passed"]
