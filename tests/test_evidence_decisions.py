"""Behavior and bounds of budgeted ODE construction decisions."""

from dataclasses import replace
from fractions import Fraction
from itertools import product
from random import Random

import pytest

from autoformalism.search.evidence_decisions import Action, DecisionProblem, Scenario


def two_possible_families(budget=4, price=0.1):
    actions = [
        Action("check_A", "check", "A", 1),
        Action("fit_A", "fit", "A", 3),
        Action("fit_B", "fit", "B", 3),
    ]
    worlds = [
        Scenario(
            1, {"check_A": "clear", "fit_A": "ok", "fit_B": "failed"}, {"A": 0, "B": 10}
        ),
        Scenario(
            1,
            {"check_A": "reject", "fit_A": "failed", "fit_B": "ok"},
            {"A": 10, "B": 0},
        ),
    ]
    return DecisionProblem(
        actions, worlds, budget=budget, cost_per_unit=price, abstention_loss=10
    )


def test_complementary_information_can_make_check_the_optimal_first_action():
    problem = two_possible_families()
    choice = problem.decide({})
    assert choice.action == "check_A"
    assert choice.model is None and choice.stop_value == 10
    assert choice.action_values == {"check_A": 0.4, "fit_A": 5.3, "fit_B": 5.3}
    assert choice.clairvoyant_lower_bound == 0
    assert problem.decide({}, depth=1).action == "fit_A"
    assert problem.decide({}, depth=1).value == 5.3
    assert problem.decide({}, depth=2) == choice
    assert problem.decide({"check_A": "clear"}).action == "fit_A"
    rejected = problem.decide({"check_A": "reject"})
    assert rejected.action == "fit_B" and "fit_A" not in rejected.action_values
    assert rejected.scenarios_remaining == 1
    done = problem.decide({"check_A": "reject", "fit_B": "ok"})
    assert done.action is None and done.model == "B" and done.value == 0


def test_unaffordable_followup_costly_exploration_and_qualification():
    assert two_possible_families(budget=1).decide({}).action is None
    assert two_possible_families(price=100).decide({}).action is None
    problem = two_possible_families()
    assert problem.decide({"fit_A": "failed"}).model is None
    with pytest.raises(ValueError, match="budget"):
        problem.decide({"fit_A": "failed", "fit_B": "ok"})
    with pytest.raises(ValueError, match="scenario support"):
        problem.decide({"fit_A": "failed", "check_A": "clear"})


def test_correlated_check_updates_risk_of_a_previously_fitted_family():
    actions = [
        Action("diag", "check", "A", 1),
        Action("a", "fit", "A", 1),
        Action("b", "fit", "B", 1),
    ]
    worlds = [
        Scenario(1, {"diag": "reject", "a": "ok", "b": "ok"}, {"A": 12, "B": 3}),
        Scenario(3, {"diag": "clear", "a": "ok", "b": "ok"}, {"A": 1, "B": 10}),
    ]
    problem = DecisionProblem(
        actions, worlds, budget=3, cost_per_unit=0.1, abstention_loss=6
    )
    result = problem.decide({"a": "ok", "b": "ok"})
    assert result.model == "A" and result.stop_value == 3.75
    assert result.action == "diag" and result.value == 1.6
    removed = problem.decide({"a": "ok", "b": "ok", "diag": "reject"})
    assert removed.model == "B" and removed.value == 3


def test_ties_favor_stopping_then_a_qualified_family_then_lexical_ids():
    acts = [Action(n, "fit", n, 1) for n in ("b", "a")]
    world = Scenario(1, {"b": "ok", "a": "ok"}, {"b": 0.2, "a": 0.2})
    p = DecisionProblem(acts, [world], budget=2, cost_per_unit=0, abstention_loss=0.2)
    assert p.decide({}).action is None
    assert p.decide({"a": "ok", "b": "ok"}).model == "a"
    assert p.decide({"b": "ok"}).action is None


def test_exact_solver_matches_exhaustively_enumerated_two_step_policies():
    # Enumerate every possible initial action and second action conditional on
    # each possible first-fit outcome, explicitly scoring every joint scenario.
    rng = Random(109)
    names = ["a", "b", "c"]
    actions = [Action(n, "fit", n, 1) for n in names]
    worlds = [
        Scenario(
            rng.randint(1, 5),
            dict(zip(names, outcomes, strict=True)),
            {n: rng.randint(1, 13) for n in names},
        )
        for outcomes in product(("failed", "ok"), repeat=3)
    ]
    price, abstention = Fraction(1, 10), 8
    problem = DecisionProblem(
        actions,
        worlds,
        budget=2,
        cost_per_unit=float(price),
        abstention_loss=abstention,
    )

    def stop_risk(group, fitted):
        common = [n for n in fitted if group[0].outcomes[n] == "ok"]
        mass = sum(w.weight for w in group)
        return min(
            [abstention * mass]
            + [sum(w.weight * w.risks[n] for w in group) for n in common]
        )

    total = sum(w.weight for w in worlds)
    values = []
    for first in names:
        alternatives = [None, *[n for n in names if n != first]]
        for second_choices in product(alternatives, repeat=2):
            value = price * total
            for observed, second in zip(("failed", "ok"), second_choices, strict=True):
                group = [w for w in worlds if w.outcomes[first] == observed]
                if second is None:
                    value += stop_risk(group, [first])
                else:
                    value += price * sum(w.weight for w in group)
                    for second_result in ("failed", "ok"):
                        children = [
                            w for w in group if w.outcomes[second] == second_result
                        ]
                        value += stop_risk(children, [first, second])
            values.append((value / total, first))
    best_value, first = min(
        [*values, (Fraction(abstention), None)], key=lambda item: item[0]
    )
    decision = problem.decide({})
    assert decision.action == first and decision.value == float(best_value)
    assert decision.clairvoyant_lower_bound <= decision.value <= decision.stop_value


def test_invalid_evidence_and_costs():
    action = Action("fit", "fit", "A", 1)
    world = Scenario(1, {"fit": "ok"}, {"A": 2})

    def create(actions, worlds):
        return DecisionProblem(
            actions, worlds, budget=1, cost_per_unit=0.1, abstention_loss=3
        )

    for invalid in (
        [replace(action, cost=0)],
        [action, action],
        [replace(action, kind="check")],
    ):
        with pytest.raises(ValueError):
            create(invalid, [world])
    for invalid in (
        [replace(world, weight=0)],
        [replace(world, risks={"A": -1})],
        [replace(world, outcomes={})],
        [replace(world, risks={})],
    ):
        with pytest.raises(ValueError):
            create([action], invalid)
    problem = create([action], [world])
    with pytest.raises(ValueError, match="Unknown actions"):
        problem.decide({"other": "ok"})
    with pytest.raises(ValueError, match="depth"):
        problem.decide({}, depth=-1)
