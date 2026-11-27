"""Choose ODE construction computations under a finite joint probability model.

Each scenario specifies what every check and fit would reveal, and the terminal
loss of returning each fitted family. A fitted family's numerical parameters are
outside the planner. The caller defines a fit outcome from observable training
evidence, including any qualifications required before return. Only a check
with a verified family-wide contradiction should emit the label "reject".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Literal


@dataclass(frozen=True)
class Action:
    """A check or a single registered fitting procedure with an integer cost."""

    name: str
    kind: Literal["check", "fit"]
    family: str
    cost: int


@dataclass(frozen=True)
class Scenario:
    """One joint possibility, including risk conditional on its latent outcome."""

    weight: float
    outcomes: Mapping[str, str]
    risks: Mapping[str, float]


@dataclass(frozen=True)
class Decision:
    """The optimal next action and the best family available if stopping now."""

    action: str | None
    model: str | None
    value: float
    stop_value: float
    action_values: dict[str, float]
    clairvoyant_lower_bound: float
    scenarios_remaining: int


def _amount(value: float, name: str) -> Fraction:
    """Use the supplied decimal numbers exactly for comparing close decisions."""
    try:
        amount = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number") from error
    if amount < 0:
        raise ValueError(f"{name} must be nonnegative")
    return amount


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


class DecisionProblem:
    """Exactly solve small finite problems or impose a planning depth horizon.

    The terminal option is abstention or any fitted, unrejected family. Check
    outcome "reject" certifies a family contradiction. Fit outcome "failed"
    indicates an unusable fit, without rejecting any other family. Other labels
    can distinguish observations, e.g. predetermined selection-loss bins.
    Previously unobserved mandatory qualifications must be included in the fit
    outcome when a model can be returned immediately after fitting it.
    """

    def __init__(
        self,
        actions: Sequence[Action],
        scenarios: Sequence[Scenario],
        *,
        abstention_loss: float,
        cost_per_unit: float,
        budget: int,
    ) -> None:
        _integer(budget, 0, "budget")
        self.budget = budget
        self.actions = tuple(sorted(actions, key=lambda a: a.name))
        names = {a.name for a in self.actions}
        families = {a.family for a in self.actions}
        fits = [a.family for a in self.actions if a.kind == "fit"]
        if len(names) != len(actions) or any(
            not a.name or not a.family or a.kind not in {"check", "fit"}
            for a in self.actions
        ):
            raise ValueError("Actions need unique names, families, and valid kinds")
        if set(fits) != families or len(fits) != len(families):
            raise ValueError("Supply exactly one fit action per family")
        for a in self.actions:
            _integer(a.cost, 1, "action cost")
        if not scenarios:
            raise ValueError("Supply at least one positive-weight scenario")
        self._abstain = _amount(abstention_loss, "abstention_loss")
        self._price = _amount(cost_per_unit, "cost_per_unit")
        self._weights = tuple(_amount(s.weight, "weight") for s in scenarios)
        if not all(self._weights):
            raise ValueError("Scenario weights must be positive")
        for s in scenarios:
            if set(s.outcomes) != names or set(s.risks) != families:
                raise ValueError("Match scenario outcomes and risks to the action menu")
            if any(not isinstance(out, str) or not out for out in s.outcomes.values()):
                raise ValueError("Evidence labels must be nonempty strings")
        self._outcomes = tuple(
            tuple(s.outcomes[a.name] for s in scenarios) for a in self.actions
        )
        self._risk_mass = {
            family: tuple(
                w * _amount(s.risks[family], "terminal risk")
                for w, s in zip(self._weights, scenarios, strict=True)
            )
            for family in families
        }
        self._index = {a.name: i for i, a in enumerate(self.actions)}

    def _options(self, done: int, witness: int) -> tuple[set[str], set[str]]:
        """Observed evidence is identical for all surviving scenarios."""
        eligible, rejected = set(), set()
        for i, a in enumerate(self.actions):
            if done & (1 << i):
                outcome = self._outcomes[i][witness]
                if a.kind == "check" and outcome == "reject":
                    rejected.add(a.family)
                if a.kind == "fit" and outcome != "failed":
                    eligible.add(a.family)
        return eligible - rejected, rejected

    def decide(
        self, history: Mapping[str, str], *, depth: int | None = None
    ) -> Decision:
        """Condition on paid observations. depth=None explores all affordable steps.

        'value' and each action value include future computation cost only. At a
        depth limit the policy stops and uses the terminal risk. The resulting
        value is an upper bound for the unrestricted optimum. The lower bound
        assumes free knowledge of the scenario and zero future cost.
        """
        if depth is not None:
            _integer(depth, 0, "depth")
        if unknown := set(history) - self._index.keys():
            raise ValueError(f"Unknown actions in history: {sorted(unknown)}")
        observed = {self._index[n]: out for n, out in history.items()}
        worlds = tuple(
            w
            for w in range(len(self._weights))
            if all(self._outcomes[i][w] == out for i, out in observed.items())
        )
        if not worlds:
            raise ValueError("Observed history is outside the scenario support")
        done = sum(1 << i for i in observed)
        remaining = self.budget - sum(self.actions[i].cost for i in observed)
        if remaining < 0:
            raise ValueError("Observed actions exceed the computation budget")
        horizon = len(self.actions) - len(observed)
        if depth is not None:
            horizon = min(depth, horizon)
        first_values: dict[str, Fraction] = {}

        @cache
        def solve(
            ws: tuple[int, ...], used: int, left: int, steps: int
        ) -> tuple[Fraction, str | None, Fraction, str | None]:
            # Costs and risk are summed with the original weights. The sum
            # over disjoint child groups then integrates over observations.
            mass = sum(self._weights[w] for w in ws)
            eligible, rejected = self._options(used, ws[0])
            stop = mass * self._abstain
            model = None
            for family in sorted(eligible):
                risk = sum(self._risk_mass[family][w] for w in ws)
                if risk < stop or (risk == stop and model is None):
                    stop, model = risk, family
            best, first = stop, None
            if steps:
                for i, action in enumerate(self.actions):
                    if (
                        used & (1 << i)
                        or action.family in rejected
                        or action.cost > left
                    ):
                        continue
                    groups = defaultdict(list)
                    for w in ws:
                        groups[self._outcomes[i][w]].append(w)
                    value = mass * self._price * action.cost
                    for group in groups.values():
                        value += solve(
                            tuple(group), used | (1 << i), left - action.cost, steps - 1
                        )[0]
                    if used == done:
                        first_values[action.name] = value
                    if value < best:
                        best, first = value, action.name
            return best, first, stop, model

        value, first, stop, model = solve(worlds, done, remaining, horizon)
        solve.cache_clear()
        mass = sum(self._weights[w] for w in worlds)
        eligible, rejected = self._options(done, worlds[0])
        possible_fits = [
            (i, a)
            for i, a in enumerate(self.actions)
            if a.kind == "fit"
            and i not in observed
            and a.family not in rejected
            and a.cost <= remaining
        ]
        lower = Fraction(0)
        for w in worlds:
            returnable = eligible | {
                a.family for i, a in possible_fits if self._outcomes[i][w] != "failed"
            }
            lower += min(
                [self._weights[w] * self._abstain]
                + [self._risk_mass[family][w] for family in returnable]
            )
        return Decision(
            action=first,
            model=model,
            value=float(value / mass),
            stop_value=float(stop / mass),
            action_values={n: float(v / mass) for n, v in first_values.items()},
            clairvoyant_lower_bound=float(lower / mass),
            scenarios_remaining=len(worlds),
        )
