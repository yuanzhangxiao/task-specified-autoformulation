"""Certified positive-rate coordinates for an isolated linear meal reservoir.

The optional chart changes outer optimizer coordinates only. The original ODE
and the bounded least-squares initial-root profiler remain unchanged. Flux-state
initials are reported as B0=k*A0; they are not additional fitted variables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.fitting import prefix_structural_reduction as structural
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "positive-meal-flux-coordinates-1"


def _monomial(*names):
    return tuple(sorted((name, 1) for name in names))


def _mentions(polynomial, names):
    return any(set(dict(powers)) & set(names) for powers in polynomial)


def _nonnegative_unbounded(oracle, names):
    for name in names:
        if name not in oracle.names:
            raise ValueError(f"meal chart omitted coordinate: {name}")
        index = oracle.names.index(name)
        if oracle.lower[index] != 0 or oracle.upper[index] != np.inf:
            raise ValueError("meal chart requires nonnegative unbounded domains")


def _proof(oracle, state, rate, input_gain, input_name, output_state):
    original = oracle.original
    candidate = original.system.model.validated.candidate
    if candidate.constraints or original.problem.base_candidate.constraints:
        raise ValueError("meal chart does not support additional constraints")
    if len({state, rate, input_gain, input_name, output_state}) != 5:
        raise ValueError("meal chart symbols must be distinct")
    if (
        input_name not in original.system.inputs
        or input_name not in original.problem.context.external_inputs
    ):
        raise ValueError("meal chart requires a declared external forcing input")
    if state not in original.system.model.state_names:
        raise ValueError("meal chart requires a declared reservoir state")
    binding = original.audit["bindings"].get(state, {})
    root = original.audit["trajectory_root_parameters"].get(state)
    if (
        root not in original.roots
        or binding.get("mode") != "value"
        or binding.get("scope") != "trajectory"
        or binding.get("expanded_expression") != root
    ):
        raise ValueError("meal chart needs an independent trajectory initial root")
    root_names = tuple(
        f"trajectory:{row.trajectory_id}:{root}" for row in original.rows
    )
    _nonnegative_unbounded(original, (rate, input_gain, *root_names))
    _nonnegative_unbounded(oracle, (rate, input_gain, *root_names))
    reduction = oracle.reduction
    if isinstance(reduction, timescale.TimescaleReduction):
        if {rate, input_gain, *root_names} & set(reduction.renamed):
            raise ValueError("meal coordinates overlap a prior timescale chart")
        reduction = reduction.base
    lift = timescale._lift_polynomials(reduction)
    for name in (rate, input_gain, *root_names):
        if lift.get(name) != {_monomial(name): Fraction(1)}:
            raise ValueError("meal coordinate must have an identity prior lift")
    trees = gains._expressions(original)
    processes = {
        key.removeprefix("process:"): tree
        for key, tree in trees.items()
        if key.startswith("process:")
    }
    parameters = {p.name for p in candidate.parameters}
    reservoir = {
        _monomial(input_gain, input_name): Fraction(1),
        _monomial(rate, state): Fraction(-1),
    }
    consumers = []
    for location, tree in trees.items():
        if location.startswith("process:"):
            continue
        polynomial = structural._polynomial(tree, processes, parameters)
        if location == f"rhs:{state}":
            if polynomial != reservoir:
                raise ValueError("meal reservoir equation differs from q*r-k*A")
        elif location == f"rhs:{output_state}":
            if polynomial.get(_monomial(rate, state)) != 1:
                raise ValueError("meal output balance must contain exactly +k*A")
            remainder = dict(polynomial)
            remainder.pop(_monomial(rate, state))
            if _mentions(remainder, (state, rate, input_gain)):
                raise ValueError("meal output balance has overlapping uses")
            consumers.append(location)
        elif location == f"initial:{state}":
            if polynomial != {_monomial(root): Fraction(1)}:
                raise ValueError("meal initializer differs from the independent root")
        elif _mentions(polynomial, (state, rate, input_gain, root)):
            raise ValueError(f"meal coordinates have an additional use: {location}")
    if f"rhs:{state}" not in trees or consumers != [f"rhs:{output_state}"]:
        raise ValueError("meal chart lacks the required reservoir/output equations")
    return root_names


@dataclass(frozen=True)
class MealCoordinates:
    """Exact k>0 shared chart g=k*q, leaving original root profiling untouched."""

    oracle: object
    state: str
    rate: str
    input_gain: str
    input_name: str
    output_state: str
    root_names: tuple[str, ...]
    audit: dict

    @property
    def flux_gain(self):
        return f"meal_flux_gain:{self.state}"

    @property
    def names(self):
        return tuple(
            self.flux_gain if name == self.input_gain else name
            for name in self.oracle.names
        )

    @property
    def lower(self):
        return self.oracle.lower.copy()

    @property
    def upper(self):
        return self.oracle.upper.copy()

    def validate_use(self, oracle, fixed: Mapping[str, float]):
        """Bind the chart to its certified oracle and reject overlapping fixes."""
        if oracle is not self.oracle:
            raise ValueError("meal chart belongs to a different oracle")
        if (
            content_sha256(oracle.original.problem) != self.audit["problem_sha256"]
            or oracle.reduction.receipt["sha256"] != self.audit["reduction_sha256"]
        ):
            raise ValueError("meal chart source identity changed")
        if set(fixed) & {self.rate, self.input_gain, self.flux_gain}:
            raise ValueError("fixed coordinates overlap the meal coordinate chart")

    def encode(self, parameters: Mapping[str, float]):
        """Convert a complete feasible effective endpoint into k,g coordinates."""
        point = self.oracle.vector(parameters).copy()
        k, q = parameters[self.rate], parameters[self.input_gain]
        if k <= 0:
            raise NumericalInvalidTrial("positive meal chart excludes zero rate")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            g = float(k * q)
        if not np.isfinite(g) or (q > 0 and g == 0):
            raise NumericalInvalidTrial("nonrepresentable meal flux product")
        point[self.oracle.names.index(self.input_gain)] = g
        return point

    def decode(self, point):
        """Lift every trial, including a changed k with unchanged g, finitely."""
        point = gains._checked_vector(point, self.names, self.lower, self.upper).copy()
        k = point[self.oracle.names.index(self.rate)]
        g = point[self.oracle.names.index(self.input_gain)]
        if k <= 0:
            raise NumericalInvalidTrial("positive meal chart excludes zero rate")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            q = float(g / k)
        if not np.isfinite(q) or (g > 0 and q == 0):
            raise NumericalInvalidTrial("nonrepresentable inverse meal gain")
        point[self.oracle.names.index(self.input_gain)] = q
        values = self.oracle.parameters(point)
        continuation.to_original(self.oracle.original, self.oracle, values)
        return self.oracle.vector(values)

    def units(self, parameters):
        """Scale g by the transformed original gain unit at the frozen seed."""
        units = self.oracle.units.copy()
        index = self.oracle.names.index(self.input_gain)
        units[index] *= parameters[self.rate]
        if not np.isfinite(units).all() or units[index] <= 0:
            raise NumericalInvalidTrial("nonrepresentable meal coordinate units")
        return units

    def scaled_initials(self, parameters):
        """Report every B0=k*A0 while retaining finite original initials."""
        self.oracle.vector(parameters)
        k = parameters[self.rate]
        if k <= 0:
            raise NumericalInvalidTrial("positive meal chart excludes zero rate")
        result = {}
        for name in self.root_names:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                value = float(k * parameters[name])
            if not np.isfinite(value) or (parameters[name] > 0 and value == 0):
                raise NumericalInvalidTrial("nonrepresentable scaled meal initial")
            result[name] = value
        return result

    def zero_stratum(self, parameters):
        """Certify the original k=0 stratum without inventing a closure inverse."""
        self.oracle.vector(parameters)
        original = continuation.to_original(
            self.oracle.original, self.oracle, parameters
        )
        if parameters[self.rate] != 0:
            raise ValueError("zero-stratum control requires exactly zero rate")
        return {
            "rate": 0.0,
            "flux_gain": 0.0,
            "scaled_initials": dict.fromkeys(self.root_names, 0.0),
            "parameters": dict(parameters),
            "original_parameters": original,
            "inverse_unique": False,
            "nonzero_flux_at_zero_rate_permitted": False,
        }

    def reseed(self, parameters, new_rate: float):
        """Change the positive rate while preserving flux gain and every B0."""
        point = self.encode(parameters)
        scaled = self.scaled_initials(parameters)
        if not np.isfinite(new_rate) or new_rate <= 0:
            raise ValueError("meal reseed requires a finite strictly positive rate")
        point[self.oracle.names.index(self.rate)] = new_rate
        values = self.oracle.parameters(self.decode(point))
        for name, value in scaled.items():
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                values[name] = float(value / new_rate)
            if not np.isfinite(values[name]) or (value > 0 and values[name] == 0):
                raise NumericalInvalidTrial("nonrepresentable reseeded meal initial")
        self.oracle.vector(values)
        continuation.to_original(self.oracle.original, self.oracle, values)
        return values


def certify(
    oracle,
    parameters: Mapping[str, float],
    *,
    state: str = "A",
    rate: str = "k_abs",
    input_gain: str = "k_abs_rate",
    input_name: str = "meal_rate_g_per_min",
    output_state: str = "Gp",
) -> MealCoordinates:
    """Certify exact restricted algebra, exclusive uses and all original domains."""
    roots = _proof(oracle, state, rate, input_gain, input_name, output_state)
    record = {
        "protocol": PROTOCOL,
        "problem_sha256": content_sha256(oracle.original.problem),
        "reduction_sha256": oracle.reduction.receipt["sha256"],
        "state": state,
        "rate": rate,
        "input_gain": input_gain,
        "input": input_name,
        "output_state": output_state,
        "root_names": list(roots),
        "proof": "exact isolated q*r-k*A reservoir and +k*A output contribution",
        "coordinate": f"meal_flux_gain:{state}",
        "positive_branch": "k>0; g=k*q; B0_i=k*A0_i",
        "inverse": "q=g/k; A0_i=B0_i/k; all original values finite",
        "zero_branch": "original k=0 implies g=0 and every B0_i=0",
        "zero_branch_in_positive_search": False,
        "strict_positive_floor_added": False,
        "root_solver": "unchanged original-root bounded least squares",
        "state_equations_rewritten": False,
        "equation_family_changed": False,
        "parameter_confidence_certified": False,
    }
    chart = MealCoordinates(
        oracle,
        state,
        rate,
        input_gain,
        input_name,
        output_state,
        roots,
        record | {"sha256": content_sha256(record)},
    )
    chart.decode(chart.encode(parameters))
    chart.scaled_initials(parameters)
    return chart
