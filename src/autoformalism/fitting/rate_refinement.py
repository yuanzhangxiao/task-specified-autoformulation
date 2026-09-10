"""Opt-in direct rate equations for the frozen collocation handoff diagnostic."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

import numpy as np

from autoformalism.expressions import CompiledModel, compile_candidate
from autoformalism.fitting.casadi_initializer import _DIVISION_EPSILON
from autoformalism.schemas import CandidateModel


@dataclass
class RateCoordinates:
    """Rewrite only declared, simple time-constant divisors before differentiation.

    Rates remain authoritative throughout fitting and replay. The positive floor
    represents the largest finite physical time constant; it is not a scientific
    bound. No chain rule through a reciprocal is used by the rate solver.
    """

    physical: CompiledModel

    def __post_init__(self) -> None:
        candidate = self.physical.validated.candidate
        self.aliases = {
            p.name: f"rate_{p.name}"
            for p in candidate.parameters
            if p.role.value == "time_constant"
        }
        if not self.aliases:
            raise ValueError("rate diagnostic requires declared time constants")
        symbols = {
            *self.physical.parameter_names,
            *self.physical.state_names,
            *self.physical.validated.process_order,
            *self.physical.validated.forcing_symbols,
            *self.physical.validated.context.targets,
            self.physical.validated.context.time_symbol,
        }
        if symbols.intersection(self.aliases.values()):
            raise ValueError("rate symbol collision")
        if candidate.constraints or any(
            p.scope.value != "global" for p in candidate.parameters
        ):
            raise ValueError("rate diagnostic requires unconstrained global equations")
        payload = candidate.model_dump(mode="json")
        for field, key, expressions, index in (
            (
                "state_equations",
                "rhs",
                self.physical.validated.equation_expressions,
                "state",
            ),
            (
                "processes",
                "expression",
                self.physical.validated.process_expressions,
                "name",
            ),
            (
                "observation_mappings",
                "expression",
                self.physical.validated.observation_expressions,
                "channel",
            ),
        ):
            for row in payload[field]:
                tree = expressions[row[index]].tree
                for parent in ast.walk(tree):
                    for child in ast.iter_child_nodes(parent):
                        if (
                            isinstance(child, ast.Name)
                            and child.id in self.aliases
                            and not (
                                isinstance(parent, ast.BinOp)
                                and isinstance(parent.op, ast.Div)
                                and parent.right is child
                            )
                        ):
                            raise ValueError(
                                "time constant outside a simple denominator"
                            )
                aliases = self.aliases

                class Rewrite(ast.NodeTransformer):
                    def visit_BinOp(self, node, aliases=aliases):
                        node = self.generic_visit(node)
                        if (
                            isinstance(node.op, ast.Div)
                            and isinstance(node.right, ast.Name)
                            and node.right.id in aliases
                        ):
                            return ast.BinOp(
                                node.left,
                                ast.Mult(),
                                ast.Name(aliases[node.right.id], ast.Load()),
                            )
                        return node

                row[key] = ast.unparse(
                    ast.fix_missing_locations(Rewrite().visit(copy.deepcopy(tree)))
                )
        for p in payload["parameters"]:
            if p["name"] in self.aliases:
                # Bounds are mapped from the original fitter layout, explicitly.
                # Nonnegative internal role avoids the generic positive-domain
                # epsilon floor; the mapped strictly positive floor is retained.
                p.update(
                    name=self.aliases[p["name"]],
                    role="nonnegative_coefficient",
                    domain="nonnegative",
                    bounds=None,
                    initialization_range=None,
                    unit="unspecified",
                    description="Internal inverse time constant",
                )
        self.model = compile_candidate(
            CandidateModel.model_validate(payload), self.physical.validated.context
        )
        self.names = self.model.parameter_names
        self.reverse = {v: k for k, v in self.aliases.items()}

    def start(self, parameters: dict[str, float]) -> dict[str, float]:
        """Map the saved physical point, including the existing division guard."""
        if set(parameters) != set(self.physical.parameter_names):
            raise ValueError("physical parameter keys differ")
        if not all(np.isfinite(v) for v in parameters.values()) or any(
            parameters[n] <= 0 for n in self.aliases
        ):
            raise ValueError("invalid physical parameters")
        return {
            self.aliases.get(n, n): 1 / max(v, _DIVISION_EPSILON)
            if n in self.aliases
            else v
            for n, v in parameters.items()
        }

    def bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert actual physical domains and reorder into explicit rate names."""
        mapped = {}
        floor = 1 / np.finfo(float).max
        for name, lo, hi in zip(
            self.physical.parameter_names, lower, upper, strict=True
        ):
            if name in self.aliases:
                if lo <= 0:
                    raise ValueError("time constant lower bound must be positive")
                lo, hi = (
                    max(floor, 1 / max(hi, _DIVISION_EPSILON)),
                    1 / max(lo, _DIVISION_EPSILON),
                )
            mapped[self.aliases.get(name, name)] = (lo, hi)
        low, high = np.array([mapped[n] for n in self.names]).T
        if np.any(low >= high):
            raise ValueError("rate domains must have nonzero width")
        return low, high

    def physical_view(self, parameters: dict[str, float], horizon: float) -> dict:
        """Label roundoff-scale decay without exporting enormous pseudo-estimates."""
        values, notes = {}, {}
        for name, value in parameters.items():
            original = self.reverse.get(name, name)
            if name not in self.reverse:
                values[original] = value
            elif value <= 0 or not np.isfinite(value):
                raise ValueError("rate must be positive and finite")
            elif value * horizon <= np.finfo(float).eps:
                values[original] = None
                notes[original] = {
                    "rate": value,
                    "rate_times_horizon": value * horizon,
                    "reason": (
                        "decay below float64 relative resolution over training horizon"
                    ),
                }
            else:
                values[original] = 1 / value
        return {
            "parameters": values,
            "unresolved_time_constants": notes,
            "informational_only": True,
            "training_horizon": horizon,
        }
