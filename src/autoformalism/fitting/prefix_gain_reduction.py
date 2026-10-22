"""Exact isolated gain quotients around an unchanged scoped training oracle.

Certification is syntactic and domain based, never based on small sensitivities.
Only an isolated product ``c = gain * amplitude`` is removed. The effective
coordinate retains the amplitude's name; the receipt records its new meaning.
The canonical lift fixes the gain to one and puts c in the original amplitude.
Shared-consumer charts, reciprocal gains and overlapping terms are not reduced.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import content_sha256

if TYPE_CHECKING:
    from autoformalism.fitting.prefix_initialization import TrainingOracle

PROTOCOL = "prefix-isolated-gain-reduction-1"


@dataclass(frozen=True)
class GainProduct:
    """One certified product and its single syntactic consumer."""

    amplitude: str
    gain: str
    consumer: str
    process: str | None = None


def _checked_vector(values, names, lower, upper) -> np.ndarray:
    """Reject incomplete, nonfinite and out-of-domain coordinate vectors."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(names),):
        raise ValueError("coordinate vector shape differs")
    if (
        not np.isfinite(vector).all()
        or np.any(vector < lower)
        or np.any(vector > upper)
    ):
        raise ValueError("coordinate vector outside declared domain")
    return vector


def _mapping_vector(values: Mapping[str, float], names: tuple[str, ...]) -> np.ndarray:
    """Require exactly the declared coordinate keys, including trajectory roots."""
    if set(values) != set(names):
        raise ValueError("incomplete optimizer parameter vector")
    return np.asarray([values[name] for name in names], dtype=float)


def _bound(value: float) -> float | str:
    """Encode unbounded domains in an ordinary finite JSON receipt."""
    return float(value) if np.isfinite(value) else ("-inf" if value < 0 else "+inf")


@dataclass(frozen=True)
class GainReduction:
    """A deterministic exact quotient, with a canonical original-model lift.

    ``reduce(lift(c)) == c``; the reverse reconstructs an equivalent original
    model parameterization, not the original nonidentifiable factorization.
    """

    original_names: tuple[str, ...]
    original_lower: tuple[float, ...]
    original_upper: tuple[float, ...]
    products: tuple[GainProduct, ...]
    retained: tuple[tuple[str, str], ...]
    source_sha256: str

    @property
    def names(self) -> tuple[str, ...]:
        """Keep original coordinate order, omitting only certified gain factors."""
        removed = {product.gain for product in self.products}
        return tuple(name for name in self.original_names if name not in removed)

    @property
    def lower(self) -> np.ndarray:
        """Return the exact effective lower bounds, certified closed under products."""
        return np.asarray(
            [self.original_lower[self.original_names.index(n)] for n in self.names]
        )

    @property
    def upper(self) -> np.ndarray:
        """Return the exact effective upper bounds."""
        return np.asarray(
            [self.original_upper[self.original_names.index(n)] for n in self.names]
        )

    @property
    def receipt(self) -> dict[str, Any]:
        """Return a detached, sealed symbolic/domain certificate and refusal audit."""
        record = {
            "protocol": PROTOCOL,
            "source_sha256": self.source_sha256,
            "original_names": list(self.original_names),
            "effective_names": list(self.names),
            "original_domains": {
                name: [_bound(low), _bound(high)]
                for name, low, high in zip(
                    self.original_names,
                    self.original_lower,
                    self.original_upper,
                    strict=True,
                )
            },
            "effective_domains": {
                name: [_bound(low), _bound(high)]
                for name, low, high in zip(
                    self.names, self.lower, self.upper, strict=True
                )
            },
            "products": [
                {
                    "effective_coordinate": product.amplitude,
                    "meaning": f"{product.amplitude} * {product.gain}",
                    "amplitude": product.amplitude,
                    "gain": product.gain,
                    "consumer": product.consumer,
                    "process": product.process,
                    "lift": {product.gain: 1.0, product.amplitude: "effective_value"},
                    "proof": "isolated_multiplicative_factors_with_closed_domain",
                }
                for product in self.products
            ],
            "retained": [
                {"parameter": name, "reason": reason} for name, reason in self.retained
            ],
            "original_model_unchanged": True,
            "initializer_scopes_unchanged": True,
            "shared_consumer_charts_supported": False,
            "original_factorization_identified": False,
        }
        return record | {"sha256": content_sha256(record)}

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        """Validate an effective-coordinate dictionary."""
        return _checked_vector(
            _mapping_vector(values, self.names), self.names, self.lower, self.upper
        )

    def reduce(self, original_vector: np.ndarray) -> np.ndarray:
        """Project a complete original endpoint without fitting or changing roots."""
        original = _checked_vector(
            original_vector,
            self.original_names,
            self.original_lower,
            self.original_upper,
        )
        values = dict(zip(self.original_names, original.tolist(), strict=True))
        for product in self.products:
            values[product.amplitude] *= values[product.gain]
        return self.vector({name: values[name] for name in self.names})

    def reduce_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Project an original-coordinate parameter dictionary."""
        vector = self.reduce(_mapping_vector(parameters, self.original_names))
        return dict(zip(self.names, vector.tolist(), strict=True))

    def lift(self, effective_vector: np.ndarray) -> np.ndarray:
        """Reconstruct an in-domain original parameter vector with every gain one."""
        vector = _checked_vector(effective_vector, self.names, self.lower, self.upper)
        values = dict(zip(self.names, vector.tolist(), strict=True))
        values.update({product.gain: 1.0 for product in self.products})
        return _checked_vector(
            _mapping_vector(values, self.original_names),
            self.original_names,
            self.original_lower,
            self.original_upper,
        )

    def lift_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Return original parameters, suitable for unchanged replay and reports."""
        vector = self.lift(self.vector(parameters))
        return dict(zip(self.original_names, vector.tolist(), strict=True))

    def lift_jacobian(self) -> np.ndarray:
        """Return the constant original-by-effective derivative of the lift."""
        result = np.zeros((len(self.original_names), len(self.names)))
        for column, name in enumerate(self.names):
            result[self.original_names.index(name), column] = 1.0
        return result


def _source_identity(oracle: TrainingOracle) -> str:
    """Bind the certificate to model, scoped graph, coordinate order and domains."""
    return content_sha256(
        {
            "problem": oracle.problem.model_dump(mode="json"),
            "compiled_candidate": oracle.system.model.validated.candidate.model_dump(
                mode="json"
            ),
            "names": list(oracle.names),
            "lower": [_bound(value) for value in oracle.lower],
            "upper": [_bound(value) for value in oracle.upper],
        }
    )


def _expressions(oracle: TrainingOracle) -> dict[str, ast.expr]:
    """Parse all executable definitions, including expanded scoped initializers."""
    candidate = oracle.system.model.validated.candidate
    sources = {
        **{f"process:{p.name}": p.expression for p in candidate.processes},
        **{f"rhs:{e.state}": e.rhs for e in candidate.state_equations},
        **{
            f"observation:{o.channel}": o.expression
            for o in candidate.observation_mappings
        },
        **{
            f"initial:{i.state}": i.expression
            for i in candidate.initial_conditions
            if i.expression is not None
        },
        **{
            f"base_initial:{i.state}": i.expression
            for i in oracle.problem.base_candidate.initial_conditions
            if i.expression is not None
        },
    }
    parser = RestrictedParser()
    return {
        location: parser.parse(expression, location=location).tree.body
        for location, expression in sources.items()
    }


def _factors(node: ast.expr) -> tuple[str, ...]:
    """Find bare factors connected only by multiplication and unary signs."""
    if isinstance(node, ast.UnaryOp):
        return _factors(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _factors(node.left) + _factors(node.right)
    return (node.id,) if isinstance(node, ast.Name) else ()


def _closed_domain(amplitude: tuple[float, float], gain: tuple[float, float]) -> bool:
    """Prove product-domain equality using closure and the gain-one section."""
    low, high = amplitude
    gain_low, gain_high = gain
    if not (0 <= gain_low <= 1 <= gain_high):
        return False
    if low == -np.inf and high == np.inf:
        return True
    if low == 0 and high == np.inf:
        return True
    # Any interval containing zero is closed under nonnegative contractions.
    return low <= 0 <= high and gain_high <= 1


def certify_gain_reduction(oracle: TrainingOracle) -> GainReduction:
    """Certify disjoint single-consumer products; retain every unsupported block."""
    trees = _expressions(oracle)
    candidate = oracle.system.model.validated.candidate
    base_names = {p.name for p in oracle.problem.base_candidate.parameters}
    shared = set(oracle.shared) & base_names
    domains = {
        name: (float(low), float(high))
        for name, low, high in zip(
            oracle.names, oracle.lower, oracle.upper, strict=True
        )
    }
    occurrences: dict[str, list[str]] = {}
    for location, root in trees.items():
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                occurrences.setdefault(node.id, []).append(location)
    constrained = {
        c.subject
        for c in (*candidate.constraints, *oracle.problem.base_candidate.constraints)
    }
    process_names = {p.name for p in candidate.processes}
    products: list[GainProduct] = []
    retained: list[tuple[str, str]] = []
    used: set[str] = set()
    for gain in oracle.names:
        if gain not in shared or domains[gain][0] < 0 or gain in used:
            continue
        reason = "no_isolated_multiplicative_amplitude"
        locations = occurrences.get(gain, [])
        if gain in constrained:
            reason = "gain_has_explicit_constraint"
        elif not (domains[gain][0] <= 1 <= domains[gain][1]):
            reason = "gain_domain_excludes_one"
        elif len(locations) != 1:
            reason = "gain_used_elsewhere_or_unused"
        elif not locations[0].startswith(("rhs:", "process:")):
            reason = "gain_used_in_observation_or_initializer"
        else:
            consumer = locations[0]
            factor_groups = [
                _factors(node)
                for node in ast.walk(trees[consumer])
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
            ]
            peers = set().union(
                *(set(group) for group in factor_groups if gain in group)
            )
            options: list[tuple[str, str | None]] = [
                (name, None) for name in oracle.names if name in peers & shared - {gain}
            ]
            for process in sorted(peers & process_names):
                if len(occurrences.get(process, [])) != 1:
                    reason = "process_has_other_consumers_or_boundary_uses"
                    continue
                if process in constrained:
                    reason = "process_has_explicit_constraint"
                    continue
                factors = set(_factors(trees[f"process:{process}"]))
                options.extend(
                    (name, process) for name in oracle.names if name in factors & shared
                )
            for amplitude, process in options:
                if amplitude in used or amplitude == gain:
                    reason = "overlapping_gain_block"
                    continue
                if amplitude in constrained:
                    reason = "amplitude_has_explicit_constraint"
                    continue
                expected = f"process:{process}" if process is not None else consumer
                if occurrences.get(amplitude) != [expected]:
                    reason = "amplitude_used_elsewhere"
                    continue
                if not _closed_domain(domains[amplitude], domains[gain]):
                    reason = "product_domain_has_no_certified_gain_one_lift"
                    continue
                products.append(GainProduct(amplitude, gain, consumer, process))
                used.update((amplitude, gain))
                break
            else:
                retained.append((gain, reason))
            continue
        retained.append((gain, reason))
    return GainReduction(
        tuple(oracle.names),
        tuple(float(value) for value in oracle.lower),
        tuple(float(value) for value in oracle.upper),
        tuple(products),
        tuple((name, reason) for name, reason in retained if name not in used),
        _source_identity(oracle),
    )


class ReducedTrainingOracle:
    """Search adapter using exact lifts and the original full residual evaluator."""

    def __init__(
        self,
        oracle: TrainingOracle,
        reduction: GainReduction | None = None,
        *,
        original_start: Mapping[str, float] | None = None,
    ) -> None:
        self.original = oracle
        certified = certify_gain_reduction(oracle)
        if reduction is not None and reduction != certified:
            raise ValueError(
                "gain reduction certificate differs or belongs to another oracle"
            )
        self.reduction = certified
        self.names = self.reduction.names
        self.lower, self.upper = self.reduction.lower, self.reduction.upper
        self.start = self.reduction.reduce_parameters(
            oracle.start if original_start is None else original_start
        )
        self.units = np.asarray(
            [oracle.units[oracle.names.index(name)] for name in self.names], float
        )
        for product in self.reduction.products:
            self.units[self.names.index(product.amplitude)] = max(
                abs(self.start[product.amplitude]), 1.0
            )
        self._lift_jacobian = self.reduction.lift_jacobian()

    def vector(self, parameters: Mapping[str, float]) -> np.ndarray:
        """Return validated effective coordinates, never original factors."""
        return self.reduction.vector(parameters)

    def parameters(self, vector: np.ndarray) -> dict[str, float]:
        """Return effective parameters for optimizer checkpoints."""
        values = _checked_vector(vector, self.names, self.lower, self.upper)
        return dict(zip(self.names, values.tolist(), strict=True))

    def lift_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Reconstruct the original shared and per-trajectory coordinates."""
        return self.reduction.lift_parameters(parameters)

    def evaluate(
        self, vector: np.ndarray, deadline: float, *, jacobian: bool = True, method=None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate the original model, then apply the exact lift chain rule."""
        residual, derivative = self.original.evaluate(
            self.reduction.lift(vector), deadline, jacobian=jacobian, method=method
        )
        return residual, derivative @ self._lift_jacobian if jacobian else None
