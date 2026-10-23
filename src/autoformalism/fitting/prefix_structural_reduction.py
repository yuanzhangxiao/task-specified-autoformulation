"""Opt-in exact inactive coordinates and private affine decay coefficients.

Certificates use restricted syntax and declared domains, never fitted derivative
sizes. Original definitions remain executable; every effective point is lifted
to the unchanged model, including anchored unused parameters.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "prefix-certified-structural-reduction-1"
MAX_TERMS = 1024


def policy_receipt() -> dict[str, Any]:
    """Describe the deliberately narrow optional certificate family."""
    return {
        "protocol": PROTOCOL,
        "inactive": "syntactically_unreachable_with_anchored_definition_domains",
        "private_decay": (
            "one_over_positive_tau_plus_nonnegative_gain_over_positive_tau"
        ),
        "effective_decay_domain": "(0,infinity)",
        "definitions_deleted": False,
        "constraint_policy": "retain_all_when_explicit_constraints_present",
        "unsupported_blocks": "retained",
        "floating_point_limit": "nonrepresentable_lifts_or_derivatives_fail_closed",
    }


def _symbols(node: ast.expr) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _dependencies(node, processes, seen=frozenset()):
    """Expand process dependencies transitively without evaluating expressions."""
    names = _symbols(node)
    for name in sorted(names & processes.keys()):
        if name in seen:
            raise ValueError("cyclic process definition in structural reduction")
        names |= _dependencies(processes[name], processes, seen | {name})
    return names


def _sum(left, right, sign=1):
    result = dict(left)
    for powers, coefficient in right.items():
        result[powers] = result.get(powers, Fraction(0)) + sign * coefficient
        if not result[powers]:
            del result[powers]
    if len(result) > MAX_TERMS:
        raise ValueError("structural polynomial term limit")
    return result


def _product(left, right):
    result = {}
    if len(left) * len(right) > MAX_TERMS:
        raise ValueError("structural polynomial product limit")
    for a, ca in left.items():
        for b, cb in right.items():
            powers = dict(a)
            for name, degree in b:
                powers[name] = powers.get(name, 0) + degree
            key = tuple(
                sorted((name, degree) for name, degree in powers.items() if degree)
            )
            result = _sum(result, {key: ca * cb})
    return result


def _polynomial(node, processes, parameters, seen=frozenset()):
    """Collect bounded Laurent monomials; division is parameter-only monomial."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = Fraction(str(node.value))
        return {(): value} if value else {}
    if isinstance(node, ast.Name):
        if node.id in processes:
            if node.id in seen:
                raise ValueError("cyclic process definition")
            return _polynomial(
                processes[node.id], processes, parameters, seen | {node.id}
            )
        return {((node.id, 1),): Fraction(1)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _polynomial(node.operand, processes, parameters, seen)
        return {
            key: (-coef if isinstance(node.op, ast.USub) else coef)
            for key, coef in value.items()
        }
    if isinstance(node, ast.BinOp):
        left = _polynomial(node.left, processes, parameters, seen)
        right = _polynomial(node.right, processes, parameters, seen)
        if isinstance(node.op, ast.Add):
            return _sum(left, right)
        if isinstance(node.op, ast.Sub):
            return _sum(left, right, -1)
        if isinstance(node.op, ast.Mult):
            return _product(left, right)
        if isinstance(node.op, ast.Div) and len(right) == 1:
            ((powers, coefficient),) = right.items()
            if coefficient and set(dict(powers)) <= parameters:
                return _product(
                    left, {tuple((n, -p) for n, p in powers): 1 / coefficient}
                )
    raise ValueError("unsupported structural coefficient grammar")


def _safe_definition(node, processes, positive, frozen, anchor, seen=frozenset()):
    """Certify retained dead definitions contain no unguarded partial functions."""
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float)
    if isinstance(node, ast.Name):
        if node.id not in processes:
            return True
        return node.id not in seen and _safe_definition(
            processes[node.id], processes, positive, frozen, anchor, seen | {node.id}
        )
    if isinstance(node, ast.UnaryOp):
        return _safe_definition(node.operand, processes, positive, frozen, anchor, seen)
    if not isinstance(node, ast.BinOp):
        return False
    if isinstance(node.op, ast.Div):
        denominator = node.right
        safe_denominator = (
            isinstance(denominator, ast.Constant) and denominator.value != 0
        ) or (
            isinstance(denominator, ast.Name)
            and (
                denominator.id in positive
                or (denominator.id in frozen and anchor[denominator.id] != 0)
            )
        )
        return safe_denominator and _safe_definition(
            node.left, processes, positive, frozen, anchor, seen
        )
    return isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)) and all(
        _safe_definition(child, processes, positive, frozen, anchor, seen)
        for child in (node.left, node.right)
    )


@dataclass(frozen=True)
class PrivateDecay:
    """One private coefficient and its exact smooth original-domain section."""

    coordinate: str
    state: str
    tau: str
    gain: str
    divisor: str
    tau_lower: float


class StructuralReduction:
    """Compose an existing exact chart with anchored and nonlinear sections."""

    def __init__(self, base, anchor, inactive, decays, proof):
        self.base = base
        self.anchor = base.reduce(anchor)
        self.inactive = tuple(inactive)
        self.decays = tuple(decays)
        self.proof = deepcopy(proof)
        removed = set(self.inactive) | {n for d in decays for n in (d.gain, d.divisor)}
        renamed = {d.tau: d.coordinate for d in decays}
        self.names = tuple(renamed.get(n, n) for n in base.names if n not in removed)
        self.lower = np.asarray(
            [
                0.0 if name in renamed.values() else base.lower[base.names.index(name)]
                for name in self.names
            ]
        )
        self.upper = np.asarray(
            [
                np.inf
                if name in renamed.values()
                else base.upper[base.names.index(name)]
                for name in self.names
            ]
        )

    @property
    def receipt(self):
        """Bind the proof, original anchors and the complete composed chart."""
        record = {
            **policy_receipt(),
            "base_reduction": self.base.receipt,
            "proof": deepcopy(self.proof),
            "effective_names": list(self.names),
            "anchored_coordinates": {
                n: float(self.anchor[self.base.names.index(n)]) for n in self.inactive
            },
            "private_decays": [
                asdict(d)
                | {
                    "meaning": f"1/{d.tau}+{d.gain}/{d.divisor}",
                    "domain": "(0,infinity)",
                    "lift": {
                        d.tau: "L+1/lambda",
                        d.divisor: "1",
                        d.gain: "lambda*(L*lambda/(1+L*lambda))",
                    },
                    "L": d.tau_lower,
                    "proof": "private_consumers_and_exact_domain_surjection",
                }
                for d in self.decays
            ],
            "lift_projection_identity": (
                "reduce(lift(effective)) == effective algebraically"
            ),
            "original_model_unchanged": True,
        }
        return record | {"sha256": content_sha256(record)}

    def vector(self, parameters: Mapping[str, float]) -> np.ndarray:
        """Reject zero/negative effective rates, including the open boundary."""
        vector = gains._checked_vector(
            gains._mapping_vector(parameters, self.names),
            self.names,
            self.lower,
            self.upper,
        )
        if any(vector[self.names.index(d.coordinate)] <= 0 for d in self.decays):
            raise ValueError("effective decay requires strictly positive rate")
        return vector

    def _checked(self, vector):
        values = gains._checked_vector(vector, self.names, self.lower, self.upper)
        return self.vector(dict(zip(self.names, values, strict=True)))

    def reduce(self, original_vector):
        values = dict(
            zip(self.base.names, self.base.reduce(original_vector), strict=True)
        )
        for decay in self.decays:
            values[decay.coordinate] = (
                1 / values[decay.tau] + values[decay.gain] / values[decay.divisor]
            )
        return self.vector({n: values[n] for n in self.names})

    def _section(self, vector):
        point = self._checked(vector)
        values = dict(zip(self.base.names, self.anchor, strict=True))
        values.update(
            {n: v for n, v in zip(self.names, point, strict=True) if n in values}
        )
        derivative = np.zeros((len(self.base.names), len(self.names)))
        for column, name in enumerate(self.names):
            if name in self.base.names:
                derivative[self.base.names.index(name), column] = 1
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for decay in self.decays:
                column = self.names.index(decay.coordinate)
                rate = point[column]
                reciprocal = 1 / rate
                t = decay.tau_lower * rate
                ratio = t / (1 + t)
                values[decay.tau] = decay.tau_lower + reciprocal
                values[decay.divisor] = 1.0
                values[decay.gain] = rate * ratio
                derivative[self.base.names.index(decay.tau), column] = (
                    -reciprocal * reciprocal
                )
                derivative[self.base.names.index(decay.gain), column] = ratio * (
                    2 - ratio
                )
        # The effective point has already passed the exact domain checks. These
        # failures are finite-precision execution failures, not invalid domains:
        # search may reject the trial while retaining its previous finite point.
        if not np.isfinite(tuple(values.values())).all():
            raise NumericalInvalidTrial("nonrepresentable structural lift")
        if not np.isfinite(derivative).all():
            raise NumericalInvalidTrial("nonrepresentable structural lift derivative")
        base_vector = self.base.vector(values)
        return base_vector, derivative

    def lift(self, vector):
        """Preserve every original coordinate and all retained process definitions."""
        section, _ = self._section(vector)
        return self.base.lift(section)

    def lift_jacobian(self, vector):
        """Apply the analytic nonlinear lift derivative to the existing chart."""
        _, derivative = self._section(vector)
        return self.base.lift_jacobian() @ derivative


def certify_structural_reduction(original, base, anchor) -> StructuralReduction:
    """Certify narrow generic blocks; unsupported/shared uses remain unchanged."""
    trees = gains._expressions(original)
    processes = {k[8:]: v for k, v in trees.items() if k.startswith("process:")}
    candidate = original.system.model.validated.candidate
    specs = {p.name: p for p in candidate.parameters}
    base_parameters = {p.name for p in original.problem.base_candidate.parameters}
    eligible = base_parameters & set(base.names)
    existing_products = getattr(getattr(base, "gain_reduction", None), "products", ())
    eligible -= {
        n for product in existing_products for n in (product.amplitude, product.gain)
    }
    anchored = dict(zip(original.names, anchor, strict=True))
    original.vector(anchored)
    roots = {k: v for k, v in trees.items() if not k.startswith("process:")}
    dependencies = {k: _dependencies(v, processes) for k, v in roots.items()}
    active = set().union(*dependencies.values())
    constrained = bool(
        candidate.constraints or original.problem.base_candidate.constraints
    )
    positive = {n for n, p in specs.items() if p.domain.value == "positive"}
    unused = eligible - active if not constrained else set()
    inactive = []
    refused = []
    for name in sorted(unused):
        affected = [
            v for v in processes.values() if name in _dependencies(v, processes)
        ]
        if affected and all(
            _safe_definition(v, processes, positive, unused, anchored) for v in affected
        ):
            inactive.append(name)
        else:
            refused.append(
                {"parameter": name, "reason": "unused_definition_domain_not_certified"}
            )
    decays = []
    consumed = set(inactive)
    for location, tree in sorted(roots.items()):
        if constrained or not location.startswith("rhs:"):
            continue
        state = location[4:]
        try:
            polynomial = _polynomial(tree, processes, set(specs))
        except ValueError:
            continue
        singles, pairs = [], []
        for monomial, coefficient in polynomial.items():
            powers = dict(monomial)
            if coefficient != -1 or powers.pop(state, None) != 1:
                continue
            if len(powers) == 1:
                tau, degree = next(iter(powers.items()))
                if degree == -1:
                    singles.append((tau, monomial))
            elif len(powers) == 2:
                numerators = [n for n, p in powers.items() if p == 1]
                denominators = [n for n, p in powers.items() if p == -1]
                if len(numerators) == len(denominators) == 1:
                    pairs.append((numerators[0], denominators[0], monomial))
        for tau, first in singles:
            for gain, divisor, second in pairs:
                block = {tau, gain, divisor}
                if len(block) != 3 or not block <= eligible or block & consumed:
                    continue
                reason = None
                if any(specs[n].bounds is not None for n in block):
                    reason = "explicit_parameter_bounds"
                elif (
                    tau not in positive
                    or divisor not in positive
                    or specs[gain].domain.value != "nonnegative"
                ):
                    reason = "unsupported_parameter_domains"
                elif any(
                    block & names
                    for key, names in dependencies.items()
                    if key != location
                ):
                    reason = "shared_output_or_initializer_consumer"
                elif any(
                    block & _dependencies(definition, processes)
                    and not _safe_definition(
                        definition, processes, positive, set(), anchored
                    )
                    for definition in processes.values()
                ):
                    reason = "retained_definition_domain_not_certified"
                elif any(
                    block & set(dict(term))
                    for term in polynomial
                    if term not in (first, second)
                ):
                    reason = "additional_coefficient_consumer"
                elif any(
                    n not in base.names
                    or base.lower[base.names.index(n)]
                    != original.lower[original.names.index(n)]
                    for n in block
                ):
                    reason = "overlapping_existing_chart"
                else:
                    lows = {n: base.lower[base.names.index(n)] for n in block}
                    highs = {n: base.upper[base.names.index(n)] for n in block}
                    if (
                        lows[gain] != 0
                        or not (0 < lows[tau] <= 1 and 0 < lows[divisor] <= 1)
                        or any(v != np.inf for v in highs.values())
                    ):
                        reason = "unsupported_numerical_domains"
                if reason:
                    refused.append({"parameters": sorted(block), "reason": reason})
                    continue
                coordinate = f"effective_decay:{state}:{tau}"
                if coordinate in base.names:
                    raise ValueError("structural effective coordinate collision")
                decays.append(
                    PrivateDecay(
                        coordinate, state, tau, gain, divisor, float(lows[tau])
                    )
                )
                consumed |= block
    proof = {
        "source_sha256": gains._source_identity(original),
        "active_dependency_roots": sorted(roots),
        "inactive_parameters": inactive,
        "retained": refused,
        "explicit_constraints_preclude_reduction": constrained,
        "definition_domain_scope": (
            "retained arithmetic definitions, nonzero denominators; "
            "finite execution checked separately"
        ),
        "offset_coefficients": (
            "unchanged; e.g. 1+nonnegative_gain retains lower bound 1"
        ),
    }
    return StructuralReduction(base, anchor, inactive, decays, proof)


class StructuralTrainingOracle:
    """Wrap the existing recovery oracle with the optional certified chart."""

    def __init__(self, oracle, retained):
        self.base_oracle = oracle
        self.original = oracle.original
        self.precision_mode = oracle.precision_mode
        self.reduction = certify_structural_reduction(
            self.original, oracle.reduction, self.original.vector(retained)
        )
        self.names = self.reduction.names
        self.lower, self.upper = self.reduction.lower, self.reduction.upper
        self.start = self.parameters(
            self.reduction.reduce(self.original.vector(self.original.start))
        )
        effective = {d.coordinate for d in self.reduction.decays}
        self.units = np.asarray(
            [
                max(abs(self.start[n]), 1.0)
                if n in effective
                else oracle.units[oracle.names.index(n)]
                for n in self.names
            ]
        )
        self.sensitive_names = tuple(
            n for n in oracle.sensitive_names if n in self.names
        )

    @property
    def recovery_audit(self):
        return self.base_oracle.recovery_audit | {
            "coordinate_reduction": self.reduction.receipt,
            "structural_reduction_policy": policy_receipt(),
            "sensitive_names": list(self.sensitive_names),
            "optimizer_coordinates": len(self.names),
        }

    def vector(self, parameters):
        return self.reduction.vector(parameters)

    def parameters(self, vector):
        return dict(
            zip(self.names, self.reduction._checked(vector).tolist(), strict=True)
        )

    def local_coordinate_scales(self, vector):
        values = self.reduction._checked(vector)
        scales = np.maximum(abs(values), self.units)
        for i, name in enumerate(self.names):
            if name in self.sensitive_names and values[i] != 0:
                scales[i] = min(self.units[i], abs(values[i]))
        return scales

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        residual, derivative = self.original.evaluate(
            self.reduction.lift(vector), deadline, jacobian=jacobian, method=method
        )
        return residual, derivative @ self.reduction.lift_jacobian(
            vector
        ) if jacobian else None
