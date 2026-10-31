"""Exact opt-in additive quotients composed after existing gain/unused charts.

Complete restricted-polynomial consumer signatures certify dependence on one
weighted sum. A group is removed only when an unrestricted real member supplies
a globally feasible constant-anchor section; no numerical smallness is a proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import prefix_structural_reduction as structural
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "prefix-certified-additive-reduction-1"


@dataclass(frozen=True)
class AdditiveGroup:
    """One weighted sum and its unrestricted-real canonical pivot."""

    coordinate: str
    members: tuple[str, ...]
    weights: tuple[Fraction, ...]
    pivot: str
    constants: tuple[tuple[str, float], ...]
    consumers: tuple[str, ...]


def _substitute(polynomial, constants):
    """Substitute only certified fixed coordinates using exact rational arithmetic."""
    result = {}
    for powers, coefficient in polynomial.items():
        remaining = []
        for name, degree in powers:
            if name in constants:
                coefficient *= Fraction(str(constants[name])) ** degree
            else:
                remaining.append((name, degree))
        result = structural._sum(result, {tuple(remaining): coefficient})
    return result


def _base_affine(base, anchor):
    """Recognize the existing constant sections without sampling a nonlinear map."""
    if not isinstance(base, structural.StructuralReduction) or base.decays:
        return None
    # StructuralReduction composes the existing gain-one and inactive sections.
    from autoformalism.fitting.prefix_construction_recovery import ActiveReduction

    if not isinstance(base.base, ActiveReduction) or not isinstance(
        base.base.gain_reduction, gains.GainReduction
    ):
        return None
    point = base.reduce(anchor)
    return base.lift(point), base.lift_jacobian(point)


class AdditiveReduction:
    """Preserve the full feasible model family with an exact original-model lift."""

    def __init__(self, base, anchor, groups, proof):
        self.base, self.groups = base, tuple(groups)
        self.anchor = base.reduce(anchor)
        self.proof = deepcopy(proof)
        removed = {n for group in groups for n in group.members if n != group.pivot}
        renamed = {group.pivot: group.coordinate for group in groups}
        self.names = tuple(renamed.get(n, n) for n in base.names if n not in removed)
        self.lower = np.asarray(
            [
                -np.inf if n in renamed.values() else base.lower[base.names.index(n)]
                for n in self.names
            ]
        )
        self.upper = np.asarray(
            [
                np.inf if n in renamed.values() else base.upper[base.names.index(n)]
                for n in self.names
            ]
        )

    @property
    def receipt(self):
        """Bind consumer identities, domains, constant sections and prior reductions."""
        record = {
            "protocol": PROTOCOL,
            "base_reduction": self.base.receipt,
            "proof": deepcopy(self.proof),
            "effective_names": list(self.names),
            "groups": [
                {
                    "coordinate": g.coordinate,
                    "members": list(g.members),
                    "weights": [str(w) for w in g.weights],
                    "pivot": g.pivot,
                    "constants": dict(g.constants),
                    "consumers": list(g.consumers),
                    "effective_domain": ["-inf", "+inf"],
                    "certificate": "complete_proportional_consumers_with_real_pivot",
                    "lift": "pivot=(effective-sum(weight*constant))/pivot_weight",
                }
                for g in self.groups
            ],
            "original_model_unchanged": True,
            "original_parameter_identifiability_certified": False,
            "scope": "exact model-family quotient; no numerical convergence claim",
        }
        return record | {"sha256": content_sha256(record)}

    def vector(self, parameters: Mapping[str, float]) -> np.ndarray:
        """Validate exactly the reduced domain and coordinate inventory."""
        return self._checked(gains._mapping_vector(parameters, self.names))

    def _checked(self, vector):
        point = gains._checked_vector(vector, self.names, self.lower, self.upper)
        if not self.groups:
            return self.base.vector(dict(zip(self.names, point, strict=True)))
        return point

    def reduce(self, original_vector):
        values = dict(
            zip(self.base.names, self.base.reduce(original_vector), strict=True)
        )
        for group in self.groups:
            # Summation order follows the original coordinate order, and the
            # canonical lift is independently replayed before permitting search.
            values[group.coordinate] = float(
                sum(
                    Fraction.from_float(float(values[n])) * w
                    for n, w in zip(group.members, group.weights, strict=True)
                )
            )
        return self.vector({n: values[n] for n in self.names})

    def _section(self, vector):
        point = self._checked(vector)
        values = dict(zip(self.base.names, self.anchor, strict=True))
        derivative = np.zeros((len(self.base.names), len(self.names)))
        for column, name in enumerate(self.names):
            if name in values:
                values[name] = point[column]
                derivative[self.base.names.index(name), column] = 1
        for group in self.groups:
            column = self.names.index(group.coordinate)
            weights = dict(zip(group.members, group.weights, strict=True))
            values.update(group.constants)
            offset = sum(
                Fraction.from_float(value) * weights[name]
                for name, value in group.constants
            )
            pivot = (Fraction.from_float(float(point[column])) - offset) / weights[
                group.pivot
            ]
            values[group.pivot] = float(pivot)
            derivative[self.base.names.index(group.pivot), column] = float(
                1 / weights[group.pivot]
            )
        if (
            not np.isfinite(list(values.values())).all()
            or not np.isfinite(derivative).all()
        ):
            raise NumericalInvalidTrial("nonrepresentable additive lift")
        return self.base.vector(values), derivative

    def lift(self, vector):
        """Lift through every prior exact chart, including anchored declarations."""
        section, _ = self._section(vector)
        return self.base.lift(section)

    def lift_jacobian(self, vector):
        """Compose the complete original-by-reduced chain rule."""
        section, derivative = self._section(vector)
        return self.base.lift_jacobian(section) @ derivative


def certify_additive_reduction(original, base, anchor) -> AdditiveReduction:
    """Certify complete exact coefficient signatures; retain unsupported groups."""
    trees = gains._expressions(original)
    candidate = original.system.model.validated.candidate
    proof = {
        "source_sha256": gains._source_identity(original),
        "retained": [],
        "constraint_policy": "explicit_constraints_preclude_reduction",
    }
    if candidate.constraints or original.problem.base_candidate.constraints:
        proof["retained"].append({"reason": "explicit_constraints"})
        return AdditiveReduction(base, anchor, (), proof)
    canonical = _base_affine(base, anchor)
    if canonical is None:
        proof["retained"].append({"reason": "unsupported_existing_nonlinear_chart"})
        return AdditiveReduction(base, anchor, (), proof)
    lifted, matrix = canonical
    constants = {
        n: float(lifted[i])
        for i, n in enumerate(original.names)
        if not np.any(matrix[i]) and not n.startswith("trajectory:")
    }
    eligible = []
    for name in base.names:
        if name not in original.shared:
            continue
        i, j = original.names.index(name), base.names.index(name)
        expected = np.zeros(len(base.names))
        expected[j] = 1
        if np.array_equal(matrix[i], expected) and np.count_nonzero(matrix[:, j]) == 1:
            eligible.append(name)
    processes = {
        key[8:]: tree for key, tree in trees.items() if key.startswith("process:")
    }
    roots = {key: tree for key, tree in trees.items() if not key.startswith("process:")}
    polynomials = {}
    blocked = set()
    for location, tree in roots.items():
        try:
            polynomials[location] = _substitute(
                structural._polynomial(
                    tree,
                    processes,
                    {parameter.name for parameter in candidate.parameters},
                ),
                constants,
            )
        except (ValueError, ZeroDivisionError):
            blocked |= structural._dependencies(tree, processes) & set(eligible)
    signatures = {}
    for name in eligible:
        if name in blocked:
            continue
        profile = {}
        invalid = False
        for location, polynomial in polynomials.items():
            for powers, coefficient in polynomial.items():
                exponent = dict(powers).get(name, 0)
                if not exponent:
                    continue
                if exponent != 1:
                    invalid = True
                    break
                remaining = tuple((n, degree) for n, degree in powers if n != name)
                profile[(location, remaining)] = coefficient
            if invalid:
                break
        if invalid or not profile:
            continue
        weight = profile[sorted(profile)[0]]
        signature = tuple(
            (key, value / weight) for key, value in sorted(profile.items())
        )
        signatures.setdefault(signature, []).append((name, weight))
    groups = []
    positive = {p.name for p in candidate.parameters if p.domain.value == "positive"}
    anchored = dict(zip(original.names, anchor, strict=True))
    for signature, members in signatures.items():
        if len(members) < 2:
            continue
        names = tuple(n for n, _ in members)
        pivot = next(
            (
                n
                for n in names
                if base.lower[base.names.index(n)] == -np.inf
                and base.upper[base.names.index(n)] == np.inf
            ),
            None,
        )
        reason = None
        if pivot is None:
            reason = "no_unrestricted_real_pivot"
        elif any(set(dict(powers)) & set(names) for (_, powers), _ in signature):
            reason = "nonlinear_group_consumer"
        elif any(
            set(names) & structural._dependencies(tree, processes)
            and not structural._safe_definition(
                tree, processes, positive, set(), anchored
            )
            for tree in processes.values()
        ):
            reason = "retained_definition_domain_not_certified"
        if reason:
            proof["retained"].append({"parameters": list(names), "reason": reason})
            continue
        fixed = tuple(
            (
                n,
                0.0
                if base.lower[base.names.index(n)]
                <= 0
                <= base.upper[base.names.index(n)]
                else float(base.reduce(anchor)[base.names.index(n)]),
            )
            for n in names
            if n != pivot
        )
        coordinate = f"effective_sum:{pivot}"
        if coordinate in base.names:
            raise ValueError("additive coordinate collision")
        groups.append(
            AdditiveGroup(
                coordinate,
                names,
                tuple(w for _, w in members),
                pivot,
                fixed,
                tuple(sorted({location for (location, _), _ in signature})),
            )
        )
    proof["consumer_roots"] = sorted(roots)
    proof["fixed_prior_coordinates"] = constants
    return AdditiveReduction(base, anchor, groups, proof)


class AdditiveTrainingOracle:
    """Evaluate only surviving local derivative columns with the exact full lift."""

    def __init__(self, oracle, retained):
        self.base_oracle, self.original = oracle, oracle.original
        self.reduction = certify_additive_reduction(
            self.original, oracle.reduction, self.original.vector(retained)
        )
        self.names = self.reduction.names
        self.lower, self.upper = self.reduction.lower, self.reduction.upper
        self.start = self.parameters(
            self.reduction.reduce(self.original.vector(retained))
        )
        self.units = np.maximum(np.abs(self.vector(self.start)), 1.0)
        self.sensitive_names = tuple(
            n for n in oracle.sensitive_names if n in self.names
        )
        self.precision_mode = oracle.precision_mode
        self._constant_lift = (
            _base_affine(oracle.reduction, self.original.vector(retained)) is not None
        )
        self._propagation = None
        if self._constant_lift:
            derivative = self.reduction.lift_jacobian(self.vector(self.start))
            self._columns = tuple(
                j
                for j in range(len(self.original.system.names))
                if any(
                    np.any(derivative[indices[j]])
                    for indices in self.original.indices.values()
                )
            )
            self._propagation = affine._Propagation(
                self.original.system, self._columns, self.original.precision_policy
            )

    def vector(self, parameters):
        return self.reduction.vector(parameters)

    def parameters(self, vector):
        return dict(
            zip(self.names, self.reduction._checked(vector).tolist(), strict=True)
        )

    def local_coordinate_scales(self, vector):
        values = self.reduction._checked(vector)
        scales = np.maximum(np.abs(values), self.units)
        for i, name in enumerate(self.names):
            if name in self.sensitive_names and values[i] != 0:
                scales[i] = min(self.units[i], abs(values[i]))
        return scales

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        """Preserve all readout and initializer derivatives in retained columns."""
        point = self.reduction._checked(vector)
        lifted = self.reduction.lift(point)
        if method is not None or self._propagation is None:
            residual, derivative = self.original.evaluate(
                lifted, deadline, jacobian=jacobian, method=method
            )
            return residual, derivative @ self.reduction.lift_jacobian(
                point
            ) if jacobian else None
        affine._check_deadline(deadline)
        scales = np.asarray(
            [self.original.scales[n] for n in self.original.system.channels]
        )
        chain = self.reduction.lift_jacobian(point) if jacobian else None
        residuals, derivatives, cache = [], [], {}
        for row in self.original.rows:
            values, derivative = self._propagation.rollout(
                row,
                self.original.local_vector(lifted, row),
                deadline,
                jacobian=jacobian,
                cache=cache,
            )
            observed = np.column_stack(
                [row.targets[n] for n in self.original.system.channels]
            )
            residuals.append(((values - observed) / scales).ravel())
            if jacobian:
                local = (derivative / scales[None, :, None]).reshape(
                    values.size, len(self._columns)
                )
                selected = self.original.indices[row.trajectory_id][list(self._columns)]
                derivatives.append(local @ chain[selected])
        affine._check_deadline(deadline)
        return affine._finite(np.concatenate(residuals), "additive residual"), (
            affine._finite(np.vstack(derivatives), "additive derivative")
            if jacobian
            else None
        )
