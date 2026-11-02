"""Certified isolated nonlinear gains and additive terms, with reduced AD systems.

Only complete, exclusive syntactic consumers are quotiented. Arbitrary nonlinear
factors remain opaque and unchanged. The constant lift is composed into each
trajectory's CasADi system before differentiating, avoiding discarded sensitivity
columns. Source equations, initializer graphs and parameter domains are immutable.
"""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction

import numpy as np

from autoformalism.fitting import prefix_additive_reduction as additive
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_qualification as qualification
from autoformalism.fitting.sensitivity_probe import symbolic_rollout

PROTOCOL = "portfolio-nonlinear-consumer-reduction-1"


def _symbols(tree):
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _expand(tree, processes, constants, seen=frozenset()):
    """Expand restricted parsed processes; never execute proposal source."""
    if isinstance(tree, ast.Name):
        if tree.id in constants:
            return ast.Constant(constants[tree.id])
        if tree.id in processes:
            if tree.id in seen:
                raise ValueError("recursive process expansion")
            return _expand(processes[tree.id], processes, constants, seen | {tree.id})
    result = deepcopy(tree)
    for field, value in ast.iter_fields(result):
        if isinstance(value, ast.expr):
            setattr(result, field, _expand(value, processes, constants, seen))
        elif isinstance(value, list):
            setattr(
                result,
                field,
                [
                    _expand(v, processes, constants, seen)
                    if isinstance(v, ast.expr)
                    else v
                    for v in value
                ],
            )
    return result


def _terms(tree, sign=Fraction(1)):
    if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, (ast.UAdd, ast.USub)):
        return _terms(tree.operand, -sign if isinstance(tree.op, ast.USub) else sign)
    if isinstance(tree, ast.BinOp) and isinstance(tree.op, (ast.Add, ast.Sub)):
        return _terms(tree.left, sign) + _terms(
            tree.right, -sign if isinstance(tree.op, ast.Sub) else sign
        )
    return [(sign, tree)]


def _factors(tree):
    """Factor only products and numerator divisions, preserving denominator ASTs."""
    if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, (ast.UAdd, ast.USub)):
        weight, factors = _factors(tree.operand)
        return (-weight if isinstance(tree.op, ast.USub) else weight), factors
    if isinstance(tree, ast.Constant) and isinstance(tree.value, (int, float)):
        return Fraction(str(tree.value)), []
    if isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Mult):
        left, a = _factors(tree.left)
        right, b = _factors(tree.right)
        return left * right, a + b
    if isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Div):
        weight, factors = _factors(tree.left)
        return weight, [*factors, ast.BinOp(ast.Constant(1), ast.Div(), tree.right)]
    return Fraction(1), [tree]


def _consumers(original, base, anchor):
    """Resolve the preceding certified constant section, with explicit refusal."""
    canonical = additive._base_affine(base, anchor)
    if canonical is None:
        raise ValueError("nonlinear reduction requires a certified constant base lift")
    lifted, matrix = canonical
    constants = {
        name: float(lifted[i])
        for i, name in enumerate(original.names)
        if not np.any(matrix[i]) and not name.startswith("trajectory:")
    }
    trees = gains._expressions(original)
    processes = {k[8:]: v for k, v in trees.items() if k.startswith("process:")}
    roots = {
        k: _expand(v, processes, constants)
        for k, v in trees.items()
        if not k.startswith("process:")
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
    # Domain-changing positions remain forbidden even in unused definitions.
    unsafe = set()
    for tree in trees.values():
        expanded = _expand(tree, processes, constants)
        for node in ast.walk(expanded):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                unsafe |= _symbols(node.right)
            elif isinstance(node, ast.Call) or (
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
            ):
                unsafe |= _symbols(node)
    count = Counter(
        n.id
        for tree in roots.values()
        for n in ast.walk(tree)
        if isinstance(n, ast.Name)
    )
    eligible = {n for n in eligible if count[n] == 1 and n not in unsafe}
    terms = []
    for location, tree in roots.items():
        for sign, term in _terms(tree):
            weight, factors = _factors(term)
            terms.append((location, sign * weight, factors))
    return roots, eligible, terms, constants


class _ProductComposition:
    """Compose disjoint numerator-product certificates with the old constant lift."""

    def __init__(self, base, products, source):
        self.base = base
        self.product = gains.GainReduction(
            tuple(base.names),
            tuple(base.lower),
            tuple(base.upper),
            tuple(products),
            (),
            source,
        )
        self.names = self.product.names
        self.lower, self.upper = self.product.lower, self.product.upper

    @property
    def receipt(self):
        return {
            "protocol": PROTOCOL,
            "base_reduction": self.base.receipt,
            "numerator_products": self.product.receipt,
        }

    def vector(self, parameters):
        return self.product.vector(parameters)

    def reduce(self, vector):
        return self.product.reduce(self.base.reduce(vector))

    def lift(self, vector):
        return self.base.lift(self.product.lift(vector))

    def lift_jacobian(self, vector):
        section = self.product.lift(vector)
        return self.base.lift_jacobian(section) @ self.product.lift_jacobian()


def certify(original, base, anchor):
    """Reduce isolated products and matching additive atoms without approximations."""
    candidate = original.system.model.validated.candidate
    if candidate.constraints or original.problem.base_candidate.constraints:
        raise ValueError("explicit constraints preclude nonlinear reduction")
    roots, eligible, terms, constants = _consumers(original, base, anchor)
    domains = {
        n: (float(lo), float(hi))
        for n, lo, hi in zip(base.names, base.lower, base.upper, strict=True)
    }
    used, products = set(), []
    for location, _, factors in terms:
        peers = [
            node.id
            for node in factors
            if isinstance(node, ast.Name) and node.id in eligible
        ]
        for gain in peers:
            if gain in used or domains[gain][0] < 0:
                continue
            for amplitude in peers:
                if amplitude == gain or amplitude in used:
                    continue
                if gains._closed_domain(domains[amplitude], domains[gain]):
                    products.append(gains.GainProduct(amplitude, gain, location))
                    used.update((gain, amplitude))
                    break
    product = _ProductComposition(base, products, gains._source_identity(original))
    # Product gain-one constants allow additive matching after the product quotient.
    extra_constants = {item.gain: 1.0 for item in products}
    signatures = {}
    for location, tree in roots.items():
        tree = _expand(tree, {}, extra_constants)
        for sign, term in _terms(tree):
            weight, factors = _factors(term)
            if not weight:
                continue
            for node in factors:
                if not isinstance(node, ast.Name) or node.id not in eligible:
                    continue
                name = node.id
                if name not in product.names:
                    continue
                other = [f for f in factors if f is not node]
                signature = (location, tuple(sorted(ast.dump(f) for f in other)))
                signatures.setdefault(signature, []).append((name, sign * weight))
    groups, grouped = [], set()
    for (location, _signature), members in signatures.items():
        names = tuple(n for n, _ in members)
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or grouped.intersection(names)
        ):
            continue
        pivot = next((n for n in names if domains[n] == (-np.inf, np.inf)), None)
        if pivot is None:
            continue
        # Every member occurs exactly once in complete executable consumers.
        # Thus none can also occur in a shared nonlinear atom or initializer.
        fixed = tuple(
            (
                n,
                0.0
                if domains[n][0] <= 0 <= domains[n][1]
                else float(product.reduce(anchor)[product.names.index(n)]),
            )
            for n in names
            if n != pivot
        )
        groups.append(
            additive.AdditiveGroup(
                f"effective_sum:{pivot}",
                names,
                tuple(w for _, w in members),
                pivot,
                fixed,
                (location,),
            )
        )
        grouped.update(names)
    proof = {
        "protocol": PROTOCOL,
        "source_sha256": gains._source_identity(original),
        "complete_consumer_roots": sorted(roots),
        "fixed_prior_coordinates": constants,
        "proof": "exclusive numerator factors and identical additive atoms",
        "zero_product_branch_preserved": True,
        "unsupported_consumers_retained": True,
    }
    return additive.AdditiveReduction(product, anchor, groups, proof)


class _LocalSystem:
    """Original symbolic functions composed with one constant per-row lift."""

    def __init__(self, system, offset, matrix, names):
        import casadi as ca

        self.source, self.offset, self.matrix = system, offset.copy(), matrix.copy()
        self.names = tuple(names)
        n, p = system.state_count, len(names)
        t, x = ca.SX.sym("t"), ca.SX.sym("x", n)
        q, u = ca.SX.sym("q", p), ca.SX.sym("u", len(system.inputs))
        theta = ca.DM(offset) + ca.DM(matrix) @ q
        rhs, obs = system.rhs(t, x, theta, u), system.observe(t, x, theta, u)
        fx, fq, hx, hq = (
            ca.jacobian(rhs, x),
            ca.jacobian(rhs, q),
            ca.jacobian(obs, x),
            ca.jacobian(obs, q),
        )
        args = [t, x, q, u]
        self.rhs = ca.Function("reduced_rhs", args, [rhs])
        self.observe = ca.Function("reduced_observe", args, [obs])
        self.state_jacobian = ca.Function("reduced_fx", args, [fx])
        self.local = ca.Function("reduced_partials", args, [fx, fq, hx, hq])
        s = ca.SX.sym("S", n, p)
        augmented = ca.vertcat(x, ca.reshape(s, n * p, 1))
        augmented_rhs = ca.vertcat(rhs, ca.reshape(fx @ s + fq, n * p, 1))
        args = [t, augmented, q, u]
        self.augmented_rhs = ca.Function("reduced_augmented_rhs", args, [augmented_rhs])
        self.augmented_jacobian = (
            None
            if system.augmented_jacobian is None
            else (
                ca.Function(
                    "reduced_augmented_jacobian",
                    args,
                    [ca.jacobian(augmented_rhs, augmented)],
                )
            )
        )

    def __getattr__(self, name):
        return getattr(self.source, name)

    def initial_for(self, row, vector):
        return self.source.initial_for(row, self.offset + self.matrix @ vector)

    def initial_sensitivity(self, row, vector):
        return (
            self.source.initial_sensitivity(row, self.offset + self.matrix @ vector)
            @ self.matrix
        )


class ReducedOracle:
    """Integrate surviving local columns only; use BDF for every default solve."""

    def __init__(self, original, previous, retained):
        self.original = original
        self.incumbent_original_parameters = dict(retained)
        self.reduction = certify(
            original, previous.reduction, original.vector(retained)
        )
        self.names = self.reduction.names
        self.lower, self.upper = self.reduction.lower, self.reduction.upper
        self.start = self.parameters(self.reduction.reduce(original.vector(retained)))
        point = self.vector(self.start)
        self.units = np.maximum(np.abs(point), 1.0)
        self.sensitive_names = tuple(
            n for n in previous.sensitive_names if n in self.names
        )
        self.precision_mode = previous.precision_mode
        self.chain = self.reduction.lift_jacobian(point)
        self.offset = self.reduction.lift(point) - self.chain @ point
        self.systems = {}
        for row in original.rows:
            indices = original.indices[row.trajectory_id]
            local_chain = self.chain[indices]
            active = np.flatnonzero(np.any(local_chain != 0, axis=0))
            system = _LocalSystem(
                original.system,
                self.offset[indices],
                local_chain[:, active],
                [self.names[j] for j in active],
            )
            self.systems[row.trajectory_id] = active, system

    def __getattr__(self, name):
        return getattr(self.original, name)

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
        portfolio._check(deadline)
        point = self.reduction._checked(vector)
        self.original.vector(self.original.parameters(self.reduction.lift(point)))
        settings = self.original.settings.model_copy(
            update={"integration_method": method or "BDF"}
        )
        scales = np.asarray(
            [self.original.scales[n] for n in self.original.system.channels]
        )
        residuals, derivatives = [], []
        for row in self.original.rows:
            active, system = self.systems[row.trajectory_id]
            values, derivative, _, _ = symbolic_rollout(
                system, row, point[active], settings, deadline, sensitivities=jacobian
            )
            observed = np.column_stack([row.targets[n] for n in system.channels])
            residuals.append(((values - observed) / scales).ravel())
            if jacobian:
                local = (derivative / scales[None, :, None]).reshape(
                    values.size, len(active)
                )
                full = np.zeros((len(local), len(self.names)))
                full[:, active] = local
                derivatives.append(full)
        return np.concatenate(residuals), np.vstack(derivatives) if jacobian else None


def make_oracle(problem, training, retained, numerical=None):
    """Build a training-only reduced oracle; qualification is a separate operation."""
    numerical = numerical or qualification.numerical_policy()
    original, previous = portfolio.make_oracle(problem, training, numerical)
    # Generic qualification calls the original oracle without a method override.
    # Freeze that default here too; explicit Radau verification remains separate.
    original.settings = original.settings.model_copy(
        update={"integration_method": "BDF"}
    )
    raw = ReducedOracle(original, previous, retained)
    audit = deepcopy(previous.portfolio_audit) | {
        "protocol": PROTOCOL,
        "optimizer_coordinates": len(raw.names),
        "reduction": raw.reduction.receipt,
        "fixed_default_method": "BDF",
        "sensitivity_columns": {
            name: len(active) for name, (active, _) in raw.systems.items()
        },
        "sensitivity_path": "differentiate_composed_constant_lift_before_integration",
    }
    original.portfolio_audit = deepcopy(audit)
    return original, portfolio.PortfolioOracle(raw, audit)


def chart_equivalence(original, oracle, seconds, checkpoint):
    """Compare supplied, canonical-lift and independent original-ODE predictions."""
    from autoformalism.fitting import prefix_portfolio_timescale as timescale

    def normalized(result):
        value = deepcopy(result)
        started, completed = (
            value.get("calls_started", 0),
            value.get("calls_completed", 0),
        )
        value.update(
            protocol=PROTOCOL,
            verification_calls_attempted=started,
            verification_calls_started=started,
            verification_calls_completed=completed,
            verification_calls_failed=started - completed,
        )
        return value

    value = timescale.chart_equivalence(
        original, oracle, seconds, lambda row: checkpoint(normalized(row))
    )
    return normalized(value)
