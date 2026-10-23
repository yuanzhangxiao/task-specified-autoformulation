"""Fixed-endpoint, training-only comparison of complete affine residual precision.

Unlike transition-only multiprecision, this diagnostic evaluates the certified
coordinate lift, coefficients, initial maps, propagation, observations and
residual subtraction in private mpmath contexts. Finite differences are formed
before converting to float64. It neither searches nor grants qualification.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import ceil, isfinite, log10
from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.fitting import prefix_execution, prefix_fit_adequacy
from autoformalism.fitting.prefix_local_derivatives import _initial_step
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.fitting.simulation import trajectory_forcing

PROTOCOL = "scoped-prefix-endpoint-precision-probe-1"


@dataclass(frozen=True)
class PrecisionProbePolicy:
    """Finite diagnostic work limits, independent of fitting allowances."""

    decimal_digits: tuple[int, int] = (40, 70)
    maximum_columns: int = 6
    maximum_dimension: int = 32
    maximum_symbolic_nodes: int = 20000
    maximum_scalings: int = 128
    maximum_taylor_terms: int = 128

    def __post_init__(self):
        if (
            not isinstance(self.decimal_digits, tuple)
            or len(self.decimal_digits) != 2
            or any(type(n) is not int for n in self.decimal_digits)
            or not 30 <= self.decimal_digits[0] < self.decimal_digits[1] <= 120
        ):
            raise ValueError("two increasing decimal precisions in [30,120] required")
        for key in asdict(self).keys() - {"decimal_digits"}:
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"positive integer {key} required")


def policy_receipt(policy: PrecisionProbePolicy | None = None) -> dict:
    """Freeze the probe algorithm without modifying any historical policy."""
    policy = policy or PrecisionProbePolicy()
    record = {
        "protocol": PROTOCOL,
        "policy": asdict(policy),
        "local_policy": _local_policy().model_dump(mode="json"),
        "scales": "two local scales h and h/2; no probe enlargement",
        "nodes": "same representable float64 probe coordinates in every backend",
        "precision_scope": (
            "restricted symbolic lift/generator/initials/observations; affine "
            "propagation, normalization, subtraction and differencing in mpmath"
        ),
        "constants": "exact binary64 source constants and data; no invented digits",
        "guard_digits": 10 + ceil(policy.maximum_scalings * log10(4)),
        "scientific_or_convergence_certificate": False,
        "changes_parameters_or_scores": False,
    }
    return record | {"sha256": content_sha256(record)}


def _local_policy():
    return prefix_fit_adequacy.Policy(
        precision_aware=True, local_adaptive=True, local_bounds_aware=True
    )


def _deadline(deadline):
    if not isfinite(deadline) or monotonic() >= deadline:
        raise TimeoutError("fixed-endpoint precision probe allowance exhausted")


class _Expression:
    """Interpret only a bounded whitelist of compiled scalar SX operations."""

    def __init__(self, expression, maximum_nodes):
        self.shape = expression.shape
        self.nodes, known = [], {}
        supported = {
            ca.OP_ADD,
            ca.OP_SUB,
            ca.OP_MUL,
            ca.OP_DIV,
            ca.OP_NEG,
            ca.OP_SQ,
            ca.OP_SQRT,
            ca.OP_EXP,
            ca.OP_LOG,
            ca.OP_POW,
            ca.OP_CONSTPOW,
            ca.OP_SIN,
            ca.OP_COS,
            ca.OP_TANH,
            ca.OP_INV,
            ca.OP_TWICE,
            ca.OP_FABS,
            ca.OP_FMIN,
            ca.OP_FMAX,
            ca.OP_SIGN,
            ca.OP_IF_ELSE_ZERO,
            ca.OP_LT,
            ca.OP_LE,
            ca.OP_EQ,
            ca.OP_NOT,
        }

        def visit(node):
            key = hash(node)
            if key in known:
                return known[key]
            if len(known) >= maximum_nodes:
                raise ValueError("precision symbolic node allowance exceeded")
            if node.is_constant():
                record = ("constant", float(node))
                if not isfinite(record[1]):
                    raise ValueError("nonfinite symbolic constant")
            elif node.is_symbolic():
                record = ("symbol", node.name())
            else:
                operation = node.op()
                if operation not in supported:
                    raise ValueError(
                        f"unsupported precision symbolic operation {operation}"
                    )
                record = (
                    operation,
                    tuple(visit(node.dep(j)) for j in range(node.n_dep())),
                )
            known[key] = len(self.nodes)
            self.nodes.append(record)
            return known[key]

        self.outputs = [visit(expression[i]) for i in range(expression.numel())]

    def __call__(self, context, values):
        unary = {
            ca.OP_NEG: lambda a: -a,
            ca.OP_SQ: lambda a: a * a,
            ca.OP_INV: lambda a: 1 / a,
            ca.OP_TWICE: lambda a: 2 * a,
            ca.OP_SQRT: context.sqrt,
            ca.OP_EXP: context.exp,
            ca.OP_LOG: context.log,
            ca.OP_SIN: context.sin,
            ca.OP_COS: context.cos,
            ca.OP_TANH: context.tanh,
            ca.OP_FABS: abs,
            ca.OP_SIGN: context.sign,
            ca.OP_NOT: lambda a: int(not a),
        }
        binary = {
            ca.OP_ADD: lambda a, b: a + b,
            ca.OP_SUB: lambda a, b: a - b,
            ca.OP_MUL: lambda a, b: a * b,
            ca.OP_DIV: lambda a, b: a / b,
            ca.OP_POW: context.power,
            ca.OP_CONSTPOW: context.power,
            ca.OP_FMIN: min,
            ca.OP_FMAX: max,
            ca.OP_IF_ELSE_ZERO: lambda a, b: b if a else context.mpf(0),
            ca.OP_LT: lambda a, b: int(a < b),
            ca.OP_LE: lambda a, b: int(a <= b),
            ca.OP_EQ: lambda a, b: int(a == b),
        }
        result = []
        for operation, args in self.nodes:
            if operation == "constant":
                value = context.mpf(args)
            elif operation == "symbol":
                value = values[args]
            elif operation in unary:
                value = unary[operation](result[args[0]])
            else:
                value = binary[operation](*(result[i] for i in args))
            if not context.isfinite(value) or context.im(value):
                raise NumericalInvalidTrial("nonfinite or complex precision expression")
            result.append(value)
        matrix = context.zeros(*self.shape)
        for i, key in enumerate(self.outputs):
            matrix[i % self.shape[0], i // self.shape[0]] = result[key]
        return matrix


def _symbolic_lift(reduction, vector):
    """Compose only the three existing typed, certified coordinate sections."""
    from autoformalism.fitting.prefix_construction_recovery import ActiveReduction
    from autoformalism.fitting.prefix_gain_reduction import GainReduction
    from autoformalism.fitting.prefix_structural_reduction import StructuralReduction

    if isinstance(reduction, StructuralReduction):
        values = dict(
            zip(reduction.base.names, map(float, reduction.anchor), strict=True)
        )
        values.update(
            {n: vector[i] for i, n in enumerate(reduction.names) if n in values}
        )
        for decay in reduction.decays:
            rate = vector[reduction.names.index(decay.coordinate)]
            t = decay.tau_lower * rate
            values.update(
                {
                    decay.tau: decay.tau_lower + 1 / rate,
                    decay.divisor: 1,
                    decay.gain: rate * t / (1 + t),
                }
            )
        return _symbolic_lift(
            reduction.base, ca.vertcat(*(values[n] for n in reduction.base.names))
        )
    if isinstance(reduction, ActiveReduction):
        values = list(map(float, reduction.anchor))
        for i, j in enumerate(reduction.indices):
            values[j] = vector[i]
        return _symbolic_lift(reduction.gain_reduction, ca.vertcat(*values))
    if isinstance(reduction, GainReduction):
        values = {n: vector[i] for i, n in enumerate(reduction.names)}
        values.update({p.gain: 1 for p in reduction.products})
        return ca.vertcat(*(values[n] for n in reduction.original_names))
    raise ValueError("unsupported precision coordinate lift")


def _exponential(context, matrix, policy, deadline):
    """Bounded scaling/squaring Taylor evaluation with no float64 matrix cast."""
    norm = context.mnorm(matrix, 1)
    scaling = max(0, int(context.ceil(context.log(norm, 2))) + 5) if norm else 0
    if scaling > policy.maximum_scalings:
        raise NumericalInvalidTrial("precision scaling allowance exceeded")
    base = matrix / context.mpf(2) ** scaling
    term = value = context.eye(matrix.rows)
    bound, scaled_norm = context.mpf(1), context.mnorm(base, 1)
    epsilon = context.mpf(10) ** -(context.dps - 5)
    for order in range(1, policy.maximum_taylor_terms + 1):
        _deadline(deadline)
        term = term * base / order
        value = value + term
        bound *= scaled_norm / order
        if order > 1 and bound < epsilon:
            break
    else:
        raise NumericalInvalidTrial("precision Taylor allowance exceeded")
    for _ in range(scaling):
        _deadline(deadline)
        value = value * value
    _deadline(deadline)
    return value


class _Residual:
    """Evaluate one affine training residual completely in the supplied context."""

    def __init__(self, oracle, policy):
        self.oracle, self.policy = oracle, policy
        self.original = getattr(oracle, "original", oracle)
        original, system = self.original, self.original.system
        n, m = system.state_count, len(system.inputs)
        self.dimension = n + 2 * m + 1
        if self.dimension > policy.maximum_dimension:
            raise ValueError("precision generator dimension allowance exceeded")
        q = ca.SX.sym("probe_q", len(oracle.names))
        lifted = _symbolic_lift(oracle.reduction, q) if oracle is not original else q
        self.lift = _Expression(lifted, policy.maximum_symbolic_nodes)
        self.qnames = [q[i].name() for i in range(q.numel())]
        p = ca.SX.sym("probe_p", len(system.names))
        self.pnames = [p[i].name() for i in range(p.numel())]
        x, u, t = ca.SX.sym("probe_x", n), ca.SX.sym("probe_u", m), ca.SX.sym("probe_t")
        self.xnames = [x[i].name() for i in range(n)]
        self.unames = [u[i].name() for i in range(m)]
        self.tname = t.name()
        self.generator = _Expression(
            original._propagation.generator(p), policy.maximum_symbolic_nodes
        )
        self.observe = _Expression(
            system.observe(t, x, p, u), policy.maximum_symbolic_nodes
        )
        self.initials = {
            row.trajectory_id: _Expression(
                system.initial_symbolic(row, p), policy.maximum_symbolic_nodes
            )
            for row in original.rows
        }
        self.inputs = {}
        for row in original.rows:
            forcing = trajectory_forcing(system.model, row)
            self.inputs[row.trajectory_id] = [
                [forcing.value(name, time) for name in system.inputs]
                for time in row.time
            ]

    def evaluate(self, vector, context, deadline):
        _deadline(deadline)
        original, system = self.original, self.original.system
        full = self.lift(
            context, dict(zip(self.qnames, map(context.mpf, vector), strict=True))
        )
        scales = [context.mpf(original.scales[c]) for c in system.channels]
        n, cache, residual = system.state_count, {}, []
        for row in original.rows:
            _deadline(deadline)
            local = [full[int(i)] for i in original.indices[row.trajectory_id]]
            env = dict(zip(self.pnames, local, strict=True))
            generator = self.generator(context, env)
            initial = self.initials[row.trajectory_id](context, env)
            state = context.matrix(initial)
            inputs = [
                list(map(context.mpf, values))
                for values in self.inputs[row.trajectory_id]
            ]
            dynamic_key = tuple(local[i] for i in original._propagation.dynamic)
            for i, time in enumerate(row.time):
                _deadline(deadline)
                if i:
                    # Preserve the supplied binary64 interval length, while all
                    # coefficient multiplication and state arithmetic stay MP.
                    dt = context.mpf(float(time - row.time[i - 1]))
                    key = dynamic_key, dt
                    if key not in cache:
                        cache[key] = _exponential(
                            context, generator * dt, self.policy, deadline
                        )
                    extended = context.matrix(
                        list(state)
                        + inputs[i - 1]
                        + [
                            (b - a) / dt
                            for a, b in zip(inputs[i - 1], inputs[i], strict=True)
                        ]
                        + [1]
                    )
                    state = (cache[key] * extended)[:n, :]
                observed = self.observe(
                    context,
                    env
                    | dict(zip(self.xnames, state, strict=True))
                    | dict(zip(self.unames, inputs[i], strict=True))
                    | {self.tname: context.mpf(float(time))},
                )
                for j, channel in enumerate(system.channels):
                    value = (
                        observed[j] - context.mpf(float(row.targets[channel][i]))
                    ) / scales[j]
                    if not context.isfinite(value) or context.im(value):
                        raise NumericalInvalidTrial("nonfinite high precision residual")
                    residual.append(value)
        _deadline(deadline)
        return residual


def _rms(context, values):
    value = context.sqrt(context.fsum(v * v for v in values) / len(values))
    result = float(value)
    if not isfinite(result):
        raise NumericalInvalidTrial("nonrepresentable precision diagnostic")
    return result


def _difference(context, base, first, second, offsets, unit):
    """Derivative of the interpolating quadratic at zero, on actual probe nodes."""
    a, b = offsets
    weights = (-(a + b) / (a * b), b / (a * (b - a)), -a / (b * (b - a)))
    return [
        context.fsum(w * v for w, v in zip(weights, values, strict=True)) * unit
        for values in zip(base, first, second, strict=True)
    ]


def compare(
    oracle, parameters, seconds, checkpoint, *, indices, steps=None, policy=None
):
    """Compare fixed local columns, preserving partial evidence and all spent time.

    The caller supplies a durable nonrenewable operation reservation and selected
    coordinate indices. No automatic retry, search, endpoint selection, or fitting
    certificate is supplied by this function.
    """
    import mpmath

    policy = policy or PrecisionProbePolicy()
    if not isinstance(policy, PrecisionProbePolicy):
        raise TypeError("typed precision probe policy required")
    indices = tuple(indices)
    if (
        not indices
        or len(indices) > policy.maximum_columns
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or not 0 <= i < len(oracle.names) for i in indices)
    ):
        raise ValueError("unique bounded precision coordinate indices required")
    if steps is not None:
        if not set(steps) <= set(indices) or any(type(i) is not int for i in steps):
            raise ValueError("saved precision steps must use selected integer indices")
        for pair in steps.values():
            if (
                len(pair) != 2
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in pair)
                or not np.isclose(pair[0], 2 * pair[1], rtol=1e-12, atol=0)
            ):
                raise ValueError("saved precision steps require positive local h,h/2")
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {
        "protocol": PROTOCOL,
        "policy": policy_receipt(policy),
        "status": "incomplete",
        "selected_indices": list(indices),
        "step_source": "saved_local_pairs"
        if steps is not None
        else "initial_local_pairs",
        "rows": [],
        "calls_started": 0,
        "calls_completed": 0,
        "calls_failed": 0,
        "search_performed": False,
        "qualification_override": False,
        "scope": (
            "Empirical fixed-endpoint precision comparison; "
            "no rigorous error bound or parameter confidence."
        ),
    }

    def save():
        result["timing"] = budget.snapshot()
        if checkpoint is not None:
            budget.callback(lambda: checkpoint(deepcopy(result)))

    def call(function):
        result["calls_started"] += 1
        try:
            value = budget.numerical(function)
        except BaseException:
            result["calls_failed"] += 1
            raise
        result["calls_completed"] += 1
        budget.check()
        return value

    interrupted = None
    try:
        vector = oracle.vector(parameters)
        original = getattr(oracle, "original", oracle)
        physical = oracle.reduction.lift(vector) if oracle is not original else vector
        result.update(
            parameters_sha256=content_sha256(oracle.parameters(vector)),
            original_parameters_sha256=content_sha256(original.parameters(physical)),
            source_sha256=content_sha256(original.problem),
            training_sha256=content_sha256(original.training),
            reduction_sha256=oracle.reduction.receipt["sha256"]
            if oracle is not original
            else None,
            baseline_precision_mode=getattr(oracle, "precision_mode", "original"),
        )
        model = _Residual(oracle, policy)
        budget.check()
        contexts = []
        for digits in policy.decimal_digits:
            context = mpmath.mp.clone()
            context.dps = digits + result["policy"]["guard_digits"]
            contexts.append(context)
        base, jacobian = call(lambda deadline: oracle.evaluate(vector, deadline))
        base = np.asarray(base, dtype=float)
        jacobian = np.asarray(jacobian, dtype=float)
        if not np.isfinite(base).all() or not np.isfinite(jacobian).all():
            raise NumericalInvalidTrial("nonfinite guarded residual/Jacobian")
        references = [
            call(lambda deadline, c=c: model.evaluate(vector, c, deadline))
            for c in contexts
        ]
        if any(len(r) != len(base) for r in references):
            raise ValueError("precision residual shape differs")
        high = contexts[-1]
        result["base_residual_discrepancy_rms"] = {
            "guarded_vs_high": _rms(
                high,
                [high.mpf(a) - b for a, b in zip(base, references[-1], strict=True)],
            ),
            "low_vs_high": _rms(
                high,
                [
                    high.mpf(a) - b
                    for a, b in zip(references[0], references[1], strict=True)
                ],
            ),
        }
        save()
        local_policy = _local_policy()
        scales = oracle.local_coordinate_scales(vector)
        for index in indices:
            row = {
                "index": index,
                "name": oracle.names[index],
                "status": "incomplete",
                "scales": [],
                "step_source": "saved_local_pair"
                if steps is not None and index in steps
                else "initial_local_pair",
            }
            result["rows"].append(row)
            initial_step = _initial_step(
                vector[index],
                oracle.lower[index],
                oracle.upper[index],
                oracle.units[index],
                local_policy,
                scales[index],
            )
            pair_steps = (
                steps[index]
                if steps is not None and index in steps
                else (initial_step, initial_step / 2)
            )
            if pair_steps[0] > initial_step * (1 + 1e-12):
                raise ValueError("saved precision steps exceed declared local radius")
            previous = None
            for h in pair_steps:
                h = float(h)
                factor = h / initial_step
                central = (
                    vector[index] - h >= oracle.lower[index]
                    and vector[index] + h <= oracle.upper[index]
                )
                signs = (
                    (-1, 1)
                    if central
                    else (
                        (1, 2)
                        if vector[index] + 2 * h <= oracle.upper[index]
                        else (-1, -2)
                    )
                )
                points = []
                for sign in signs:
                    point = vector.copy()
                    point[index] += sign * h
                    oracle.vector(oracle.parameters(point))
                    if point[index] == vector[index]:
                        raise NumericalInvalidTrial(
                            "precision probe displacement not representable"
                        )
                    points.append(point)
                if points[0][index] == points[1][index]:
                    raise NumericalInvalidTrial("precision probe nodes coincide")
                guarded = [
                    call(lambda d, x=x: oracle.evaluate(x, d, jacobian=False)[0])
                    for x in points
                ]
                mp_values = [
                    [call(lambda d, x=x, c=c: model.evaluate(x, c, d)) for x in points]
                    for c in contexts
                ]
                offsets = [
                    high.mpf(float(x[index])) - high.mpf(float(vector[index]))
                    for x in points
                ]
                unit = high.mpf(float(oracle.units[index]))
                # Guarded residual values are lifted exactly before subtraction:
                # this isolates residual evaluation from the differencing algebra.
                guarded_fd = _difference(
                    high,
                    list(map(high.mpf, base)),
                    *[list(map(high.mpf, r)) for r in guarded],
                    offsets,
                    unit,
                )
                derivatives = [
                    _difference(
                        c,
                        references[j],
                        *mp_values[j],
                        [c.mpf(v) for v in offsets],
                        c.mpf(unit),
                    )
                    for j, c in enumerate(contexts)
                ]
                analytic = [
                    high.mpf(float(v * oracle.units[index])) for v in jacobian[:, index]
                ]

                def error(values, reference=analytic):
                    return _rms(
                        high,
                        [
                            high.mpf(a) - b
                            for a, b in zip(values, reference, strict=True)
                        ],
                    )

                record = {
                    "factor": factor,
                    "step": h,
                    "scheme": "central" if central else "one_sided_quadratic",
                    "actual_offsets": list(map(float, offsets)),
                    "analytic_rms": _rms(high, analytic),
                    "guarded_discrepancy_rms": error(guarded_fd),
                    "multiprecision_discrepancy_rms": [error(v) for v in derivatives],
                    "precision_difference_rms": _rms(
                        high,
                        [high.mpf(a) - b for a, b in zip(*derivatives, strict=True)],
                    ),
                    "local_step_difference_rms": None
                    if previous is None
                    else _rms(
                        high,
                        [a - b for a, b in zip(previous, derivatives[-1], strict=True)],
                    ),
                }
                row["scales"].append(record)
                previous = derivatives[-1]
                save()
            row["status"] = "complete"
            save()
        result["status"] = "complete"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        interrupted = error
    finally:
        if "original" in locals():
            result["baseline_affine_execution"] = deepcopy(original.affine_audit)
        try:
            save()
        finally:
            result["timing"] = budget.snapshot()
    if interrupted is not None:
        raise interrupted
    return result
