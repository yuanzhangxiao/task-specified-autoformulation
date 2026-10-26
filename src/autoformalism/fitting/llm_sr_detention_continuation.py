"""Bounded continuation charts for the reviewed detention numerical controls.

Model 15's triangular chart covers its correlated quotient domain; it does not
replace that domain by a box in physical combinations. Model 14's restricted
chart is explicitly temporary. All residuals use the unchanged original RHS,
and saved search points still require independent original-equation verification.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from fractions import Fraction
from math import fsum, isfinite
from time import monotonic
from typing import Literal

import numpy as np

from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_reduction as reduction
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as recovery

PROTOCOL = "llm-sr-detention-continuation-numerics-1"
Mode = Literal["original", "reduced_model15", "subset_model14"]
Policy = previous.Policy


def effective_policy(policy=None, *, mode: Mode) -> Policy:
    """Use closer independent chart probes without relaxing any error threshold.

    The reduced chart starts its shrinking ladder at at most 1e-5 of its local
    physical scale (five times the tight optimizer stencil). Original-coordinate
    evidence retains its unchanged policy, including historical imported audits.
    """
    value = previous._policy(policy)
    if mode not in ("original", "reduced_model15", "subset_model14"):
        raise ValueError("unknown detention coordinate mode")
    if mode == "reduced_model15":
        derivative = value.numerical.derivative.model_copy(
            update={
                "local_relative_step_cap": min(
                    value.numerical.derivative.local_relative_step_cap, 1e-5
                )
            }
        )
        value = value.model_copy(
            update={
                "numerical": value.numerical.model_copy(
                    update={"derivative": derivative}
                )
            }
        )
    return previous._policy(value)


def _f(value):
    return Fraction(float(value)) if not isinstance(value, Fraction) else value


def _fraction_position(value, low, high):
    if not low <= value <= high:
        raise ValueError("physical point outside conditional chart interval")
    return Fraction(1, 2) if low == high else (value - low) / (high - low)


class CoordinateChart:
    """A fixed original, temporary subset, or surjective quotient chart.

    For Model 15, b selects the feasible groundwater/loss difference, then M
    selects capacity within its conditional range, and d selects the remaining
    product range. Selecting the signed product last avoids a capacity jump at
    d=0. Exact rational arithmetic handles signs and degenerate intervals.
    """

    def __init__(self, physical_domain, parameters, mode: Mode):
        if mode not in ("original", "reduced_model15", "subset_model14"):
            raise ValueError("unknown detention coordinate mode")
        self.physical = deepcopy(physical_domain)
        self.anchor = dict(parameters)
        self.mode = mode
        self.bounds = self.physical["bounds"]
        if set(parameters) != set(self.physical["names"]) or any(
            isinstance(v, bool)
            or not isfinite(v)
            or not self.bounds[n][0] <= v <= self.bounds[n][1]
            for n, v in parameters.items()
        ):
            raise ValueError("original parameter vector outside reviewed domain")
        if mode == "subset_model14" and self.physical["task"] != 14:
            raise ValueError("temporary h_max restriction requires Model 14")
        if mode == "reduced_model15" and self.physical["task"] != 15:
            raise ValueError("exact quotient requires Model 15")
        self.rational_bounds = {
            n: tuple(_f(v) for v in pair) for n, pair in self.bounds.items()
        }
        if mode == "reduced_model15":
            # The original domain checker validates all branch assumptions.
            reduction.reduction_certificate(parameters, self.bounds)
            self.names = (
                *reduction.UNCHANGED,
                "b_fraction",
                "d_fraction",
                "M_fraction",
            )
            glo, ghi = self.rational_bounds["k_gw"]
            llo, lhi = self.rational_bounds["k_loss"]
            self.b_interval = glo - lhi, ghi - llo
            chart_bounds = {n: self.bounds[n] for n in reduction.UNCHANGED}
            chart_bounds |= {
                "b_fraction": [0.0, 1.0],
                "d_fraction": [0.0, 1.0],
                "M_fraction": [0.0, 1.0],
            }
            initial = reduction.reduce_parameters(parameters)
            mlo, mhi = self._m_interval(initial["b"], initial["dt"])
            dlo, dhi = self._d_interval(initial["b"], initial["M"], initial["dt"])
            height = self.physical["public_anchors"]["height"]
            rate = self.physical["public_anchors"]["rate"]

            def fraction_unit(physical_unit, low, high):
                return (
                    1.0 if low == high else min(1.0, physical_unit / float(high - low))
                )

            units = {n: self.physical["units"][n] for n in reduction.UNCHANGED}
            units |= {
                "b_fraction": fraction_unit(rate, *self.b_interval),
                "d_fraction": fraction_unit(rate / height, dlo, dhi),
                "M_fraction": fraction_unit(height, mlo, mhi),
            }
        else:
            self.names = tuple(
                n
                for n in self.physical["names"]
                if mode != "subset_model14" or n != "h_max"
            )
            chart_bounds = {n: self.bounds[n] for n in self.names}
            units = {n: self.physical["units"][n] for n in self.names}
        self.domain = {
            "protocol": "detention-preserved-family-chart-1",
            "mode": mode,
            "names": list(self.names),
            "bounds": chart_bounds,
            "units": units,
            "anchor_parameters_sha256": public.content_sha256(parameters),
            "original_domain_sha256": public.content_sha256(self.physical),
            "physical_family_preserved": mode != "subset_model14",
            "temporary_fixed_parameters": {"h_max": parameters["h_max"]}
            if mode == "subset_model14"
            else {},
            "full_model_convergence_claimed": False,
            "physical_combinations_are_independent_box": False,
            "conditional_order": ["b", "M", "d"] if mode == "reduced_model15" else [],
            "fraction_probe_scales": "frozen_physical_units_not_fraction_origin",
        }
        self.lower, self.upper = np.asarray([chart_bounds[n] for n in self.names]).T
        self.units = np.asarray([units[n] for n in self.names])
        self.anchor_vector = self.encode(parameters)

    def _g_interval(self, b, m=None, t=None):
        glo, ghi = self.rational_bounds["k_gw"]
        llo, lhi = self.rational_bounds["k_loss"]
        low, high = max(glo, b + llo), min(ghi, b + lhi)
        if m is not None:
            hlo, hhi = self.rational_bounds["h_max"]
            low, high = max(low, (m - hhi) / t), min(high, (m - hlo) / t)
        if low > high:
            raise ValueError("empty conditional groundwater interval")
        return low, high

    def _d_interval(self, b, m, t):
        low, high = self._g_interval(b, m, t)
        products = [g * k for g in (low, high) for k in self.rational_bounds["kappa"]]
        return min(products), max(products)

    def _m_interval(self, b, t):
        low, high = self._g_interval(b)
        hlo, hhi = self.rational_bounds["h_max"]
        return hlo + t * low, hhi + t * high

    def encode(self, parameters):
        """Encode a feasible original representative without choosing a gauge."""
        if set(parameters) != set(self.anchor) or any(
            isinstance(v, bool)
            or not isfinite(v)
            or not self.bounds[n][0] <= v <= self.bounds[n][1]
            for n, v in parameters.items()
        ):
            raise ValueError("original parameter vector outside reviewed domain")
        if (
            self.mode == "subset_model14"
            and parameters["h_max"] != self.anchor["h_max"]
        ):
            raise ValueError("temporary fixed h_max differs")
        if self.mode == "reduced_model15":
            point = reduction.reduce_parameters(parameters)
            mlo, mhi = self._m_interval(point["b"], point["dt"])
            dlo, dhi = self._d_interval(point["b"], point["M"], point["dt"])
            values = {n: point[n] for n in reduction.UNCHANGED}
            values |= {
                "b_fraction": _fraction_position(point["b"], *self.b_interval),
                "d_fraction": _fraction_position(point["d"], dlo, dhi),
                "M_fraction": _fraction_position(point["M"], mlo, mhi),
            }
        else:
            values = parameters
        return np.asarray([float(values[n]) for n in self.names])

    def decode(self, vector):
        """Lift a chart point to a feasible original vector for exact RHS replay."""
        point = np.asarray(vector, dtype=float)
        if (
            point.shape != self.units.shape
            or not np.isfinite(point).all()
            or np.any(point < self.lower)
            or np.any(point > self.upper)
        ):
            raise ValueError("optimizer point outside fixed chart bounds")
        if np.array_equal(point, self.anchor_vector):
            return dict(self.anchor)
        if self.mode != "reduced_model15":
            return self.anchor | dict(zip(self.names, point.tolist(), strict=True))
        chart = dict(zip(self.names, map(_f, point), strict=True))
        reduced = {n: chart[n] for n in reduction.UNCHANGED}
        blo, bhi = self.b_interval
        b = blo + chart["b_fraction"] * (bhi - blo)
        mlo, mhi = self._m_interval(b, chart["dt"])
        m = mlo + chart["M_fraction"] * (mhi - mlo)
        dlo, dhi = self._d_interval(b, m, chart["dt"])
        d = dlo + chart["d_fraction"] * (dhi - dlo)
        reduced |= {"b": b, "d": d, "M": m}
        lifted = reduction.lift_parameters(
            reduced,
            self.bounds,
            preferred_g=self.anchor["k_gw"],
            preferred_kappa=self.anchor["kappa"],
        )
        return {n: float(v) for n, v in lifted.items()}


def domain(item, context, training, parameters, mode: Mode):
    """Return the immutable chart contract used by the numerical search."""
    return CoordinateChart(
        original.domain_policy(item, context, training), parameters, mode
    ).domain


class ChartOracle:
    """Finite-difference the actual chart while evaluating the original model."""

    def __init__(
        self,
        item,
        context,
        training,
        parameters,
        mode,
        policy,
        precision,
        checkpoint=None,
    ):
        policy = effective_policy(policy, mode=mode)
        self.native = previous._oracle(
            item, context, training, policy, precision, checkpoint
        )
        self.chart = CoordinateChart(self.native.domain, parameters, mode)
        self.names = self.chart.names
        self.lower, self.upper, self.units = (
            self.chart.lower,
            self.chart.upper,
            self.chart.units,
        )
        self.policy, self.cost = self.native.policy, self.native.cost
        self.settings = self.native.settings
        self.start = dict(parameters)
        self.checkpoint = checkpoint or (lambda _: None)
        self.best_residual = None
        self.retain_base_checkpoints = False
        self.jacobian_cache = OrderedDict()

    def vector(self, parameters):
        return self.chart.encode(parameters)

    def parameters(self, vector):
        return self.chart.decode(vector)

    def local_coordinate_scales(self, vector):
        point = np.asarray(vector)
        scale = np.maximum(np.abs(point), self.units)
        for name in ("h_max", "eps_s", "dt"):
            if name in self.names:
                i = self.names.index(name)
                scale[i] = min(scale[i], abs(point[i]))
        for name in ("b_fraction", "d_fraction", "M_fraction"):
            if name in self.names:
                i = self.names.index(name)
                scale[i] = self.units[i]
        return scale

    def residual(self, vector, deadline, method="Radau"):
        return self.native.residual(
            self.native.vector(self.parameters(vector)), deadline, method
        )

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        point = np.asarray(vector)
        residual = self.residual(point, deadline, method or "Radau")
        if not jacobian:
            return residual, None
        if self.retain_base_checkpoints:
            loss = float(np.mean(residual**2))
            if self.best_residual is None or loss < self.best_residual["training_nmse"]:
                self.best_residual = {
                    "parameters": self.parameters(point),
                    "training_nmse": loss,
                    "origin": "search_base_rollout_before_derivative_completion",
                    "independent_verification_required": True,
                }
                self.checkpoint({"best_rollout_checkpoint": self.best_residual})
        key = point.tobytes()
        if key in self.jacobian_cache:
            return residual, self.jacobian_cache[key].copy()
        self.cost["jacobian_calls"] += 1
        matrix = np.column_stack(
            [previous._column(self, point, i, deadline)[1] for i in range(len(point))]
        )
        self.jacobian_cache[key] = matrix.copy()
        if len(self.jacobian_cache) > 8:
            self.jacobian_cache.popitem(last=False)
        return residual, matrix


def _identity(item, context, training, parameters, mode, purpose, precision, policy):
    if purpose != "search" or precision not in ("standard", "tight"):
        raise ValueError("continuation chart evidence is for qualified search only")
    contract = domain(item, context, training, parameters, mode)
    policy = effective_policy(policy, mode=mode)
    return previous._identity(
        item, context, training, parameters, purpose, precision, policy
    ) | {
        "protocol": PROTOCOL,
        "coordinate_chart_sha256": public.content_sha256(contract),
        "mode": mode,
    }


def direction(
    item,
    context,
    training,
    parameters,
    index,
    *,
    mode: Mode,
    purpose="search",
    precision="tight",
    seconds=120,
    checkpoint=None,
    policy=None,
):
    """Audit one actual chart coordinate, retaining a separately sealed result."""
    began, cpu, deadline = original._clock(seconds)
    policy = effective_policy(policy, mode=mode)
    identity = _identity(
        item, context, training, parameters, mode, purpose, precision, policy
    )
    oracle = ChartOracle(
        item, context, training, parameters, mode, policy, precision, checkpoint
    )
    oracle.native.role = "verification"
    point = oracle.vector(parameters)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(point)
    ):
        raise ValueError("direction index outside chart roster")
    state = {}

    def evaluate(value, stop):
        full = point.copy()
        full[index] = value[0]
        residual, column = previous._column(oracle, full, index, stop)
        state.update(base_residual=residual, jacobian_column=column)
        return residual, column[:, None]

    def residual_only(value, stop):
        full = point.copy()
        full[index] = value[0]
        return oracle.residual(full, stop)

    check = adequacy.derivative_check(
        evaluate,
        point[index : index + 1],
        oracle.lower[index : index + 1],
        oracle.upper[index : index + 1],
        deadline,
        coordinate_units=oracle.units[index : index + 1],
        policy=policy.numerical.derivative,
        residual_only=residual_only,
        local_coordinate_scales=oracle.local_coordinate_scales(point)[
            index : index + 1
        ],
    )
    result = {
        "identity": identity,
        "index": index,
        "name": oracle.names[index],
        "purpose": purpose,
        "precision": precision,
        "parameters": dict(zip(oracle.names, point.tolist(), strict=True)),
        "original_parameters": dict(parameters),
        "coordinate_names": list(oracle.names),
        "coordinate_units": oracle.units.tolist(),
        "lower_bounds": oracle.lower.tolist(),
        "upper_bounds": oracle.upper.tolist(),
        "integration_settings": oracle.settings.model_dump(mode="json"),
        "optimizer_fd_relative_step": oracle.policy.finite_difference_relative_step,
        "derivative": public._jsonable(check),
        "status": "complete"
        if check.checked_indices == (0,) and not check.error
        else "unavailable",
        **{key: value.tolist() for key, value in state.items()},
        "parameter_information_qualified": False,
        "full_model_convergence_claimed": False,
        "validation_or_test_access": False,
        "imported_evidence": None,
    }
    result |= original._accounting(oracle, began, cpu, "verification") | {
        "numerical_cost": dict(oracle.cost),
        "wall_seconds": monotonic() - began,
        "budget_exhausted": monotonic() >= deadline,
    }
    result["evidence_sha256"] = public.content_sha256(result)
    if checkpoint:
        checkpoint(result)
    return result


def import_original_directions(
    item, context, training, parameters, directions, *, mode: Mode, policy=None
):
    """Reuse authenticated tight physical columns only when their meaning is unchanged.

    Historical computation remains charged to the parent. The new record lists
    its exact source receipt; it changes no numeric array, local probe or bound.
    Quotient coordinates cannot reuse original columns without a fresh audit.
    """
    if mode not in ("original", "subset_model14"):
        raise ValueError("nonlinear chart requires fresh derivative evidence")
    policy = effective_policy(policy, mode=mode)
    previous.qualification(
        item,
        context,
        training,
        parameters,
        directions,
        purpose="diagnosis",
        precision="tight",
        policy=policy,
    )
    chart = CoordinateChart(
        original.domain_policy(item, context, training), parameters, mode
    )
    identity = _identity(
        item, context, training, parameters, mode, "search", "tight", policy
    )
    point = dict(zip(chart.names, chart.anchor_vector.tolist(), strict=True))
    result = []
    for saved in directions:
        if saved["name"] not in chart.names:
            continue
        row = deepcopy(saved)
        row |= {
            "identity": identity,
            "index": chart.names.index(saved["name"]),
            "purpose": "search",
            "parameters": point,
            "original_parameters": dict(parameters),
            "coordinate_names": list(chart.names),
            "coordinate_units": chart.units.tolist(),
            "lower_bounds": chart.lower.tolist(),
            "upper_bounds": chart.upper.tolist(),
            "full_model_convergence_claimed": False,
            "imported_evidence": {
                "source_evidence_sha256": saved["evidence_sha256"],
                "source_identity": saved["identity"],
                "no_new_rollout": True,
            },
        }
        row["evidence_sha256"] = public.content_sha256(
            {k: v for k, v in row.items() if k != "evidence_sha256"}
        )
        result.append(row)
    return result


def qualification(
    item,
    context,
    training,
    parameters,
    directions,
    *,
    mode: Mode,
    purpose="search",
    precision="tight",
    policy=None,
):
    """Qualify all selected chart directions without granting full-model confidence."""
    policy = effective_policy(policy, mode=mode)
    identity = _identity(
        item, context, training, parameters, mode, purpose, precision, policy
    )
    chart = CoordinateChart(
        original.domain_policy(item, context, training), parameters, mode
    )
    point = dict(zip(chart.names, chart.anchor_vector.tolist(), strict=True))
    indexed, rows, columns, residual = {}, [], {}, None
    for record in directions:
        if record.get("original_parameters") != parameters:
            raise ValueError("coordinate evidence original endpoint differs")
        i, check = previous._validate_row(
            record, identity, point, chart.domain, policy, precision
        )
        if i in indexed:
            raise ValueError("duplicate chart coordinate receipt")
        indexed[i] = record
        if (
            record.get("status") != "complete"
            or check.error
            or check.checked_indices != (0,)
        ):
            continue
        r, column = (
            np.asarray(record["base_residual"]),
            np.asarray(record["jacobian_column"]),
        )
        if (
            r.ndim != 1
            or not len(r)
            or column.shape != r.shape
            or not np.isfinite(r).all()
            or not np.isfinite(column).all()
        ):
            raise ValueError("invalid residual/Jacobian coordinate arrays")
        if (
            adequacy._digest(r) != check.base_residual_sha256
            or adequacy._digest(column[:, None]) != check.base_jacobian_sha256
        ):
            raise ValueError("coordinate arrays differ from audit")
        if residual is not None and not np.array_equal(residual, r):
            raise ValueError("coordinate residual arrays differ")
        residual, columns[i] = r, column
        audited = check.directions[0]
        gradient = previous._gradient(r, column, chart.units[i])
        limit = max(
            policy.search_gradient_absolute_error,
            policy.search_gradient_relative_error * abs(gradient),
        )
        rows.append(
            {
                "index": i,
                "name": chart.names[i],
                "status": audited.precision_status,
                "gradient": gradient,
                "gradient_error_bound": audited.gradient_error_bound,
                "projected_gradient_error_bound": (
                    audited.projected_gradient_error_bound
                ),
                "search_error_limit": limit,
                "search_qualified": audited.precision_status != "failed"
                and audited.gradient_error_bound is not None
                and audited.gradient_error_bound <= limit,
            }
        )
    rows.sort(key=lambda row: row["index"])
    complete = len(columns) == len(chart.names)
    failed = [r["index"] for r in rows if r["status"] == "failed"]
    result = {
        "identity": identity,
        "parameters": dict(parameters),
        "mode": mode,
        "purpose": purpose,
        "precision": precision,
        "chart": chart.domain,
        "directions": deepcopy(list(directions)),
        "coordinate_names": list(chart.names),
        "coordinate_units": chart.units.tolist(),
        "rows": rows,
        "status": "failed" if failed else "complete" if complete else "incomplete",
        "complete": complete,
        "failed_indices": failed,
        "missing_indices": [i for i in range(len(chart.names)) if i not in columns],
        "search_qualified": complete and all(r["search_qualified"] for r in rows),
        "parameter_information_qualified": False,
        "full_model_convergence_claimed": False,
        "validation_or_test_access": False,
    }
    if complete:
        result |= {
            "base_residual": residual.tolist(),
            "jacobian": np.column_stack(
                [columns[i] for i in range(len(chart.names))]
            ).tolist(),
        }
    result["qualification_sha256"] = public.content_sha256(result)
    return result


def validate_qualification(
    item,
    context,
    training,
    parameters,
    directions,
    *,
    mode: Mode,
    qualification: dict,
    purpose="search",
    precision="tight",
    policy=None,
):
    """Reconstruct gates with exact evidence and narrowly bounded summation roundoff."""
    policy = effective_policy(policy, mode=mode)
    expected = globals()["qualification"](
        item,
        context,
        training,
        parameters,
        directions,
        mode=mode,
        purpose=purpose,
        precision=precision,
        policy=policy,
    )
    if qualification.get("qualification_sha256") != public.content_sha256(
        {k: v for k, v in qualification.items() if k != "qualification_sha256"}
    ):
        raise ValueError("qualification digest differs")
    if len(qualification.get("rows", [])) != len(expected["rows"]):
        raise ValueError("qualification coordinate coverage differs")
    source = {r["index"]: r for r in directions}
    for want, saved in zip(expected["rows"], qualification["rows"], strict=True):
        if want["index"] != saved.get("index"):
            raise ValueError("qualification row order differs")
        row = source[want["index"]]
        magnitude = (
            2
            * fsum(
                abs(float(a) * float(b))
                for a, b in zip(
                    row["base_residual"], row["jacobian_column"], strict=True
                )
            )
            / len(row["base_residual"])
            * expected["coordinate_units"][want["index"]]
        )
        tolerance = 32 * np.finfo(float).eps * magnitude + np.finfo(float).tiny
        value = saved.get("gradient")
        if (
            not isinstance(value, (int, float))
            or not isfinite(value)
            or abs(value - want["gradient"]) > tolerance
        ):
            raise ValueError("qualification gradient exceeds summation roundoff")
        want["gradient"] = value
        want["search_error_limit"] = max(
            policy.search_gradient_absolute_error,
            policy.search_gradient_relative_error * abs(value),
        )
        want["search_qualified"] = (
            want["status"] != "failed"
            and want["gradient_error_bound"] is not None
            and want["gradient_error_bound"] <= want["search_error_limit"]
        )
    expected["search_qualified"] = expected["complete"] and all(
        r["search_qualified"] for r in expected["rows"]
    )
    expected["qualification_sha256"] = public.content_sha256(
        {k: v for k, v in expected.items() if k != "qualification_sha256"}
    )
    if expected != qualification:
        raise ValueError("qualification differs from reconstructed evidence")


def fit(
    item,
    context,
    training,
    parameters,
    *,
    mode: Mode,
    qualification: dict,
    seconds=1800,
    checkpoint=None,
    policy=None,
):
    """Search one certified chart with an independently charged bounded allowance."""
    began, cpu, deadline = original._clock(seconds)
    policy = effective_policy(policy, mode=mode)
    precision = qualification.get("precision", "tight")
    validate_qualification(
        item,
        context,
        training,
        parameters,
        qualification.get("directions", []),
        mode=mode,
        qualification=qualification,
        precision=precision,
        policy=policy,
    )
    if not qualification["search_qualified"]:
        return {
            "stop_reason": "derivatives_unqualified",
            "calls": 0,
            "best": None,
            "qualification_sha256": qualification["qualification_sha256"],
            "mode": mode,
            "full_model_convergence_claimed": False,
        }
    state = {}

    def save(value):
        state.update(value)
        candidate = value.get("best_rollout_checkpoint") or (
            value.get("search") or {}
        ).get("best")
        if candidate and (
            not state.get("best")
            or candidate["training_nmse"] < state["best"]["training_nmse"]
        ):
            state["best"] = candidate
        if checkpoint:
            checkpoint(dict(state))

    oracle = ChartOracle(
        item, context, training, parameters, mode, policy, precision, save
    )
    oracle.retain_base_checkpoints = True
    try:
        result = recovery.search(
            oracle,
            parameters,
            deadline,
            lambda value: save({"search": value}),
            calls=policy.numerical.maximum_calls,
            target=policy.numerical.target_nmse,
            optimizer_scaling="guarded_relative",
            scaling_policy=policy.numerical.scaling,
            normalized_objective=True,
            poor_fit=policy.numerical.stopping,
            reject_invalid_trials=True,
            telemetry=True,
            record_steps=True,
            record_full_steps=True,
        )
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result = {
            "best": state.get("best"),
            "stop_reason": "budget_limited"
            if isinstance(error, TimeoutError)
            else "numerical_failure",
            "error": f"{type(error).__name__}: {error}",
        }
    if state.get("best") and (
        not result.get("best")
        or state["best"]["training_nmse"] < result["best"]["training_nmse"]
    ):
        result["best"] = state["best"]
    return (
        result
        | original._accounting(oracle, began, cpu, "search")
        | {
            "protocol": PROTOCOL,
            "mode": mode,
            "chart": oracle.chart.domain,
            "qualification_sha256": qualification["qualification_sha256"],
            "integration_settings": oracle.settings.model_dump(mode="json"),
            "numerical_cost": dict(oracle.cost),
            "wall_seconds": monotonic() - began,
            "budget_exhausted": monotonic() >= deadline,
            "global_convergence_certified": False,
            "full_model_convergence_claimed": False,
            "independent_verification_required": True,
            "validation_or_test_access": False,
        }
    )
