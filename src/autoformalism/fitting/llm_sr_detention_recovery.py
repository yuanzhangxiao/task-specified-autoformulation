"""Opt-in coordinate-wise recovery for the frozen detention numerical controls.

Each local derivative direction is an independently journalable operation.
Search-useful precision is separate from strict endpoint diagnostic precision;
neither numerical agreement nor low error certifies scientific correctness.
"""

from __future__ import annotations

from copy import deepcopy
from math import fsum, isfinite, sqrt
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as recovery
from autoformalism.fitting.llm_sr_detention_reduction import reduction_certificate
from autoformalism.schemas.base import FiniteFloat, StrictSchema

PROTOCOL = "llm-sr-detention-recovery-numerics-1"
Purpose = Literal["search", "diagnosis"]
Precision = Literal["standard", "tight"]


class Policy(StrictSchema):
    """Fixed useful-search thresholds, distinct from endpoint stationarity."""

    numerical: original.Policy = Field(default_factory=original.Policy)
    tight_fd_relative_step: FiniteFloat = Field(default=2e-6, gt=0, le=1e-4)
    search_gradient_absolute_error: FiniteFloat = Field(default=1e-7, gt=0)
    search_gradient_relative_error: FiniteFloat = Field(default=0.01, gt=0, le=0.1)

    @model_validator(mode="after")
    def distinct_local_steps(self):
        if (
            self.tight_fd_relative_step
            >= self.numerical.finite_difference_relative_step
        ):
            raise ValueError("tight derivative step must shrink the optimizer stencil")
        if not self.numerical.derivative.diagnostic_precision:
            raise ValueError("recovery requires diagnostic uncertainty envelopes")
        return self


def _policy(policy: Policy | None) -> Policy:
    return Policy.model_validate((policy or Policy()).model_dump(mode="json"))


def _gradient(residual, column, unit):
    """Deterministic scalar accumulation; no platform BLAS reduction in hashes."""
    return (
        2
        * fsum(float(a) * float(b) for a, b in zip(residual, column, strict=True))
        / len(residual)
        * float(unit)
    )


def _rank_bounds(singular, errors, relative):
    """Empirical Weyl envelope using a reproducible scalar norm."""
    error = sqrt(fsum(float(v) * float(v) for v in errors))
    largest = float(singular[0]) if len(singular) else 0.0
    thresholds = (relative * max(0.0, largest - error), relative * (largest + error))
    interval = (
        sum(max(0.0, float(s) - error) > thresholds[1] for s in singular),
        sum(float(s) + error > thresholds[0] for s in singular),
    )
    return interval, error, thresholds


def _identity(item, context, training, parameters, purpose, precision, policy):
    if purpose not in ("search", "diagnosis") or precision not in ("standard", "tight"):
        raise ValueError("unknown derivative purpose or integration precision")
    if purpose == "diagnosis" and precision != "tight":
        raise ValueError("endpoint diagnostics require tight integration")
    # Revalidate the public-data boundary even when aggregating saved evidence.
    original.supplied.checked_training(training, context)
    return {
        "protocol": PROTOCOL,
        "model_sha256": public.content_sha256(item),
        "context_sha256": public.content_sha256(context),
        "training_sha256": public.content_sha256(training),
        "parameters_sha256": public.content_sha256(parameters),
        "policy_sha256": public.content_sha256(policy),
        "purpose": purpose,
        "precision": precision,
    }


def _oracle(item, context, training, policy, precision, checkpoint=None):
    numerical = policy.numerical.model_copy(
        update={"finite_difference_relative_step": policy.tight_fd_relative_step}
        if precision == "tight"
        else {}
    )
    oracle = original.DetentionOracle(item, context, training, numerical, checkpoint)
    if precision == "tight":
        oracle.tighten()
    return oracle


def _column(oracle, point, index, deadline):
    """Compute precisely one optimizer FD column in physical coordinates."""
    residual = oracle.residual(point, deadline)
    scale = oracle.local_coordinate_scales(point)[index]
    step = min(
        oracle.policy.finite_difference_relative_step * scale,
        (oracle.upper[index] - oracle.lower[index]) / 4,
    )
    left, right = point[index] - oracle.lower[index], oracle.upper[index] - point[index]
    one, two = point.copy(), point.copy()
    if min(left, right) >= step:
        one[index] += step
        two[index] -= step
        if one[index] == two[index]:
            raise recovery.NumericalInvalidTrial("unrepresentable central step")
        column = (oracle.residual(one, deadline) - oracle.residual(two, deadline)) / (
            one[index] - two[index]
        )
    else:
        sign = 1 if right >= left else -1
        step = min(step, max(left, right) / 2)
        one[index] += sign * step
        two[index] += sign * 2 * step
        if one[index] == point[index] or two[index] == one[index]:
            raise recovery.NumericalInvalidTrial("unrepresentable boundary step")
        column = (
            sign
            * (
                -3 * residual
                + 4 * oracle.residual(one, deadline)
                - oracle.residual(two, deadline)
            )
            / (2 * step)
        )
    return residual, column


def direction(
    item,
    context,
    training,
    parameters,
    index: int,
    *,
    purpose: Purpose = "search",
    precision: Precision = "standard",
    seconds: float = 120,
    checkpoint=None,
    policy: Policy | None = None,
) -> dict:
    """Check one local coordinate without computing any other Jacobian column.

    The campaign journals each returned operation separately, so a later timeout
    never renews already consumed allowances or discards completed directions.
    """
    began, cpu, deadline = original._clock(seconds)
    policy = _policy(policy)
    identity = _identity(
        item, context, training, parameters, purpose, precision, policy
    )
    oracle = _oracle(item, context, training, policy, precision, checkpoint)
    oracle.role = "verification"
    point = oracle.vector(parameters)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(point)
    ):
        raise ValueError("direction index outside parameter roster")
    state = {}

    def evaluate(value, stop):
        full = point.copy()
        full[index] = value[0]
        residual, column = _column(oracle, full, index, stop)
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
        "parameters": dict(parameters),
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
        "validation_or_test_access": False,
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


def _validate_row(row, identity, parameters, domain, policy, precision):
    """Bind raw arrays and scalar derivative evidence to one exact operation."""
    if row.get("identity") != identity or row.get("parameters") != parameters:
        raise ValueError("coordinate evidence belongs to another endpoint or contract")
    if row.get("evidence_sha256") != public.content_sha256(
        {k: v for k, v in row.items() if k != "evidence_sha256"}
    ):
        raise ValueError("coordinate evidence digest differs")
    i = row.get("index")
    if (
        isinstance(i, bool)
        or not isinstance(i, int)
        or not 0 <= i < len(domain["names"])
    ):
        raise ValueError("coordinate index outside roster")
    if (
        row.get("name") != domain["names"][i]
        or row.get("coordinate_names") != domain["names"]
    ):
        raise ValueError("coordinate evidence names differ")
    units = [domain["units"][n] for n in domain["names"]]
    lower, upper = np.asarray([domain["bounds"][n] for n in domain["names"]]).T
    if (
        row.get("coordinate_units") != units
        or row.get("lower_bounds") != lower.tolist()
        or row.get("upper_bounds") != upper.tolist()
    ):
        raise ValueError("coordinate evidence domain differs")
    if row.get("purpose") != identity["purpose"] or row.get("precision") != precision:
        raise ValueError("coordinate evidence mode differs")
    expected_settings = {
        "relative_tolerance": getattr(
            policy.numerical,
            "tight_relative_tolerance"
            if precision == "tight"
            else "relative_tolerance",
        ),
        "absolute_tolerance": getattr(
            policy.numerical,
            "tight_absolute_tolerance"
            if precision == "tight"
            else "absolute_tolerance",
        ),
    }
    if row.get("integration_settings") != expected_settings or row.get(
        "optimizer_fd_relative_step"
    ) != (
        policy.tight_fd_relative_step
        if precision == "tight"
        else policy.numerical.finite_difference_relative_step
    ):
        raise ValueError("coordinate evidence numerical settings differ")
    check = adequacy.DerivativeCheck.model_validate(row["derivative"])
    if (
        check.policy != policy.numerical.derivative
        or check.reduced_vector != (parameters[domain["names"][i]],)
        or check.coordinate_units != (units[i],)
    ):
        raise ValueError("scalar audit coordinates or policy differ")
    return i, check


def qualification(
    item,
    context,
    training,
    parameters,
    directions,
    *,
    purpose: Purpose = "search",
    precision: Precision = "standard",
    policy: Policy | None = None,
) -> dict:
    """Assemble all exact directions; incompleteness never qualifies a search."""
    policy = _policy(policy)
    identity = _identity(
        item, context, training, parameters, purpose, precision, policy
    )
    domain = original.domain_policy(item, context, training)
    names = domain["names"]
    if set(parameters) != set(names) or any(
        not isfinite(parameters[n])
        or not domain["bounds"][n][0] <= parameters[n] <= domain["bounds"][n][1]
        for n in names
    ):
        raise ValueError("parameters outside frozen domain")
    indexed, rows, columns, residual = {}, [], {}, None
    for row in directions:
        i, check = _validate_row(row, identity, parameters, domain, policy, precision)
        if i in indexed:
            raise ValueError("duplicate coordinate receipt")
        indexed[i] = row
        if (
            row.get("status") != "complete"
            or check.error
            or check.checked_indices != (0,)
        ):
            continue
        r, column = (
            np.asarray(row["base_residual"], dtype=float),
            np.asarray(row["jacobian_column"], dtype=float),
        )
        if (
            r.ndim != 1
            or not len(r)
            or column.shape != r.shape
            or not np.isfinite(r).all()
            or not np.isfinite(column).all()
        ):
            raise ValueError("invalid coordinate residual or Jacobian array")
        if (
            adequacy._digest(r) != check.base_residual_sha256
            or adequacy._digest(column[:, None]) != check.base_jacobian_sha256
        ):
            raise ValueError("coordinate arrays differ from derivative audit")
        if residual is not None and not np.array_equal(r, residual):
            raise ValueError("coordinate base residuals differ at identical precision")
        residual, columns[i] = r, column
        audited = check.directions[0]
        gradient = _gradient(r, column, domain["units"][names[i]])
        search_limit = max(
            policy.search_gradient_absolute_error,
            policy.search_gradient_relative_error * abs(gradient),
        )
        rows.append(
            {
                "index": i,
                "name": names[i],
                "status": audited.precision_status,
                "gradient": gradient,
                "gradient_error_bound": audited.gradient_error_bound,
                "projected_gradient_error_bound": (
                    audited.projected_gradient_error_bound
                ),
                "jacobian_error_bound_l2": audited.jacobian_error_bound_l2,
                "search_error_limit": search_limit,
                "search_qualified": audited.precision_status != "failed"
                and audited.gradient_error_bound is not None
                and audited.gradient_error_bound <= search_limit,
                "strict_gradient_qualified": audited.precision_status != "failed"
                and audited.projected_gradient_error_bound is not None
                and audited.projected_gradient_error_bound
                <= policy.numerical.derivative.gradient_tolerance
                * policy.numerical.derivative.precision_gradient_fraction,
            }
        )
    rows.sort(key=lambda row: row["index"])
    complete = len(columns) == len(names)
    failed = [r["index"] for r in rows if r["status"] == "failed"]
    result = {
        "identity": identity,
        "parameters": dict(parameters),
        "purpose": purpose,
        "precision": precision,
        "directions": deepcopy(list(directions)),
        "coordinate_names": names,
        "coordinate_units": [domain["units"][n] for n in names],
        "status": "failed" if failed else "complete" if complete else "incomplete",
        "complete": complete,
        "rows": rows,
        "failed_indices": failed,
        "missing_indices": [i for i in range(len(names)) if i not in columns],
        "search_qualified": complete and all(r["search_qualified"] for r in rows),
        "strict_gradient_qualified": complete
        and all(r["strict_gradient_qualified"] for r in rows),
        "parameter_information_qualified": False,
        "scope": (
            "Search-useful derivative precision is not endpoint convergence "
            "or confidence."
        ),
    }
    if complete:
        result |= {
            "base_residual": residual.tolist(),
            "jacobian": np.column_stack(
                [columns[i] for i in range(len(names))]
            ).tolist(),
        }
    result["qualification_sha256"] = public.content_sha256(result)
    return result


def fit(
    item,
    context,
    training,
    start,
    *,
    qualification: dict,
    seconds: float = 600,
    checkpoint=None,
    policy: Policy | None = None,
) -> dict:
    """Search only after complete qualification; keep improving base checkpoints."""
    began, cpu, deadline = original._clock(seconds)
    policy = _policy(policy)
    validate_qualification(
        item,
        context,
        training,
        start,
        qualification.get("directions", []),
        qualification=qualification,
        purpose="search",
        precision=qualification.get("precision", "standard"),
        policy=policy,
    )
    gate = qualification
    if not gate["search_qualified"]:
        return {
            "stop_reason": "derivatives_unqualified",
            "calls": 0,
            "best": None,
            "qualification_sha256": gate["qualification_sha256"],
            "global_convergence_certified": False,
        }
    state = {}

    def save(value):
        state.update(value)
        candidate = value.get("best_rollout_checkpoint") or (
            value.get("search") or {}
        ).get("best")
        if candidate and (
            state.get("best") is None
            or candidate["training_nmse"] < state["best"]["training_nmse"]
        ):
            state["best"] = candidate
        if checkpoint:
            checkpoint(dict(state))

    oracle = _oracle(item, context, training, policy, gate["precision"], save)
    oracle.retain_base_checkpoints = True
    try:
        result = recovery.search(
            oracle,
            start,
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
            "qualification_sha256": gate["qualification_sha256"],
            "integration_settings": oracle.settings.model_dump(mode="json"),
            "numerical_cost": dict(oracle.cost),
            "wall_seconds": monotonic() - began,
            "budget_exhausted": monotonic() >= deadline,
            "global_convergence_certified": False,
            "independent_verification_required": True,
            "validation_or_test_access": False,
        }
    )


def _diagnose(
    item,
    context,
    training,
    parameters,
    *,
    directions,
    verification,
    policy: Policy | None = None,
    decomposition=None,
    frozen_gate=None,
) -> dict:
    """Pure exact-endpoint aggregation with empirical gradient and rank bounds."""
    policy = _policy(policy)
    gate = frozen_gate or qualification(
        item,
        context,
        training,
        parameters,
        directions,
        purpose="diagnosis",
        precision="tight",
        policy=policy,
    )
    result = {
        "protocol": PROTOCOL,
        "parameters": dict(parameters),
        "qualification": gate,
        "status": "unavailable",
        "parameter_information_qualified": False,
        "global_optimum_certified": False,
        "global_identifiability_certified": False,
        "coefficient_accuracy_claimed": False,
        "confidence_probability": None,
        "local_first_order_stationarity": "unresolved",
        "validation_or_test_access": False,
    }
    domain = original.domain_policy(item, context, training)
    if item.task == 15:
        result["exact_reduction"] = reduction_certificate(parameters, domain["bounds"])
        g, loss, k, h, t = (
            parameters[n] for n in ("k_gw", "k_loss", "kappa", "h_max", "dt")
        )
        result["exact_ambiguity"] = {
            "rank_ceiling": 8,
            "declared_coordinates": 9,
            "combinations": {"b": g - loss, "d": g * k, "M": h + t * g},
            "scope": (
                "Exact preserved-family parameter ambiguity; original coefficients "
                "cannot be uniquely identified."
            ),
        }
    if (
        verification.get("parameters") != parameters
        or not verification.get("usable_for_retention")
        or not recovery.usable(verification)
    ):
        return result | {
            "reason": "matching independent original-equation verification unavailable"
        }
    if not gate["complete"]:
        return result | {"reason": "incomplete retained-endpoint coordinate coverage"}
    r, jacobian = np.asarray(gate["base_residual"]), np.asarray(gate["jacobian"])
    if not np.isclose(
        np.mean(r * r), verification["training_nmse"], rtol=1e-10, atol=1e-14
    ):
        raise ValueError("diagnostic residual differs from tight retained verification")
    names, units = gate["coordinate_names"], np.asarray(gate["coordinate_units"])
    lo, hi = np.asarray([domain["bounds"][n] for n in names]).T
    point = np.asarray([parameters[n] for n in names])
    scaled = jacobian * units
    if decomposition is None:
        _, singular, right = np.linalg.svd(scaled, full_matrices=False)
    else:
        singular, right = decomposition
    errors = [row["jacobian_error_bound_l2"] for row in gate["rows"]]
    interval, error, thresholds = _rank_bounds(
        singular, errors, policy.numerical.derivative.rank_relative_tolerance
    )
    gradient = np.asarray(
        [_gradient(r, jacobian[:, i], units[i]) for i in range(len(names))]
    )
    tol = policy.numerical.derivative.bound_tolerance * np.maximum(units, np.abs(point))
    low, high = point - lo <= tol, hi - point <= tol
    projected = gradient.copy()
    projected[low & (gradient > 0)] = 0
    projected[high & (gradient < 0)] = 0
    radii = np.asarray([row["projected_gradient_error_bound"] for row in gate["rows"]])
    upper = float(np.max(np.abs(projected) + radii))
    active = [name for name, flag in zip(names, low | high, strict=True) if flag]
    qualified = (
        all(row["status"] == "passed" for row in gate["rows"])
        and interval[0] == interval[1]
        and not active
    )
    result |= {
        "status": "complete",
        "training_nmse": fsum(float(v) * float(v) for v in r) / len(r),
        "coordinate_names": names,
        "coordinate_units": units.tolist(),
        "singular_values": singular.tolist(),
        "parameter_rank_interval": list(interval),
        "jacobian_error_bound_frobenius": error,
        "error_bound_interpretation": (
            "Empirical finite-difference truncation/roundoff envelopes, "
            "not rigorous interval bounds or statistical confidence intervals."
        ),
        "rank_threshold_interval": list(thresholds),
        "active_bounds": active,
        "projected_gradient_inf": float(np.max(np.abs(projected))),
        "projected_gradient_upper_bound": upper,
        "local_first_order_stationarity": "qualified"
        if not gate["failed_indices"]
        and gate["strict_gradient_qualified"]
        and upper <= policy.numerical.derivative.gradient_tolerance
        else "unresolved",
        "parameter_information_qualified": qualified and item.task != 15,
        "weak_combinations": [
            {
                "singular_value": float(singular[i]),
                "dimensionless_direction": dict(
                    zip(names, right[i].tolist(), strict=True)
                ),
            }
            for i in range(max(0, len(singular) - 2), len(singular))
        ],
        "scope": (
            "Local numerical first-order/rank evidence; no minimum, uniqueness, "
            "statistical coverage or scientific-validity certificate."
        ),
    }
    if item.task == 15:
        result["numeric_rank_conflicts_with_exact_ceiling"] = interval[0] > 8
        result["original_parameter_uniqueness"] = "excluded_by_exact_ambiguity"
    return result


def validate_qualification(
    item,
    context,
    training,
    parameters,
    directions,
    *,
    qualification,
    purpose: Purpose = "search",
    precision: Precision = "standard",
    policy: Policy | None = None,
) -> None:
    """Validate frozen gate without replacing it with platform-derived numbers.

    Only a scalar dot product may differ within floating-point summation error.
    Every identity, raw array, direction classification and decision is exact.
    """
    policy = _policy(policy)
    expected = globals()["qualification"](
        item,
        context,
        training,
        parameters,
        directions,
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
    source = {row["index"]: row for row in directions}
    for want, saved in zip(expected["rows"], qualification["rows"], strict=True):
        if saved.get("index") != want["index"]:
            raise ValueError("qualification coordinate order differs")
        row = source[want["index"]]
        residual, column = (
            np.asarray(row["base_residual"]),
            np.asarray(row["jacobian_column"]),
        )
        unit = expected["coordinate_units"][want["index"]]
        magnitude = (
            2
            * fsum(
                abs(float(a) * float(b)) for a, b in zip(residual, column, strict=True)
            )
            / len(residual)
            * unit
        )
        rounding = 32 * np.finfo(float).eps * magnitude + np.finfo(float).tiny
        value = saved.get("gradient")
        if (
            not isinstance(value, (int, float))
            or not isfinite(value)
            or abs(value - want["gradient"]) > rounding
        ):
            raise ValueError("qualification gradient differs beyond summation roundoff")
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
        row["search_qualified"] for row in expected["rows"]
    )
    expected["qualification_sha256"] = public.content_sha256(
        {k: v for k, v in expected.items() if k != "qualification_sha256"}
    )
    if expected != qualification:
        raise ValueError("supplied qualification differs from reconstructed evidence")


def diagnose(
    item,
    context,
    training,
    parameters,
    *,
    directions,
    verification,
    policy: Policy | None = None,
) -> dict:
    """Compute once; reporters validate and preserve this saved decomposition."""
    return _diagnose(
        item,
        context,
        training,
        parameters,
        directions=directions,
        verification=verification,
        policy=policy,
    )


def validate_diagnosis(
    item,
    context,
    training,
    parameters,
    *,
    directions,
    verification,
    diagnosis,
    policy: Policy | None = None,
) -> None:
    """Validate SVD invariants rather than comparing arbitrary singular-vector signs.

    Accepted singular values have only dimension-scaled machine roundoff slack.
    Weak vectors must be normalized, mutually orthogonal eigenvectors of J^T J.
    All scientific decisions are reconstructed from the accepted saved values.
    """
    policy = _policy(policy)
    saved = deepcopy(diagnosis)
    for key, value in {
        "diagnosed_parameters_sha256": public.content_sha256(parameters),
        "verification_sha256": public.content_sha256(verification),
    }.items():
        if key in saved and saved.pop(key) != value:
            raise ValueError("diagnosis endpoint binding differs")
    gate = saved.get("qualification", {})
    validate_qualification(
        item,
        context,
        training,
        parameters,
        directions,
        qualification=gate,
        purpose="diagnosis",
        precision="tight",
        policy=policy,
    )
    decomposition = None
    if saved.get("status") == "complete":
        if not gate["complete"]:
            raise ValueError("complete diagnosis lacks all coordinate evidence")
        matrix = np.asarray(gate["jacobian"]) * np.asarray(gate["coordinate_units"])
        singular = np.asarray(saved.get("singular_values", []), dtype=float)
        dimension = min(matrix.shape)
        if (
            singular.shape != (dimension,)
            or not np.isfinite(singular).all()
            or np.any(singular < 0)
            or np.any(np.diff(singular) > 0)
        ):
            raise ValueError("invalid saved singular values")
        fresh = np.linalg.svd(matrix, compute_uv=False)
        frobenius = sqrt(fsum(float(v) * float(v) for v in matrix.flat))
        tolerance = 64 * np.finfo(float).eps * max(matrix.shape)
        if np.any(
            np.abs(fresh - singular) > tolerance * frobenius + np.finfo(float).tiny
        ):
            raise ValueError("saved singular values exceed SVD roundoff envelope")
        expected_indices = list(range(max(0, dimension - 2), dimension))
        weak = saved.get("weak_combinations", [])
        if len(weak) != len(expected_indices):
            raise ValueError("saved weak direction count differs")
        right = np.zeros((dimension, matrix.shape[1]))
        vectors = []
        for index, record in zip(expected_indices, weak, strict=True):
            values = record.get("dimensionless_direction", {})
            if (
                set(values) != set(gate["coordinate_names"])
                or record.get("singular_value") != singular[index]
            ):
                raise ValueError("saved weak direction roster/value differs")
            vector = np.asarray(
                [values[n] for n in gate["coordinate_names"]], dtype=float
            )
            if (
                not np.isfinite(vector).all()
                or abs(float(vector @ vector) - 1) > tolerance
            ):
                raise ValueError("saved weak vector is not normalized")
            if any(abs(float(vector @ other)) > tolerance for other in vectors):
                raise ValueError("saved weak vectors are not orthogonal")
            residual = matrix.T @ (matrix @ vector) - singular[index] ** 2 * vector
            if (
                np.linalg.norm(residual)
                > tolerance * frobenius**2 + np.finfo(float).tiny
            ):
                raise ValueError(
                    "saved weak vector does not satisfy SVD eigen-equation"
                )
            vectors.append(vector)
            right[index] = vector
        decomposition = (singular, right)
    expected = _diagnose(
        item,
        context,
        training,
        parameters,
        directions=directions,
        verification=verification,
        policy=policy,
        decomposition=decomposition,
        frozen_gate=gate,
    )
    # Gradient reductions use fsum, so all post-SVD scalar decisions are exact.
    # Arbitrary SVD signs/degenerate bases were validated invariantly above and
    # then reused when reconstructing the saved explanatory weak combinations.
    if expected != saved:
        raise ValueError(
            "saved diagnosis differs from raw evidence and frozen decisions"
        )
