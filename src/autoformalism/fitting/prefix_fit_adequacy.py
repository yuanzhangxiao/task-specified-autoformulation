"""Opt-in local fitting adequacy for a scoped, reduced-coordinate residual.

The caller owns training-only data, durable budgets, oracle construction and the
independent verification of the *original* equations. This module never fits or
changes a model. A local first-order assessment is not a minimum, identifiability,
scientific-validity, or global optimality certificate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from time import monotonic
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, model_serializer, model_validator

from autoformalism.schemas.base import FiniteFloat, StrictSchema

Array = NDArray[np.float64]
Evaluate = Callable[[Array, float], tuple[ArrayLike, ArrayLike]]
# Three predeclared coarse/fine pairs; never enlarge a probe in response to a fit.
PRECISION_PROBE_MULTIPLIERS = (1.0, 0.5, 16.0, 8.0, 1024.0, 512.0)


class Policy(StrictSchema):
    """Frozen thresholds in explicit optimizer coordinates; no adaptive loosening."""

    accuracy_nmse: FiniteFloat = Field(default=1e-8, gt=0)
    gradient_tolerance: FiniteFloat = Field(default=1e-8, gt=0)
    correlation_tolerance: FiniteFloat = Field(default=1e-6, gt=0, lt=1)
    history_window: int = Field(default=3, ge=2)
    loss_tolerance: FiniteFloat = Field(default=1e-10, gt=0)
    step_tolerance: FiniteFloat = Field(default=1e-6, gt=0)
    bound_tolerance: FiniteFloat = Field(default=1e-10, gt=0)
    derivative_relative_step: FiniteFloat = Field(default=1e-4, gt=0, le=0.1)
    derivative_absolute_step: FiniteFloat = Field(default=1e-6, gt=0, le=0.1)
    derivative_relative_tolerance: FiniteFloat = Field(default=1e-3, gt=0, lt=1)
    derivative_absolute_tolerance: FiniteFloat = Field(default=1e-7, gt=0)
    max_directions: int = Field(default=64, ge=1)
    rank_relative_tolerance: FiniteFloat = Field(default=1e-8, gt=0, lt=1)
    precision_aware: bool = False
    precision_roundoff_factor: FiniteFloat = Field(default=32, ge=1)
    precision_gradient_fraction: FiniteFloat = Field(default=0.1, gt=0, le=1)
    local_adaptive: bool = False
    local_max_scales: int = Field(default=6, ge=2, le=12)
    local_relative_step_cap: FiniteFloat = Field(default=1e-4, gt=0, le=1e-2)
    local_bounds_aware: bool = False
    diagnostic_precision: bool = False

    @model_validator(mode="after")
    def local_requires_precision(self):
        if self.local_adaptive and not self.precision_aware:
            raise ValueError("local adaptive checks require precision_aware")
        if self.local_bounds_aware and not self.local_adaptive:
            raise ValueError("bounds-aware checks require local_adaptive")
        if self.diagnostic_precision and not self.local_bounds_aware:
            raise ValueError("diagnostic precision requires local_bounds_aware")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy_policy(self, handler):
        """Keep historical policy fingerprints unchanged when the opt-in is off."""
        result = handler(self)
        if not self.diagnostic_precision:
            result.pop("diagnostic_precision", None)
        if not self.local_bounds_aware:
            result.pop("local_bounds_aware", None)
        if not self.local_adaptive:
            for name in (
                "local_adaptive",
                "local_max_scales",
                "local_relative_step_cap",
            ):
                result.pop(name, None)
        if not self.precision_aware:
            for name in (
                "precision_aware",
                "precision_roundoff_factor",
                "precision_gradient_fraction",
            ):
                result.pop(name, None)
        return result


class HistoryPoint(StrictSchema):
    """An accepted optimizer step, never an arbitrary residual evaluation.

    Project these fields from ``recovery_numerics.search(..., record_steps=True)``
    ``accepted_steps``. Step norms must use the same coordinate units as assessment.
    Repeated checkpoints of the same evaluation must not become separate steps.
    """

    training_nmse: FiniteFloat = Field(ge=0)
    coordinate_step_norm: FiniteFloat = Field(ge=0)
    accepted: bool = True
    evaluation_count: int | None = Field(default=None, ge=1)


class OriginalVerification(StrictSchema):
    """Caller-authenticated original-equation replay at this reduced endpoint.

    The two NMSEs/discrepancies have the ``recovery_numerics.usable`` semantics:
    same training residual normalization, two independent original-equation solvers,
    and maximum normalized prediction discrepancy. Reduced-oracle agreement alone
    does not supply ``original_equations_verified``.
    """

    reduced_vector: tuple[FiniteFloat, ...]
    training_nmse: FiniteFloat = Field(ge=0)
    alternate_training_nmse: FiniteFloat = Field(ge=0)
    solver_loss_discrepancy: FiniteFloat = Field(ge=0)
    solver_prediction_discrepancy: FiniteFloat = Field(ge=0)
    original_equations_verified: bool


class PrecisionProbe(StrictSchema):
    """One fixed-scale probe in the same declared optimizer coordinate units."""

    multiplier: FiniteFloat
    step: FiniteFloat
    derivative_rms: FiniteFloat = Field(ge=0)
    error_rms: FiniteFloat = Field(ge=0)
    roundoff_rms: FiniteFloat = Field(ge=0)


class DerivativeDirection(StrictSchema):
    """One reduced coordinate, with two representable in-domain probe sizes."""

    index: int = Field(ge=0)
    scheme: Literal["central", "forward", "backward"]
    steps: tuple[FiniteFloat, FiniteFloat]
    error_rms: tuple[FiniteFloat, FiniteFloat]
    relative_errors: tuple[FiniteFloat, FiniteFloat]
    between_steps_relative_error: FiniteFloat
    one_sided_disagreement: tuple[FiniteFloat, FiniteFloat] | None = None
    passed: bool
    precision_status: Literal["passed", "failed", "inconclusive"] | None = None
    precision_probes: tuple[PrecisionProbe, ...] = ()
    reference_probe_index: int | None = None
    derivative_uncertainty_rms: FiniteFloat | None = Field(default=None, ge=0)
    gradient_error_bound: FiniteFloat | None = Field(default=None, ge=0)
    projected_gradient_error_bound: FiniteFloat | None = Field(default=None, ge=0)
    jacobian_error_bound_l2: FiniteFloat | None = Field(default=None, ge=0)

    @model_serializer(mode="wrap")
    def preserve_historical_direction(self, handler):
        """New diagnostic envelopes do not alter earlier direction receipts."""
        result = handler(self)
        for key in ("projected_gradient_error_bound", "jacobian_error_bound_l2"):
            if result.get(key) is None:
                result.pop(key, None)
        return result


class LocalProbeRecord(StrictSchema):
    """One bounded local probe, retained even when a later probe cannot finish."""

    index: int = Field(ge=0)
    scale_index: int = Field(ge=0)
    scheme: Literal["central", "forward", "backward"]
    step: FiniteFloat = Field(gt=0)
    status: Literal["complete", "numerical_failure", "unavailable", "deadline"]
    derivative_rms: FiniteFloat | None = Field(default=None, ge=0)
    error_rms: FiniteFloat | None = Field(default=None, ge=0)
    roundoff_rms: FiniteFloat | None = Field(default=None, ge=0)
    error: str | None = None


class DerivativeCheck(StrictSchema):
    """Finite local audit with explicit uncovered columns and call accounting."""

    status: Literal["passed", "failed", "inconclusive", "incomplete", "unavailable"]
    reduced_vector: tuple[FiniteFloat, ...]
    coordinate_units: tuple[FiniteFloat, ...]
    base_residual_sha256: str | None = None
    base_jacobian_sha256: str | None = None
    base_training_nmse: FiniteFloat | None = None
    lower_bounds_sha256: str | None = None
    upper_bounds_sha256: str | None = None
    policy: Policy
    required_indices: tuple[int, ...]
    checked_indices: tuple[int, ...]
    omitted_indices: tuple[int, ...]
    directions: tuple[DerivativeDirection, ...] = ()
    calls_started: int = Field(ge=0)
    calls_completed: int = Field(ge=0)
    jacobian_calls_started: int = Field(ge=0)
    residual_only_calls_started: int = Field(ge=0)
    elapsed_seconds: FiniteFloat = Field(ge=0)
    error: str | None = None
    search_qualified: bool = False
    inconclusive_indices: tuple[int, ...] = ()
    failed_indices: tuple[int, ...] = ()
    gradient_error_bound_inf: FiniteFloat | None = Field(default=None, ge=0)
    parameter_information_qualified: bool = False
    local_probe_attempts: tuple[LocalProbeRecord, ...] = ()
    local_coordinate_scales: tuple[FiniteFloat, ...] = ()
    gradient_precision_qualified: bool | None = None
    scope: str = (
        "Two-scale finite differences at one reduced endpoint; not a global "
        "smoothness or minimum certificate. Deadlines are cooperative."
    )

    @model_serializer(mode="wrap")
    def preserve_historical_audit(self, handler):
        """Do not add local-policy payload fields to historical receipts."""
        result = handler(self)
        if not self.policy.diagnostic_precision:
            result.pop("gradient_precision_qualified", None)
        if not self.policy.local_adaptive:
            result.pop("local_probe_attempts", None)
        if not self.policy.local_bounds_aware:
            result.pop("local_coordinate_scales", None)
        return result


class RestartEvidence(StrictSchema):
    """Predeclared, distinct starts strengthen only observed search reliability."""

    start_id: str = Field(min_length=1)
    predeclared: bool
    distinct_from_incumbent: bool
    status: Literal["complete", "failed", "incomplete"]
    verified_training_nmse: FiniteFloat | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def complete_has_loss(self):
        if self.status == "complete" and self.verified_training_nmse is None:
            raise ValueError("complete restart requires a verified training loss")
        return self


class Assessment(StrictSchema):
    """Separate fit adequacy, local parameter information and restart coverage."""

    classification: Literal[
        "optimization_unresolved",
        "locally_converged_accurate_fit",
        "locally_converged_poor_fit",
    ]
    policy: Policy
    reasons: tuple[str, ...]
    training_nmse: FiniteFloat
    projected_gradient_inf: FiniteFloat
    maximum_residual_column_correlation: FiniteFloat
    active_bound_indices: tuple[int, ...]
    accepted_history_points: int = Field(ge=0)
    derivative_status: str
    original_rollout_verified: bool
    parameter_rank: int = Field(ge=0)
    parameter_singular_values: tuple[FiniteFloat, ...]
    free_coordinates: int = Field(ge=0)
    parameter_condition_number: FiniteFloat | None
    parameter_information: Literal[
        "full_local_rank",
        "rank_deficient",
        "no_free_coordinates",
        "derivative_precision_unresolved",
    ]
    projected_gradient_upper_bound: FiniteFloat | None = Field(default=None, ge=0)
    gradient_error_bound_inf: FiniteFloat | None = Field(default=None, ge=0)
    correlation_unresolved_indices: tuple[int, ...] = ()
    parameter_information_qualified: bool = False
    parameter_rank_interval: tuple[int, int] | None = None
    jacobian_error_bound_frobenius: FiniteFloat | None = Field(default=None, ge=0)
    rank_threshold_interval: tuple[FiniteFloat, FiniteFloat] | None = None
    complete_predeclared_restarts: int = Field(ge=0)
    improving_predeclared_restarts: int = Field(ge=0)
    global_minimum_certified: Literal[False] = False
    structural_revision_justified: Literal[False] = False
    scope: str = (
        "Local first-order evidence in declared coordinates, not proof of a "
        "local/global minimum, coefficient accuracy, or a structural defect. "
        "Parameter rank is local and separate from optimization adequacy; full "
        "rank alone is not a statistical parameter-confidence conclusion."
    )

    @model_serializer(mode="wrap")
    def preserve_historical_assessment(self, handler):
        """Retain historical assessment payloads when diagnostic precision is off."""
        result = handler(self)
        if not self.policy.diagnostic_precision:
            for key in (
                "parameter_rank_interval",
                "jacobian_error_bound_frobenius",
                "rank_threshold_interval",
            ):
                result.pop(key, None)
        return result


def _rank_precision(singular, column_errors, relative_tolerance):
    """Bound a thresholded numerical rank under the empirical column envelope.

    Frobenius bounds spectral error; Weyl bounds singular-value changes. The
    relative rank threshold changes with the largest singular value as well.
    This is conditional numerical stability, not structural identifiability.
    """
    error = float(np.linalg.norm(column_errors))
    maximum = float(singular[0]) if len(singular) else 0.0
    thresholds = (
        relative_tolerance * max(0.0, maximum - error),
        relative_tolerance * (maximum + error),
    )
    interval = (
        int(np.sum(np.maximum(0, singular - error) > thresholds[1])),
        int(np.sum(singular + error > thresholds[0])),
    )
    return interval, error, thresholds


def _coordinates(vector, lower, upper, coordinate_units):
    x, lo, hi = (np.asarray(v, dtype=float) for v in (vector, lower, upper))
    if x.ndim != 1 or lo.shape != x.shape or hi.shape != x.shape:
        raise ValueError("coordinate and bound shapes differ")
    if not np.isfinite(x).all() or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("nonfinite coordinates or NaN bounds")
    if np.any(lo > hi) or np.any(x < lo) or np.any(x > hi):
        raise ValueError("coordinates outside ordered bounds")
    units = (
        np.ones_like(x)
        if coordinate_units is None
        else np.asarray(coordinate_units, dtype=float)
    )
    if units.shape != x.shape or not np.isfinite(units).all() or np.any(units <= 0):
        raise ValueError("coordinate units must be finite positive and shape matched")
    return x, lo, hi, units


def _arrays(residual, jacobian, dimension):
    r, j = np.asarray(residual, dtype=float), np.asarray(jacobian, dtype=float)
    if r.ndim != 1 or not len(r) or j.shape != (len(r), dimension):
        raise ValueError("invalid residual or reduced Jacobian shape")
    if not np.isfinite(r).all() or not np.isfinite(j).all():
        raise ValueError("nonfinite residual or reduced Jacobian")
    if not np.isfinite(np.mean(r * r)):
        raise ValueError("nonfinite residual loss")
    return r, j


def _rms(value):
    scale = float(np.max(np.abs(value), initial=0))
    return scale * float(np.sqrt(np.mean((value / scale) ** 2))) if scale else 0.0


def _digest(value):
    """Bind exact float64 array shape/content without serializing large Jacobians."""
    data = np.ascontiguousarray(value, dtype="<f8")
    return sha256(str(data.shape).encode("ascii") + data.tobytes()).hexdigest()


def _agreement(left, right, policy):
    error = _rms(left - right)
    scale = max(_rms(left), _rms(right))
    relative = error / scale if scale else 0.0
    # The absolute floor alone must not certify a completely wrong tiny column.
    passed = relative <= policy.derivative_relative_tolerance and error <= (
        policy.derivative_absolute_tolerance
        + policy.derivative_relative_tolerance * scale
    )
    return error, relative, passed


def _precision_result(
    estimates: Sequence[Array],
    roundoffs: Sequence[Array],
    unilateral: Sequence[float],
    unilateral_roundoff: Sequence[float],
    steps: Sequence[float],
    residual: Array,
    column: Array,
    policy: Policy,
) -> dict:
    """Bound an empirical derivative interval using all fixed coarse/fine pairs.

    The enclosure assumes locally smooth leading-order finite-difference error
    and the declared floating-point residual envelope. It is numerical evidence,
    not an interval-arithmetic guarantee for an arbitrary black-box oracle.
    A resolved disagreement is a failure regardless of the column's magnitude.
    """
    order_factor = 4 if unilateral else 2
    candidates = []
    for fine in range(1, len(estimates), 2):
        coarse = fine - 1
        # Inflate observed truncation and rounding uncertainty; do not infer
        # accuracy merely from a small analytic column or an absolute tolerance.
        radius = roundoffs[fine] + 4 / (order_factor - 1) * (
            np.abs(estimates[fine] - estimates[coarse])
            + roundoffs[fine]
            + roundoffs[coarse]
        )
        sides_ok = not unilateral or (
            unilateral[fine]
            <= unilateral_roundoff[fine]
            + policy.derivative_relative_tolerance * _rms(column)
            or unilateral[fine]
            <= 0.75 * unilateral[coarse]
            + unilateral_roundoff[fine]
            + 0.75 * unilateral_roundoff[coarse]
        )
        candidates.append((_rms(radius), fine, radius, sides_ok))
    admissible = [item for item in candidates if item[3]]
    uncertainty, fine, radius, sides_ok = min(
        admissible or candidates, key=lambda item: (item[0], item[1])
    )
    estimate = estimates[fine]
    error, _, agrees = _agreement(estimate, column, policy)
    scale = max(_rms(estimate), _rms(column))
    contradiction = error > (uncertainty + policy.derivative_relative_tolerance * scale)
    stable = _agreement(estimates[fine - 1], estimate, policy)[2]
    resolved = uncertainty <= policy.derivative_relative_tolerance * scale
    status = (
        "failed"
        if contradiction or not sides_ok
        else "passed"
        if agrees and stable and resolved
        else "inconclusive"
    )
    gradient_error = float(
        2 * np.mean(np.abs(residual) * (np.abs(column - estimate) + radius))
    )
    if not np.isfinite(gradient_error):
        raise ValueError("nonfinite derivative gradient uncertainty")
    return {
        "passed": status == "passed",
        "precision_status": status,
        "reference_probe_index": fine,
        "derivative_uncertainty_rms": uncertainty,
        "gradient_error_bound": gradient_error,
        "precision_probes": tuple(
            PrecisionProbe(
                multiplier=multiplier,
                step=step,
                derivative_rms=_rms(value),
                error_rms=_rms(value - column),
                roundoff_rms=_rms(roundoff),
            )
            for multiplier, step, value, roundoff in zip(
                PRECISION_PROBE_MULTIPLIERS,
                steps,
                estimates,
                roundoffs,
                strict=True,
            )
        ),
    }


def derivative_check(
    evaluate: Evaluate,
    vector: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    deadline: float,
    *,
    coordinate_units: ArrayLike | None = None,
    policy: Policy | None = None,
    directions: Sequence[int] | None = None,
    residual_only: Callable[[Array, float], ArrayLike] | None = None,
    local_coordinate_scales: ArrayLike | None = None,
) -> DerivativeCheck:
    """Audit the actual reduced residual, including profiled reconstruction.

    ``evaluate(x, deadline)`` returns residual and derivative with respect to ``x``.
    The caller must use the same residual weighting and reconstruction as optimization.
    All nonfixed coordinates are required for a pass; selected or capped subsets are
    explicitly incomplete. At a domain boundary use one-sided inward differences.
    Each started oracle call is counted even when it fails or overruns the deadline.
    Optional ``residual_only`` avoids sensitivity integrations at perturbed points;
    its base residual is first checked exactly against ``evaluate``'s residual.
    The precision-aware opt-in retains unresolved columns and can qualify search
    only when their mean-loss gradient error is negligible in the fixed units.
    Bounds-aware probes optionally take physical ``local_coordinate_scales``;
    these are caller-declared local radii, not proofs that a residual is affine.
    Singular/nonlinear small scales must remain restricted by that caller.
    """
    policy = policy or Policy()
    if local_coordinate_scales is not None and not policy.local_bounds_aware:
        raise ValueError("local coordinate scales require local_bounds_aware")
    if policy.local_adaptive:
        from autoformalism.fitting.prefix_local_derivatives import (
            local_derivative_check,
        )

        return local_derivative_check(
            evaluate,
            vector,
            lower,
            upper,
            deadline,
            coordinate_units=coordinate_units,
            policy=policy,
            directions=directions,
            residual_only=residual_only,
            local_coordinate_scales=local_coordinate_scales,
        )
    x, lo, hi, units = _coordinates(vector, lower, upper, coordinate_units)
    if not np.isfinite(deadline):
        raise ValueError("finite monotonic deadline required")
    required = tuple(int(i) for i in np.flatnonzero(lo < hi))
    selected = list(required if directions is None else directions)
    if len(set(selected)) != len(selected) or any(
        not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i not in required
        for i in selected
    ):
        raise ValueError("directions must be distinct nonfixed coordinate indices")
    selected = selected[: policy.max_directions]
    begun, started, completed = monotonic(), 0, 0
    jacobian_calls, residual_calls = 0, 0
    base_r, base_j, rows, error = None, None, [], None

    def call(v, *, full=False):
        nonlocal started, completed, jacobian_calls, residual_calls
        if monotonic() >= deadline:
            raise TimeoutError("derivative audit deadline reached")
        started += 1
        if full or residual_only is None:
            jacobian_calls += 1
            r, j = _arrays(*evaluate(v.copy(), deadline), len(x))
        else:
            residual_calls += 1
            r = np.asarray(residual_only(v.copy(), deadline), dtype=float)
            if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
                raise ValueError("invalid residual-only oracle result")
            j = None
        completed += 1
        if monotonic() >= deadline:
            raise TimeoutError("derivative oracle returned after deadline")
        if base_r is not None and r.shape != base_r.shape:
            raise ValueError("oracle residual size changed across probes")
        return r, j

    try:
        base_r, base_j = call(x, full=True)
        if residual_only is not None and not np.array_equal(call(x)[0], base_r):
            raise ValueError(
                "residual-only callback differs from audited oracle at base"
            )
        for index in selected:
            h = max(
                policy.derivative_absolute_step * units[index],
                policy.derivative_relative_step * max(abs(x[index]), units[index]),
            )
            left, right = x[index] - lo[index], hi[index] - x[index]
            multipliers = (
                PRECISION_PROBE_MULTIPLIERS if policy.precision_aware else (1, 0.5)
            )
            largest = max(multipliers)
            if left >= h * largest and right >= h * largest:
                scheme = "central"
            elif right >= left:
                scheme, h = "forward", min(h, right / (2 * largest))
            else:
                scheme, h = "backward", min(h, left / (2 * largest))
            estimates, steps, errors, relatives, agrees = [], [], [], [], []
            unilateral, roundoffs, unilateral_roundoff = [], [], []
            for multiplier in multipliers:
                step = h * multiplier
                a, b = x.copy(), x.copy()
                if scheme != "backward":
                    a[index] += step
                if scheme != "forward":
                    b[index] -= step
                displacement = a[index] - b[index]
                if not np.isfinite(displacement) or displacement <= 0:
                    raise ValueError(
                        "finite-difference displacement is not representable"
                    )
                if np.any(a > hi) or np.any(b < lo):
                    raise ValueError("finite-difference probe outside domain")
                ra = base_r if np.array_equal(a, x) else call(a)[0]
                rb = base_r if np.array_equal(b, x) else call(b)[0]
                # Report derivatives in the fixed optimizer coordinate units.
                estimate = (ra - rb) / displacement * units[index]
                if not np.isfinite(estimate).all():
                    raise ValueError("nonfinite finite-difference estimate")
                e, rel, ok = _agreement(
                    estimate, base_j[:, index] * units[index], policy
                )
                estimates.append(estimate)
                steps.append(float(step))
                errors.append(e)
                relatives.append(rel)
                agrees.append(ok)
                if policy.precision_aware:
                    epsilon = np.finfo(float).eps * policy.precision_roundoff_factor
                    roundoffs.append(
                        epsilon
                        * (np.abs(ra) + np.abs(rb))
                        / displacement
                        * units[index]
                    )
                if scheme == "central":
                    forward = (ra - base_r) / (a[index] - x[index]) * units[index]
                    backward = (base_r - rb) / (x[index] - b[index]) * units[index]
                    unilateral.append(_rms(forward - backward))
                    if policy.precision_aware:
                        unilateral_roundoff.append(
                            _rms(
                                epsilon
                                * units[index]
                                * (
                                    (np.abs(ra) + np.abs(base_r))
                                    / (a[index] - x[index])
                                    + (np.abs(base_r) + np.abs(rb))
                                    / (x[index] - b[index])
                                )
                            )
                        )
            _, between, stable = _agreement(*estimates[:2], policy)
            # Central cancellation must not certify abs(x) at its kink. Smooth
            # curvature may disagree at finite h, but its discrepancy contracts.
            sides_ok = not unilateral or (
                unilateral[1]
                <= policy.derivative_absolute_tolerance
                + policy.derivative_relative_tolerance
                * _rms(base_j[:, index] * units[index])
                or unilateral[1] <= 0.75 * unilateral[0]
            )
            precision = (
                _precision_result(
                    estimates,
                    roundoffs,
                    unilateral,
                    unilateral_roundoff,
                    steps,
                    base_r,
                    base_j[:, index] * units[index],
                    policy,
                )
                if policy.precision_aware
                else {"passed": all(agrees) and stable and sides_ok}
            )
            rows.append(
                DerivativeDirection(
                    index=index,
                    scheme=scheme,
                    steps=tuple(steps[:2]),
                    error_rms=tuple(errors[:2]),
                    relative_errors=tuple(relatives[:2]),
                    between_steps_relative_error=between,
                    one_sided_disagreement=tuple(unilateral[:2])
                    if unilateral
                    else None,
                    **precision,
                )
            )
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    checked = tuple(row.index for row in rows)
    omitted = tuple(i for i in required if i not in checked)
    inconclusive = tuple(
        row.index for row in rows if row.precision_status == "inconclusive"
    )
    failed = tuple(
        row.index
        for row in rows
        if not row.passed and row.precision_status != "inconclusive"
    )
    status = (
        "failed"
        if failed
        else (
            "unavailable"
            if base_r is None
            else "incomplete"
            if omitted or error
            else "inconclusive"
            if inconclusive
            else "passed"
        )
    )
    return DerivativeCheck(
        status=status,
        reduced_vector=tuple(x),
        coordinate_units=tuple(units),
        base_residual_sha256=None if base_r is None else _digest(base_r),
        base_jacobian_sha256=None if base_j is None else _digest(base_j),
        base_training_nmse=None if base_r is None else float(np.mean(base_r**2)),
        lower_bounds_sha256=_digest(lo) if policy.precision_aware else None,
        upper_bounds_sha256=_digest(hi) if policy.precision_aware else None,
        policy=policy,
        required_indices=required,
        checked_indices=checked,
        omitted_indices=omitted,
        directions=tuple(rows),
        calls_started=started,
        calls_completed=completed,
        jacobian_calls_started=jacobian_calls,
        residual_only_calls_started=residual_calls,
        elapsed_seconds=monotonic() - begun,
        error=error,
        search_qualified=status == "passed"
        or (
            policy.precision_aware
            and status == "inconclusive"
            and all(
                row.gradient_error_bound
                <= policy.gradient_tolerance * policy.precision_gradient_fraction
                for row in rows
                if row.index in inconclusive
            )
        ),
        inconclusive_indices=inconclusive,
        failed_indices=failed,
        gradient_error_bound_inf=(
            max((row.gradient_error_bound for row in rows), default=0.0)
            if policy.precision_aware and not omitted and error is None
            else None
        ),
        parameter_information_qualified=status == "passed",
        **(
            {
                "scope": (
                    "Fixed six-scale finite differences at one endpoint. Gradient "
                    "error bounds assume smooth local truncation and the declared "
                    "floating-point "
                    "residual envelope; not rigorous interval bounds or a solver-error "
                    "certificate. Inconclusive columns may qualify search only, not "
                    "parameter information. Deadlines are cooperative."
                )
            }
            if policy.precision_aware
            else {}
        ),
    )


def assess(
    vector: ArrayLike,
    residual: ArrayLike,
    jacobian: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    verification: OriginalVerification | Mapping | None,
    derivative: DerivativeCheck,
    stop_reason: str | None,
    budget_exhausted: bool | None,
    history: Sequence[HistoryPoint | Mapping] = (),
    coordinate_units: ArrayLike | None = None,
    restarts: Sequence[RestartEvidence | Mapping] = (),
    policy: Policy | None = None,
) -> Assessment:
    """Require independent replay, current derivatives and accepted local progress.

    Neither an optimizer success flag, a budget stop, a small step alone, nor a
    stationary starting point qualifies. Repeated diverse restarts can strengthen
    observed reliability, but never override the local gates or certify a global fit.
    """
    policy = policy or Policy()
    x, lo, hi, units = _coordinates(vector, lower, upper, coordinate_units)
    r, j = _arrays(residual, jacobian, len(x))
    loss, scaled = float(np.mean(r * r)), j * units
    free = lo < hi
    product = scaled.T @ r
    tolerance = policy.bound_tolerance * np.maximum(units, np.abs(x))
    blocked = (
        ((x - lo <= tolerance) & (product > 0))
        | ((hi - x <= tolerance) & (product < 0))
        | ~free
    )
    projected = np.where(blocked, 0.0, product)
    gradient = float(np.max(2 * np.abs(projected) / len(r), initial=0))
    denominator = np.linalg.norm(scaled, axis=0) * np.linalg.norm(r)
    correlation = np.divide(
        np.abs(projected),
        denominator,
        out=np.zeros_like(projected),
        where=denominator > 0,
    )
    max_correlation = float(np.max(correlation, initial=0))
    reasons = []
    verified = False
    if verification is not None:
        v = OriginalVerification.model_validate(verification)
        loss_allowance = max(
            1e-12, 1e-8 * max(v.training_nmse, v.alternate_training_nmse)
        )
        verified = (
            v.original_equations_verified
            and np.array_equal(x, v.reduced_vector)
            and v.solver_prediction_discrepancy <= 1e-7
            and v.solver_loss_discrepancy <= loss_allowance
            and abs(v.training_nmse - v.alternate_training_nmse) <= loss_allowance
            and abs(loss - v.training_nmse) <= loss_allowance
        )
    if not verified:
        reasons.append("original_equation_rollout_unverified_or_mismatched")
    derivative_qualified = (
        (
            derivative.status == "passed"
            or (policy.precision_aware and derivative.search_qualified)
        )
        and np.array_equal(x, derivative.reduced_vector)
        and np.array_equal(units, derivative.coordinate_units)
        and _digest(r) == derivative.base_residual_sha256
        and _digest(j) == derivative.base_jacobian_sha256
        and derivative.policy.precision_aware == policy.precision_aware
        and derivative.policy.derivative_relative_tolerance
        <= policy.derivative_relative_tolerance
        and derivative.policy.derivative_absolute_tolerance
        <= policy.derivative_absolute_tolerance
        and set(derivative.checked_indices) == set(np.flatnonzero(free))
        and not derivative.omitted_indices
        and (
            not policy.precision_aware
            or (
                derivative.policy == policy
                and derivative.lower_bounds_sha256 == _digest(lo)
                and derivative.upper_bounds_sha256 == _digest(hi)
                and derivative.gradient_error_bound_inf is not None
                and not derivative.failed_indices
                and derivative.error is None
            )
        )
    )
    if not derivative_qualified:
        reasons.append("reduced_derivative_audit_unqualified_or_mismatched")
    gradient_upper = None
    unresolved_correlation = ()
    correlation_for_gate = max_correlation
    if policy.precision_aware and derivative_qualified:
        uncertainty = np.zeros(len(x))
        for direction in derivative.directions:
            uncertainty[direction.index] = direction.gradient_error_bound
        nominal = 2 * product / len(r)
        gradient_low, gradient_high = nominal - uncertainty, nominal + uncertainty
        # Projection is applied to the whole interval. A nominal outward gradient
        # at a bound cannot hide a potentially material inward gradient.
        worst = np.maximum(np.abs(gradient_low), np.abs(gradient_high))
        at_lower, at_upper = x - lo <= tolerance, hi - x <= tolerance
        worst = np.where(at_lower & ~at_upper, np.maximum(0, -gradient_low), worst)
        worst = np.where(at_upper & ~at_lower, np.maximum(0, gradient_high), worst)
        worst = np.where(free, worst, 0)
        gradient_upper = float(np.max(worst, initial=0))
        unresolved_correlation = derivative.inconclusive_indices
        resolved_correlation = correlation.copy()
        resolved_correlation[list(unresolved_correlation)] = 0
        correlation_for_gate = float(np.max(resolved_correlation, initial=0))
    if budget_exhausted is not False or stop_reason in {
        "budget_limited",
        "wall_clock_limit",
        "numerical_failure",
        "interrupted",
    }:
        reasons.append("search_incomplete_or_budget_status_unknown")
    if stop_reason not in {
        "step_or_gradient_converged",
        "gradient_converged",
        "stationary_poor_fit",
        "numerical_target_reached",
        "output_problem_solved",
        "no_free_coordinates",
    }:
        reasons.append("termination_not_supported")
    if gradient > policy.gradient_tolerance:
        reasons.append("projected_gradient_not_small")
    if gradient_upper is not None and gradient_upper > policy.gradient_tolerance:
        reasons.append("projected_gradient_uncertainty_not_small")
    if correlation_for_gate > policy.correlation_tolerance:
        reasons.append("residual_column_correlation_not_small")
    accepted = [HistoryPoint.model_validate(p) for p in history]
    accepted = [p for p in accepted if p.accepted and p.coordinate_step_norm > 0]
    counts = [p.evaluation_count for p in accepted if p.evaluation_count is not None]
    if counts != sorted(set(counts)):
        reasons.append("accepted_history_repeated_or_out_of_order")
    recent = accepted[-policy.history_window :]
    if np.any(free):
        if len(recent) < policy.history_window:
            reasons.append("accepted_progress_history_insufficient")
        elif (
            max(p.coordinate_step_norm for p in recent) > policy.step_tolerance
            or max(p.training_nmse for p in recent)
            - min(p.training_nmse for p in recent)
            > policy.loss_tolerance * max(1, loss)
            or abs(recent[-1].training_nmse - loss)
            > policy.loss_tolerance * max(1, loss)
        ):
            reasons.append("accepted_history_not_stable_at_endpoint")
    singular = np.linalg.svd(scaled[:, free], compute_uv=False)
    rank = (
        int(np.sum(singular > policy.rank_relative_tolerance * singular[0]))
        if len(singular)
        else 0
    )
    nfree = int(np.sum(free))
    condition = float(singular[0] / singular[-1]) if rank == nfree and nfree else None
    if condition is not None and not np.isfinite(condition):
        condition = None
    parameter_information_qualified = (
        derivative_qualified and derivative.parameter_information_qualified
    )
    rank_interval, jacobian_error, threshold_interval = None, None, None
    if policy.diagnostic_precision:
        envelopes = {
            row.index: row.jacobian_error_bound_l2 for row in derivative.directions
        }
        if derivative_qualified and all(
            envelopes.get(int(i)) is not None for i in np.flatnonzero(free)
        ):
            rank_interval, jacobian_error, threshold_interval = _rank_precision(
                singular,
                [envelopes[int(i)] for i in np.flatnonzero(free)],
                policy.rank_relative_tolerance,
            )
        parameter_information_qualified = (
            parameter_information_qualified
            and rank_interval is not None
            and rank_interval[0] == rank_interval[1] == rank
        )
    if policy.precision_aware and not parameter_information_qualified:
        condition = None
    records = [RestartEvidence.model_validate(item) for item in restarts]
    ids = [item.start_id for item in records]
    if len(set(ids)) != len(ids):
        raise ValueError("restart identities must be unique")
    complete = [
        item
        for item in records
        if item.predeclared
        and item.distinct_from_incumbent
        and item.status == "complete"
    ]
    improving = sum(
        item.verified_training_nmse < loss - policy.loss_tolerance * max(1, loss)
        for item in complete
    )
    if improving:
        reasons.append("predeclared_restart_has_better_verified_endpoint")
    classification = (
        "optimization_unresolved"
        if reasons
        else (
            "locally_converged_accurate_fit"
            if loss <= policy.accuracy_nmse
            else "locally_converged_poor_fit"
        )
    )
    return Assessment(
        classification=classification,
        policy=policy,
        reasons=tuple(reasons),
        training_nmse=loss,
        projected_gradient_inf=gradient,
        maximum_residual_column_correlation=max_correlation,
        active_bound_indices=tuple(int(i) for i in np.flatnonzero(blocked & free)),
        accepted_history_points=len(accepted),
        derivative_status=derivative.status,
        original_rollout_verified=verified,
        parameter_rank=rank,
        parameter_singular_values=tuple(singular),
        free_coordinates=nfree,
        parameter_condition_number=condition,
        parameter_information="no_free_coordinates"
        if not nfree
        else "derivative_precision_unresolved"
        if policy.precision_aware and not parameter_information_qualified
        else ("full_local_rank" if rank == nfree else "rank_deficient"),
        projected_gradient_upper_bound=gradient_upper,
        gradient_error_bound_inf=derivative.gradient_error_bound_inf
        if policy.precision_aware and derivative_qualified
        else None,
        correlation_unresolved_indices=unresolved_correlation,
        parameter_information_qualified=parameter_information_qualified,
        parameter_rank_interval=rank_interval,
        jacobian_error_bound_frobenius=jacobian_error,
        rank_threshold_interval=threshold_interval,
        complete_predeclared_restarts=len(complete),
        improving_predeclared_restarts=improving,
    )
