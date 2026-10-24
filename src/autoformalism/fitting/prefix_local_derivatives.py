"""Local shrinking finite-difference diagnostics, separate from historical policy.

Only a bounded set of shrinking local probes is used. Returned uncertainty is
conditional numerical evidence under a smooth-truncation/roundoff model; it does
not certify an arbitrary black-box residual or turn a zero column into identified
parameter information. The caller owns durable operation budgets and receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from autoformalism.fitting import prefix_fit_adequacy as a
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial


@dataclass
class _Probe:
    scale_index: int
    step: float
    scheme: str
    estimate: a.Array
    roundoff: a.Array
    unilateral: float | None
    unilateral_roundoff: float
    order: int = 1


def _initial_step(value, lower, upper, unit, policy, local_scale=None):
    """Cap the physical perturbation by local magnitude and declared domain.

    By default tiny positive time constants remain magnitude-capped. The separate
    bounds-aware policy accepts a caller-declared scale for each coordinate, so
    nonsingular gains near zero need not inherit that reciprocal-scale restriction.
    No data-dependent increase follows a failed probe.
    """
    # A declared physical radius is distinct from the optimizer's coordinate
    # unit. Capping a large nonsingular coefficient by that unit can produce
    # roundoff-dominated differences far below a meaningful relative step.
    scales = [local_scale] if local_scale is not None else [unit]
    if local_scale is None and value != 0:
        scales.append(abs(value))
    width = upper - lower
    if np.isfinite(width):
        scales.append(width)
    scale = min(scales)
    proposed = max(
        policy.derivative_absolute_step * unit,
        policy.derivative_relative_step * max(abs(value), unit),
    )
    step = min(proposed, policy.local_relative_step_cap * scale)
    left, right = value - lower, upper - value
    if policy.local_bounds_aware:
        # Preserve a meaningful local radius near a bound. A second-order
        # inward stencil needs two steps of room, not symmetric distance from
        # an arbitrarily tiny positive coefficient to its lower bound.
        if min(left, right) < step:
            step = min(step / 2, max(left, right) / 2)
    elif left > 0 and right > 0:
        step = min(step, left / 2, right / 2)
    else:
        step = min(step, max(left, right) / 2)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("finite local probe size is not representable")
    return step


def _pair(coarse, fine, residual, column, policy, *, residual_error=None):
    """Assess one same-scheme geometric pair without using larger-scale data."""
    central = fine.scheme == "central"
    order_factor = 4 if central or fine.order == 2 else 2
    radius = fine.roundoff + 4 / (order_factor - 1) * (
        np.abs(fine.estimate - coarse.estimate) + fine.roundoff + coarse.roundoff
    )
    uncertainty = a._rms(radius)
    error, relative, agrees = a._agreement(fine.estimate, column, policy)
    _, between, stable = a._agreement(coarse.estimate, fine.estimate, policy)
    scale = max(a._rms(column), a._rms(fine.estimate))
    sides_ok = not central or (
        fine.unilateral
        <= fine.unilateral_roundoff + policy.derivative_relative_tolerance * scale
        or fine.unilateral
        <= 0.75 * coarse.unilateral
        + fine.unilateral_roundoff
        + 0.75 * coarse.unilateral_roundoff
    )
    contradiction = error > uncertainty + policy.derivative_relative_tolerance * scale
    # Zero analytic/FD columns give no positive information about parameter rank.
    # A small estimated gradient envelope can still permit a search to proceed.
    resolved = scale > 0 and uncertainty <= policy.derivative_relative_tolerance * scale
    status = (
        "failed"
        if stable and (contradiction or not sides_ok)
        else "passed"
        if agrees and stable and resolved and sides_ok
        else "inconclusive"
    )
    gradient_error = float(
        2 * np.mean(np.abs(residual) * (np.abs(column - fine.estimate) + radius))
    )
    gradient_radius = float(2 * np.mean(np.abs(residual) * radius))
    if residual_error is not None:
        # The gradient contains both residual and Jacobian factors. Accurate
        # derivatives alone cannot resolve a cancellation-contaminated residual.
        gradient_error += float(
            2 * np.mean(residual_error * (np.abs(fine.estimate) + radius))
        )
        gradient_radius += float(2 * np.mean(residual_error * radius))
    if not np.isfinite(gradient_error):
        raise ValueError("nonfinite local derivative gradient uncertainty")
    return {
        "status": status,
        "stable": stable,
        "sides_ok": sides_ok,
        "uncertainty": uncertainty,
        "gradient_error": gradient_error,
        "gradient_radius": gradient_radius,
        "column_error_l2": float(
            np.linalg.norm(np.abs(column - fine.estimate) + radius)
        ),
        "between": between,
        "relative": relative,
        "error": error,
        "roundoff_dominated": uncertainty > policy.derivative_relative_tolerance * scale
        and a._rms(fine.estimate - coarse.estimate)
        <= a._rms(fine.roundoff + coarse.roundoff),
        "evaluation_floor_dominated": a._rms(fine.estimate - coarse.estimate)
        <= a._rms(fine.roundoff + coarse.roundoff),
    }


def _projected_uncertainty(pair, residual, column, value, lower, upper, unit, policy):
    """Uncertainty in the bound-projected gradient, in optimizer coordinates."""
    nominal = float(2 * np.mean(residual * column))
    error = pair["gradient_error"]
    tolerance = policy.bound_tolerance * max(unit, abs(value))
    low, high = nominal - error, nominal + error
    at_lower, at_upper = value - lower <= tolerance, upper - value <= tolerance
    if at_lower and not at_upper:
        return max(0.0, -low) - max(0.0, -nominal)
    if at_upper and not at_lower:
        return max(0.0, high) - max(0.0, nominal)
    return error


def _direction(index, probes, pair, reference, column, initial_step, policy):
    """Save the chosen finite pair together with every finite probe so far."""
    coarse, fine = probes[reference - 1 : reference + 1]
    agreements = [a._agreement(p.estimate, column, policy) for p in (coarse, fine)]
    return a.DerivativeDirection(
        index=index,
        scheme=fine.scheme,
        steps=(coarse.step, fine.step),
        error_rms=tuple(v[0] for v in agreements),
        relative_errors=tuple(v[1] for v in agreements),
        between_steps_relative_error=pair["between"],
        one_sided_disagreement=(coarse.unilateral, fine.unilateral)
        if fine.scheme == "central"
        else None,
        passed=pair["status"] == "passed",
        precision_status=pair["status"],
        reference_probe_index=reference,
        derivative_uncertainty_rms=pair["uncertainty"],
        gradient_error_bound=pair["gradient_error"],
        **(
            {
                "projected_gradient_error_bound": pair["projected_gradient_error"],
                "jacobian_error_bound_l2": pair["column_error_l2"],
            }
            if policy.diagnostic_precision
            else {}
        ),
        precision_probes=tuple(
            a.PrecisionProbe(
                multiplier=p.step / initial_step,
                step=p.step,
                derivative_rms=a._rms(p.estimate),
                error_rms=a._rms(p.estimate - column),
                roundoff_rms=a._rms(p.roundoff),
            )
            for p in probes
        ),
    )


def _best_pair(previous, candidate, *, diagnostic=False):
    """Keep the least uncertain smooth pair without hiding resolved failures.

    Selection uses estimated truncation/roundoff, not agreement with the analytic
    column. A newly observed unresolved kink overrides earlier smooth evidence;
    later smaller probes may resolve such a coarse-scale curvature warning.
    """
    if previous is None or candidate[0]["status"] == "failed":
        return candidate
    old, new = previous[0], candidate[0]
    if new["status"] == "passed" and not diagnostic:
        return candidate
    if old["status"] == "failed":
        return candidate if new["sides_ok"] else previous
    metric = "gradient_radius" if diagnostic else "uncertainty"
    old_key = (not old["sides_ok"], not old["stable"], old[metric])
    new_key = (not new["sides_ok"], not new["stable"], new[metric])
    return candidate if new_key < old_key else previous


def local_derivative_check(
    evaluate,
    vector,
    lower,
    upper,
    deadline,
    *,
    coordinate_units,
    policy,
    directions=None,
    residual_only=None,
    local_coordinate_scales=None,
    finite_difference_probe=None,
    residual_error_bound=None,
):
    """Check all nonfixed coordinates with local, bounded shrinking differences.

    Contract errors abort; only typed numerical failures (or shape-valid nonfinite
    oracle values) permit a smaller probe. Finite earlier directions and all probe
    attempts survive failures. A failed or partial audit never qualifies search.
    A valid pair stops a direction early, avoiding unconditional six-scale work.
    The opt-in probe callback supplies already formed differences in the declared
    units; a residual-error callback contributes a normalized residual envelope
    to the gradient bound without altering the Jacobian/rank tolerance.
    """
    x, lo, hi, units = a._coordinates(vector, lower, upper, coordinate_units)
    if local_coordinate_scales is not None and not policy.local_bounds_aware:
        raise ValueError("local coordinate scales require local_bounds_aware")
    scales = (
        np.where(x == 0, units, np.minimum(np.abs(x), units))
        if local_coordinate_scales is None
        else np.asarray(local_coordinate_scales, dtype=float)
    )
    if scales.shape != x.shape or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("local coordinate scales must be finite positive full vector")
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
    base_r, base_j, rows, records, error = None, None, {}, [], None
    probe_call_times = []

    def call(v, *, full=False):
        nonlocal started, completed, jacobian_calls, residual_calls
        if monotonic() >= deadline:
            raise TimeoutError("local derivative audit deadline reached")
        call_start = monotonic()
        started += 1
        if full or residual_only is None:
            jacobian_calls += 1
            raw_r, raw_j = evaluate(v.copy(), deadline)
            r, j = np.asarray(raw_r, dtype=float), np.asarray(raw_j, dtype=float)
            if r.ndim != 1 or not len(r) or j.shape != (len(r), len(x)):
                raise ValueError("invalid residual or reduced Jacobian shape")
        else:
            residual_calls += 1
            r, j = np.asarray(residual_only(v.copy(), deadline), dtype=float), None
            if r.ndim != 1 or not len(r):
                raise ValueError("invalid residual-only oracle shape")
        if base_r is not None and r.shape != base_r.shape:
            raise ValueError("oracle residual size changed across probes")
        if not np.isfinite(r).all() or (j is not None and not np.isfinite(j).all()):
            raise NumericalInvalidTrial("nonfinite local derivative oracle result")
        if not np.isfinite(np.mean(r * r)):
            raise NumericalInvalidTrial("nonfinite local derivative residual loss")
        completed += 1
        if not full:
            # The initial sensitivity calculation may be much more expensive
            # than a residual-only probe. Reserve future work using actual probe
            # calls, not that unrelated one-time Jacobian construction.
            probe_call_times.append(monotonic() - call_start)
        if monotonic() >= deadline:
            raise TimeoutError("local derivative oracle returned after deadline")
        return r, j

    def probe(index, scale_index, step):
        left, right = x[index] - lo[index], hi[index] - x[index]
        scheme = (
            "central"
            if min(left, right) >= step
            else ("forward" if right >= left else "backward")
        )
        metadata = {
            "index": index,
            "scale_index": scale_index,
            "scheme": scheme,
            "step": step,
        }
        try:
            if finite_difference_probe is not None:
                result = finite_difference_probe(index, scale_index, step, deadline)
                metadata["scheme"] = result.scheme
                return save_probe(
                    metadata,
                    result.estimate,
                    result.roundoff,
                    result.unilateral,
                    result.unilateral_roundoff,
                    order=result.order,
                )
            pa, pb = x.copy(), x.copy()
            if policy.local_bounds_aware and scheme != "central":
                sign = 1 if scheme == "forward" else -1
                pa[index] += sign * step
                pb[index] += sign * 2 * step
                h1 = sign * (pa[index] - x[index])
                h2 = sign * (pb[index] - x[index])
                if not np.isfinite(h2) or not (0 < h1 < h2):
                    raise ValueError(
                        "finite local probe displacement is not representable"
                    )
                if (
                    np.any(pa < lo)
                    or np.any(pa > hi)
                    or np.any(pb < lo)
                    or np.any(pb > hi)
                ):
                    raise ValueError("finite local derivative probe outside domain")
                r1, r2 = call(pa)[0], call(pb)[0]
                w1, w2 = h2 / (h1 * (h2 - h1)), -h1 / (h2 * (h2 - h1))
                estimate = (
                    sign * units[index] * (w1 * (r1 - base_r) + w2 * (r2 - base_r))
                )
                epsilon = np.finfo(float).eps * policy.precision_roundoff_factor
                roundoff = (
                    epsilon
                    * units[index]
                    * (
                        abs(w1 + w2) * np.abs(base_r)
                        + abs(w1) * np.abs(r1)
                        + abs(w2) * np.abs(r2)
                    )
                )
                unilateral, side_roundoff = None, 0.0
                return save_probe(
                    metadata, estimate, roundoff, unilateral, side_roundoff, order=2
                )
            if scheme != "backward":
                pa[index] += step
            if scheme != "forward":
                pb[index] -= step
            da, db = pa[index] - x[index], x[index] - pb[index]
            displacement = da + db
            if (
                not np.isfinite(displacement)
                or displacement <= 0
                or (scheme == "central" and (da <= 0 or db <= 0))
            ):
                raise ValueError("finite local probe displacement is not representable")
            if np.any(pa > hi) or np.any(pb < lo):
                raise ValueError("finite local derivative probe outside domain")
            ra = base_r if np.array_equal(pa, x) else call(pa)[0]
            rb = base_r if np.array_equal(pb, x) else call(pb)[0]
            epsilon = np.finfo(float).eps * policy.precision_roundoff_factor
            if scheme == "central":
                forward = (ra - base_r) / da * units[index]
                backward = (base_r - rb) / db * units[index]
                estimate = (db * forward + da * backward) / displacement
                roundoff = (
                    epsilon
                    * units[index]
                    / displacement
                    * (
                        db * (np.abs(ra) + np.abs(base_r)) / da
                        + da * (np.abs(base_r) + np.abs(rb)) / db
                    )
                )
                # Equal symmetric steps cancel the base residual from the
                # centered difference, so avoid charging its roundoff twice.
                if da == db:
                    roundoff = (
                        epsilon
                        * (np.abs(ra) + np.abs(rb))
                        / displacement
                        * units[index]
                    )
                unilateral = a._rms(forward - backward)
                side_roundoff = a._rms(
                    epsilon
                    * units[index]
                    * (
                        (np.abs(ra) + np.abs(base_r)) / da
                        + (np.abs(base_r) + np.abs(rb)) / db
                    )
                )
            else:
                estimate = (ra - rb) / displacement * units[index]
                roundoff = (
                    epsilon * (np.abs(ra) + np.abs(rb)) / displacement * units[index]
                )
                unilateral, side_roundoff = None, 0.0
            return save_probe(
                metadata,
                estimate,
                roundoff,
                unilateral,
                side_roundoff,
                order=2 if scheme == "central" else 1,
            )
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as exc:
            records.append(
                a.LocalProbeRecord(
                    **metadata,
                    status="numerical_failure"
                    if isinstance(exc, NumericalInvalidTrial)
                    else "deadline"
                    if isinstance(exc, TimeoutError)
                    else "unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

    def save_probe(metadata, estimate, roundoff, unilateral, side_roundoff, *, order):
        """Retain one finite stencil, including its truncation order internally."""
        if not np.isfinite(estimate).all() or not np.isfinite(roundoff).all():
            raise NumericalInvalidTrial("nonfinite local finite-difference estimate")
        records.append(
            a.LocalProbeRecord(
                **metadata,
                status="complete",
                derivative_rms=a._rms(estimate),
                error_rms=a._rms(
                    estimate - base_j[:, metadata["index"]] * units[metadata["index"]]
                ),
                roundoff_rms=a._rms(roundoff),
            )
        )
        return _Probe(
            metadata["scale_index"],
            metadata["step"],
            metadata["scheme"],
            estimate,
            roundoff,
            unilateral,
            side_roundoff,
            order,
        )

    try:
        base_r, base_j = call(x, full=True)
        residual_error = (
            None
            if residual_error_bound is None
            else np.asarray(residual_error_bound(), dtype=float)
        )
        if residual_error is not None and (
            residual_error.shape != base_r.shape
            or not np.isfinite(residual_error).all()
            or np.any(residual_error < 0)
        ):
            raise ValueError("invalid residual uncertainty envelope")
        if residual_only is not None and not np.array_equal(call(x)[0], base_r):
            raise ValueError(
                "residual-only callback differs from audited oracle at base"
            )
        for position, index in enumerate(selected):
            initial_step = _initial_step(
                x[index],
                lo[index],
                hi[index],
                units[index],
                policy,
                scales[index] if policy.local_bounds_aware else None,
            )
            column, probes = base_j[:, index] * units[index], []
            selected_pair = None
            for scale_index in range(policy.local_max_scales):
                step = initial_step * 0.5**scale_index
                # Reserve estimated minimum work for remaining directions. This
                # controls optional shrinking only; actual calls share one deadline.
                if len(probes) >= 2 and probe_call_times:
                    remaining = len(selected) - position - 1
                    reserve = 4 * remaining * max(probe_call_times)
                    if deadline - monotonic() <= reserve + 2 * max(probe_call_times):
                        break
                try:
                    current = probe(index, scale_index, step)
                except NumericalInvalidTrial:
                    continue
                probes.append(current)
                if len(probes) < 2:
                    continue
                coarse, fine = probes[-2:]
                if (
                    fine.scheme != coarse.scheme
                    or fine.scale_index != coarse.scale_index + 1
                ):
                    continue
                pair = _pair(
                    coarse,
                    fine,
                    base_r,
                    column,
                    policy,
                    residual_error=residual_error,
                )
                if policy.diagnostic_precision:
                    pair["projected_gradient_error"] = _projected_uncertainty(
                        pair,
                        base_r,
                        column,
                        x[index],
                        lo[index],
                        hi[index],
                        units[index],
                        policy,
                    )
                selected_pair = (
                    _best_pair(
                        selected_pair,
                        (pair, len(probes) - 1),
                        diagnostic=policy.diagnostic_precision,
                    )
                    if policy.local_bounds_aware
                    else (pair, len(probes) - 1)
                )
                row = _direction(
                    index, probes, *selected_pair, column, initial_step, policy
                )
                # Preserve finite evidence through later failures. Bounds-aware
                # mode retains the least uncertain smooth pair, while a resolved
                # contradiction or unresolved kink cannot be hidden by selection.
                rows[index] = row
                if pair["status"] == "passed" and (
                    not policy.diagnostic_precision
                    or row.projected_gradient_error_bound
                    <= policy.gradient_tolerance * policy.precision_gradient_fraction
                ):
                    break
                if pair["status"] == "failed" and pair["sides_ok"]:
                    break  # Stable, resolved disagreement with the analytic column.
                floor = (
                    pair["evaluation_floor_dominated"]
                    if policy.diagnostic_precision
                    else pair["roundoff_dominated"]
                )
                if floor and pair["sides_ok"]:
                    break  # Shrinking further only amplifies the evaluation floor.
            if index not in rows:
                # Other directions may still be checked; this one remains omitted.
                continue
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    checked = tuple(rows)
    omitted = tuple(i for i in required if i not in rows)
    inconclusive = tuple(
        i for i, row in rows.items() if row.precision_status == "inconclusive"
    )
    failed = tuple(i for i, row in rows.items() if row.precision_status == "failed")
    status = (
        "failed"
        if failed
        else "unavailable"
        if base_r is None
        else "incomplete"
        if omitted or error
        else "inconclusive"
        if inconclusive
        else "passed"
    )
    search_qualified = status == "passed" or (
        status == "inconclusive"
        and all(
            rows[i].gradient_error_bound
            <= policy.gradient_tolerance * policy.precision_gradient_fraction
            for i in inconclusive
        )
    )
    information_qualified = status == "passed"
    if policy.diagnostic_precision and information_qualified:
        singular = np.linalg.svd(
            base_j[:, required] * units[list(required)], compute_uv=False
        )
        interval, _, _ = a._rank_precision(
            singular,
            [rows[i].jacobian_error_bound_l2 for i in required],
            policy.rank_relative_tolerance,
        )
        information_qualified = interval[0] == interval[1]
    return a.DerivativeCheck(
        status=status,
        reduced_vector=tuple(x),
        coordinate_units=tuple(units),
        base_residual_sha256=None if base_r is None else a._digest(base_r),
        base_jacobian_sha256=None if base_j is None else a._digest(base_j),
        base_training_nmse=None if base_r is None else float(np.mean(base_r**2)),
        lower_bounds_sha256=a._digest(lo),
        upper_bounds_sha256=a._digest(hi),
        policy=policy,
        required_indices=required,
        checked_indices=checked,
        omitted_indices=omitted,
        directions=tuple(rows.values()),
        calls_started=started,
        calls_completed=completed,
        jacobian_calls_started=jacobian_calls,
        residual_only_calls_started=residual_calls,
        elapsed_seconds=monotonic() - begun,
        error=error,
        search_qualified=search_qualified,
        inconclusive_indices=inconclusive,
        failed_indices=failed,
        gradient_error_bound_inf=max(
            (row.gradient_error_bound for row in rows.values()), default=0.0
        )
        if not omitted and error is None
        else None,
        parameter_information_qualified=information_qualified,
        gradient_precision_qualified=(
            search_qualified
            and all(
                row.projected_gradient_error_bound
                <= policy.gradient_tolerance * policy.precision_gradient_fraction
                for row in rows.values()
            )
            if policy.diagnostic_precision
            else None
        ),
        local_probe_attempts=tuple(records),
        local_coordinate_scales=tuple(scales) if policy.local_bounds_aware else (),
        scope=(
            "Bounded local shrinking finite differences at one endpoint. No probe "
            "enlargement or global smoothness claim. Gradient uncertainty assumes "
            "smooth local truncation and a floating-point residual envelope; it is "
            "not a rigorous solver-error bound. Unresolved and exactly zero columns "
            "can qualify search only, not parameter information. All nonfixed "
            "coordinates are required; deadlines are cooperative."
        ),
    )
