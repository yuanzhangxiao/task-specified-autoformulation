"""Bounded training searches, training-only starts, and original-equation scoring."""

from __future__ import annotations

from time import monotonic, process_time

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import least_squares, minimize
from scipy.sparse import csr_matrix

from autoformalism.fitting import remaining_portfolio_search as existing
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.research import meal_revision_metrics as meal

FLOOR = "basal_floor"
DECAY = "k_insulin_term_2_0_071bd10aef"
SOURCE = "k_insulin_term_1_0_4e94c3cad2"
DELAY = "k_insulin_delay"
DOMAIN_TOL = 1e-8


def starts(model, *, control=False):
    """Build starts from the archived parameters and public training scales."""
    warm = model.warm.copy()
    if control:
        warm[FLOOR] = 0.0
    candidates = {"inherited": warm}
    if model.branch == "c15-4" and not control:
        candidates["nested_zero_basal"] = warm | {FLOOR: 0.0}
        candidates["basal_width_from_training"] = warm | {
            "eps": float(np.median(model.observed[:, 0, model.channels.index("I")]))
        }
    elif model.branch == "c15-3":
        candidates["nested_zero_memory"] = warm | {"gain_M": 0.0}
        candidates["finite_meal_memory"] = warm | {
            "gain_M": 0.02,
            "k_M_rise": 1.0,
            "tau_M": 30.0,
            "c_I": 0.5 * warm["c_I"],
        }
    elif model.branch == "c15-2":
        glucose = model.observed[:, :, model.channels.index("Gp")]
        basal = float(np.median(model.observed[:, 0, model.channels.index("I")]))
        candidates["glucose_scaled_sigmoid"] = warm | {
            "Gp_thr": float(np.median(glucose)),
            "k_prod": 1.0 / max(float(np.std(glucose)), 1e-8),
            "prod_max": 2 * basal / 30,
            DECAY: (1.0 / 30 - 1) / warm["af_process_gain_2_0"],
        }
    elif model.branch == "c15-0":
        basal = np.median(model.observed[:, 0, :], axis=0)
        candidates["basal_balance_probe"] = warm | {
            DELAY: 0.0,
            SOURCE: float(
                basal[model.channels.index("I")] / basal[model.channels.index("Gp")]
            )
            / warm["af_process_gain_1_0"],
        }
    elif model.branch == "c21-2":
        candidates["unit_memory_rate"] = warm | {"k_X_decay": 1.0}
    for p in candidates.values():
        model.reduce(p)
    return candidates


def score_grid(model, values, subdivisions):
    data = (values[:, ::subdivisions] - model.observed) / model.scales
    minima = np.min(values, axis=1) / model.scales
    return float(np.mean(data**2)), bool(np.all(minima >= -DOMAIN_TOL))


def search(model, candidates, *, physical, control, seconds, calls, save):
    """Fit shared coefficients and initials and keep both endpoint categories."""
    if not (0 < seconds <= 1800 and type(calls) is int and 1 <= calls <= 2000):
        raise ValueError("invalid numerical allowance")
    start_time, cpu = monotonic(), process_time()
    spent_io = 0.0
    attempts = 0
    invalid = 0
    best = {}
    stage_best = None
    stage_loss = np.inf
    seed_scores = {}
    errors = []
    phase_status = "running"
    last_save = start_time
    effective_calls = 0
    solver_status = None

    def record():
        return {
            "status": phase_status,
            "solver_status": solver_status,
            "search_calls_attempted": attempts,
            "search_calls_started": attempts,
            "search_calls_completed": effective_calls,
            "search_calls_failed": invalid,
            "errors": errors.copy(),
            "seeds": dict(seed_scores),
            "best": {k: v.copy() for k, v in best.items()},
            "best_search_objective": float(stage_loss)
            if np.isfinite(stage_loss)
            else None,
            "objective": "NMSE + 100 * mean(scaled minimum shortfall squared)"
            if physical
            else "pooled training NMSE",
            "search_timing": {
                "numerical_active_wall_seconds": max(
                    0.0, monotonic() - start_time - spent_io
                ),
                "callback_wall_seconds": spent_io,
            },
            "cpu_seconds": process_time() - cpu,
            "checkpoint_wall_seconds": spent_io,
            "conditional_or_global_convergence_certified": False,
        }

    def persist():
        nonlocal spent_io, last_save
        begin = monotonic()
        save(record())
        last_save = monotonic()
        spent_io += last_save - begin

    def evaluate(point):
        nonlocal attempts, invalid, effective_calls, stage_best, stage_loss
        if (
            attempts >= calls
            or monotonic() - start_time - spent_io >= seconds
            or monotonic() - start_time >= seconds + 450
        ):
            raise TimeoutError("reserved search allowance exhausted")
        attempts += 1
        try:
            p = model.lift(point)
            y = model.fast(p)
            loss, admissible = score_grid(model, y, 2)
            diff = (y[:, ::2] - model.observed) / model.scales
            if physical:
                # A small interior search margin helps a numerically feasible checkpoint
                # pass the finer independent replay.
                shortfall = np.maximum(1e-5 - np.min(y, axis=1) / model.scales, 0.0)
                diff = np.concatenate(
                    (diff, np.sqrt(100 * len(model.time)) * shortfall[:, None, :]),
                    axis=1,
                )
            scalar = float(np.sum(diff**2) / model.observed.size)
            if not np.isfinite(scalar):
                raise ValueError("nonfinite training loss")
        except (ValueError, FloatingPointError, OverflowError, RuntimeError) as exc:
            invalid += 1
            if len(errors) < 3:
                errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            shape = (
                len(model.rows),
                len(model.time) + int(physical),
                len(model.channels),
            )
            return np.full(shape, 1e16).ravel(), 1e32
        effective_calls += 1
        improved = False
        for key in ("numerical", "admissible") if admissible else ("numerical",):
            if key not in best or loss < best[key]["fast_nmse"]:
                best[key] = {
                    "parameters": p,
                    "fast_nmse": loss,
                    "attempt": attempts,
                    "sampled_minima": dict(
                        zip(model.channels, y.min(axis=(0, 1)).tolist(), strict=True)
                    ),
                }
                improved = True
        if scalar < stage_loss:
            stage_loss, stage_best = scalar, point.copy()
        if improved or monotonic() - last_save >= 30:
            persist()
        return diff.ravel(), scalar

    try:
        for name, p in candidates.items():
            v = model.reduce(p)
            _, score = evaluate(v)
            seed_scores[name] = score
        if stage_best is None:
            raise ValueError("no finite search seed")
        origin = stage_best.copy()
        # All certified reduced coordinates are released; the clip uses value-only
        # original coordinates and skips exactly unused dynamics names.
        active = [
            j
            for j, n in enumerate(model.names)
            if model.lower[j] < model.upper[j]
            and (
                model.chart is not None
                or n in model.used_shared
                or n.startswith("trajectory:")
            )
            and not (control and n in (FLOOR, "eps"))
        ]
        if control and (FLOOR not in model.names or model.lift(origin)[FLOOR] != 0.0):
            raise ValueError("zero-basal control lost its fixed coordinate")
        units = np.maximum(np.abs(origin), 0.01)
        for j, n in enumerate(model.names):
            if n.startswith("trajectory:"):
                units[j] = max(units[j], 1.0)
            if n == DECAY:
                units[j] = max(abs(1 + origin[j]), 0.01)
            if n == "Gp_thr":
                units[j] = model.scales[model.channels.index("Gp")]
        ix = np.asarray(active, int)
        scales = units[ix]
        lower = 1 + (model.lower[ix] - origin[ix]) / scales
        upper = 1 + (model.upper[ix] - origin[ix]) / scales

        def residual(z):
            point = origin.copy()
            point[ix] = np.clip(
                origin[ix] + scales * (z - 1), model.lower[ix], model.upper[ix]
            )
            return evaluate(point)

        x0 = np.ones(len(ix))
        if model.branch == "c15-0":
            out = minimize(
                lambda z: residual(z)[1],
                x0,
                method="Powell",
                bounds=list(zip(lower, upper, strict=True)),
                options={"xtol": 1e-6, "ftol": 1e-9, "maxfev": calls},
            )
            solver_status = {
                "method": "derivative-free Powell",
                "raw_success": bool(out.success),
                "message": str(out.message),
            }
        else:
            ntime = len(model.time) + int(physical)
            width = ntime * len(model.channels)
            structure = np.ones((len(model.rows) * width, len(ix)), bool)
            row_index = {r.trajectory_id: j for j, r in enumerate(model.rows)}
            for j, i in enumerate(ix):
                n = model.names[i]
                if n.startswith("trajectory:"):
                    idx = row_index[n.split(":")[1]]
                    structure[:, j] = False
                    structure[idx * width : (idx + 1) * width, j] = True
            out = least_squares(
                lambda z: residual(z)[0],
                x0,
                bounds=(lower, upper),
                jac_sparsity=csr_matrix(structure),
                x_scale="jac",
                max_nfev=calls,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-9,
            )
            solver_status = {
                "method": "bounded trust-region LS with sparse finite differences",
                "raw_success": bool(out.success),
                "message": str(out.message),
                "iterations": out.nfev,
            }
        phase_status = "solver_returned" if out.success else "solver_stopped"
    except TimeoutError:
        phase_status = "budget_exhausted"
    persist()
    return record()


def affine_root_seed(model, point, seconds=120):
    """Use the existing certified conditional root solve on the two affine families."""
    original, oracle = model.chart.original, model.chart
    start = oracle.parameters(model.reduce(point))
    result = existing.qualify(
        original, oracle, start, seconds, lambda _: None, "affine"
    )
    if not result.get("passed"):
        return {
            "passed": False,
            "reason": result.get("error", "root profile unavailable"),
        }
    return {
        "passed": True,
        "parameters": result["original_parameters"],
        "training_nmse": result["training_nmse"],
        "certificate_sha256": result["certificate_sha256"],
    }


def diagnostics(model, point, data, *, seconds=300):
    """Score original equations and report physical checks independently of search."""
    started = monotonic()
    values, states = model.original(point, 8, seconds=seconds)
    observed_grid = values[:, ::8]
    loss = model.nmse(observed_grid)
    errors = (observed_grid - model.observed) / model.scales
    metrics = meal.summarize(
        model.training,
        model.channels,
        dict(zip(model.channels, model.scales.tolist(), strict=True)),
        errors.ravel(),
    )
    minima = values.min(axis=(0, 1))
    qualifiers = {
        c: {
            "minimum": float(minima[j]),
            "below_tolerance": int(
                np.sum(values[:, :, j] < -DOMAIN_TOL * model.scales[j])
            ),
            "negative_trajectories_at_source_threshold": int(
                np.sum(np.min(observed_grid[:, :, j], axis=1) < -1e-4)
            ),
        }
        for j, c in enumerate(model.channels)
    }
    row_index = {r.trajectory_id: i for i, r in enumerate(model.rows)}
    w = data["boundary"]
    tid = w["trajectory"]
    i = row_index[tid]
    end = states[i, -1].copy()
    p = {n: point[n] for n in model.shared}
    zero = trajectory_forcing(model.model, model.rows[i])
    if any(zero.value(n, model.time[-1]) != 0 for n in model.inputs):
        raise ValueError("tail requires zero terminal forcing")

    def force(t, x):
        return model.equations.rhs(t, x, p, zero)

    si = model.states.index("I")
    own = end.copy()
    own[si] = 0.0
    own_rate = float(force(model.time[-1], own)[si])
    common = None
    if model.branch.startswith("c15"):
        state = dict(w["state_at_boundary"])
        if "M" in model.states:
            state["M"] = float(end[model.states.index("M")])
        common = float(
            force(w["time_min"], np.array([state[n] for n in model.states]))[si]
        )
    # Specify the diagnostic continuation inputs explicitly.
    from autoformalism.expressions.compiler import PiecewiseLinearForcing

    off = PiecewiseLinearForcing(
        [300.0, 330.0],
        {n: [0.0, 0.0] for n in model.inputs},
        allowed_channels=frozenset(model.inputs),
    )

    def outside(t, y):
        if monotonic() - started > seconds:
            raise TimeoutError("tail diagnostic deadline")
        return model.equations.rhs(t, y, p, off)

    tail = solve_ivp(
        outside,
        (300.0, 330.0),
        end,
        method="Radau",
        t_eval=np.arange(300.0, 331.0),
        rtol=1e-9,
        atol=1e-11,
    )
    if not tail.success or tail.y.shape != (model.n, 31):
        raise ValueError("zero-input diagnostic integration failed")
    shifts = [
        v
        for v in metrics["rows"]
        if v["matches_shifted_90g_schedule"] and v["external_insulin_zero"]
    ]
    trajectories = np.mean(errors**2, axis=(1, 2))
    center = np.median(trajectories)
    result = {
        "parameters": point,
        "training_nmse": loss,
        "target_nmse": metrics["target_training_nmse"],
        "trajectory_median_nmse": float(center),
        "trajectory_unscaled_mad": float(np.median(abs(trajectories - center))),
        "sampled_domain_passed": all(
            v["below_tolerance"] == 0 for v in qualifiers.values()
        ),
        "domain": qualifiers,
        "domain_times_per_channel": values.shape[0] * values.shape[1],
        "concentration_tolerance_scaled": DOMAIN_TOL,
        "own_depletion_rate": own_rate,
        "common_depletion_rate": common,
        "extension_150g": {
            "I300": float(end[si]),
            "I301": float(tail.y[si, 1]),
            "I330": float(tail.y[si, -1]),
            "minimum": float(tail.y[si].min()),
            "observed_I300": model.rows[i].targets["I"][-1],
            "inputs": "zero ingestion and infusion",
            "observations_after_300": False,
        },
        "shifted_90g": shifts,
        "no_meal": metrics["no_meal_controls"],
        "original_equations_replayed": True,
        "validation_or_test_access": False,
        "continuous_positivity_certified": False,
        "convergence_certified": False,
        "parameter_sha256": content_sha256(point),
    }
    windows = []
    for tid, ranges in (
        ("train_000", [(0.0, 60.0)]),
        ("train_003", [(v["start"], v["stop"]) for v in data["glucose_windows"]]),
    ):
        ri = row_index[tid]
        for a, b in ranges:
            sl = (model.time >= a) & (model.time <= b)
            t = model.time[sl]

            def supply(y, sl=sl, t=t):
                G, U = (
                    y[sl, model.channels.index("Gp")],
                    y[sl, model.channels.index("U")],
                )
                return float((G[-1] - G[0] + trapezoid(U, t)) / (t[-1] - t[0]))

            windows.append(
                {
                    "trajectory": tid,
                    "start": a,
                    "stop": b,
                    "observed_mean_net_supply": supply(model.observed[ri]),
                    "fitted_mean_net_supply": supply(observed_grid[ri]),
                }
            )
    result["glucose_supply_windows"] = windows
    result["_observed_predictions"] = observed_grid
    return result
