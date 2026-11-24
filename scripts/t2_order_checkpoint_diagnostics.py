"""Independent public-training diagnostics for the delayed disposal model."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.integrate import trapezoid

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.research.trajectory_function_fitting import _Rollout
from autoformalism.staged_topology import content_hash
from scripts import t2_initial_memory_forecast as forecast
from scripts.fit_t2_disposal_constrained import DOMAIN_TOL
from scripts.positive_insulin_refit_inputs import TARGETS, TRAIN_IDS
from scripts.t2_challenge_folds import aggregate
from scripts.t2_disposal_coupled_replay import parameter_violations
from scripts.validate_positive_insulin_transfer import Inputs, mesh

MINUTE = 15
LATER_DISTINGUISHABLE_GAP = 0.01  # mg/kg/min, declared prior to inspecting the outcomes
CHECKPOINT_RTOL = 1e-7  # maximum difference in training SD units
STEPS = 8


def _comparisons(ids, observed, predicted, fraction, allowance):
    """Evaluate every distinguishable pair and the absolute errors in native units."""
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if (
        not len(ids)
        or len(ids) != len(set(ids))
        or y.shape != (len(ids),)
        or y.shape != p.shape
        or not np.isfinite([y, p]).all()
        or not 0 <= fraction <= 1
        or not np.isfinite(allowance)
        or allowance < 0
    ):
        raise ValueError("Invalid event observations or ordering settings")
    details = []
    for hi, lo in zip(*np.where((y[:, None] - y) > allowance), strict=True):
        obs_gap, pred_gap = y[hi] - y[lo], p[hi] - p[lo]
        slack = float(pred_gap - fraction * obs_gap)
        details.append(
            {
                "high": ids[hi],
                "low": ids[lo],
                "observed_gap": float(obs_gap),
                "predicted_gap": float(pred_gap),
                "signed_slack": slack,
                "violated": slack < -DOMAIN_TOL,
            }
        )
    worst = min(details, key=lambda pair: pair["signed_slack"]) if details else None
    i = int(np.argmax(abs(p - y)))
    return {
        "observed_gap_allowance": allowance,
        "required_fraction": fraction,
        "feasibility_tolerance": DOMAIN_TOL,
        "pairs": len(details),
        "reversed_pairs": sum(pair["predicted_gap"] < -DOMAIN_TOL for pair in details),
        "violations": sum(pair["violated"] for pair in details),
        "worst_signed_slack": worst["signed_slack"] if worst else None,
        "worst_order_violation": max(0.0, -worst["signed_slack"]) if worst else 0.0,
        "worst_pair": worst,
        "worst_absolute_error": float(abs(p[i] - y[i])),
        "worst_absolute_event": ids[i],
        "pair_details": details,
    }


def _event(pub: dict, end: float | None) -> Inputs:
    """Truncate public forcings and time together before constructing a rollout mesh."""
    t = np.asarray(pub["time"], dtype=float)
    stop = len(t) if end is None else int(np.searchsorted(t, end)) + 1
    if end is not None and (stop > len(t) or t[stop - 1] != end):
        raise ValueError("Missing predeclared checkpoint on the observed grid")
    supplied = {
        k: np.asarray(v[:stop], dtype=float) for k, v in pub["supplied"].items()
    }
    initial = pub["initial"] | {k: vals[0] for k, vals in supplied.items()}
    return Inputs(pub["event"], t[:stop], supplied, initial)


def _flow(engine, model, event):
    """Propagate from public initials using only the supplied prefix."""
    if (
        set(engine.readings) - event.initial.keys()
        or set(engine.forcings) - event.supplied.keys()
    ):
        raise ValueError("A required initializer or forcing is unavailable")
    t = mesh(event, model["knots"])
    u = np.array([np.interp(t, event.time, event.supplied[k]) for k in engine.forcings])
    p = [model["formula"]["parameters"][k] for k in engine.parameters]
    z = [event.initial[k] for k in engine.readings]
    args = (z, u, np.diff(t)[None, :])
    values, _ = engine.run([(engine._flow(len(t)), args)], p)[0]
    return values[:, np.searchsorted(t, event.time)], len(t) - 1


def _affine_insulin_drive(model, engine):
    """Compile the declared action ODE and verify it is affine in insulin alone."""
    ca = engine.ca
    sources = model["graph"]["sources"]
    if set(sources["I_act"]) != {"I"} or set(sources["U"]) != {
        "Gp",
        "Gt",
        "Uii",
        "I_act",
    }:
        raise ValueError("Conditional reconstruction is defined for the delayed graph")
    insulin = ca.SX.sym("measured_insulin")
    rate = _translate(
        ca,
        RestrictedParser().parse(
            model["formula"]["equations"]["I_act"], location="I_act"
        ),
        model["formula"]["parameters"] | {"I": insulin},
    )
    if ca.depends_on(ca.jacobian(rate, insulin), insulin):
        raise ValueError("Conditional trapezoid requires an affine insulin action law")
    return ca.Function("measured_action_rate", [insulin], [rate])


def _conditional(row, event, model, engine, drive):
    """Read disposal at minute 15 given measured plasma glucose and insulin history."""
    n = len(event.time)
    insulin = np.asarray(row["targets"]["I"][:n])
    p = [model["formula"]["parameters"][k] for k in engine.parameters]
    z = engine.initialize(p, [event.initial[k] for k in engine.readings])
    rates = np.asarray(drive(insulin.reshape(1, -1))).ravel()
    z[engine.states.index("I_act")] += float(trapezoid(rates, event.time))
    z[engine.states.index("Gp")] = float(row["targets"]["Gp"][n - 1])
    f = [event.supplied[k][-1] for k in engine.forcings]
    return float(engine.observe(z, p, f)[TARGETS.index("U")])


def diagnose(model: dict, rows: list[dict], *, rho: float = 0.5) -> dict:
    """Assess a fixed vector using two independently constructed free rollouts."""
    ids = [row["trajectory_id"] for row in rows]
    if len(ids) != len(TRAIN_IDS) or set(ids) != set(TRAIN_IDS):
        raise ValueError("Expected the complete sixteen-event training cohort")
    if not 0 <= rho <= 1:
        raise ValueError("Invalid minimum observed-gap fraction")
    scales = np.array([model["scales"][key] for key in TARGETS])
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("Expected positive training target scales")
    public = [forecast.public_event(row) for row in rows]
    phases = {}
    durations, steps = {}, {}
    for name, end in (("partial", MINUTE), ("matched_full", None)):
        # Both timings include fresh graph construction, mesh, flow compilation,
        # integration and sampled readout at identical solver settings.
        tick = perf_counter()
        engine = _Rollout(model["graph"], model["formula"], TARGETS, STEPS)
        phase = [_flow(engine, model, _event(pub, end)) for pub in public]
        durations[name] = perf_counter() - tick
        phases[name] = [v for v, _ in phase]
        steps[name] = sum(m * STEPS for _, m in phase)

    tick = perf_counter()
    checker = _Rollout(model["graph"], model["formula"], TARGETS, 2)
    full = [forecast._predict(model, pub, checker) for pub in public]
    durations["independent_full"] = perf_counter() - tick
    drive = _affine_insulin_drive(model, checker)
    events, metrics = [], []
    delta_partial = np.zeros(len(TARGETS))
    delta_matched = np.zeros(len(TARGETS))
    params_bad = parameter_violations(model)
    u = TARGETS.index("U")
    for i, (row, pub, f) in enumerate(zip(rows, public, full, strict=True)):
        prefix = _event(pub, MINUTE)
        j = len(prefix.time) - 1
        a = phases["partial"][i]
        b = phases["matched_full"][i][:, : j + 1]
        c = np.array([f["predicted"][k][: j + 1] for k in TARGETS])
        delta_partial = np.maximum(delta_partial, np.max(abs(a - c), axis=1))
        delta_matched = np.maximum(delta_matched, np.max(abs(a - b), axis=1))
        m, obs = forecast._metrics(model, f, row)
        metrics.append(m)
        errors = {}
        for key in TARGETS:
            err = np.asarray(f["predicted"][key]) - np.asarray(obs[key])
            ix = int(np.argmax(abs(err)))
            errors[key] = {
                "max_absolute_error": float(abs(err[ix])),
                "max_error_time": pub["time"][ix],
                "early_nmse": float(
                    np.mean((err[: j + 1] / model["scales"][key]) ** 2)
                ),
                "later_nmse": float(
                    np.mean((err[j + 1 :] / model["scales"][key]) ** 2)
                ),
            }
        events.append(
            {
                "event": ids[i],
                "per_target_nmse": m["per_target_nmse"],
                "pooled_nmse": m["pooled_nmse"],
                "errors": errors,
                "observed0": obs["U"][0],
                "actual0": float(a[u, 0]),
                "observed15": obs["U"][j],
                "actual15": float(a[u, -1]),
                "matched_full15": float(b[u, -1]),
                "independent_full15": float(c[u, -1]),
                "conditional15": _conditional(row, prefix, model, checker, drive),
                "domain": f["qualification"],
            }
        )
    if (
        max(np.max(delta_partial / scales), np.max(delta_matched / scales))
        > CHECKPOINT_RTOL
    ):
        raise ArithmeticError(
            "Partial training prefix differs from full trajectory replay"
        )
    if not all(f["qualification"]["numerically_resolved"] for f in full):
        raise ArithmeticError("Unresolved dense training trajectory")

    def by_key(key):
        return [e[key] for e in events]

    domains = [f["qualification"] for f in full]
    summary = aggregate(metrics, len(events))
    initial = _comparisons(ids, by_key("observed0"), by_key("actual0"), rho, 0)
    minute15 = _comparisons(
        ids, by_key("observed15"), by_key("actual15"), rho, LATER_DISTINGUISHABLE_GAP
    )
    conditional = _comparisons(
        ids,
        by_key("observed15"),
        by_key("conditional15"),
        rho,
        LATER_DISTINGUISHABLE_GAP,
    )
    return {
        "model_sha256": content_hash(model),
        "minute": MINUTE,
        "pooled_nmse": summary["pooled_nmse"],
        "per_target_nmse": summary["available_per_target_nmse"],
        "events": events,
        "domain": {
            "eligible": not params_bad and all(d["eligible"] for d in domains),
            "parameter_violations": params_bad,
            "minimum_U_minus_Uii": min(
                d["minimum_values"]["U_minus_Uii"] for d in domains
            ),
        },
        "initial": initial,
        "minute15": minute15,
        "conditional_minute15": conditional,
        "verification": {
            "max_partial_vs_dense_full_by_target": dict(
                zip(TARGETS, delta_partial.tolist(), strict=True)
            ),
            "max_partial_vs_matched_full_by_target": dict(
                zip(TARGETS, delta_matched.tolist(), strict=True)
            ),
            "max_prefix_discrepancy_in_target_sd": float(
                np.max(delta_partial / scales)
            ),
            "tolerance_in_target_sd": CHECKPOINT_RTOL,
        },
        "cost": {
            "partial_event_simulations": len(rows),
            "matched_full_event_simulations": len(rows),
            "independent_dense_full_event_simulations": 2 * len(rows),
            "scalar_full_event_checks": len(rows),
            "partial_rk_steps": steps["partial"],
            "matched_full_rk_steps": steps["matched_full"],
            "partial_seconds": durations["partial"],
            "matched_full_seconds": durations["matched_full"],
            "independent_full_seconds": durations["independent_full"],
        },
    }


def screen_bank(models: dict[str, dict], rows: list[dict]) -> dict:
    """Compare actual prefix rejection with full replay on the same frozen bank."""
    ids = [r["trajectory_id"] for r in rows]
    if not models or sorted(ids) != sorted(TRAIN_IDS):
        raise ValueError("Expected fixed vectors and the registered training events")
    structure = {
        content_hash(
            [
                m["graph"],
                m["formula"]["equations"],
                m["formula"]["initializers"],
                sorted(m["formula"]["parameters"]),
            ]
        )
        for m in models.values()
    }
    if len(structure) != 1:
        raise ValueError("This bank must contain one declared equation family")
    first = next(iter(models.values()))
    public = [forecast.public_event(row) for row in rows]
    prefixes = [_event(pub, MINUTE) for pub in public]
    indices = [len(prefix.time) - 1 for prefix in prefixes]
    observed = [r["targets"]["U"][i] for r, i in zip(rows, indices, strict=True)]
    whole = [_event(pub, None) for pub in public]
    paths, timings = {}, {}
    u = TARGETS.index("U")
    for path in ("prefix_first", "full_only"):
        started = perf_counter()
        engine = _Rollout(first["graph"], first["formula"], TARGETS, STEPS)
        counts = {"partial": 0, "full": 0}
        steps = {"partial": 0, "full": 0}
        seconds = {"partial": 0.0, "full": 0.0}
        decisions = {}
        for label, model in models.items():
            for run in ("partial", "full") if path == "prefix_first" else ("full",):
                tick = perf_counter()
                events = prefixes if run == "partial" else whole
                phase = [_flow(engine, model, event) for event in events]
                seconds[run] += perf_counter() - tick
                counts[run] += len(phase)
                steps[run] += STEPS * sum(intervals for _, intervals in phase)
                values = [
                    matrix[u, index]
                    for (matrix, _), index in zip(phase, indices, strict=True)
                ]
                order = _comparisons(
                    ids, observed, values, 0.5, LATER_DISTINGUISHABLE_GAP
                )
                passed = order["violations"] == 0
                decision = {
                    "passed": passed,
                    "violations": order["violations"],
                    "worst_signed_slack": order["worst_signed_slack"],
                    "predicted_u15": dict(zip(ids, map(float, values), strict=True)),
                }
                if label in decisions and decisions[label] != decision:
                    raise ArithmeticError("Survivor prefix differs from full replay")
                decisions[label] = decision
                if run == "full" and passed:
                    decisions[label]["full_predictions_sha256"] = content_hash(
                        {
                            event_id: a.tolist()
                            for event_id, (a, _) in zip(ids, phase, strict=True)
                        }
                    )
                if not passed:
                    break
        paths[path] = decisions
        timings[path] = {
            "wall_seconds": perf_counter() - started,
            "partial_seconds": seconds["partial"],
            "full_seconds": seconds["full"],
            "partial_event_simulations": counts["partial"],
            "full_event_simulations": counts["full"],
            "partial_rk_steps": steps["partial"],
            "full_rk_steps": steps["full"],
        }
    # A matching compiled mesh ending at the time of a sample has an identical prefix.
    if paths["prefix_first"] != paths["full_only"]:
        raise ArithmeticError("Prefix and full-only fixed-bank screening disagree")
    return {
        "protocol": "fixed-bank-minute15-screen-1",
        "semantics": "Screen on fixed vectors only. "
        "Survivors are replayed from time zero. Each path reuses one compiled graph.",
        "minute": MINUTE,
        "observed_gap_allowance": LATER_DISTINGUISHABLE_GAP,
        "required_fraction": 0.5,
        "feasibility_tolerance": DOMAIN_TOL,
        "model_sha256": {k: content_hash(m) for k, m in models.items()},
        "candidates": paths["prefix_first"],
        "decisions_agree": True,
        "passed_candidates": sum(d["passed"] for d in paths["prefix_first"].values()),
        "rejected_candidates": sum(
            not d["passed"] for d in paths["prefix_first"].values()
        ),
        "cost": timings,
    }
