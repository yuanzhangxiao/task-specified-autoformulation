"""Single-round disposal revisions on the fitted delayed T2-easy parent."""

from __future__ import annotations

import math
from copy import deepcopy
from functools import cache

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.trajectory_function_fitting import validate_formula
from autoformalism.staged_topology import content_hash
from scripts import fit_t2_initial_disposal_order as order
from scripts import t2_disposal_coupled_replay as replay
from scripts.positive_insulin_refit_kernel import FitProblem

PARENT = {"initial_source": "none", "half_life_min": 0}
SOURCES = ("Gt", "Gp", "I")
TARGETS = replay.TARGETS
COMMON = ("afcoef11", "disp_tissue")
GAIN = "initial_disposal_gain"


@cache
def _intake() -> tuple[dict, list[dict]]:
    saved = sealed_read(order.SOURCE)
    if saved["artifact_sha256"] != order.FREEZE or saved["validation_opened_at_freeze"]:
        raise ValueError("Expected the registered training-only delayed parent")
    return saved["models"]["delayed_refit"], replay.load_training(
        order.coupled.INPUTS / "train.json"
    )


def _check_edit(edit: dict) -> None:
    if not isinstance(edit, dict) or set(edit) != set(PARENT):
        raise ValueError("Expected initial_source and half_life_min")
    src, half = edit["initial_source"], edit["half_life_min"]
    if not isinstance(src, str) or src not in (*SOURCES, "none"):
        raise ValueError("Unknown initial source")
    if type(half) not in (int, float) or not (half == 0 or 2 <= half <= 300):
        raise ValueError("Expected zero (persistent) or a half-life from 2 to 300 min")
    if src == "none" and half != 0:
        raise ValueError("An unchanged model has no decay time")


def _assemble(edit: dict, parent: dict, reference: dict) -> dict:
    """Compile the signed initial effect into the saved graph and its actual U sink."""
    _check_edit(edit)
    model = deepcopy(parent)
    src = edit["initial_source"]
    if src == "none":
        return model
    half = edit["half_life_min"]
    graph, form = model["graph"], model["formula"]
    state = src in graph["variable_types"]
    group = "targets" if state else "auxiliaries"
    x0 = float(reference[group][src][0])
    init = ("state0_" if state else "") + src
    graph["variable_types"]["M_init"] = "differential"
    graph["initializers"]["M_init"] = {
        "mode": "map",
        "state_initials": [src] if state else [],
        "public_initials": [] if state else [src],
    }
    graph["sources"]["M_init"] = ["M_init"] if half else []
    graph["sources"]["U"] = sorted([*graph["sources"]["U"], "M_init"])
    form["equations"]["U"] += "+M_init"
    form["equations"]["M_init"] = f"-{math.log(2) / half!r}*M_init" if half else "0"
    form["initializers"]["M_init"] = f"{GAIN}*(({init}-({x0!r}))/100)"
    form["parameters"][GAIN] = 0.0
    validate_formula(graph, form)
    return model


def packet() -> dict:
    """Report the parent's predicted and measured disposal at training time zero."""
    parent, rows = _intake()
    problem = FitProblem(parent, rows)
    err = problem.residual(problem._vector(parent["formula"]["parameters"]))
    idx = np.cumsum([0, *[len(row["time"]) for row in rows]])[:-1]
    obs = np.array([row["targets"]["U"][0] for row in rows])
    pred = obs + err.reshape(-1, 3)[idx, 2] * parent["scales"]["U"]
    high, low = order.order_pairs(obs)
    return {
        "parent": dict(PARENT),
        "algorithm_edits": [
            {"initial_source": "Gt", "half_life_min": half} for half in (0, 5, 30)
        ],
        "evidence": {
            "parent_model_sha256": content_hash(parent),
            "task": (
                "Generate glucose mass Gp, insulin I and disposal U. Keep the meal "
                "balance and the delayed I -> insulin action -> U pathway."
            ),
            "equations": {
                name: parent["formula"]["equations"][name]
                for name in ("Gp", "I_act", "U")
            },
            "training_events": len(rows),
            "distinct_initial_pairs": len(high),
            "reversed_initial_pairs": int(np.sum(pred[high] < pred[low])),
            "initial_observations": [
                {
                    "event": row["trajectory_id"],
                    "Gp": row["targets"]["Gp"][0],
                    "Gt": row["auxiliaries"]["Gt"][0],
                    "I": row["targets"]["I"][0],
                    "U": float(obs[i]),
                    "predicted_U": float(pred[i]),
                }
                for i, row in enumerate(rows)
                if i in (0, 11, 12)
            ],
            "fit": (
                "Refit the same disposal intercept and ongoing tissue slope in each "
                "model and the signed amplitude of any added initial effect. Keep "
                "the remaining fitted coefficients fixed, propagate all three targets, "
                "preserve half the measured initial U gaps and nonnegative disposal "
                "above Uii on the training grid."
            ),
        },
        "edit_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["initial_source", "half_life_min"],
            "properties": {
                "initial_source": {"type": "string", "enum": ["none", *SOURCES]},
                "half_life_min": {"type": "number", "minimum": 0, "maximum": 300},
            },
        },
        "edit_contract": {
            "initial_source": ["none", *SOURCES],
            "half_life_min": (
                "0 for a persistent effect or a half-life from 2 to 300 minutes. "
                "Use 0 with none. M(0) = gain*(source(0)-fasting_reference)/100, "
                "U gains +M(t), M decays exponentially for positive half-life."
            ),
        },
    }


def _restricted_fit(model: dict, rows: list[dict], names: tuple[str, ...]) -> dict:
    """Fit active coefficients subject to the linear physical constraints."""
    midrows = [order.base._midpoints(row) for row in rows]
    engine = FitProblem(model, midrows, substeps=2)
    p = engine._vector(model["formula"]["parameters"])
    offsets = engine.residual(p).reshape(-1, len(TARGETS))
    columns = []
    for name in names:
        unit = p.copy()
        unit[engine.names.index(name)] += 1
        columns.append(engine.residual(unit).reshape(offsets.shape) - offsets)
    sensitivity = np.stack(columns, axis=-1)
    y = np.vstack([np.array([r["targets"][k] for k in TARGETS]).T for r in midrows])
    y0 = y + offsets * engine.scales
    dy = sensitivity * engine.scales[None, :, None]
    first = np.cumsum([0, *[len(row["time"]) for row in midrows]])[:-1]
    high, low = order.order_pairs(y[first, 2])
    margin = (
        y0[first[high], 2]
        - y0[first[low], 2]
        - 0.5 * (y[first[high], 2] - y[first[low], 2])
    )
    uii = np.concatenate([r["auxiliaries"]["Uii"] for r in midrows])
    mats = np.vstack([dy[:, 0], dy[:, 2], dy[first[high], 2] - dy[first[low], 2]])
    bounds = np.r_[-y0[:, 0], uii + order.base.DISPOSAL_BUFFER - y0[:, 2], -margin]
    samples = np.concatenate(
        [
            np.arange(start, start + len(row["time"]), 2)
            for start, row in zip(first, midrows, strict=True)
        ]
    )
    matrix = sensitivity[samples].reshape(-1, len(names))
    resid = offsets[samples].ravel()
    hess, grad = matrix.T @ matrix / len(resid), matrix.T @ resid / len(resid)
    feasible = linprog(
        np.zeros(len(names)),
        A_ub=-mats,
        b_ub=-bounds,
        bounds=[(None, None)] * len(names),
        method="highs",
    )
    if not feasible.success:
        raise ArithmeticError(f"Conditional feasible-start search: {feasible.message}")
    fit = minimize(
        lambda w: 0.5 * w @ hess @ w + grad @ w,
        feasible.x,
        jac=lambda w: hess @ w + grad,
        method="SLSQP",
        constraints=LinearConstraint(mats, bounds, np.inf),
        options={"ftol": 1e-12, "maxiter": 100},
    )
    if not fit.success or np.min(mats @ fit.x - bounds) < -1e-8:
        raise ArithmeticError(f"Unqualified conditional fit: {fit.message}")
    for name, delta in zip(names, fit.x, strict=True):
        model["formula"]["parameters"][name] += float(delta)
    verified = engine.residual(engine._vector(model["formula"]["parameters"]))
    if not np.allclose(
        verified.reshape(offsets.shape),
        offsets + sensitivity @ fit.x,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ArithmeticError("Affine response disagrees with nonlinear graph rollout")
    return {
        "updated_coefficients": {n: model["formula"]["parameters"][n] for n in names},
        "forward_evaluations": 2 + len(names),
        "iterations": fit.nit,
    }


def _metrics(result: dict) -> dict:
    events = result["events"]
    if result["summary"]["failed_events"]:
        raise ArithmeticError("Unresolved full-model replay")
    obs = np.array([e["observed"]["U"][0] for e in events])
    pred = np.array([e["predicted"]["U"][0] for e in events])
    high, low = order.order_pairs(obs)
    margins = pred[high] - pred[low] - 0.5 * (obs[high] - obs[low])
    return {
        "nmse": result["summary"]["pooled_nmse"],
        "per_target_nmse": result["summary"]["available_per_target_nmse"],
        "reversed_initial_pairs": int(np.sum(pred[high] < pred[low])),
        "comparable_initial_pairs": len(high),
        "half_gap_passed": bool(np.all(margins >= -1e-8)),
        "physical_checks_passed": result["domain"]["eligible"],
        "min_U_minus_Uii": result["domain"]["minimum_U_minus_Uii"],
        "domain_reasons": result["domain"]["reasons"],
        "event_count": len(events),
    }


def score(edit: dict) -> dict:
    """Fit and independently replay one conditional revision on training events."""
    try:
        _check_edit(edit)
    except ValueError as exc:
        return {"eligible": False, "reason": str(exc)}
    parent, rows = _intake()
    model = _assemble(edit, parent, rows[0])
    names = COMMON if edit["initial_source"] == "none" else (*COMMON, GAIN)
    try:
        fit = _restricted_fit(model, rows, names)
        metrics = _metrics(replay.replay(model, rows))
    except ArithmeticError as exc:
        return {"eligible": False, "reason": str(exc)}
    eligible = metrics["physical_checks_passed"] and metrics["half_gap_passed"]
    return {
        "eligible": eligible,
        "train_nmse": metrics["nmse"],
        "edit": dict(edit),
        "reason": "eligible" if eligible else "Physical domain or initial order failed",
        "fit": fit,
        "metrics": metrics,
        "model_sha256": content_hash(model),
    }


def validate(edit: dict, score_result: dict) -> dict:
    """Reconstruct the selected model before reading the four development events."""
    if not score_result.get("eligible") or score_result.get("edit") != edit:
        raise ValueError("Expected an eligible training score for this edit")
    parent, train = _intake()
    model = _assemble(edit, parent, train[0])
    model["formula"]["parameters"].update(score_result["fit"]["updated_coefficients"])
    if content_hash(model) != score_result["model_sha256"]:
        raise ValueError("Selected model differs from training")
    rows = replay.load_validation(order.coupled.INPUTS / "validation.json")
    return _metrics(replay.replay(model, rows))


def _conditional_order_loss(
    columns: np.ndarray, prediction: np.ndarray, observed: np.ndarray, first: np.ndarray
) -> float | None:
    """Fit signed gains to disposal residuals with the training initial-order rule."""
    high, low = order.order_pairs(observed[first])
    lhs = columns[first[high]] - columns[first[low]]
    rhs = 0.5 * (observed[first[high]] - observed[first[low]]) - (
        prediction[first[high]] - prediction[first[low]]
    )
    residual = prediction - observed
    hessian = columns.T @ columns / len(columns)
    gradient = columns.T @ residual / len(columns)
    feasibility = linprog(
        np.zeros(columns.shape[1]),
        A_ub=-lhs,
        b_ub=-rhs,
        bounds=[(None, None)] * columns.shape[1],
        method="highs",
    )
    if not feasibility.success:
        return None
    solver = minimize(
        lambda w: 0.5 * w @ hessian @ w + gradient @ w,
        feasibility.x,
        jac=lambda w: hessian @ w + gradient,
        method="SLSQP",
        constraints=[LinearConstraint(lhs, rhs, np.inf)] if len(high) else [],
        options={"maxiter": 100, "ftol": 1e-12},
    )
    if not solver.success or np.any(lhs @ solver.x < rhs - 1e-8):
        return None
    return float(np.mean((residual + columns @ solver.x) ** 2))


def _initial_contrasts(rows: list[dict]) -> tuple[dict, dict]:
    """Identify equivalent initial signals up to one nonzero shared scale."""
    unique, alias = {}, {}
    for source in SOURCES:
        group = "auxiliaries" if source == "Gt" else "targets"
        values = np.asarray([r[group][source][0] for r in rows], dtype=float)
        centered = (values - values[0]) / 100
        scale = np.max(np.abs(centered))
        if not scale:
            alias[source] = "zero on training"
            continue
        for other, feature in unique.items():
            j = np.argmax(abs(feature))
            ratio = centered[j] / feature[j]
            if ratio and np.allclose(
                centered / scale, ratio * feature / scale, atol=1e-12, rtol=1e-12
            ):
                alias[source] = other
                break
        else:
            unique[source] = centered
    return unique, alias


def screen() -> dict:
    """Rank initial effects by disposal error conditional on observed glucose."""
    parent, rows = _intake()
    engine = FitProblem(parent, rows)
    base_error = engine.residual(engine._vector(parent["formula"]["parameters"]))
    base_error = base_error.reshape(-1, 3) * engine.scales
    durations = [np.asarray(r["time"]) - r["time"][0] for r in rows]
    times = np.concatenate(durations)
    lengths = [len(t) for t in durations]
    first = np.cumsum([0, *lengths])[:-1]
    # This slope belongs to the authenticated delayed parent. At fixed predicted
    # action, substituting the measured Gp adds slope * (observed Gp - predicted Gp).
    gp_slope = parent["formula"]["parameters"]["afcoef12"] / 49.9160934007
    observed = np.concatenate([r["targets"]["U"] for r in rows])
    prediction = observed + base_error[:, 2] - gp_slope * base_error[:, 0]
    ongoing = (np.concatenate([r["auxiliaries"]["Gt"] for r in rows]) - 100) / 100
    contrasts, aliases = _initial_contrasts(rows)
    horizon = min(300, max(t[-1] for t in durations))
    half_lives = [0] + [2**i for i in range(1, 9) if 2**i <= horizon]
    profiles = []
    for source, levels in contrasts.items():
        amplitude = np.repeat(levels, lengths)
        for half in half_lives:
            memory = amplitude * np.exp2(-times / half) if half else amplitude
            columns = np.column_stack((np.ones(len(times)), ongoing, memory))
            loss = _conditional_order_loss(columns, prediction, observed, first)
            profiles.append(
                {
                    "edit": {"initial_source": source, "half_life_min": half},
                    "conditional_U_mse": loss,
                }
            )
    ranked = sorted(
        (p for p in profiles if p["conditional_U_mse"] is not None),
        key=lambda p: (
            p["conditional_U_mse"],
            p["edit"]["half_life_min"],
            p["edit"]["initial_source"],
        ),
    )
    return {
        "edits": [item["edit"] for item in ranked[:3]],
        "screened": len(SOURCES) * len(half_lives),
        "distinct_profiles": len(profiles),
        "training_equivalent_or_inactive_sources": aliases,
        "criterion": (
            "Lowest training U mean squared error conditional on measured Gp and "
            "the parent-predicted insulin action. Refit the signed disposal offset, "
            "ongoing Gt slope and initial-effect gain with half-gap constraints. "
            "All shortlisted edits still require the complete free-trajectory fit."
        ),
        "observations_used": {
            "training_events": len(rows),
            "time_samples": len(times),
            "target_histories": ["Gp", "U"],
            "target_initials": ["Gp", "I"],
            "supplied_histories": "Declared parent forcings and tissue glucose",
        },
        "cost": {
            "full_parent_residual_evaluations": 1,
            "conditional_fit_attempts": len(profiles),
            "candidate_full_rollouts": 0,
        },
        "profiles": profiles,
    }
