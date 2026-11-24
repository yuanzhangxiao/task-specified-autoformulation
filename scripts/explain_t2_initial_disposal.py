"""Attribute two frozen training-holdout forecasts to the fitted disposal terms."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import run_t2_initial_memory_holdouts as study
from scripts import t2_initial_memory_fit as fitting
from scripts import t2_initial_memory_forecast as forecast

OUTPUT = Path("output/t2-disposal-dip-equations-20261124")
TIMES = (0, 9, 13, 15, 300)


def disposal_terms(model: dict, row: dict, pred: dict) -> dict[str, np.ndarray]:
    """Evaluate additive contributions and require closure against the saved rollout."""
    p = model["formula"]["parameters"]
    c, s = model["centers"], model["scales"]
    t = np.asarray(row["time"])
    g = np.asarray(pred["Gp"])
    gt = np.asarray(row["auxiliaries"]["Gt"])
    ins, primitive = fitting._scalar_insulin(
        row, model["knots"], model["scalar_model"], t
    )
    if not np.allclose(ins, pred["I"], rtol=1e-7, atol=1e-6):
        raise ValueError("Reconstructed insulin differs from the frozen forecast")
    drift = p["afcoef9"] * t
    insulin = p["afcoef10"] * (primitive - c["I"] * t) / s["I"]
    memory = p["init_tissue_gain"] * (gt[0] - c["Gt_ref"]) / 100
    rate = p.get("memory_rate", 0.0)
    action_gain = p["afcoef13"]
    terms = {
        "constant": np.full(t.shape, p["afcoef11"], dtype=float),
        "Gp": p["afcoef12"] * (g - c["Gp"]) / s["Gp"],
        "Gt": p["disp_tissue"] * (gt - 100) / 100,
        "Uii": p["afcoef14"] * (np.asarray(row["auxiliaries"]["Uii"]) - 1),
        "action_initial_baseline": np.full(
            t.shape, action_gain * (p["afcoef15"] + p["afcoef16"] * ins[0])
        ),
        "action_initial_tissue": action_gain * memory * np.exp(-rate * t),
        "action_drift": action_gain * drift,
        "action_insulin_drive": action_gain * insulin,
    }
    if not np.allclose(sum(terms.values()), pred["U"], rtol=1e-7, atol=1e-7):
        raise ValueError("Fitted-term sum differs from the compiled disposal forecast")
    return terms


def glucose_budget(time, t, source, predicted_gp, observed_gp, predicted_u, observed_u):
    """Compare integrated modeled flows with the source implied by observations."""
    j = _sample(time, t)
    span = slice(j + 1)
    delta_pred = float(predicted_gp[j] - predicted_gp[0])
    delta_obs = float(observed_gp[j] - observed_gp[0])
    src = float(np.trapezoid(source[span], x=time[span]))
    u_obs = float(np.trapezoid(observed_u[span], x=time[span]))
    # The reconstructed affine source is piecewise linear on the recorded grid.
    # Its integral minus the simulated change in plasma mass gives the model's
    # exact integrated disposal up to the independent rollout's integration error.
    u_model = src - delta_pred
    integrated_curve = float(np.trapezoid(predicted_u[span], x=time[span]))
    return {
        "delta_Gp_predicted": delta_pred,
        "delta_Gp_observed": delta_obs,
        "source_area_model": src,
        "source_area_implied_by_observations": delta_obs + u_obs,
        "disposal_area_model": u_model,
        "disposal_area_observed": u_obs,
        "source_area_error": src - (delta_obs + u_obs),
        "disposal_area_error": u_model - u_obs,
        "minute_grid_model_disposal_quadrature_error": integrated_curve - u_model,
    }


def observed_storage_balance(row: dict) -> dict:
    """Evaluate two balances on the recorded zero-ingestion prefix of each event."""
    active = np.flatnonzero(np.asarray(row["inputs"]["meal_rate_g_per_min"]) != 0)
    stop = int(active[0]) if active.size else len(row["time"])
    if stop < 2:
        return {"status": "no_positive_duration_before_ingestion", "samples": stop}
    time = np.asarray(row["time"][:stop], dtype=float)
    gp = np.asarray(row["targets"]["Gp"][:stop])
    gt = np.asarray(row["auxiliaries"]["Gt"][:stop])
    net = (
        np.asarray(row["auxiliaries"]["EGP"][:stop])
        - np.asarray(row["auxiliaries"]["E"][:stop])
        - np.asarray(row["targets"]["U"][:stop])
    )
    flux = cumulative_trapezoid(net, time, initial=0)
    plasma_error = gp - gp[0] - flux
    total_error = plasma_error + (gt - gt[0])
    snapshots = {}
    for t in (t for t in TIMES if t <= time[-1]):
        j = _sample(time, t)
        snapshots[str(t)] = {
            "delta_Gt": float(gt[j] - gt[0]),
            "delta_Gp": float(gp[j] - gp[0]),
            "net_flow_integral": float(flux[j]),
            "plasma_error": float(plasma_error[j]),
            "total_error": float(total_error[j]),
        }
    return {
        "status": "through_end" if not active.size else "pre_ingestion",
        "samples": stop,
        "last_minute": float(time[-1]),
        "max_abs_plasma_error": float(np.max(abs(plasma_error))),
        "max_abs_total_error": float(np.max(abs(total_error))),
        "by_minute": snapshots,
    }


def _sample(time: np.ndarray, t: int) -> int:
    indices = np.flatnonzero(time == t)
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one recorded sample at minute {t}")
    return int(indices[0])


def _no_fade_control(model: dict, pub: dict, freeze: dict, term0: float) -> dict:
    """Switch off attenuation at fixed gains and verify its linear feedback response."""
    control = deepcopy(model)
    p = control["formula"]["parameters"]
    rate = p["memory_rate"]
    p["memory_rate"] = 0.0
    alt = forecast.predict(control, pub)
    if not alt["qualification"]["numerically_resolved"]:
        raise ArithmeticError("Unresolved fixed-vector counterfactual")
    slope = p["afcoef12"] / model["scales"]["Gp"]
    values = {}
    for t in TIMES:
        idx = _sample(np.asarray(pub["time"]), t)

        # The sole changed forcing in d(Gp_fade-Gp_control)/dt is the
        # disappearance of a fraction of the initial contribution to disposal.
        def transfer(v, stop=t):
            return np.exp(-slope * (stop - v)) * (-np.expm1(-rate * v))

        delta_gp = term0 * quad(transfer, 0, t)[0]
        delta_u = slope * delta_gp + term0 * np.expm1(-rate * t)
        compiled = [freeze[k][idx] - alt["predicted"][k][idx] for k in ("Gp", "U")]
        if not np.allclose([delta_gp, delta_u], compiled, rtol=1e-7, atol=1e-7):
            raise ArithmeticError("Analytic memory contrast differs from graph replay")
        values[str(t)] = {
            "Gp_rate_zero": alt["predicted"]["Gp"][idx],
            "U_rate_zero": alt["predicted"]["U"][idx],
            "Gp_difference_fading_minus_zero": float(delta_gp),
            "U_difference_fading_minus_zero": float(delta_u),
        }
    return {
        "fixed_vector_with_rate_zero": values,
        "rate_zero_glucose_curve": alt["predicted"]["Gp"],
        "qualification": alt["qualification"],
    }


def inspect_arm(model: dict, row: dict, frozen: dict) -> dict:
    """Trace a saved vector on a no-ingestion initial-condition challenge."""
    pub = forecast.public_event(row)
    if frozen["model_sha256"] != content_hash(model) or frozen[
        "public_inputs_sha256"
    ] != content_hash(pub):
        raise ValueError("Forecast, fitted vector and public inputs disagree")
    if not frozen["qualification"]["numerically_resolved"]:
        raise ValueError("Numerically unresolved frozen forecast")
    meal = pub["supplied"]["meal_rate_g_per_min"]
    if (
        np.any(np.asarray(meal) != 0)
        or model["formula"]["initializers"]["D_meal"] != "0"
    ):
        raise ValueError(
            "Expected a meal-free event and empty initial meal compartment"
        )
    pred, obs = frozen["predicted"], row["targets"]
    time = np.asarray(row["time"])
    terms = disposal_terms(model, row, pred)
    p = model["formula"]["parameters"]
    src = p["meal_balance_offset"] + p["meal_balance_egp"] * np.asarray(
        row["auxiliaries"]["EGP"]
    )
    snapshots = {}
    for t in TIMES:
        j = _sample(time, t)
        snapshots[str(t)] = {
            "terms": {k: float(v[j]) for k, v in terms.items()},
            "change_from_zero": {k: float(v[j] - v[0]) for k, v in terms.items()},
            "observed": {k: v[j] for k, v in obs.items()},
            "predicted": {k: v[j] for k, v in pred.items()},
            "source": float(src[j]),
        }
    ans = {
        "model_sha256": content_hash(model),
        "snapshots": snapshots,
        "glucose_budget": {
            str(t): glucose_budget(
                time, t, src, pred["Gp"], obs["Gp"], pred["U"], obs["U"]
            )
            for t in TIMES
            if t > 0
        },
    }
    if model["memory_arm"] == "fading":
        ans.update(
            _no_fade_control(model, pub, pred, terms["action_initial_tissue"][0])
        )
    return ans


def run(root: Path, output: Path) -> dict:
    """Produce a sealed term audit for each original model and its frozen forecast."""
    records = sealed_read(root / "forecasts.json")
    rows = {r["trajectory_id"]: r for r in study.intake.load_training(study.TRAIN)}
    comparisons = {}
    for event in study.folds.HOLDOUTS:
        selected = sealed_read(root / event / "selected.json")
        if records["identity"][event] != selected["artifact_sha256"]:
            raise ValueError("Forecast selection digest differs")
        comparisons[event] = {
            arm: inspect_arm(selected["models"][arm]["model"], rows[event], values)
            for arm, values in records["result"][event].items()
        }
    report = sealed_write(
        output / "equation-budget.json",
        {
            "forecasts_sha256": records["artifact_sha256"],
            "events": comparisons,
            "observed_pre_ingestion_balances": {
                name: observed_storage_balance(row) for name, row in rows.items()
            },
        },
    )
    plot_dip(
        comparisons["train_011"]["fading"],
        rows["train_011"],
        records["result"]["train_011"]["fading"],
        output / "term-attribution.png",
    )
    return report


def plot_dip(audit: dict, row: dict, frozen: dict, output: Path) -> None:
    """Draw the early disposal budget and fixed-vector glucose contrast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    left, right = axes
    change = audit["snapshots"]["9"]["change_from_zero"]
    vals = [
        change["action_initial_tissue"],
        change["Gt"],
        change["Gp"],
        change["action_drift"] + change["action_insulin_drive"],
    ]
    observed = row["targets"]["U"]
    vals.extend(
        [sum(vals), observed[_sample(np.asarray(row["time"]), 9)] - observed[0]]
    )
    labels = [
        "Initial tissue offset",
        "Current tissue glucose",
        "Current plasma glucose",
        "Ongoing insulin action",
        "Predicted sum",
        "Observed change",
    ]
    colors = ["#a2412c", "#3d7990", "#3d7990", "#3d7990", "#26292c", "#656973"]
    left.barh(labels, vals, color=colors, height=0.58)
    left.invert_yaxis()
    left.axvline(0, color="0.5", lw=0.7)
    for i, v in enumerate(vals):
        left.text(
            v + (0.06 if v >= 0 else -0.06),
            i,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )
    left.set_xlim(-2.68, 1.96)
    left.set_xlabel("Change in disposal (mg/kg/min)")
    left.set_title("Disposal, minute 0 to minute 9", loc="left", fontsize=12)
    time = row["time"]
    right.plot(time, row["targets"]["Gp"], color="#26292c", lw=2.5, label="Observed")
    right.plot(
        time,
        frozen["predicted"]["Gp"],
        color="#bc531c",
        ls="--",
        lw=2,
        label="Fitted fading effect",
    )
    right.plot(
        time,
        audit["rate_zero_glucose_curve"],
        color="#236887",
        ls="-.",
        lw=2,
        label="Decay switched off, same gains",
    )
    right.set_title(
        "Plasma glucose with the same fitted gains", loc="left", fontsize=12
    )
    right.set_xlabel("Time (min)")
    right.set_ylabel("Plasma glucose (mg/kg)")
    right.legend(frameon=False, fontsize=8.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=study.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.source, args.output)
    print(args.output / "equation-budget.json")
