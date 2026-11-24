"""Audit the integrated meal source and disposal on fixed T2 training meal forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_simpson, cumulative_trapezoid

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import _Rollout
from autoformalism.staged_topology import content_hash
from scripts import t2_initial_memory_forecast as replay
from scripts.run_t2_initial_memory_holdouts import TRAIN
from scripts.run_t2_total_glucose_holdouts import ROOT
from scripts.t2_initial_memory_folds import HOLDOUTS, load_fit_rows
from scripts.t2_total_glucose_meals import MEAL, TARGETS, meal_schedule


def filter_source(t, q, gain, tau):
    """Integrate dD/dt=(q-D)/tau with D(0)=0 for the declared linear input knots."""
    if tau <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Expected a positive time constant and increasing input knots")
    delivery = np.zeros(len(t))
    for i, z in enumerate(np.diff(t) / tau):
        mass = -np.expm1(-z)
        right = z * (0.5 - z / 6 + z**2 / 24) if z < 1e-4 else 1 - mass / z
        delivery[i + 1] = (
            delivery[i] + (q[i] - delivery[i]) * mass + (q[i + 1] - q[i]) * right
        )
    return gain * (cumulative_trapezoid(q, t, initial=0) - tau * delivery)


def _areas(model, engine, row, curve):
    pub = replay.public_event(row)
    t = np.asarray(pub["time"])
    supplied = {k: np.asarray(v) for k, v in pub["supplied"].items()}
    initial = {k: v[0] for k, v in supplied.items()} | pub["initial"]
    event = replay.scalar.Inputs(pub["event"], t, supplied, initial)
    mesh = replay.scalar.mesh(event, model["knots"])
    parts = curve["qualification"]["total_mass"]["mesh_subdivisions"]
    fine, values, _, _, _ = replay.dense._fine_arrays(engine, model, event, mesh, parts)
    idx = np.searchsorted(fine, t)
    if (
        not np.array_equal(t, fine[idx])
        or any(
            not np.allclose(values[k][idx], curve["predicted"][k], atol=1e-7, rtol=1e-9)
            for k in TARGETS
        )
        or curve["observed"] != row["targets"]
    ):
        raise ValueError("Independent rollout differs from frozen event")
    p = model["formula"]["parameters"]
    meal = filter_source(
        t, supplied[MEAL], p["meal_balance_gain"], p["meal_balance_tau"]
    )
    g = np.array(row["targets"]["Gp"])
    s = g + supplied["Gt"]
    u = np.asarray(row["targets"]["U"])
    obs = (
        s
        - s[0]
        + cumulative_trapezoid(u + supplied["E"] - supplied["EGP"], t, initial=0)
    )
    disposal = cumulative_simpson(values["U"], x=fine, initial=0)[
        idx
    ] - cumulative_trapezoid(u, t, initial=0)
    gp_err = values["Gp"][idx] - g
    err = float(max(abs(meal - obs - disposal - gp_err)))
    if err > 1e-4:
        raise ArithmeticError("Native mass error decomposition did not close")
    return (
        t,
        {
            "modeled_appearance": meal,
            "implied_appearance": obs,
            "excess_appearance": meal - obs,
            "excess_disposal": disposal,
            "glucose_error": gp_err,
        },
        err,
    )


def _window(t, areas, begin, end):
    left, right = np.searchsorted(t, [begin, end])
    if right >= len(t) or t[left] != begin or t[right] != end:
        raise ValueError("Requested report times must be measured")
    return {k: float(v[right] - v[left]) for k, v in areas.items()}


def audit(root: Path, train_file: Path):
    """Quantify excess appearance and disposal after all model selection is frozen."""
    curves = sealed_read(root / "training-curves.json")
    result = {}
    for held in HOLDOUTS:
        selected = sealed_read(root / held / "selected.json")
        if curves["identity"][held] != selected["artifact_sha256"]:
            raise ValueError("Incompatible frozen training curves")
        model = selected["models"]["total"]["model"]
        if (
            any(
                model["formula"]["equations"][k] != v
                for k, v in {
                    "S": "EGP-E-U+meal_balance_gain*D_meal",
                    "D_meal": "(meal_rate_g_per_min-D_meal)/meal_balance_tau",
                    "Gp": "S-Gt",
                }.items()
            )
            or model["formula"]["initializers"]["D_meal"] != "0"
        ):
            raise ValueError("Expected total balance with a zero-start meal filter")
        engine = _Rollout(model["graph"], model["formula"], TARGETS, 2)
        events = {}
        for row in load_fit_rows(train_file, held):
            schedule = meal_schedule(row)
            if not schedule["pulses"]:
                continue
            name = row["trajectory_id"]
            t, areas, err = _areas(
                model, engine, row, curves["result"][held]["total"][name]
            )
            checkpoints = {}

            for pulse in schedule["pulses"]:
                start = pulse["start_min"]
                checkpoints[f"{start:g}-{start + 14:g}"] = _window(
                    t, areas, start, start + 14
                )
            for minute in (14, 30, 60, 120, 300):
                checkpoints[f"0-{minute}"] = _window(t, areas, 0, minute)
            events[name] = {
                "schedule": schedule,
                "windows_min": checkpoints,
                "max_closure_error_mg_per_kg": err,
            }
        result[held] = {"model_sha256": content_hash(model), "events": events}
    return sealed_write(
        root / "meal-appearance-budget.json",
        {
            "training_curves_sha256": curves["artifact_sha256"],
            "folds": result,
            "unit": "mg/kg",
            "observed_appearance": "Measured mass change + trapezoid integral(U+E-EGP)",
            "modeled_appearance": (
                "Analytic integral of fitted filter on the linear input knots"
            ),
            "excess_disposal": (
                "Fine-grid Simpson integral of model U minus native-grid "
                "trapezoid integral of measured U"
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    audit(args.root, TRAIN)
    print(args.root / "meal-appearance-budget.json")
