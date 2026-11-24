"""Independent complete-graph and glucose-mass checks for the two fitting folds."""

from __future__ import annotations

import numpy as np

from autoformalism.research.trajectory_function_fitting import _Rollout
from scripts import t2_initial_memory_forecast as prior
from scripts.positive_insulin_refit_inputs import TARGETS

# Units are mg/kg. The analytic meal integral and numerical U integral are
# compared with the change in glucose mass at every supplied sample.
BALANCE_ATOL = 1e-4
READOUT_ATOL = 1e-9
MEAL = "meal_rate_g_per_min"


def mass_audit(model: dict, pub: dict, engine: _Rollout) -> dict:
    """Verify total mass at initialization and throughout the free rollout."""
    if "S" not in engine.states:
        raise ValueError("Expected a differential total-glucose state")
    time = np.asarray(pub["time"])
    supplied = {k: np.asarray(v) for k, v in pub["supplied"].items()}
    gt = supplied["Gt"]
    p = model["formula"]["parameters"]
    gain, tau = p["meal_balance_gain"], p["meal_balance_tau"]
    initial = {k: v[0] for k, v in supplied.items()} | pub["initial"]
    event = prior.scalar.Inputs(pub["event"], time, supplied, initial)
    base = prior.scalar.mesh(event, model["knots"])
    for factor in (1, 2, 4, 8):
        parts = factor * prior.dense.SUBDIVISIONS
        fine, values, states, _, _ = prior.dense._fine_arrays(
            engine, model, event, base, parts
        )
        s = states[engine.states.index("S")]
        gap = s - np.interp(fine, time, gt) - values["Gp"]
        initial_error = abs(s[0] - (initial["Gp"] + gt[0]))
        readout_error = float(np.max(abs(gap)))
        # Simpson panels lie inside the mesh intervals where forcing is affine.
        u = values["U"]
        integ_u = np.r_[
            0.0, np.cumsum(np.diff(fine[::2]) * (u[:-2:2] + 4 * u[1::2] + u[2::2]) / 6)
        ]
        idx = np.searchsorted(fine, time)
        if not np.array_equal(fine[idx], time) or np.any(idx % 2):
            raise ArithmeticError("Quadrature panels do not cover native observations")
        delivery = values["D_meal"][idx]

        def area(v):
            return np.r_[0.0, np.cumsum(np.diff(time) * (v[1:] + v[:-1]) / 2)]

        # D'=(r-D)/tau implies integral(D)=integral(r)-tau*(D(t)-D(0)).
        # This check uses the actual supplied meal amount and an independent
        # quadrature of U rather than re-integrating the model's S derivative.
        meal = gain * (area(supplied[MEAL]) - tau * (delivery - delivery[0]))
        expected = area(supplied["EGP"] - supplied["E"]) + meal - integ_u[idx // 2]
        imbalance = s[idx] - s[0] - expected
        maximum = float(np.max(abs(imbalance)))
        resolved = bool(
            initial_error <= READOUT_ATOL
            and readout_error <= READOUT_ATOL
            and abs(delivery[0]) <= READOUT_ATOL
            and maximum <= BALANCE_ATOL
        )
        if resolved:
            break
    return {
        "resolved": resolved,
        "absolute_mass_tolerance": BALANCE_ATOL,
        "minimum_S": float(np.min(s)),
        "initial_error": float(initial_error),
        "max_abs_readout_error": readout_error,
        "max_abs_integrated_mass_error": maximum,
        "minute_of_max_abs_error": float(time[np.argmax(abs(imbalance))]),
        "final_integrated_mass_error": float(imbalance[-1]),
        "mesh_subdivisions": parts,
    }


def _with_balance(model, pub, engine):
    f = prior._predict(model, pub, engine)
    if "S" in engine.states:
        a = mass_audit(model, pub, engine)
        q = f["qualification"]
        q["numerically_resolved"] &= a["resolved"]
        q["physical"] &= a["minimum_S"] >= -prior.dense.DOMAIN_TOL
        q["minimum_values"]["S"] = a["minimum_S"]
        q["eligible"] = q["numerically_resolved"] and q["physical"]
        q["total_mass"] = a
    return f


def predict(model: dict, pub: dict) -> dict:
    """Roll out the graph from its public schedule, Gp(0) and I(0)."""
    engine = _Rollout(model["graph"], model["formula"], TARGETS, 2)
    return _with_balance(model, pub, engine)


def _check_training(model, rows):
    ids = sorted(r["trajectory_id"] for r in rows)
    if (
        len(ids) != 15
        or ids != sorted(model["fit_training_ids"])
        or not any(set(ids) == set(prior.TRAIN_IDS) - {h} for h in prior.OMISSIONS)
    ):
        raise ValueError("Model and rows must belong to exactly the same fitting fold")
    for k in TARGETS:
        y = np.concatenate([r["targets"][k] for r in rows])
        if not np.isclose(model["scales"][k], max(1e-8, y.std()), rtol=1e-12, atol=0):
            raise ValueError(f"Target scale differs from the fitting fold for {k}")


def training_curves(model: dict, rows15: list[dict]) -> dict:
    """Score each fitting event and record its meal exposure."""
    _check_training(model, rows15)
    engine = _Rollout(model["graph"], model["formula"], TARGETS, 2)
    curves = {}
    for row in rows15:
        pub = prior.public_event(row)
        f = _with_balance(model, pub, engine)
        m, obs = prior._metrics(model, f, row)
        dose = float(np.trapezoid(pub["supplied"][MEAL], x=pub["time"]))
        curves[m["event"]] = m | {
            "meal_grams": dose,
            "time": pub["time"],
            "predicted": f["predicted"],
            "observed": obs,
            "qualification": f["qualification"],
        }
    return curves


def training_replay(model: dict, rows15: list[dict]) -> dict:
    """Check physical bounds, integrated mass and ordering on the fitting fold."""
    curves = training_curves(model, rows15)
    observed = [e["observed"]["U"][0] for e in curves.values()]
    predicted = [e["predicted"]["U"][0] for e in curves.values()]
    order = prior._ordering(observed, predicted)
    events = [
        {k: v for k, v in e.items() if k not in {"time", "predicted", "observed"}}
        for e in curves.values()
    ]
    good = all(e["qualification"]["eligible"] for e in events)
    domain = {
        "eligible": good,
        "minimum_U_minus_Uii": min(
            e["qualification"]["minimum_values"]["U_minus_Uii"] for e in events
        ),
    }
    return {
        "eligible": good and order["eligible"],
        "domain": domain,
        "ordering": order,
        "summary": prior.metrics.aggregate(events, len(events)),
        "events": events,
    }
