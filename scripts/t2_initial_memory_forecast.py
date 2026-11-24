"""Predict omitted T2 events from permitted readings, then score frozen curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autoformalism.research.trajectory_function_fitting import _Rollout
from autoformalism.staged_topology import content_hash
from scripts import compare_positive_insulin_rates as insulin
from scripts import t2_challenge_folds as metrics
from scripts import t2_disposal_coupled_replay as dense
from scripts import validate_positive_insulin_transfer as scalar
from scripts.positive_insulin_refit_inputs import TARGETS, TRAIN_IDS

OMISSIONS = ("train_011", "train_012")
INITIALS = ("Gp", "I")
ORDER_TOL = 1e-8


def public_event(row: dict) -> dict:
    """Read the public schedule and only the two permitted initial target readings."""
    name = row["trajectory_id"]
    if name not in TRAIN_IDS:
        raise ValueError("Expected a registered training event")
    time = np.asarray(row["time"], dtype=float)
    if (
        time.ndim != 1
        or len(time) < 2
        or time[0] != 0
        or not np.all(np.isfinite(time))
        or not np.all(np.diff(time) > 0)
    ):
        raise ValueError("Expected an increasing finite grid starting at minute zero")
    inputs, auxiliaries = row["inputs"], row["auxiliaries"]
    covariates = row.get("covariates", {})
    names = [*inputs, *auxiliaries, *covariates]
    if len(names) != len(set(names)) or set(names) & set(TARGETS):
        raise ValueError("Duplicated or prohibited supplied channel roles")
    supplied = {}
    for name_in, vals in (
        inputs | auxiliaries | {k: np.full(len(time), v) for k, v in covariates.items()}
    ).items():
        y = np.asarray(vals, dtype=float)
        if y.shape != time.shape or not np.all(np.isfinite(y)):
            raise ValueError(f"Invalid supplied history for {name_in}")
        supplied[name_in] = y.tolist()
    initial = {k: float(row["targets"][k][0]) for k in INITIALS}
    if not all(np.isfinite(list(initial.values()))):
        raise ValueError("Missing finite public initial reading")
    return {
        "event": name,
        "time": time.tolist(),
        "initial": initial,
        "supplied": supplied,
    }


def _predict(model: dict, pub: dict, engine: _Rollout) -> dict:
    if (
        pub["event"] not in TRAIN_IDS
        or set(pub["initial"]) != set(INITIALS)
        or set(pub["supplied"]) & set(TARGETS)
    ):
        raise ValueError("Invalid forecast roles or event")
    supplied = {k: np.asarray(v, dtype=float) for k, v in pub["supplied"].items()}
    initial = {k: v[0] for k, v in supplied.items()} | pub["initial"]
    if set(engine.readings) - initial.keys() or set(engine.forcings) - supplied.keys():
        raise ValueError("Model requires an unavailable channel")
    event = scalar.Inputs(pub["event"], np.asarray(pub["time"]), supplied, initial)
    scales = model["scales"]
    if set(scales) != set(TARGETS) or not all(
        np.isfinite(s) and s > 0 for s in scales.values()
    ):
        raise ValueError("Expected finite positive training scales for each target")

    base = scalar.mesh(event, model["knots"])
    fine, values, states, u, uii = dense._fine_arrays(
        engine, model, event, base, dense.SUBDIVISIONS
    )
    coarse, prev, *_ = dense._fine_arrays(
        engine, model, event, base, dense.SUBDIVISIONS // 2
    )
    if not np.allclose(fine[::2], coarse, atol=1e-12, rtol=0):
        raise ArithmeticError("Unable to match verification grids")
    idx = np.searchsorted(fine, event.time)
    if not np.array_equal(fine[idx], event.time):
        raise ArithmeticError("Observation times missing from verification grid")
    delta = max(
        float(np.max(abs(values[k][::2] - prev[k])) / scales[k]) for k in TARGETS
    )
    pred = {k: values[k][idx].tolist() for k in TARGETS}
    # This independent scalar solver integrates the affine forcing analytically.
    # It receives no observed insulin beyond its initial reading.
    ref = insulin.predict(
        {
            "trajectory_id": event.name,
            "time": pub["time"],
            "targets": {"I": [initial["I"]]},
            "inputs": {insulin.prior.DRUG: supplied[insulin.prior.DRUG]},
            "auxiliaries": {
                k: v for k, v in supplied.items() if k != insulin.prior.DRUG
            },
        },
        model["knots"],
        model["scalar_model"],
    )
    numerically_resolved = bool(
        delta <= dense.GRID_TOL and np.allclose(pred["I"], ref, rtol=1e-7, atol=1e-5)
    )
    minima = {k: float(np.min(v)) for k, v in values.items()}
    minima["U_minus_Uii"], gap_time = dense._refined_minimum(
        fine,
        values["U_minus_Uii"],
        states,
        u,
        uii,
        engine,
        [model["formula"]["parameters"][k] for k in engine.parameters],
    )
    physical = all(np.isfinite(v) and v >= -dense.DOMAIN_TOL for v in minima.values())
    return {
        "event": event.name,
        "time": pub["time"],
        "public_inputs_sha256": content_hash(pub),
        "model_sha256": content_hash(model),
        "predicted": pred,
        "qualification": {
            "eligible": numerically_resolved and physical,
            "numerically_resolved": numerically_resolved,
            "physical": physical,
            "minimum_values": minima,
            "minimum_U_minus_Uii_time": gap_time,
            "max_scaled_dense_grid_change": delta,
            "max_absolute_scalar_insulin_error": float(
                np.max(abs(np.asarray(pred["I"]) - ref))
            ),
        },
    }


def predict(model: dict, pub: dict) -> dict:
    """Run the complete frozen graph without an observed disposal or future targets."""
    return _predict(model, pub, _Rollout(model["graph"], model["formula"], TARGETS, 2))


def _metrics(model: dict, forecast: dict, truth: dict) -> tuple[dict, dict]:
    name = forecast["event"]
    if name != truth["trajectory_id"] or not np.array_equal(
        forecast["time"], truth["time"]
    ):
        raise ValueError(
            "Forecast and observations concern different events or time grids"
        )
    observed = {}
    n = len(forecast["time"])
    for k in TARGETS:
        y = np.asarray(truth["targets"][k], dtype=float)
        yhat = np.asarray(forecast["predicted"][k], dtype=float)
        if y.shape != (n,) or yhat.shape != (n,) or not np.isfinite([y, yhat]).all():
            raise ValueError(f"Invalid scored trajectory for {name} {k}")
        observed[k] = y.tolist()
    curve = {
        "trajectory_id": name,
        "time": forecast["time"],
        "predicted": forecast["predicted"],
        "observed": observed,
    }
    return metrics.event_metrics(curve, model["scales"]), observed


def score(model: dict, forecast: dict, held: dict) -> dict:
    """Score a held forecast with the frozen fifteen-event target scales."""
    if (
        forecast["event"] not in OMISSIONS
        or forecast["model_sha256"] != content_hash(model)
        or forecast["public_inputs_sha256"] != content_hash(public_event(held))
    ):
        raise ValueError(
            "Held forecast does not match the frozen model and public inputs"
        )
    if not forecast["qualification"]["numerically_resolved"]:
        raise ArithmeticError("Held prediction is numerically unresolved")
    m, observed = _metrics(model, forecast, held)
    ii = np.flatnonzero(np.asarray(forecast["time"]) == 300.0)
    if len(ii) != 1:
        raise ValueError("No observed sample at minute 300")
    j = int(ii[0])
    p = forecast["predicted"]
    return m | {
        "time": forecast["time"],
        "predicted": p,
        "observed": observed,
        "observed_U0": observed["U"][0],
        "predicted_U0": p["U"][0],
        "observed_Gp300": observed["Gp"][j],
        "predicted_Gp300": p["Gp"][j],
        "model_sha256": forecast["model_sha256"],
        "forecast_sha256": content_hash(forecast),
        "qualification": forecast["qualification"],
    }


def _ordering(observed: list[float], predicted: list[float]) -> dict:
    """Check every strictly ordered observed pair using its native disposal units."""
    y, p = np.asarray(observed), np.asarray(predicted)
    high, low = np.where(y[:, None] > y[None, :])
    slack = (p[high] - p[low]) - 0.5 * (y[high] - y[low])
    return {
        "eligible": bool(np.all(slack >= -ORDER_TOL)),
        "pairs": len(slack),
        "min_slack": float(np.min(slack)) if len(slack) else None,
        "violations": int(np.count_nonzero(slack < -ORDER_TOL)),
        "tolerance": ORDER_TOL,
    }


def training_replay(model: dict, rows15: list[dict]) -> dict:
    """Independently replay a candidate, qualifying feasibility on the fitting fold."""
    ids = [r["trajectory_id"] for r in rows15]
    if len(ids) != 15 or not any(
        set(ids) == set(TRAIN_IDS) - {held} for held in OMISSIONS
    ):
        raise ValueError("Expected a registered fifteen-event training fold")
    for target in TARGETS:
        readings = np.concatenate([row["targets"][target] for row in rows15])
        scale = max(1e-8, float(readings.std()))
        if not np.isclose(model["scales"][target], scale, atol=0, rtol=1e-12):
            raise ValueError(f"Incorrect fold normalization scale for {target}")
    engine = _Rollout(model["graph"], model["formula"], TARGETS, 2)
    events, y0, p0 = [], [], []
    for row in rows15:
        f = _predict(model, public_event(row), engine)
        m, obs = _metrics(model, f, row)
        events.append(m | {"qualification": f["qualification"]})
        y0.append(obs["U"][0])
        p0.append(f["predicted"]["U"][0])
    q = [e["qualification"] for e in events]
    domain = {
        "eligible": all(e["eligible"] for e in q),
        "minimum_U_minus_Uii": min(e["minimum_values"]["U_minus_Uii"] for e in q),
    }
    order = _ordering(y0, p0)
    return {
        "eligible": domain["eligible"] and order["eligible"],
        "domain": domain,
        "ordering": order,
        "summary": metrics.aggregate(events, len(events)),
        "events": events,
    }


def plot(scores_by_fold_and_arm: dict, path: Path) -> None:
    """Draw the two omitted trajectories on native physical axes with matched limits."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.1, 5.8), sharex=True, sharey="row")
    colors = {"persistent": "#00689d", "fading": "#ca572a"}
    for col, held in enumerate(OMISSIONS):
        arms = scores_by_fold_and_arm[held]
        a, b = (arms[k] for k in ("persistent", "fading"))
        if (
            a["event"] != held
            or b["event"] != held
            or a["observed"] != b["observed"]
            or a["time"] != b["time"]
        ):
            raise ValueError("Cannot overlay curves from different observations")
        axes[0, col].set_title(held, loc="left", fontsize=12, weight="bold")
        for i, (k, label) in enumerate(
            (("Gp", "Plasma glucose (mg/kg)"), ("U", "Disposal (mg/kg/min)"))
        ):
            ax = axes[i, col]
            ax.plot(
                a["time"], a["observed"][k], color="#202633", lw=2.3, label="Observed"
            )
            for arm, linestyle in (("persistent", "--"), ("fading", "-.")):
                e = arms[arm]
                ax.plot(
                    e["time"],
                    e["predicted"][k],
                    linestyle=linestyle,
                    lw=1.8,
                    color=colors[arm],
                    label=arm.capitalize(),
                )
            ax.spines[["right", "top"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
            if col == 0:
                ax.set_ylabel(label)
        axes[1, col].set_xlabel("Time (min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
