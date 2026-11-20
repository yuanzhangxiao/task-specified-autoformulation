#!/usr/bin/env python3
"""Reconstruct the fitted 20B models, score the meal and pulse, and plot the traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import expit

from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.research import meal_revision_metrics as meal
from scripts import analyze_t2_insulin_terms as insulin
from scripts import refit_t2_20b_revisions as run

LABELS = ("Parent", "109 initial (+)", "109 correction (-)", "116 correction")
COLORS = ("#576373", "#C66A29", "#087F79", "#834995")
STYLES = ("--", "-.", "-", ":")


def changes(time, observed, predicted) -> dict:
    """Measure signed endpoint change, absolute change error and level error."""
    time = list(time)
    a, b = time.index(240), time.index(300)
    actual, estimate = (
        float(observed[b] - observed[a]),
        float(predicted[b] - predicted[a]),
    )
    return {
        "observed_240": float(observed[a]),
        "observed_300": float(observed[b]),
        "fit_240": float(predicted[a]),
        "fit_300": float(predicted[b]),
        "observed_change": actual,
        "fitted_change": estimate,
        "absolute_change_error": abs(estimate - actual),
        "rmse_240_300": float(
            np.sqrt(np.mean((predicted[a : b + 1] - observed[a : b + 1]) ** 2))
        ),
    }


def late_balance(model, point, row, times, states) -> dict:
    """Integrate the signed rates on the same fitted state and verify their sum."""
    c = insulin.coefficients(point)
    s = dict(zip(model.states, states.T, strict=True))
    g, i, x = (s[n] for n in ("Gp", "I", "X_insulin_action"))
    for name in model.inputs:
        if np.any(np.interp(times, row.time, row.external_inputs[name])):
            raise ValueError("expected the unforced 150 g tail")
    signed_product = x * (g - point["Gp_thr"])
    x_feedback = point.get("k_X_decay_glucose", 0.0) * signed_product
    if model.branch == run.MODELS[2]:
        x_feedback = -x_feedback
    x_gate = np.zeros_like(x)
    if "Gp_thr_action" in point:
        x_gate = (
            point["af_process_gain_6_0"]
            * i
            * expit(point["k_glucose_action_thr"] * (g - point["Gp_thr_action"]))
        )
    terms = {
        "I": {
            "glucose_secretion": insulin.secretion(c, g),
            "net_linear_loss": -c["clearance"] * i,
            "new_feedback": point.get("k_I_feedback", 0.0) * signed_product,
        },
        "X_insulin_action": {
            "linear_insulin_drive": c["insulin_to_X"] * i,
            "linear_action_loss": -c["X_decay"] * x,
            "new_feedback": x_feedback + x_gate,
        },
        "Gp": {
            "appearance": c["appearance_rate"] * s["A"],
            "minus_disposal": -(s["af_internal_aux_1"] + x),
        },
    }
    forcing = trajectory_forcing(model.model, row)
    original = {n: point[n] for n in model.shared}
    rhs = np.array(
        [
            model.equations.rhs(t, z, original, forcing)
            for t, z in zip(times[::8], states[::8], strict=True)
        ]
    )
    out = {}
    for state, rates in terms.items():
        total = sum(rates.values())
        np.testing.assert_allclose(
            total[::8], rhs[:, model.states.index(state)], atol=3e-9, rtol=1e-9
        )
        change = float(s[state][-1] - s[state][0])
        integral = float(trapezoid(total, times))
        if not np.isclose(change, integral, rtol=4e-5, atol=0.002):
            raise ValueError(f"signed rates do not reconstruct change in {state}")
        out[state] = {
            "integrated_terms": {
                k: float(trapezoid(v, times)) for k, v in rates.items()
            },
            "rates_at_300": {k: float(v[-1]) for k, v in rates.items()},
            "change": change,
            "integral": integral,
        }
    mask = (np.asarray(row.time) >= 240) & (np.asarray(row.time) <= 300)
    og, ou = (np.asarray(row.targets[n])[mask] for n in ("Gp", "U"))
    return {
        "balances": out,
        "X_range": [float(x.min()), float(x.max())],
        "G_minus_threshold_range": [
            float(min(g - point["Gp_thr"])),
            float(max(g - point["Gp_thr"])),
        ],
        "mean_observed_total_supply": float(
            (og[-1] - og[0] + trapezoid(ou, np.asarray(row.time)[mask])) / 60
        ),
        "mean_fitted_appearance": out["Gp"]["integrated_terms"]["appearance"] / 60,
    }


def score(model, fit_row) -> tuple[dict, dict]:
    """Measure both infusion counterfactuals from matched complete states."""
    retained = fit_row["best"]["endpoint"]
    p = retained["parameters"]
    yy, states = model.original(p, 8)
    y = yy[:, ::8]
    if not np.isclose(model.nmse(y), retained["training_nmse"], atol=1e-10, rtol=1e-8):
        raise ValueError("retained score differs on replay")
    idx = {r.trajectory_id: j for j, r in enumerate(model.rows)}
    scaled = (y - model.observed) / model.scales
    metrics = meal.summarize(
        model.training,
        model.channels,
        dict(zip(model.channels, model.scales, strict=True)),
        scaled.ravel(),
    )
    ri = idx[insulin.LARGE]
    row = model.rows[ri]
    ci = {c: model.channels.index(c) for c in model.channels}
    dense_t, _ = model.grid(8)
    mask = (dense_t >= 240) & (dense_t <= 300)
    large = {
        c: changes(model.time, model.observed[ri, :, j], y[ri, :, j])
        for c, j in ci.items()
    }
    pulse = insulin.matched_pulse(model, p, states, idx)
    reference = (
        model.observed[idx[insulin.PULSE], 29:61, ci["I"]]
        - model.observed[idx[insulin.SHIFTED[0]], 29:61, ci["I"]]
    )
    pulse_series = {}
    for origin, rates in pulse["origins"].items():
        series = np.asarray(rates.pop("series"))
        pulse_series[origin] = series.tolist()
        rates["rmse_30_44"] = float(
            np.sqrt(np.mean((series[1:16] - reference[1:16]) ** 2))
        )
    peak_rows = {
        r["trajectory"]: r
        for r in metrics["rows"]
        if r["matches_shifted_90g_schedule"] and r["external_insulin_zero"]
    }
    if set(peak_rows) != set(insulin.SHIFTED):
        raise ValueError("shifted training schedules differ")
    data = {
        "training_nmse": retained["training_nmse"],
        "per_target_nmse": metrics["target_training_nmse"],
        "target_minima": retained["minima"],
        "parameter_sha256": retained["parameter_sha256"],
        "retained_checkpoint": fit_row["best"]["origin"],
        "large_meal": large,
        "peaks": {
            n: {
                k: r[k]
                for k in (
                    "predicted_insulin_peak_min",
                    "observed_insulin_peak_min",
                    "predicted_insulin_max",
                    "observed_insulin_max",
                    "predicted_peak_unique",
                )
            }
            for n, r in peak_rows.items()
        },
        "paired_infusion": pulse,
        "late_terms": late_balance(model, p, row, dense_t[mask], states[ri, mask]),
        "new_parameters": {
            k: p[k]
            for k in (
                "k_I_feedback",
                "k_X_decay_glucose",
                "af_process_gain_6_0",
                "Gp_thr_action",
                "k_glucose_action_thr",
            )
            if k in p
        },
        "Gp_thr": p["Gp_thr"],
        "infusion_gain": insulin.coefficients(p)["infusion"],
        "max_scaled_solver_difference": retained["max_scaled_solver_difference"],
        "blocks": fit_row["blocks"],
    }
    if "Gp_thr_action" in p:
        gate = expit(
            p["k_glucose_action_thr"] * (yy[:, :, ci["Gp"]] - p["Gp_thr_action"])
        )
        base = insulin.coefficients(p)["insulin_to_X"]
        drive = base + p["af_process_gain_6_0"] * gate
        data["action_gate_all_training"] = {
            "sigmoid_range": [float(gate.min()), float(gate.max())],
            "linear_insulin_gain": base,
            "effective_insulin_gain_range": [float(drive.min()), float(drive.max())],
        }
    # Disable one added coupling for the last hour, holding the complete fitted
    # minute-240 state and all other coefficients fixed. The 109 gains are
    # strictly positive, so use the optimizer's legal lower bound for them.
    common = states[ri, np.flatnonzero(mask)[0]]
    counterfactuals = {}
    for gain in ("k_I_feedback", "k_X_decay_glucose", "af_process_gain_6_0"):
        if gain not in p:
            continue
        lower = float(model.lower[model.names.index(gain)])
        altered = p | {gain: lower}
        model.reduce(altered)
        finals = [
            model.H
            @ insulin.integrate(model, row, altered, [240.0, 300.0], common, method)[-1]
            for method in ("DOP853", "Radau")
        ]
        np.testing.assert_allclose(*finals, rtol=1e-7, atol=1e-6)
        counterfactuals[gain] = {
            "fitted_gain": p[gain],
            "diagnostic_gain": lower,
            "outputs_at_300": dict(
                zip(model.channels, finals[0].tolist(), strict=True)
            ),
            "insulin_rise": float(finals[0][ci["I"]] - large["I"]["fit_240"]),
        }
    data["late_single_gain_controls"] = counterfactuals
    series = {
        "minute": model.time.tolist(),
        "paired_minute": list(range(29, 61)),
        "observed_paired": reference.tolist(),
        "predicted_paired": pulse_series,
        "large": {},
        "shifted": {},
    }
    for schedule, tid in (("large", insulin.LARGE), ("shifted", insulin.SHIFTED[1])):
        j = idx[tid]
        series[schedule] = {
            c: {
                "observed": model.observed[j, :, col].tolist(),
                "predicted": y[j, :, col].tolist(),
            }
            for c, col in ci.items()
        }
    return data, series


def review(packet_path: Path, root: Path) -> tuple[dict, dict]:
    packet = run.load_packet(packet_path)
    plan = run.fit.read_seal(root / "plan.json")
    manifest = run.fit.read_seal(root / "fits.json")
    if (
        plan["zip_sha256"] != run.PACKET_SHA
        or manifest["status"] != "complete"
        or manifest["plan_sha256"] != run.public.content_sha256(plan)
    ):
        raise ValueError("incomplete or different campaign")
    report, curves, endpoints = {}, {}, {}
    for name in run.MODELS:
        model = run.model_for(packet, name)
        results, views, traces = {}, {}, {}
        for kind in run.STARTS:
            key = f"{name}/{kind}"
            row = run.fit.read_seal(root / key / "fit.json")
            if (
                manifest["fits"][key] != run.public.content_sha256(row)
                or row["identity"]
                != {"task": key, "plan_sha256": manifest["plan_sha256"]}
                or row["status"] != "complete"
                or not row["best"]["endpoint"]
            ):
                raise ValueError(f"missing verified fit for {key}")
            results[kind] = row["best"]["endpoint"]
            views[kind], traces[kind] = score(model, row)
        winner = run.fit._select(results, feasible=True)["origin"]
        report[name] = {"retained_start": winner, "starts": views}
        curves[name] = traces[winner]
        endpoints[name] = {
            "problem": packet["models"][name]["problem"],
            "parameters_with_training_roots": results[winner]["parameters"],
            "training_nmse": results[winner]["training_nmse"],
            "parameter_sha256": results[winner]["parameter_sha256"],
            "retained_start": winner,
        }
    summary = {
        "plan_sha256": manifest["plan_sha256"],
        "selection": plan["selection"],
        "public_training_only": True,
        "models": report,
    }
    with run.public._lock(root):
        run.fit.seal(root / "refit-results.json", summary)
        run.fit.seal(root / "trajectory-data.json", curves)
        run.fit.seal(
            root / "fitted-models.json",
            {
                "plan_sha256": manifest["plan_sha256"],
                "training_sha256": plan["training_sha256"],
                "prefix_policy": plan["prefix_policy"],
                "models": endpoints,
            },
        )
    return summary, curves


def plot(curves: dict, path: Path) -> None:
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    panels = (
        (
            "large",
            "I",
            "Late insulin recovery after the 150 g meal",
            "Insulin (pmol/L)",
        ),
        ("large", "Gp", "Glucose after the 150 g meal", "Glucose (mg/kg)"),
        (
            "shifted",
            "I",
            "Insulin after the 90 g meal at minute 60",
            "Insulin (pmol/L)",
        ),
        (
            "paired",
            "I",
            "Added insulin infusion, same starting state",
            "Infusion minus control (pmol/L)",
        ),
    )
    first = curves[run.PARENT]
    for ax, (schedule, channel, title, units) in zip(axes.flat, panels, strict=True):
        paired = schedule == "paired"
        minute = first["paired_minute" if paired else "minute"]
        observed = (
            first["observed_paired"] if paired else first[schedule][channel]["observed"]
        )
        window = (
            slice(minute.index(240), None)
            if (schedule, channel) == ("large", "I")
            else slice(None)
        )
        minute = minute[window]
        if (schedule, channel) == ("large", "Gp"):
            ax.axvspan(240, 300, color="#EEE8D8", alpha=0.55, linewidth=0)
        if paired:
            ax.axvspan(30, 44, color="#EEE8D8", alpha=0.55, linewidth=0)
            ax.axhline(0, color="#888888", lw=0.7)
        ax.plot(minute, observed[window], color="#111111", lw=2.4, label="Observed")
        for name, label, color, style in zip(
            run.MODELS, LABELS, COLORS, STYLES, strict=True
        ):
            c = curves[name]
            pred = (
                c["predicted_paired"][insulin.SHIFTED[0]]
                if paired
                else c[schedule][channel]["predicted"]
            )
            ax.plot(minute, pred[window], color=color, ls=style, lw=1.8, label=label)
        ax.set(
            title=title, xlabel="Time (min)", ylabel=units, xlim=(minute[0], minute[-1])
        )
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=1.7)
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    summary, curves = review(args.packet, args.root)
    plot(curves, args.root / "trajectory-comparison.png")
    print("Verified", len(summary["models"]), "models with two trained starts each.")
