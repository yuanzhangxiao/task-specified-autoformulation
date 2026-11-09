#!/usr/bin/env python3
"""Trace and perturb the two fitted T2 insulin balances on public training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from itertools import pairwise
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import expit

from autoformalism.fitting.simulation import forcing_segment_indices, trajectory_forcing
from autoformalism.research import t2_breadth_fitting as fit
from scripts.run_t2_infusion_control import GAIN, OUTER, SIGNED_PLAN

ARCHIVE_SHA = "d0c4d0a1302e1a158d98abf11ca1f7c7a27e5faf5ac8b187da6fc1f1391d646b"
SOURCE_PLAN = "bbd73e98558b9800e8d1e61b983649e2f516f6cc952dc2b5d992ffa190b12f0b"
MEAL = "meal_rate_g_per_min"
INFUSION = "insulin_pmol_per_kg_min"
SHIFTED = ("train_003", "train_006", "train_007")
LARGE = "train_005"
PULSE = "train_015"
REVIEW = "output/t2-breadth-fitted-review-20261109/"


def attachment(path: Path) -> dict:
    """Authenticate the handoff and load the numerical records as data."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("different pulse handoff")
    hashes, records = {}, {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("invalid pulse handoff member")
            key = str(name.relative_to("t2-pulse-review"))
            if key in hashes:
                raise ValueError("duplicate pulse handoff member")
            data = archive.extractfile(member).read()
            hashes[key] = hashlib.sha256(data).hexdigest()
            if key == "SHA256SUMS":
                manifest = {
                    n: h
                    for h, n in (
                        line.split("  ", 1) for line in data.decode().splitlines()
                    )
                }
            for item in (
                "construction-feedback",
                "independent-starts",
                "sign-checked-warm-starts",
            ):
                if key == REVIEW + item + ".json":
                    records[item] = json.loads(data)
    del hashes["SHA256SUMS"]
    if (
        hashes != manifest
        or records["independent-starts"]["plan_sha256"] != SIGNED_PLAN
    ):
        raise ValueError("pulse handoff inventory or source starts differ")
    warm = records["sign-checked-warm-starts"]
    if warm["sha256"] != fit.public.content_sha256(warm["value"]):
        raise ValueError("sign-mirrored parameter seal differs")
    return records


def coefficients(p: dict) -> dict:
    """Combine actual outer gains, signed inner coefficients and explicit balances."""
    c = {
        "meal": p["c_I"],
        "infusion": p[OUTER] * p[GAIN],
        "clearance": 1 + p["af_process_gain_2_0"] * p[fit.fitting.DECAY],
        "appearance_rate": p["k_abs"],
        "absorption_gain": p["k_abs_rate"],
        "insulin_to_X": p["k_X_term_11_0_70bc44fdc2"]
        + p["af_process_gain_4_0"] * p["k_insulin_term_4_0_2f644cff88"],
        "X_decay": p["k_insulin_X_term_11_1_7dde7939d1"]
        + p["af_process_gain_5_0"] * p["k_insulin_X_term_5_0_dbe4d15663"],
        "basal_disposal_drift": p["k_Uii_term_12_0_244d72574a"],
    }
    g = p["af_process_gain_1_0"]
    if "tau_M" in p:
        c.update(
            glucose=g * p[fit.fitting.SOURCE],
            action=g * p[fit.fitting.DELAY],
            memory_to_I=g * p["gain_M"],
            memory_input_gain=p["k_M_rise"],
            memory_time=p["tau_M"],
        )
    else:
        c.update(
            max_secretion=g * p["prod_max"],
            sigmoid_slope=p["k_prod"],
            sigmoid_midpoint=p["Gp_thr"],
        )
    return c


def secretion(c: dict, glucose):
    return c["max_secretion"] * expit(
        c["sigmoid_slope"] * (glucose - c["sigmoid_midpoint"])
    )


def decomposition(m, p, row, ts, states):
    """Resolve signed rates and cross-check against all original state equations."""
    c = coefficients(p)
    s = dict(zip(m.states, states.T, strict=True))
    G, insulin, X, A, Q = (
        s[n] for n in ("Gp", "I", "X_insulin_action", "A", "af_internal_aux_1")
    )
    r, v = (np.interp(ts, row.time, row.external_inputs[k]) for k in (MEAL, INFUSION))
    terms = {
        "direct_meal": c["meal"] * r,
        "infusion": c["infusion"] * v,
        "net_self": -c["clearance"] * insulin,
    }
    rates = {
        "Gp": r + c["appearance_rate"] * A - Q - X,
        "A": c["absorption_gain"] * r - c["appearance_rate"] * A,
        "af_internal_aux_1": np.full_like(insulin, c["basal_disposal_drift"]),
        "X_insulin_action": c["insulin_to_X"] * insulin - c["X_decay"] * X,
    }
    if m.branch == "c15-3":
        terms.update(
            memory=c["memory_to_I"] * s["M"],
            glucose=c["glucose"] * G,
            action=c["action"] * X,
        )
        rates["M"] = c["memory_input_gain"] * r - s["M"] / c["memory_time"]
    else:
        terms["glucose_secretion"] = secretion(c, G)
    rates["I"] = sum(terms.values())
    params = {n: p[n] for n in m.shared}
    forcing = trajectory_forcing(m.model, row)
    stride = round(1 / (ts[1] - ts[0]))
    for j in range(0, len(ts), stride):
        rhs = m.equations.rhs(ts[j], states[j], params, forcing)
        np.testing.assert_allclose(
            [rates[n][j] for n in m.states], rhs, atol=2e-9, rtol=1e-10
        )
    return (
        terms,
        rates,
        {
            "Gp": G,
            "I": insulin,
            "X": X,
            "A": A,
            "U": Q + X,
            "appearance": c["appearance_rate"] * A,
        },
    )


def integrate(m, row, point, times, initial, method, *, transit=0.0):
    """Replay the full original RHS with optional dose-preserving secretion transit."""
    params = {k: point[k] for k in m.shared}
    forcing = trajectory_forcing(m.model, row)
    if transit:
        if m.branch != "c15-3" or transit <= 0:
            raise ValueError("transit applies to the meal-to-insulin path")
        im, ii = m.states.index("M"), m.states.index("I")
        gain = point["af_process_gain_1_0"] * point["gain_M"]
        x = np.r_[initial, 0.0]
    else:
        x = initial

    def rhs(t, z):
        value = m.equations.rhs(t, z[: m.n], params, forcing)
        if not transit:
            return value
        source = gain * z[im]
        value[ii] += z[-1] - source
        return np.r_[value, (source - z[-1]) / transit]

    times = np.asarray(times, float)
    knots = np.asarray(row.time)[forcing_segment_indices(m.model, row)]
    knots = np.r_[times[0], knots[(knots > times[0]) & (knots < times[-1])], times[-1]]
    blocks = [x[None, :]]
    for a, b in pairwise(knots):
        grid = times[(times > a) & (times <= b)]
        # The solver must carry the exact segment endpoint even if not requested.
        sol = solve_ivp(
            rhs, (a, b), x, method=method, dense_output=True, rtol=1e-10, atol=1e-12
        )
        if not sol.success:
            raise ValueError(sol.message)
        x = sol.y[:, -1]
        if len(grid):
            blocks.append(sol.sol(grid).T)
    return np.concatenate(blocks)[:, : m.n]


def matched_pulse(m, point, states, rows):
    """Condition both schedules on the same complete minute-29 state, twice."""
    base, pulse = rows[SHIFTED[0]], rows[PULSE]
    ii = m.states.index("I")
    ci = m.channels.index("I")
    for k in m.inputs:
        if k != INFUSION:
            np.testing.assert_array_equal(
                m.rows[base].external_inputs[k], m.rows[pulse].external_inputs[k]
            )
    np.testing.assert_allclose(
        m.observed[base, :30], m.observed[pulse, :30], atol=1e-8, rtol=0
    )
    obs = (m.observed[pulse, :, ci] - m.observed[base, :, ci])[29:61]
    ts = np.arange(29, 61, dtype=float)
    result = {}
    for tid, idx in ((SHIFTED[0], base), (PULSE, pulse)):
        x = states[idx, 29 * 8]
        deltas = []
        for method in ("DOP853", "Radau"):
            runs = [
                integrate(m, m.rows[i], point, ts, x, method)[:, ii]
                for i in (base, pulse)
            ]
            deltas.append(runs[1] - runs[0])
        np.testing.assert_allclose(*deltas, atol=1e-6, rtol=1e-8)
        delta = deltas[0]
        result[tid] = {
            "I40": float(delta[11]),
            "min_30_44": float(min(delta[1:16])),
            "max_30_44": float(max(delta[1:16])),
            "I60": float(delta[-1]),
            "series": delta.tolist(),
        }
    return {
        "observed_I40": float(obs[11]),
        "observed_I60": float(obs[-1]),
        "observed_min_30_44": float(min(obs[1:16])),
        "origins": result,
    }


def windows_record(terms, rates, s, row, ts, pred_peak, observed_peak, end):
    """Compare an actual zero crossing to the subsequent observed peak."""

    def idx(t):
        return int(np.searchsorted(ts, t))

    snapshots = {}
    for t in (end, pred_peak - 5, pred_peak, pred_peak + 5, observed_peak):
        i = idx(t)
        snapshots[str(int(t))] = {
            **{k: float(a[i]) for k, a in terms.items()},
            "I_dot": float(rates["I"][i]),
            "I": float(s["I"][i]),
            "observed_I": row.targets["I"][int(t)],
        }
    mask = (ts >= pred_peak) & (ts <= observed_peak)
    sums = {k: float(trapezoid(a[mask], ts[mask])) for k, a in terms.items()}
    change = s["I"][idx(observed_peak)] - s["I"][idx(pred_peak)]
    np.testing.assert_allclose(sum(sums.values()), change, atol=0.002, rtol=1e-5)
    return {
        "meal_end": end,
        "predicted_peak": pred_peak,
        "observed_peak": observed_peak,
        "memory_drive_peak_min": float(ts[np.argmax(terms["memory"])]),
        "snapshots": snapshots,
        "term_integrals_between_peaks": sums,
        "predicted_I_change": change,
        "observed_I_change": row.targets["I"][int(observed_peak)]
        - row.targets["I"][int(pred_peak)],
    }


def memory_control(m, point, row, settings):
    """Perturb one pathway at fixed fitted coefficients and initial state."""
    variant, p, lag = settings
    x = m.initial(point, row)
    series = [
        integrate(m, row, p, m.time, x, method, transit=lag)
        for method in ("DOP853", "Radau")
    ]
    np.testing.assert_allclose(*series, atol=2e-5, rtol=1e-7)
    y = series[0] @ m.H.T
    insulin = y[:, m.channels.index("I")]
    result = {
        "variant": variant,
        "peak_min": float(m.time[np.argmax(insulin)]),
        "peak": float(max(insulin)),
        "I300": float(insulin[-1]),
        "minimum_outputs": dict(zip(m.channels, y.min(axis=0).tolist(), strict=True)),
    }
    if row.trajectory_id == "train_006" and variant == "30_min_transit":
        result["insulin_curve"] = insulin.tolist()
    return result


def forced_insulin(c, times, g, initial, order=8):
    """Exact integrating factor with quadrature of sigmoid at interpolated readings."""
    times, g = np.asarray(times), np.asarray(g)
    if len(times) != len(g) or np.any(np.diff(times) <= 0):
        raise ValueError("increasing measured glucose knots required")
    x, w = np.polynomial.legendre.leggauss(order)
    half = np.diff(times)[:, None] / 2
    nodes = times[:-1, None] + half * (1 + x)
    glucose = np.interp(nodes, times, g)
    contributions = (
        half * w * np.exp(-c["clearance"] * (times[-1] - nodes)) * secretion(c, glucose)
    )
    return float(
        np.exp(-c["clearance"] * (times[-1] - times[0])) * initial + contributions.sum()
    )


def late_flux_controls(c, row, s, ts, implied):
    """Perturb one glucose flux during the last hour and propagate the other states."""
    start = int(np.searchsorted(ts, 240))
    initial = [s[n][start] for n in ("Gp", "I", "X")]
    basal = s["U"][start] - initial[2]
    flux = s["appearance"][start]
    grid = np.arange(240.0, 301.0)
    result = {}
    for key in ("original", "mean_source_from_recorded_balance", "recorded_disposal"):

        def rhs(t, z, key=key):
            glucose, insulin, action = z
            source = flux * np.exp(-c["appearance_rate"] * (t - 240))
            disposal = basal + c["basal_disposal_drift"] * (t - 240) + action
            if key == "mean_source_from_recorded_balance":
                source = implied
            if key == "recorded_disposal":
                disposal = float(np.interp(t, row.time, row.targets["U"]))
            return [
                source - disposal,
                secretion(c, glucose) - c["clearance"] * insulin,
                c["insulin_to_X"] * insulin - c["X_decay"] * action,
            ]

        runs = []
        for solver in ("DOP853", "Radau"):
            sol = solve_ivp(
                rhs,
                (240, 300),
                initial,
                t_eval=grid,
                max_step=0.5,
                method=solver,
                rtol=1e-10,
                atol=1e-12,
            )
            if not sol.success:
                raise ValueError(sol.message)
            runs.append(sol.y)
        np.testing.assert_allclose(*runs, rtol=2e-7, atol=1e-5)
        result[key] = {"Gp300": runs[0][0, -1], "I300": runs[0][1, -1]}
    np.testing.assert_allclose(
        [result["original"]["Gp300"], result["original"]["I300"]],
        [s["Gp"][-1], s["I"][-1]],
        rtol=1e-7,
        atol=1e-6,
    )
    return result


def recovery_record(c, terms, rates, s, row, ts):
    """Partition the large-meal insulin error and audit the linked glucose balance."""
    if np.any(np.asarray(row.external_inputs[MEAL])[240:]) or np.any(
        row.external_inputs[INFUSION]
    ):
        raise ValueError("late balance assumes zero ingestion and infusion")
    ix = (ts >= 240) & (ts <= 300)
    gt = np.asarray(row.targets["Gp"])[240:301]
    it = np.asarray(row.targets["I"])[240:301]
    ut = np.asarray(row.targets["U"])[240:301]
    t = np.asarray(row.time)[240:301]
    gf, ifit = s["Gp"][ix], s["I"][ix]
    e = np.exp(-c["clearance"] * 60)
    forced = forced_insulin(c, t, gt, ifit[0])
    second = forced_insulin(c, t, gt, it[0])
    np.testing.assert_allclose(
        forced, forced_insulin(c, t, gt, ifit[0], 4), rtol=1e-11, atol=1e-10
    )
    # Check the integrating-factor reconstruction against the full ODE replay.
    np.testing.assert_allclose(
        forced_insulin(c, ts[ix], gf, ifit[0]), ifit[-1], atol=2e-4, rtol=0
    )
    parts = {
        "glucose_path": ifit[-1] - forced,
        "carried_insulin_error": e * (ifit[0] - it[0]),
        "insulin_law_on_recorded_Gp": second - it[-1],
    }
    np.testing.assert_allclose(sum(parts.values()), ifit[-1] - it[-1], atol=1e-8)
    appearance = trapezoid(s["appearance"][ix], ts[ix]) / 60
    uf = trapezoid(s["U"][ix], ts[ix]) / 60
    uo = trapezoid(ut, t) / 60
    implied = (gt[-1] - gt[0]) / 60 + uo
    np.testing.assert_allclose(appearance - uf, (gf[-1] - gf[0]) / 60, atol=5e-6)
    snapshots = {}
    for minute in (240, 270, 300):
        j = np.searchsorted(ts, minute)
        snapshots[str(minute)] = {
            "Gp": s["Gp"][j],
            "observed_Gp": row.targets["Gp"][minute],
            "I": s["I"][j],
            "observed_I": row.targets["I"][minute],
            "U": s["U"][j],
            "observed_U": row.targets["U"][minute],
            "appearance": s["appearance"][j],
            "Gp_dot": rates["Gp"][j],
            "secretion": terms["glucose_secretion"][j],
            "net_self": terms["net_self"][j],
            "I_dot": rates["I"][j],
        }
    return {
        "snapshots": snapshots,
        "I300_error": ifit[-1] - it[-1],
        "error_components": parts,
        "recorded_glucose_drive_I300_from_fitted_I240": forced,
        "recorded_glucose_drive_I300_from_observed_I240": second,
        "rate_if_Gp_replaced_at_300": secretion(c, gt[-1]) - c["clearance"] * ifit[-1],
        "rate_if_Gp_and_I_replaced_at_300": secretion(c, gt[-1])
        - c["clearance"] * it[-1],
        "last_10_observed_mean_I_rate": (it[-1] - it[-11]) / 10,
        "closed_loop_flux_checks": late_flux_controls(c, row, s, ts, implied),
        "recorded_Gp_driven_curve": [float(ifit[0])]
        + [
            forced_insulin(c, t[: j + 1], gt[: j + 1], ifit[0])
            for j in range(1, len(t))
        ],
        "mean_appearance": appearance,
        "mean_fitted_disposal": uf,
        "mean_observed_disposal": uo,
        "mean_supply_implied_by_recorded_Gp_and_U": implied,
        "appearance_from_A0_at_300": c["appearance_rate"]
        * s["A"][0]
        * np.exp(-300 * c["appearance_rate"]),
        "appearance_timescale": 1 / c["appearance_rate"],
        "I_timescale": 1 / c["clearance"],
        "X_timescale": 1 / c["X_decay"],
    }


def run_one(m, entry, expected, *, interventions: bool) -> dict:
    """Replay a single fitted vector and reconcile both replication and balances."""
    point = entry["parameters"]
    if entry["parameter_sha256"] != fit.public.content_sha256(point):
        raise ValueError("fitted parameter digest differs")
    y, states = m.original(point, 8, method="DOP853")
    check, _ = m.original(point, 1, method="Radau")
    if not np.allclose(y[:, ::8], check, rtol=5e-7, atol=1e-6 * m.scales):
        raise ValueError("original adaptive solvers disagree")
    np.testing.assert_allclose(
        m.nmse(y[:, ::8]), entry["training_nmse"], rtol=2e-7, atol=1e-9
    )
    if not np.all(y.min(axis=(0, 1)) >= -fit.fitting.DOMAIN_TOL * m.scales):
        raise ValueError("nonphysical endpoint")
    by_id = {r.trajectory_id: i for i, r in enumerate(m.rows)}
    ts = m.grid(8)[0]
    params = coefficients(point)
    paired = matched_pulse(m, point, states, by_id)
    if expected:
        if expected["parameter_sha256"] != entry["parameter_sha256"]:
            raise ValueError("pulse attachment references different coefficients")
        for tid, metric in paired["origins"].items():
            np.testing.assert_allclose(
                metric["I40"], expected["pulse_at_40"][tid], rtol=1e-6, atol=2e-6
            )
    result = {
        "parameter_sha256": entry["parameter_sha256"],
        "coefficients": params,
        "training_nmse": entry["training_nmse"],
        "pulse": paired,
        "minima": dict(zip(m.channels, y.min(axis=(0, 1)).tolist(), strict=True)),
        "pulse_trajectory_nmse": float(
            np.mean(((y[by_id[PULSE], ::8] - m.observed[by_id[PULSE]]) / m.scales) ** 2)
        ),
        "shifted_peak_minutes": [
            float(m.time[np.argmax(y[by_id[tid], ::8, m.channels.index("I")])])
            for tid in SHIFTED
        ],
        "I300": float(y[by_id[LARGE], -1, m.channels.index("I")]),
    }
    if not interventions:
        return result
    shifts = {v["trajectory"]: v for v in entry["shifted_90g"]}
    if set(shifts) != set(SHIFTED):
        raise ValueError("incorrect timing witnesses")
    for tid in [*SHIFTED, LARGE]:
        idx = by_id[tid]
        row = m.rows[idx]
        if any(row.external_inputs[INFUSION]):
            raise ValueError("meal witnesses require zero external infusion")
        terms, rates, s = decomposition(m, point, row, ts, states[idx])
        if m.branch == "c15-3":
            if tid in shifts:
                v = shifts[tid]
                end = v["meal_events"][0]["end_min"]
                result.setdefault("meal_balance", {})[tid] = windows_record(
                    terms,
                    rates,
                    s,
                    row,
                    ts,
                    v["predicted_insulin_peak_min"],
                    v["observed_insulin_peak_min"],
                    end,
                )
            # Both timing interventions preserve the memory impulse-response area.
            variants = [
                ("30_min_transit", point, 30.0),
                (
                    "double_memory_time_same_area",
                    point
                    | {"tau_M": 2 * point["tau_M"], "gain_M": point["gain_M"] / 2},
                    0.0,
                ),
                (
                    "25_percent_weaker_action_inhibition",
                    point | {fit.fitting.DELAY: 0.75 * point[fit.fitting.DELAY]},
                    0.0,
                ),
            ]
            controls = result.setdefault("pathway_controls", {}).setdefault(tid, {})
            for setting in variants:
                controls[setting[0]] = memory_control(m, point, row, setting)
        elif tid == LARGE:
            result["recovery"] = recovery_record(params, terms, rates, s, row, ts)
        if (m.branch == "c15-3" and tid == "train_006") or (
            m.branch == "c15-2" and tid == LARGE
        ):
            result["plot"] = {
                "trajectory": tid,
                "time": m.time.tolist(),
                "observed": {k: list(row.targets[k]) for k in ("I", "Gp", "U")},
                "model": {k: v[::8].tolist() for k, v in s.items()},
                "insulin_rates": {k: v[::8].tolist() for k, v in terms.items()},
                "I_dot": rates["I"][::8].tolist(),
                "Gp_dot": rates["Gp"][::8].tolist(),
            }
    return result


def saved_replay(
    folder: Path, plan_hash: str, model, entry: dict, expected, perturb: bool
):
    """Resume a completed deterministic replay at the same vector and plan."""
    binding = {
        "plan": plan_hash,
        "parameters": entry["parameter_sha256"],
        "perturb": perturb,
    }
    with fit.public._lock(folder):
        path = folder / "replay.json"
        if path.is_file():
            saved = fit.read_seal(path)
            if saved["binding"] != binding:
                raise ValueError("replay checkpoint differs")
            return saved["value"]
        replay = run_one(model, entry, expected, interventions=perturb)
        fit.seal(path, {"binding": binding, "value": replay})
        return replay


def audit(source: Path, signed: Path, control: Path, pulse: Path, output: Path) -> dict:
    """Authenticate all four signed/bounded pairs, and keep training-only evidence."""
    attached = attachment(pulse)
    orig, data = fit.verify(source, runtime=False)
    signed_plan = fit.read_seal(signed / "plan.json")
    if (
        fit.public.content_sha256(orig) != SOURCE_PLAN
        or fit.public.content_sha256(signed_plan) != SIGNED_PLAN
    ):
        raise ValueError("incorrect source model exchange")
    comparison = fit.read_seal(control / "summary.json")
    if (
        comparison["status"] != "complete"
        or comparison["comparison"]["signed_plan_sha256"] != SIGNED_PLAN
    ):
        raise ValueError("incorrect bounded comparison")
    signed_summary = fit.read_seal(signed / "summary.json")
    if (
        signed_summary["status"] != "complete"
        or signed_summary["plan_sha256"] != SIGNED_PLAN
    ):
        raise ValueError("incomplete independent-start study")
    bounded_plan = fit.read_seal(control / "plan.json")
    if (
        fit.public.content_sha256(bounded_plan)
        != comparison["comparison"]["bounded_plan_sha256"]
    ):
        raise ValueError("different bounded plan")
    references = attached["independent-starts"]["models"]
    mirror_models = attached["sign-checked-warm-starts"]["value"]["models"]
    plan = {
        "protocol": "t2-insulin-term-attribution-1",
        "source_sha256": fit.public._source_identity(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matched_plan": SOURCE_PLAN,
        "independent_plan": SIGNED_PLAN,
        "pulse_archive_sha256": ARCHIVE_SHA,
        "bounded_summary_sha256": fit.public.content_sha256(comparison),
        "training_sha256": data["training_sha256"],
        "scope": "public training balances and a separate infusion-gain control",
    }
    with fit.public._lock(output):
        fit.seal(output / "analysis-plan.json", plan)
    digest_plan = fit.public.content_sha256(plan)
    result = {
        "plan": plan,
        "signed": {},
        "nonnegative_control": {},
        "mirrored_reference": {},
    }
    for name, expected in references.items():
        branch = expected["branch"]
        m = fit._model(branch, data)
        if data["models"][branch]["model_sha256"] != expected["model_sha256"]:
            raise ValueError("handoff scientific model differs")
        for group, folder, digest in (
            ("signed", signed, signed_summary["tasks"][name]["result_sha256"]),
            ("nonnegative_control", control, comparison["result_sha256"][name]),
        ):
            row = fit.read_seal(folder / name / "result.json")
            if fit.public.content_sha256(row) != digest or row["status"] != "complete":
                raise ValueError("fitted endpoint differs")
            rec = row["best_admissible"]
            result[group][name] = saved_replay(
                output / group / name,
                digest_plan,
                m,
                rec,
                expected if group == "signed" else None,
                group == "signed",
            )
        if branch not in result["mirrored_reference"]:
            mirror = mirror_models[branch]
            ref = attached["construction-feedback"]["records"][branch]
            if (
                mirror["model_sha256"] != expected["model_sha256"]
                or mirror["source_parameter_sha256"] != ref["fitted_parameter_sha256"]
                or fit.public.content_sha256(mirror["parameters"])
                != mirror["parameter_sha256"]
            ):
                raise ValueError("different sign-only control")
            current = saved_replay(
                output / "mirrored_reference" / branch,
                digest_plan,
                m,
                mirror,
                None,
                False,
            )
            np.testing.assert_allclose(
                current["pulse"]["origins"][SHIFTED[0]]["I40"],
                ref["sign_only_fixed_parameter_check"]["pulse_delta_I_at_40"],
                atol=2e-5,
            )
            result["mirrored_reference"][branch] = current
    with fit.public._lock(output):
        fit.seal(output / "equation-evidence.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "signed", "control", "pulse", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    a = parser.parse_args()
    audit(a.source, a.signed, a.control, a.pulse, a.output)
    print(a.output / "equation-evidence.json")
