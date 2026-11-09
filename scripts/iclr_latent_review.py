"""Audit frozen development models' glucose-balance terms and latent readouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting import prefix_additive_reduction as additive
from autoformalism.fitting import prefix_structural_reduction as structural
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas import CandidateModel
from scripts import evaluate_t2_latent_dynamics as probe
from scripts import iclr_anchor_fidelity as assay
from scripts import iclr_reference_fits as fits

PLAN = "007208388d9c58cbb02b82451eda131388559fc89d802f0a9866e8f6eb33d1fa"
SCORES = "9e0342d655cdfc556cc176befde52386cc8ee65f367d6701d486cf8394ebf3f8"
TRACE_ZIP = "3092a860152247773e74a6086e18f95f5e4c64145a8db51dce65b1dbcea9b98a"
MEAL = "meal_event_g"
INSULIN = "insulin_pmol_per_kg_min"
DRIVERS = {MEAL, INSULIN, "Uii"}
HIDDEN = ("Ra", "Uid", "EGP", "X")
METHODS = (
    "all_latent_affine",
    "glucose_reachable_affine",
    "generated_insulin_affine",
    "all_latent_ridge",
)
RIDGE_ALPHAS = (1.0, 0.1, 0.01, 0.001, 0.0001)
# Independent review of the signed balance and dynamic coordinates in each frozen fit.
MODEL_NOTES = {
    "52ed5620": (
        "U sums generated I and an infusion-driven X and enters dGp positively."
    ),
    "9ae80116": (
        "Only infusion-driven I_act enters U. I_delay and I_mem lack a Gp path."
    ),
    "99c3d23f": "Two distinct fitted states X and Y enter dGp with separate gains.",
    "db64e886": "U combines the supplied Uii and X behind a single fitted gain.",
    "5561d003": (
        "The subtracted A mixes direct meal forcing with insulin-driven dynamics."
    ),
    "8001b320": (
        "Uii, X and Y share a fitted glucose gain; both latent states enter dGp."
    ),
    "f16a59d0": (
        "The declared process U has no glucose path; dGp subtracts a gain times X."
    ),
    "9821a00a": "Declared U is unused; dGp adds Uii and subtracts a scaled X fed by Y.",
    "d01c5397": (
        "A meal-driven Y and insulin-driven X enter the balance with distinct gains."
    ),
    "e633d9f5": "U_delayed enters dGp twice; its underlying X_delayed also affects dI.",
    "aa1fa8ad": "U_delayed negates I times X_delayed, then dGp subtracts U_delayed.",
    "f7afd180": "Dynamic U is subtracted via three inverse-time coefficients in dGp.",
    "8af5fca3": (
        "X_I and I have identical infusion/clearance laws, with different initials."
    ),
    "4005f90a": "The sink multiplies I by X; X is driven by imposed infusion.",
    "b243e0a8": (
        "The double negative adds I times DelayedInsulinAction to dGp. X is orphaned."
    ),
    "f849d7c1": (
        "Distinct meal Xg_delayed and insulin X_delayed enter dGp with opposite signs."
    ),
}


def fitted_equations(candidate: CandidateModel, parameters: dict) -> dict:
    """Expand the restricted grammar and substitute each fitted value exactly."""
    if any(m.channel != m.expression for m in candidate.observation_mappings):
        raise ValueError("this retrospective assay requires identity observation maps")
    parser = RestrictedParser()
    processes = {
        p.name: parser.parse(p.expression, location=p.name).tree.body
        for p in candidate.processes
    }
    if set(parameters) != {p.name for p in candidate.parameters}:
        raise ValueError("coefficient roster differs from candidate")
    equations = {}
    for e in candidate.state_equations:
        tree = parser.parse(e.rhs, location=e.state).tree.body
        poly = structural._polynomial(tree, processes, set(parameters))
        equations[e.state] = additive._substitute(poly, parameters)
    return equations


def ancestors(equations: dict, names: set[str]) -> set[str]:
    """Compute the closure through nonzero terms, including external drivers."""
    visited = set()
    todo = set(names)
    while todo:
        name = todo.pop()
        if name in visited:
            continue
        visited.add(name)
        todo.update(k for terms in equations.get(name, {}) for k, _ in terms)
    return visited


def term_text(powers: tuple, coefficient: float) -> str:
    factors = [k if n == 1 else f"{k}**{n}" for k, n in powers]
    return " * ".join([repr(float(coefficient)), *factors])


def components(equations: dict) -> list[dict]:
    """Group individual glucose-balance monomials by their upstream drivers."""
    result = []
    for powers, coefficient in sorted(equations["Gp"].items()):
        drives = ancestors(equations, {name for name, _ in powers}) & DRIVERS
        if drives == {MEAL}:
            group = "meal_only"
        elif INSULIN in drives:
            group = "insulin_linked"
        else:
            group = "other"
        result.append(
            {
                "powers": powers,
                "coefficient": float(coefficient),
                "expression": term_text(powers, coefficient),
                "upstream_forcings": sorted(drives),
                "group": group,
            }
        )
    return result


def term_series(term: dict, curves: dict, reference: dict) -> np.ndarray:
    """Evaluate on saved generated states and the three permitted public forcings."""
    result = np.full(len(reference["time"]), term["coefficient"])
    for k, n in term["powers"]:
        if k in DRIVERS:
            series = reference[k]
        else:
            series = curves[k if k in ("Gp", "I") else "latent." + k]
        result *= np.asarray(series) ** n
    return result


def product_changes(term: dict, curves: dict, refs: dict, case: str, base: str) -> dict:
    """Partition a bilinear term difference at the paired control's factor values."""
    if len(term["powers"]) != 2:
        return {}
    a, b = term["powers"]

    def values(power, k):
        return term_series({"powers": [power], "coefficient": 1.0}, curves[k], refs[k])

    a0, b0 = values(a, base), values(b, base)
    da, db = values(a, case) - a0, values(b, case) - b0
    gain = term["coefficient"]
    return {
        f"change in {a[0]} alone": gain * da * b0,
        f"change in {b[0]} alone": gain * db * a0,
        "product interaction": gain * da * db,
    }


def physical_review(
    subject: dict, candidate: CandidateModel, terms: list[dict], active: set[str]
) -> list[dict]:
    """Record evidence for each candidate-level physical mapping decision."""
    states = [s for s in candidate.states if s.name not in ("I", "Gp")]
    if any(d.unit != "unspecified" for d in (*states, *candidate.processes)):
        raise ValueError("changed unit annotations require a new native review")
    components_by_driver = {
        g: " + ".join(t["expression"] for t in terms if t["group"] == g)
        for g in ("meal_only", "insulin_linked")
    }
    reasons = {
        "Ra": (
            "Meal-only dGp term has balance-rate units inferred from the public "
            "target. No candidate flux has documented physical calibration to gut "
            "appearance. The signed term is measured separately without fitting."
        ),
        "Uid": (
            "The linked term acts on plasma Gp. Simulator Uid removes glucose from "
            "tissue Gt; plasma also responds to exchange and production. A plasma "
            "balance term cannot uniquely allocate an insulin effect to tissue uptake."
        ),
        "EGP": (
            "No separately unit-annotated hepatic production output; insulin effects "
            "on production and uptake can be absorbed into one fitted plasma balance."
        ),
        "X": (
            "Latent coordinates have unspecified units, free initial scale and fitted "
            "gains. An action name or delayed pathway supplies no physiological "
            "remote-action scaling or independently verified correspondence."
        ),
    }
    return [
        {
            **subject,
            "quantity": h,
            "reference_unit": assay.UNITS[h],
            "review_status": "unavailable",
            "fitted_equation_review": MODEL_NOTES[subject["candidate_hash"][:8]],
            "declared_latents": "|".join(s.name for s in states),
            "glucose_reachable_latents": "|".join(
                s.name for s in states if s.name in active
            ),
            "declared_processes": "|".join(d.name for d in candidate.processes),
            "fitted_meal_only_glucose_terms": components_by_driver["meal_only"],
            "fitted_insulin_linked_glucose_terms": components_by_driver[
                "insulin_linked"
            ],
            "evidence": reasons[h],
        }
        for h in HIDDEN
    ]


def ridge_fit(xs: np.ndarray, y: np.ndarray, alpha: float) -> tuple:
    """Standardize and solve the sample-mean ridge objective on one training fold."""
    center, scale = xs.mean(axis=0), xs.std(axis=0)
    active = scale > 1e-12 * np.maximum(1, np.abs(center))
    scale = np.where(active, scale, 1)
    z = np.where(active, (xs - center) / scale, 0)
    mean = float(y.mean())
    lhs = (z.T @ z) / len(y) + alpha * np.eye(xs.shape[1])
    weights = np.linalg.solve(lhs, (z.T @ (y - mean)) / len(y))
    return center, scale, weights, mean


def multivariate_probe(
    names: list[str], curves: dict, refs: dict, ids: list[str], h: str
) -> dict:
    """Choose a penalty by leave-one-training-schedule-out error; refit once."""
    if not names:
        return {"status": "no_latent_coordinate"}
    if len(ids) < 2:
        raise ValueError("need at least two training schedules for blocked validation")
    xs = {k: np.column_stack([curves[k][n] for n in names]) for k in ids}
    ys = {k: np.asarray(refs[k][h]) for k in ids}
    losses = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for held in ids:
            train = [k for k in ids if k != held]
            center, scale, w, mean = ridge_fit(
                np.concatenate([xs[k] for k in train]),
                np.concatenate([ys[k] for k in train]),
                alpha,
            )
            yhat = mean + ((xs[held] - center) / scale) @ w
            errors.append(np.mean((yhat - ys[held]) ** 2))
        losses.append(float(np.mean(errors)))
    alpha = RIDGE_ALPHAS[int(np.argmin(losses))]
    x, y = np.concatenate(list(xs.values())), np.concatenate(list(ys.values()))
    center, scale, w, mean = ridge_fit(x, y, alpha)
    err = float(np.mean((mean + ((x - center) / scale) @ w - y) ** 2))
    return {
        "name": "|".join(names),
        "input_names": names,
        "weights": w.tolist(),
        "centers": center.tolist(),
        "scales": scale.tolist(),
        "offset": mean,
        "alpha": alpha,
        "blocked_validation_mse": min(losses),
        "calibration_mse": err,
    }


def decode_fit(fit: dict, curve: dict) -> np.ndarray:
    """Apply a trained affine or multivariate decoder with frozen coefficients."""
    if "weights" not in fit:
        return probe.decode(fit, curve)
    x = np.column_stack([curve[k] for k in fit["input_names"]])
    return fit["offset"] + ((x - fit["centers"]) / fit["scales"]) @ fit["weights"]


def calibration(
    curves: dict, refs: dict, schedules: dict, active: set[str], method: str, h: str
) -> dict:
    """Fit each control only on the enumerated original training schedules."""
    ids = sorted(k for k, v in schedules.items() if v["role"] == "train_calibration")
    if not ids or any(schedules[k]["protocol"]["split"] != "train" for k in ids):
        raise ValueError("unverified calibration schedules")
    names = sorted(k for k in curves[ids[0]] if k.startswith("latent."))
    if method == "glucose_reachable_affine":
        names = [k for k in names if k.removeprefix("latent.") in active]
    elif method == "generated_insulin_affine":
        names = ["I"]
    elif method not in ("all_latent_affine", "all_latent_ridge"):
        raise ValueError("unknown probe")
    x = {n: np.concatenate([curves[k][n] for k in ids]) for n in names}
    truth = np.concatenate([refs[k][h] for k in ids])
    fit = (
        multivariate_probe(names, curves, refs, ids, h)
        if method == "all_latent_ridge"
        else probe.calibrate(x, truth)
    )
    if "gain" in fit:
        fit.update(
            slope=fit["gain"] / fit["scale"],
            intercept=fit["offset"] - fit["gain"] * fit["center"] / fit["scale"],
        )
    return fit | {
        "schedules": "|".join(ids),
        "eligible_inputs": "|".join(names),
        "reference_training_sd": float(np.std(truth)),
    }


def score_paired(
    refs: dict, k: str, item: dict, h: str, pred: dict, alt: dict, scale: float
) -> dict:
    """Use the original assay metrics, response floor and two-solver agreement gate."""
    base = item["control"]
    a, b = pred[k], pred[base]
    delta = a - b
    truth = np.asarray(refs[k][h]) - np.asarray(refs[base][h])
    amp = max(1e-8, np.max(abs(truth)), np.max(abs(delta)))
    gap = float(np.max(np.abs(alt[k] - alt[base] - delta)) / amp)
    if gap > 5e-3:
        return {"status": "effect_solver_disagreement", "paired_solver_gap": gap}
    return assay.scores(
        refs[k]["time"], refs[k][h], refs[base][h], a, b, item["window"], scale
    ) | {"paired_solver_gap": gap}


def analyze_learned(
    row: dict, i: int, schedules: dict, refs: dict, saved: dict
) -> dict:
    """Analyze all six contrasts for one fixed historical T2 fit."""
    subject = {
        "subject": i,
        "source_id": row["source_id"],
        "method": row["method"],
        "seed": row["repetition"],
        "candidate_hash": row["source_model_sha256"],
        "fit_hash": row["source_fit_sha256"],
    }
    candidate = CandidateModel.model_validate(row["candidate"])
    eq = fitted_equations(candidate, row["parameters"])
    active = ancestors(eq, {"Gp"})
    terms = components(eq)
    curves = {k: r["curves"] for k, r in saved.items()}
    other = {k: r["verification_curves"] for k, r in saved.items()}
    report = {
        "physical_readout_review": physical_review(subject, candidate, terms, active),
        "signed_glucose_terms": [],
        "paired_product_factors": [],
        "glucose_balance_closure": [],
        "model_implied_flux_effects": [],
        "supplementary_probe_effects": [],
        "probe_calibration": [],
    }
    contrasts = {k: v for k, v in schedules.items() if v["role"] == "contrast"}
    for h in HIDDEN:
        for method in METHODS:
            fit = calibration(curves, refs, schedules, active, method, h)
            name = fit.get("name", "")
            record = {
                **subject,
                "quantity": h,
                "measurement": method,
                "selected_input": name,
                "reaches_glucose": all(
                    n.removeprefix("latent.") in active for n in name.split("|")
                )
                if name
                else None,
            }
            report["probe_calibration"].append(record | fit)
            for k, v in contrasts.items():
                score = {"status": "no_correspondence"}
                if name:
                    a, b = [
                        {c: decode_fit(fit, data[c]) for c in (k, v["control"])}
                        for data in (curves, other)
                    ]
                    score = score_paired(
                        refs, k, v, h, a, b, fit["reference_training_sd"]
                    )
                report["supplementary_probe_effects"].append(
                    record | {"case": k, "control": v["control"]} | score
                )

    for k, v in contrasts.items():
        base = v["control"]
        curve = np.asarray(curves[k]["Gp"]) - np.asarray(curves[base]["Gp"])
        peak = int(np.argmax(abs(curve)))
        t = np.asarray(refs[k]["time"])
        if not np.array_equal(t, refs[base]["time"]):
            raise ValueError("mismatched scheduled time grids")
        rate = np.zeros_like(t)
        fluxes = {}
        for group, sign, h in (("meal_only", 1, "Ra"), ("insulin_linked", -1, "Uid")):
            group_terms = [term for term in terms if term["group"] == group]
            q = []
            for data in (curves, other):
                q.append(
                    {
                        c: sign
                        * sum(
                            (
                                term_series(term, data[c], refs[c])
                                for term in group_terms
                            ),
                            np.zeros_like(t),
                        )
                        for c in (k, base)
                    }
                )
            fluxes[group] = q[0][k] - q[0][base]
            truth = np.concatenate(
                [
                    refs[c][h]
                    for c in sorted(schedules)
                    if schedules[c]["role"] == "train_calibration"
                ]
            )
            score = score_paired(refs, k, v, h, *q, float(np.std(truth)))
            report["model_implied_flux_effects"].append(
                {
                    **subject,
                    "case": k,
                    "control": base,
                    "equation_term_group": group,
                    "reference_quantity": h,
                    "readout": "uncalibrated_plasma_balance",
                    "n_terms": len(group_terms),
                    "positive_sign": "source" if sign == 1 else "sink",
                    **score,
                }
            )
        integrated_to_peak = []
        for term in terms:
            diff = term_series(term, curves[k], refs[k]) - term_series(
                term, curves[base], refs[base]
            )
            rate += diff
            contribution = float(np.trapezoid(diff[: peak + 1], t[: peak + 1]))
            integrated_to_peak.append(contribution)
            secondary = product_changes(term, other, refs, k, base)
            for factor, change in product_changes(term, curves, refs, k, base).items():
                report["paired_product_factors"].append(
                    {
                        **subject,
                        "case": k,
                        "control": base,
                        "term": term["expression"],
                        "factor": factor,
                        "glucose_peak_time_min": float(t[peak]),
                        "integral_to_glucose_peak_mg_per_kg": float(
                            np.trapezoid(change[: peak + 1], t[: peak + 1])
                        ),
                        "verification_integral_to_same_peak_mg_per_kg": float(
                            np.trapezoid(secondary[factor][: peak + 1], t[: peak + 1])
                        ),
                    }
                )
            report["signed_glucose_terms"].append(
                {
                    **subject,
                    "case": k,
                    "control": base,
                    "term": term["expression"],
                    "group": term["group"],
                    "drivers": "|".join(term["upstream_forcings"]),
                    "glucose_peak_time_min": float(t[peak]),
                    "integral_to_glucose_peak_mg_per_kg": contribution,
                    "integral_full_horizon_mg_per_kg": float(np.trapezoid(diff, t)),
                }
            )
        cumulative = np.r_[0, np.cumsum((rate[1:] + rate[:-1]) * np.diff(t) * 0.5)]
        report["glucose_balance_closure"].append(
            {
                **subject,
                "case": k,
                "control": base,
                "glucose_peak_time_min": float(t[peak]),
                "predicted_signed_peak_mg_per_kg": float(curve[peak]),
                "initial_glucose_difference": float(curve[0]),
                "summed_term_integrals_to_peak_mg_per_kg": sum(integrated_to_peak),
                "peak_residual_mg_per_kg": float(
                    curve[peak] - curve[0] - cumulative[peak]
                ),
                "largest_grid_quadrature_residual_mg_per_kg": float(
                    np.max(abs(curve - curve[0] - cumulative))
                ),
                "meal_contribution_to_peak_mg_per_kg": float(
                    np.trapezoid(fluxes["meal_only"][: peak + 1], t[: peak + 1])
                ),
                "insulin_linked_contribution_to_peak_mg_per_kg": -float(
                    np.trapezoid(fluxes["insulin_linked"][: peak + 1], t[: peak + 1])
                ),
            }
        )
    return report


def summarize_learned_targets(rows: list[dict]) -> list[dict]:
    """Summarize all historical algorithm/prompt configurations per intervention."""
    result = []
    legacy = [r for r in rows if r["benchmark_id"] == assay.LEGACY]
    for key in sorted({(r["case"], r["quantity"]) for r in legacy}):
        cohort = [r for r in legacy if (r["case"], r["quantity"]) == key]
        scores = [
            r["effect_rse"]
            for r in cohort
            if r["status"] == "complete" and r["effect_rse"] is not None
        ]
        result.append(
            {
                "benchmark_id": assay.LEGACY,
                "case": key[0],
                "quantity": key[1],
                "planned": len(cohort),
                "complete": len(scores),
                "lower_than_zero_change": sum(v < 1 - 1e-6 for v in scores),
                "zero_change_tolerance": 1e-6,
                "minimum": min(scores) if scores else None,
                "median": float(np.median(scores)) if scores else None,
                "maximum": max(scores) if scores else None,
            }
        )
    return result


def pair_tables(plan: dict, scores: list[dict]) -> tuple[list[dict], list[dict]]:
    """Retain every prespecified pair and independently indexed outcome for matches."""
    lookup = {(r["subject"], r["case"], r["readout"], r["quantity"]): r for r in scores}
    pairs, effects = [], []
    for item in plan["pairs"]:
        left, right = (int(item[k]) for k in ("left", "right"))
        a, b = (plan["roster"][i] for i in (left, right))
        pid = f"{a['source_id']}--{b['source_id']}"
        meta = {
            "pair_id": pid,
            "left_source": a["source_id"],
            "right_source": b["source_id"],
            "left_fit": a["source_fit_sha256"],
            "right_fit": b["source_fit_sha256"],
            "left_validation_nmse": a["validation_nmse"],
            "right_validation_nmse": b["validation_nmse"],
            **{k: v for k, v in item.items() if k != "reasons"},
            "caliper_failures": "|".join(item["reasons"]),
        }
        count = 0
        if item["eligible"]:
            for (i, k, kind, n), lrow in lookup.items():
                if i != left:
                    continue
                rrow = lookup[right, k, kind, n]
                effects.append(
                    {
                        "pair_id": pid,
                        "case": k,
                        "quantity": n,
                        "readout": kind,
                        "left_status": lrow["status"],
                        "right_status": rrow["status"],
                        "left_effect_rse": lrow.get("effect_rse"),
                        "right_effect_rse": rrow.get("effect_rse"),
                    }
                )
                count += 1
        pairs.append(meta | {"outcome_rows": count})
    return pairs, effects


def pair_plot(path: Path, plan: dict, raw: dict, calibrations: list[dict]) -> None:
    """Plot the two historical Full models and the simulator's own paired responses."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 8.1), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2, sharex=True)
    by_probe = {
        (r["subject"], r["quantity"], r["measurement"]): r for r in calibrations
    }
    ids = [
        i
        for i, r in enumerate(plan["roster"])
        if r["method"] == "autoformalism:full:c1v1s1"
    ]
    for col, (k, title) in enumerate(
        (
            ("fed", "48 g meal vs fasting"),
            ("short_high", "Fasting insulin pulse vs fasting"),
        )
    ):
        for line, (h, label) in enumerate(
            (
                ("Gp", "Plasma glucose\n(mg/kg)"),
                ("I", "Plasma insulin\n(pmol/L)"),
                ("Uid", "Disposal\n(mg/kg/min)"),
            )
        ):
            ax = axes[line, col]
            for i in ids:
                src = plan["roster"][i]
                refs, saved = raw[i]
                base = plan["schedules"][src["benchmark_id"]][k]["control"]
                t = refs[k]["time"]
                if i == ids[0]:
                    truth = np.subtract(refs[k][h], refs[base][h])
                    ax.plot(t, truth, label="Simulator", color="black", linewidth=2)
                curves = {c: saved[c]["curves"] for c in (k, base)}
                if h == "Uid":
                    fit = by_probe[i, h, "all_latent_affine"]
                    delta = probe.decode(fit, curves[k]) - probe.decode(
                        fit, curves[base]
                    )
                else:
                    delta = np.subtract(curves[k][h], curves[base][h])
                ax.plot(
                    t,
                    delta,
                    label=f"Full seed {src['repetition']}",
                    linewidth=1.5,
                    linestyle="--" if src["repetition"] == 0 else "-.",
                    color="#a7521f" if src["repetition"] == 0 else "#286c83",
                )
            ax.axhline(0, color="0.65", linewidth=0.6)
            if col == 0:
                ax.set_ylabel(label)
            if line == 0:
                ax.set_title(title)
            if line == 2:
                ax.set_xlabel("Time (min)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Historical T2-hard paired responses\nGlucose and insulin are generated "
        "targets. Disposal uses the training-calibrated latent probe.",
        fontsize=12,
    )
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})


def run(evidence_path: Path, trace_path: Path, out: Path) -> dict:
    """Authenticate the two releases and record the fixed development review."""
    expected = "7e1a64fe6ef8e9c93f3521f8e513edc2aac14f2b8144bcb081f2d53dca315663"
    if (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest() != expected
        or hashlib.sha256(trace_path.read_bytes()).hexdigest() != TRACE_ZIP
    ):
        raise ValueError(
            "require the authenticated historical evidence and trace supplements"
        )
    with (
        ZipFile(evidence_path) as evidence,
        ZipFile(trace_path) as archive,
        public._lock(out),
    ):
        plan = json.loads(evidence.read("fidelity/frozen_selection_manifest.json"))
        scored = json.loads(evidence.read("fidelity/scored_rows.json"))
        historical = json.loads(evidence.read("fidelity/summary.json"))
        if (
            plan["artifact_sha256"] != PLAN
            or scored["artifact_sha256"] != SCORES
            or plan["mode"] != "development"
        ):
            raise ValueError("wrong frozen roster or score selection")
        policy = sealed_write(
            out / "plan.json",
            {
                "cohort": PLAN,
                "published_scores": SCORES,
                "evidence_sha256": expected,
                "traces_sha256": TRACE_ZIP,
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "analysis": "retrospective, no model refits; complete roster",
                "model_selection": False,
                "new_confirmation_access": False,
                "decoder_methods": list(METHODS),
                "active_graph": (
                    "Nonzero glucose dependencies after fitting and process expansion."
                ),
                "comparison_fluxes": (
                    "Unscaled plasma balance terms grouped by upstream drivers."
                ),
                "ridge_alphas_descending": list(RIDGE_ALPHAS),
                "numerical_runtime": public._runtime(),
                "ridge_selection": "Leave one training schedule out, then refit.",
            },
        )
        records = list(
            csv.DictReader(
                io.StringIO(archive.read("fidelity/trace_index.csv").decode())
            )
        )
        raw = {}
        collected = {}
        ref_cache = {}
        for i, row in enumerate(plan["roster"]):
            cell = row["benchmark_id"]
            schedules = plan["schedules"][cell]
            saved = {}
            refs = {}
            for x in (r for r in records if int(r["subject_index"]) == i):
                k = x["case"]
                rec = json.loads(archive.read(x["path"]))
                if (
                    x["source_id"] != row["source_id"]
                    or x["selection_plan_sha256"] != PLAN
                    or rec["artifact_sha256"] != x["trace_artifact_sha256"]
                    or rec["plan"] != PLAN
                ):
                    raise ValueError("foreign rollout in indexed cohort")
                if x["reference_path"] not in ref_cache:
                    ref = json.loads(evidence.read(x["reference_path"]))
                    if ref["plan"] != PLAN:
                        raise ValueError("foreign reference")
                    ref_cache[x["reference_path"]] = ref["reference"]
                refs[k] = ref_cache[x["reference_path"]]
                saved[k] = rec
            if set(refs) != set(schedules):
                raise ValueError("missing intervention or calibration schedules")
            raw[i] = (refs, saved)
            if row["source_protocol"] == fits.VERSION:
                physical = [
                    {
                        "subject": i,
                        "source_id": row["source_id"],
                        "method": row["method"],
                        "seed": None,
                        "optimizer_start": row["fit"]["start"],
                        "fit_hash": row["fit"]["artifact_sha256"],
                        "quantity": h,
                        "reference_unit": assay.UNITS[h],
                        "review_status": "supplied_structure_physical",
                        "evidence": (
                            "Privileged equations output this simulator quantity "
                            "in physical units."
                        ),
                    }
                    for h in fits.HIDDEN[cell]
                ]
                result = {"physical_readout_review": physical}
            else:
                if cell != assay.LEGACY:
                    raise ValueError("unexpected learned release")
                result = analyze_learned(row, i, schedules, refs, saved)
                prior = historical["subjects"][i]["training_probes"]
                for c in result["probe_calibration"]:
                    if c["measurement"] == "all_latent_affine" and any(
                        c.get(n) != v for n, v in prior[c["quantity"]].items()
                    ):
                        raise ValueError("original scalar probe calibration differs")
            for name, rows in result.items():
                collected.setdefault(name, []).extend(rows)
        if len(raw) != 20 or len(records) != 208:
            raise ValueError("incomplete original assay")
        scores = scored["rows"]
        targets = [r for r in scores if r["readout"] == "target"]
        collected["target_effects"] = targets
        collected["learned_target_summary"] = summarize_learned_targets(targets)
        collected["matched_pairs"], collected["matched_pair_effects"] = pair_tables(
            plan, scores
        )
        lookup = {
            (r["subject"], r["case"], r["quantity"]): r
            for r in scores
            if r["readout"] == "oracle_affine"
        }
        for r in collected["supplementary_probe_effects"]:
            if r["measurement"] == "all_latent_affine":
                original = lookup[r["subject"], r["case"], r["quantity"]]
                if r["status"] != original["status"] or r.get(
                    "effect_rse"
                ) != original.get("effect_rse"):
                    raise ValueError(
                        "historical scalar probe metrics did not reproduce"
                    )
        for name, rows in collected.items():
            assay.csv_table(out / f"{name}.csv", rows)
        mapping = sealed_write(
            out / "physical_readout_map.json",
            {
                "cohort": PLAN,
                "review_protocol": policy["artifact_sha256"],
                "units_policy": (
                    "Native maps require independent scale and role evidence. "
                    "Signed balance terms have their own operational interpretation."
                ),
                "learned_candidate_readouts": {
                    row["source_model_sha256"]: {}
                    for row in plan["roster"]
                    if row["source_protocol"] != fits.VERSION
                },
                "operational_glucose_balance": {
                    r["candidate_hash"]: {
                        "meal_only_plasma_source": r["fitted_meal_only_glucose_terms"],
                        "insulin_linked_plasma_sink": (
                            "-("
                            + (r["fitted_insulin_linked_glucose_terms"] or "0")
                            + ")"
                        ),
                        "unit": "mg/kg/min from dGp/dt",
                        "source_fit_sha256": r["fit_hash"],
                    }
                    for r in collected["physical_readout_review"]
                    if r["quantity"] == "Uid" and r["review_status"] == "unavailable"
                },
                "decisions_sha256": hashlib.sha256(
                    (out / "physical_readout_review.csv").read_bytes()
                ).hexdigest(),
            },
        )
        pair_plot(
            out / "historical_glucose_insulin_and_disposal.pdf",
            plan,
            raw,
            collected["probe_calibration"],
        )
        return sealed_write(
            out / "summary.json",
            {
                "plan": policy["artifact_sha256"],
                "subjects": len(raw),
                "reference_fits": 4,
                "learned_fits": 16,
                "matched_candidate_pairs": len(collected["matched_pairs"]),
                "eligible_pairs": sum(
                    r["eligible"] for r in collected["matched_pairs"]
                ),
                "rows": {k: len(v) for k, v in collected.items()},
                "target_status": dict(Counter(r["status"] for r in targets)),
                "physical_map": mapping["artifact_sha256"],
                "table_sha256": {
                    k: hashlib.sha256((out / f"{k}.csv").read_bytes()).hexdigest()
                    for k in collected
                },
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.evidence, args.traces, args.output)["artifact_sha256"])
