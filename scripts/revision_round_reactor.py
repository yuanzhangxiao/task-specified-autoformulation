"""One-round heat-law revisions on the saved CSTR-easy temperature balance."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from scripts import thermal_balance_trial as thermal
from scripts import thermal_forecast_continuation as forecast

PUBLIC = (
    Path(__file__).resolve().parents[1]
    / "artifacts/phase-c-development-final-v2/public"
)
PARENT = {"heat_law": "concentration", "concentration_power": 1}
PARTITIONS = (15, 1)
EVALUATIONS = 8
COEFFICIENT_FLOOR = 1e-8
SCREEN_POWERS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def _valid(edit: dict) -> bool:
    if not isinstance(edit, dict) or set(edit) != set(PARENT):
        return False
    power = edit["concentration_power"]
    return (
        edit["heat_law"] in forecast.FAMILIES
        and type(power) in (int, float)
        and 0.5 <= power <= 3
    )


@lru_cache(maxsize=1)
def _training():
    spec = thermal._read(PUBLIC, thermal.EASY, "specification")
    if thermal.eligibility(spec)["status"] != "applicable":
        raise ValueError("Registered training balance requires supplied C, Tj and Tf")
    rows = thermal._read(PUBLIC, thermal.EASY, "train")
    reference = thermal.fit(rows, ["concentration"])["concentration"]
    scale = float(np.std(np.concatenate([r.targets["T"] for r in rows])))
    return rows, reference, scale


def _with_power(rows, power: float):
    """Transform the supplied concentration before either fitting or forecasting."""
    if power == 1:
        return rows
    updated = []
    for row in rows:
        concentration = np.asarray(row.auxiliaries["C"])
        if np.any(concentration < 0):
            raise ValueError("Fractional concentration powers require nonnegative C")
        aux = {**row.auxiliaries, "C": tuple((concentration**power).tolist())}
        updated.append(row.model_copy(update={"auxiliaries": aux}))
    return updated


def _rate_errors(rows, model: dict) -> list[dict]:
    """Summarize signed parent rate errors by measured temperature and concentration."""
    d = thermal.intervals(rows)
    temperature = d.t_nodes @ thermal.WEIGHTS
    concentration = d.c_nodes @ thermal.WEIGHTS
    p = model["best"]
    error = d.deltas / d.dt - (
        p["a"] * d.transport + p["b"] * d.jacket + p["c"] * concentration
    )
    groups = []
    for hot in (False, True):
        for rich in (False, True):
            sample = ((temperature > np.median(temperature)) == hot) & (
                (concentration > np.median(concentration)) == rich
            )
            if sample.any():
                groups.append(
                    {
                        "temperature_half": "high" if hot else "low",
                        "concentration_half": "high" if rich else "low",
                        "intervals": int(sample.sum()),
                        "median_observed_minus_parent_rate_K_per_min": float(
                            np.median(error[sample])
                        ),
                    }
                )
    return groups


def packet() -> dict:
    """Expose scientific evidence from the parent fit and a bounded revision grammar."""
    rows, reference, scale = _training()
    p = reference["best"]
    curves = thermal.score(rows, reference, scale)
    worst = max(curves["events"], key=lambda event: event["rmse_K"])
    return {
        "parent": PARENT.copy(),
        "algorithm_edits": [
            {"heat_law": "exponential_inverse_temperature", "concentration_power": 1},
            {"heat_law": "concentration", "concentration_power": 2},
            {"heat_law": "exponential_inverse_temperature", "concentration_power": 0.5},
        ],
        "edit_schema": {
            "type": "object",
            "properties": {
                "heat_law": {"type": "string", "enum": list(forecast.FAMILIES)},
                "concentration_power": {"type": "number", "minimum": 0.5, "maximum": 3},
            },
            "required": ["heat_law", "concentration_power"],
            "additionalProperties": False,
        },
        "edit_contract": {
            "fields": {
                "heat_law": list(forecast.FAMILIES),
                "concentration_power": "finite number from 0.5 through 3",
            },
            "balance": "dT/dt = a*(Tf-T) + b*(Tj-T) + c*C^concentration_power*q(T)",
            "heat_law": (
                "concentration sets q=1. exponential_inverse_temperature sets "
                "q=exp(beta*(1/T_min - 1/T)/(1/T_min - 1/T_max)); the fitter "
                "estimates beta and positive a,b,c. Temperature is in kelvins."
            ),
        },
        "evidence": {
            "task": (
                "Generate reactor temperature from measured initial T and supplied "
                "feed temperature Tf, jacket temperature Tj and concentration C. "
                "The task requires feed transport, reaction heat and jacket exchange."
            ),
            "training_events": len(rows),
            "training_temperature_range_K": [reference["T_min"], reference["T_max"]],
            "parent_coefficients": {k: p[k] for k in ("a", "b", "c")},
            "parent_training_nmse": curves["pooled_nmse"],
            "worst_parent_event": worst["event"],
            "worst_parent_event_rmse_K": worst["rmse_K"],
            "training_interval_rate_errors": _rate_errors(rows, reference),
            "diagnosis": (
                "The fitted parent retains the three terms and under-explains "
                "temperature variation. This is an empirical error, with no proof "
                "that the parent family cannot fit better."
            ),
        },
    }


def screen() -> dict:
    """Rank a fixed grid by out-of-fold temperature-balance error on training."""
    rows, _, _ = _training()
    if len(rows) < 2:
        raise ValueError("Two-fold screening needs at least two training events")
    rows = sorted(rows, key=lambda row: row.trajectory_id)
    profile = []
    for power in SCREEN_POWERS:
        adjusted = _with_power(rows, power)
        laws = [
            law
            for law in forecast.FAMILIES
            if {"heat_law": law, "concentration_power": power} != PARENT
        ]
        losses = {law: [] for law in laws}
        for fold in (0, 1):
            fit = thermal.fit(adjusted[fold::2], laws)
            for law in laws:
                losses[law].append(
                    thermal.rate_error(adjusted[1 - fold :: 2], fit[law])
                )
        for law, terms in losses.items():
            rmse = None
            if all(term is not None for term in terms):
                sse, duration = np.sum(terms, axis=0)
                rate_rmse = float(np.sqrt(sse / duration))
                if math.isfinite(rate_rmse):
                    rmse = rate_rmse
            profile.append(
                {
                    "edit": {"heat_law": law, "concentration_power": power},
                    "cv_rate_rmse_K_per_min": rmse,
                }
            )
    ranked = sorted(
        (item for item in profile if item["cv_rate_rmse_K_per_min"] is not None),
        key=lambda item: item["cv_rate_rmse_K_per_min"],
    )
    return {
        "edits": [item["edit"] for item in ranked[:3]],
        "screened": len(profile),
        "interval_family_fits": 2 * len(profile),
        "full_trajectory_fits_during_screen": 0,
        "criterion": (
            "Lowest duration-weighted held-fold interval-rate RMSE (K/min); "
            "two alternating whole-event folds in sorted training order. "
            "Equal errors follow ascending powers then registered heat-law order."
        ),
        "additional_evidence": (
            "Raw public training T, Tf, Tj and C on each interval. Learn the "
            "temperature scale, heat sensitivity and three nonnegative gains only "
            "in the fitting fold, evaluate temperature changes along the measured "
            "held training paths. Proposer packet contains four residual summaries."
        ),
        "profile": profile,
    }


def score(edit: dict) -> dict:
    """Refit a candidate on training using matched starts and short/full windows."""
    if not _valid(edit):
        return {
            "eligible": False,
            "reason": "unsupported heat law or concentration power",
        }
    training, reference, scale = _training()
    rows = _with_power(training, edit["concentration_power"])
    law = edit["heat_law"]
    template = thermal.fit(rows, [law])[law]
    initial = forecast._vector(template)
    shared = forecast._vector(reference) + ([0.125] if len(initial) == 4 else [])
    batches = {n: forecast.windows(rows, n) for n in PARTITIONS}
    candidates = [initial, shared]
    calls = 0
    for start in (initial, shared):
        vector = start
        for n in PARTITIONS:
            try:
                opt = forecast.optimize(
                    batches[n],
                    vector,
                    template["T_min"],
                    template["T_max"],
                    scale,
                    limit=EVALUATIONS,
                )
            except (ValueError, ArithmeticError, RuntimeError):
                break
            calls += opt["nfev"]
            vector = opt["vector"]
            candidates.append(vector)
    scores = []
    for vector in dict.fromkeys(tuple(v) for v in candidates):
        if min(vector[:3]) <= COEFFICIENT_FLOOR:
            continue
        model = forecast._model(template, vector)
        result = forecast._forecast(rows, model, scale)
        if result["status"] == "scored" and math.isfinite(result["pooled_nmse"]):
            scores.append((result["pooled_nmse"], vector, result))
    if not scores:
        return {
            "eligible": False,
            "reason": "no qualified full training rollout",
            "optimizer_evaluations": calls,
        }
    nmse, vector, result = min(scores, key=lambda entry: entry[0])
    return {
        "eligible": True,
        "train_nmse": nmse,
        "parameters": forecast._model(template, vector)["best"],
        "edit": edit.copy(),
        "training_temperature_range_K": [template["T_min"], template["T_max"]],
        "training_scale_K": scale,
        "rmse_K": result["rmse_K"],
        "per_event": result["per_event"],
        "max_scaled_grid_difference": result["max_scaled_grid_difference"],
        "optimizer_evaluations": calls,
        "fit_scope": {
            "starts": 2,
            "window_partitions": list(PARTITIONS),
            "max_evaluations_per_stage": EVALUATIONS,
        },
    }


def validate(edit: dict, score_result: dict) -> dict:
    """Evaluate frozen parameters on public development trajectories."""
    if not _valid(edit) or not score_result["eligible"] or score_result["edit"] != edit:
        raise ValueError("Validation requires a qualified fit of the same revision")
    lo, hi = score_result["training_temperature_range_K"]
    model = {
        "status": "fitted",
        "law": edit["heat_law"],
        "best": score_result["parameters"],
        "T_min": lo,
        "T_max": hi,
    }
    rows = _with_power(
        thermal._read(PUBLIC, thermal.EASY, "val"), edit["concentration_power"]
    )
    result = forecast._forecast(rows, model, score_result["training_scale_K"])
    return {"nmse": result.get("pooled_nmse"), **result}
