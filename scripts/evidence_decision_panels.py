"""Replay two candidate families from authenticated T2 and coupled-basin fits."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import pairwise
from math import exp, isfinite, log, sqrt
from pathlib import Path
from statistics import mean

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import trajectory_function_fitting as rollout
from autoformalism.research.preproposal_cases import BASIN, T2
from autoformalism.staged_topology import content_hash
from scripts import basin_function_family as basin
from scripts import run_t2_balanced_disposal as disposal
from scripts.matched_equation_experiment import _registered
from scripts.report_hidden_form_forecasts import _curve
from scripts.topology_forecast_fit import INPUTS, TRAIN, VALIDATION

OUTPUT = Path(__file__).resolve().parents[1] / "output"
BASIN_ROOT = (
    OUTPUT / "basin-autopilot-20261123/aces-folds/phase-c-basin-stability-original-v1"
)
BASIN_STUDY = "eea7980f79f53a5c1aaa0d11cd1dc467fd9f3ccfd6ac97610e3ed871e0b1aded"
BASIN_RANKINGS = "103850fb64952706462430a088c4ee393ac3e6208ffef57b94d2c254fd8ba38f"
DISPOSAL_PLAN = "bd7e2e69e41323b76ffd8101238934abdaa80732cd88dba5877825f6ad6b70d7"
T2_FORMS = {"simple": "balanced_action", "extended": "balanced_interaction"}
BASIN_FORMS = {"simple": (0, 1.0), "extended": (4, 1.5)}


def _integral(x: float, y: float, p: float) -> float:
    """Integral of head**(-p) from y to x above the crest."""
    if p == 1.0:
        return log(x / y)
    if p == 1.5:
        return 2 * (1 / sqrt(y) - 1 / sqrt(x))
    raise ValueError("Unsupported dry-recession power")


def _propagate(head: float, slope: float, power: float) -> float:
    """Analytical state after a dry interval with slope=k*dt/area."""
    if power == 1.0:
        return head * exp(-slope)
    return (1 / sqrt(head) + slope / 2) ** -2


def dry_intervals(rows: list[dict]) -> list[dict]:
    """Prove zero transfer by bounding upstream volume with all received inflow."""
    intervals = []
    for r in rows:
        if not r["trajectory_id"].startswith("train_"):
            raise ValueError("Dry evidence must use fitting trajectories")
        c, inlet = r["covariates"], r["inputs"]
        # Exact rational area, supplied surveyed initial depth and piecewise
        # linear input knots. Keeping all positive inflow is a storage upper bound.
        area = Fraction(c["area_up"])
        room = area * (Fraction(c["crest_up"]) - Fraction(c["initial_up"]))
        volume = Fraction()
        if area <= 0 or c["area_down"] <= 0 or min(inlet["inflow_up"]) < 0:
            raise ValueError("Expected positive areas and nonnegative runoff")
        time = r["time"]
        for j, (t0, t1) in enumerate(pairwise(time)):
            dt = Fraction(t1) - Fraction(t0)
            if dt <= 0:
                raise ValueError("Expected strictly increasing time")
            volume += dt * sum(Fraction(v) for v in inlet["inflow_up"][j : j + 2]) / 2
            if volume > room or any(inlet["inflow_down"][j : j + 2]):
                continue
            y0, y1 = r["targets"]["h_down"][j : j + 2]
            intervals.append(
                {
                    "event": r["trajectory_id"],
                    "times": [t0, t1],
                    "heads": [y0 - c["crest_down"], y1 - c["crest_down"]],
                    "area_over_dt": float(c["area_down"] / dt),
                }
            )
    return intervals


def _verify_gap(low: dict, high: dict, midpoint: float, p: float, tol: float) -> None:
    """Cross-check incompatible inequalities with the analytical flow map."""
    # If even the lowest initial head overshoots the high observed endpoint
    # at k=midpoint, every k<=midpoint fails that interval. Similarly for k>=midpoint.
    for rec, too_high in ((low, True), (high, False)):
        x, y = rec["heads"]
        initial = x - tol if too_high else x + tol
        final = _propagate(initial, midpoint / rec["area_over_dt"], p)
        checked = final > y + tol if too_high else final < y - tol
        if not checked:
            raise ArithmeticError(
                "Dry-recession analytic flow disagrees with certificate"
            )


def dry_check(intervals: list[dict], power: float, tolerance: float) -> dict:
    """Intersect necessary k intervals for the stated endpoint depth accuracy."""
    if not isfinite(tolerance) or tolerance < 0 or power not in (1.0, 1.5):
        raise ValueError("Invalid depth tolerance or power")
    bounds = []
    for r in intervals:
        x, y = r["heads"]
        if min(x, y) <= tolerance:
            continue
        a = r["area_over_dt"]
        lo = a * _integral(x - tolerance, y + tolerance, power)
        hi = a * _integral(x + tolerance, y - tolerance, power)
        # Allow conservative rounding on top of the user-facing depth tolerance.
        slack = 1e-9 * max(1, abs(lo), abs(hi))
        bounds.append((max(0.0, lo - slack), hi + slack, r))
    if not bounds:
        return {"status": "inconclusive", "intervals": 0, "k_min": 0.0, "k_max": None}
    low = max(bounds, key=lambda b: b[0])
    high = min(bounds, key=lambda b: b[1])
    rejected = low[0] > high[1]
    if rejected:
        if high[1] >= 0:
            _verify_gap(low[2], high[2], (low[0] + high[1]) / 2, power, tolerance)
        elif not high[2]["heads"][1] - tolerance > high[2]["heads"][0] + tolerance:
            raise ArithmeticError(
                "Unconfirmed impossibility of rising during recession"
            )
    return {
        "status": "certified_reject" if rejected else "inconclusive",
        "intervals": len(bounds),
        "k_min": low[0],
        "k_max": high[1],
        "lower_witness": low[2],
        "upper_witness": high[2],
        "tolerance_m": tolerance,
        "power": power,
    }


def _valid_ids(fit: list[str], selection: list[str], nfit: int, nselect: int) -> None:
    ids = fit + selection
    if len(fit) != nfit or len(selection) != nselect or len(set(ids)) != len(ids):
        raise ValueError("Wrong fitting/selection split")
    if any(not k.startswith("train_") for k in ids):
        raise ValueError("Only training observations may enter selection")


def _t2_panels() -> list[dict]:
    root = disposal.DEFAULT
    plan = _registered(root / "plan.json", DISPOSAL_PLAN)
    panels = []
    for panel in plan["panels"]:
        pid, fit_ids = panel["panel_id"], panel["fit_ids"]
        screen = _registered(root / "screens" / f"{pid}.json", panel["screen"])
        if screen["fit_ids"] != fit_ids or screen["packet"] != panel["packet"]:
            raise ValueError("Mismatched disposal fitting screen")
        for rot, split in enumerate(panel["choice_rotations"]):
            ids = split["selection_ids"]
            _valid_ids(fit_ids, ids, 8, 4)
            slots, evidence = {}, {}
            for alias, family in T2_FORMS.items():
                c = panel["candidates"][family]
                h = screen["screen"]["hidden_candidates"][c["dependency_key"]]
                reject = not h["eligible"]
                _, _, fit, fit_id = disposal.load_endpoint(root, plan, panel, family)
                folder = root / "fits" / pid / family
                best = _registered(folder / "best.json", fit_id)
                attempts = [
                    _registered(folder / f"{s}.json", v)["result"]
                    for s, v in best["starts"].items()
                ]
                curves = [_curve(root, plan, pid, family, e) for e in ids]
                outcomes = [
                    r["nmse"] if r["status"] == "scored" else None for r in curves
                ]
                raw = mean(outcomes) if all(v is not None for v in outcomes) else None
                slots[alias] = {
                    "family": family,
                    "certified_reject": reject,
                    "raw_selection_loss": raw,
                    "selection_loss": None if reject else raw,
                    "fit_status": "fitted" if fit else "no_fit",
                    "selection_status": [r["status"] for r in curves],
                    "fit_evaluations": sum(r["evaluations"] for r in attempts),
                    "fit_receipt": fit_id,
                    "endpoint_rejected": False,
                    "selection_receipts": [r["artifact_sha256"] for r in curves],
                    "nominal_fit_budget": 1000,
                }
                evidence[alias] = {
                    "status": "certified_reject" if reject else "inconclusive",
                    "repeated_groups": h["repeated_argument_groups"],
                    "minimum_uniform_error": h["minimum_uniform_error"],
                }
            panels.append(
                {
                    "panel_id": f"t2/{pid}/r{rot}",
                    "benchmark": "t2",
                    "fit_ids": fit_ids,
                    "selection_ids": ids,
                    "candidates": slots,
                    "evidence": evidence,
                    "source_digests": {
                        "plan": DISPOSAL_PLAN,
                        "screen": panel["screen"],
                        "packet": panel["packet"],
                    },
                }
            )
    return panels


def _basin_endpoint(i: int, alias: str) -> tuple:
    rankings = _registered(BASIN_ROOT / "rankings.json", BASIN_RANKINGS)
    rank = _registered(BASIN_ROOT / f"fold-{i}/ranking.json", rankings["rankings"][i])
    folder = BASIN_ROOT / f"fold-{i}"
    plan = _registered(folder / "plan.json", rank["plan"])
    slot, p = BASIN_FORMS[alias]
    model = plan["models"][slot]
    fit = _registered(
        folder / f"fits/{slot:02}/result.json", rank["fit_receipts"][slot]
    )
    if (
        plan["study"],
        rankings["study"],
        model["up"],
        model["down"],
        fit["slot"],
        fit["formula"],
        fit["plan"],
        rank["selected_fit_indices"][slot],
    ) != (
        BASIN_STUDY,
        BASIN_STUDY,
        p,
        p,
        slot,
        content_hash(model["formula"]),
        plan["artifact_sha256"],
        slot if basin._admissible(fit["result"]) else None,
    ):
        raise ValueError("Mismatched basin fit")
    return plan, rank, model, fit


def _basin_panels(tolerance: float) -> list[dict]:
    training = _registered(INPUTS / "train.json", TRAIN)["train"][BASIN]
    events = {r["trajectory_id"]: r for r in training}
    panels = []
    for i in range(6):
        slots, evidence = {}, {}
        for alias, (slot, power) in BASIN_FORMS.items():
            plan, rank, _, fit = _basin_endpoint(i, alias)
            ids, held = plan["training"], plan["held"]
            _valid_ids(ids, [held], 5, 1)
            check = dry_check(dry_intervals([events[e] for e in ids]), power, tolerance)
            rec = sealed_read(BASIN_ROOT / f"fold-{i}/scores/{slot:02}.json")
            if (rec["rankings"], rec["event"], rec["fit"]) != (
                BASIN_RANKINGS,
                held,
                fit["artifact_sha256"],
            ):
                raise ValueError("Mismatched basin selection event")
            score = rec["score"]
            valid = (
                slot in rank["training_order"]
                and score["status"] == "scored"
                and score["negative_samples"]["h_down"] == 0
            )
            raw = score["pooled_nmse"] if valid else None
            k = fit["result"].get("parameters", {}).get("k_outflow")
            reject = check["status"] == "certified_reject"
            endpoint_bad = k is not None and (
                k < check["k_min"]
                or (check["k_max"] is not None and k > check["k_max"])
            )
            slots[alias] = {
                "family": f"shared_power_{power:g}",
                "certified_reject": reject,
                "raw_selection_loss": raw,
                "selection_loss": None if reject or endpoint_bad else raw,
                "endpoint_rejected": endpoint_bad,
                "fit_status": fit["result"]["status"],
                "selection_status": [score["status"]],
                "fit_evaluations": fit["result"].get("evaluations", 0),
                "fit_receipt": fit["artifact_sha256"],
                "selection_receipts": [rec["artifact_sha256"]],
                "nominal_fit_budget": plan["budget"]["max_evaluations"],
            }
            evidence[alias] = check
        panels.append(
            {
                "panel_id": f"basin/fold{i}",
                "benchmark": "basin",
                "fit_ids": ids,
                "selection_ids": [held],
                "candidates": slots,
                "evidence": evidence,
                "source_digests": {
                    "training": TRAIN,
                    "study": BASIN_STUDY,
                    "rankings": BASIN_RANKINGS,
                    "fold_plan": plan["artifact_sha256"],
                },
            }
        )
    return panels


def load_panels(*, basin_tolerance: float = 0.0001) -> list[dict]:
    """Load training receipts for replay behind the paid-action interface."""
    return _t2_panels() + _basin_panels(basin_tolerance)


@lru_cache(maxsize=28)
def validation(panel_id: str, candidate: str) -> dict:
    """Replay the selected fit on validation after the caller freezes its choice."""
    parts = panel_id.split("/")
    if candidate not in T2_FORMS:
        raise ValueError("Unknown candidate")
    if len(parts) == 3 and parts[0] == "t2" and parts[2] in {"r0", "r1"}:
        plan = _registered(disposal.DEFAULT / "plan.json", DISPOSAL_PLAN)
        panel = next(p for p in plan["panels"] if p["panel_id"] == parts[1])
        graph, form, fit, digest = disposal.load_endpoint(
            disposal.DEFAULT, plan, panel, T2_FORMS[candidate]
        )
        dataset, nonnegative = T2, plan["nonnegative_targets"]
    elif (
        len(parts) == 2
        and parts[0] == "basin"
        and parts[1] in {f"fold{i}" for i in range(6)}
    ):
        plan, _, model, rec = _basin_endpoint(int(parts[1][-1]), candidate)
        graph, form = plan["graph"], model["formula"]
        fit, digest = rec["result"], rec["artifact_sha256"]
        dataset, nonnegative = BASIN, ["h_down"]
    else:
        raise ValueError("Unknown model panel")
    release = _registered(INPUTS / "validation.json", VALIDATION)
    if release["training_intake_sha256"] != TRAIN or release["split"] != "val":
        raise ValueError("Wrong validation release")
    outcomes = {}
    for row in release["rows"][dataset]:
        eid = row["trajectory_id"]
        if not eid.startswith("val_"):
            raise ValueError("Expected validation event")
        scored = (
            rollout.score_formula(graph, form, fit, [row])
            if fit
            else {"status": "no_fit"}
        )
        nmse = None
        status = scored["status"]
        if status == "scored" and any(
            scored["negative_samples"][k] for k in nonnegative
        ):
            status = "negative_forecast"
        if status == "scored":
            if dataset == BASIN:
                nmse = scored["pooled_nmse"]
            else:
                curve = scored["curves"][0]
                nmse = mean(
                    float(
                        np.mean(
                            np.square(
                                (
                                    np.array(curve["predicted"][k][1:])
                                    - curve["observed"][k][1:]
                                )
                                / fit["scales"][k]
                            )
                        )
                    )
                    for k in fit["scales"]
                )
        outcomes[eid] = {"status": status, "nmse": nmse}
    valid = all(v["nmse"] is not None for v in outcomes.values())
    return {
        "status": "qualified" if valid else "failed",
        "events": outcomes,
        "nmse": mean(v["nmse"] for v in outcomes.values()) if valid else None,
        "fit_receipt": digest,
        "validation_release": VALIDATION,
    }
