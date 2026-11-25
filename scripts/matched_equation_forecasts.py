"""Fit two fixed-graph candidates per arm and forecast omitted training events."""

from __future__ import annotations

import fcntl
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import (
    trajectory_function_fitting as rollout,
)
from autoformalism.research import (
    trajectory_nonnegative_fitting as bounded,
)
from autoformalism.staged_topology import content_hash
from scripts.construction_functions import file_hash

BUDGET = {"max_evaluations": 1000, "wall_seconds": 180.0, "starts": 2}
SEEDS = (100, 101)
PROTOCOL = "matched-equation-omitted-training-event-1"


def _safe_ids(*ids: str) -> None:
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", name) for name in ids):
        raise ValueError("Use a single identifier for each panel and arm")


def _fit_one(graph, formula, rows, bounds, slot) -> dict:
    if formula is None:
        return {"status": "no_formula", "training_nmse": None, "evaluations": 0}
    try:
        checked = rollout.validate_formula(graph, formula)
    except (KeyError, ValueError) as exc:
        return {"status": "invalid_formula", "reason": str(exc), "evaluations": 0}
    settings = {**BUDGET, "seed": SEEDS[slot]}
    try:
        result = (
            bounded.fit_nonnegative(graph, checked, rows, targets=bounds, **settings)
            if bounds
            else rollout.fit_formula(graph, checked, rows, **settings)
        )
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        return {"status": "fit_exception", "reason": f"{type(exc).__name__}: {exc}"}
    if (
        result["budget"] != settings
        or not 0 <= result["evaluations"] <= settings["max_evaluations"]
    ):
        raise ValueError("Fit receipt disagrees with the matched evaluation budget")
    if result["status"] == "fitted" and (
        not np.isfinite(result["training_nmse"])
        or any(result["negative_samples"][target] for target in bounds)
    ):
        raise ValueError("Invalid training winner or violated required bound")
    return result


def fit_arm(
    root: Path,
    *,
    panel_id: str,
    arm: str,
    graph: dict,
    train_rows: list,
    formulas: list,
    nonnegative_targets: list[str],
) -> dict:
    """Freeze shared training inputs, fit both slots, and select by training NMSE."""
    _safe_ids(panel_id, arm)
    if len(formulas) != len(SEEDS):
        raise ValueError("Supply exactly two candidate slots, using None for omissions")
    train = [r.model_dump(mode="json") for r in rollout._rows(train_rows, "train")]
    bounds = sorted(nonnegative_targets)
    if len(set(bounds)) != len(bounds) or set(bounds) - set(train[0]["targets"]):
        raise ValueError("Bounds must refer to distinct observed targets")
    folder = root / "fit_plans" / panel_id
    folder.mkdir(parents=True, exist_ok=True)
    # Different arms may start in parallel and must agree on their shared plan.
    with (folder / ".common.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        common = sealed_write(
            folder / "common.json",
            {
                "protocol": PROTOCOL,
                "panel_id": panel_id,
                "graph_sha256": content_hash(graph),
                "train_sha256": content_hash(train),
                "training_ids": [r["trajectory_id"] for r in train],
                "budget": dict(BUDGET),
                "seeds": list(SEEDS),
                "bounds": bounds,
                "sources": {
                    p.name: file_hash(p)
                    for p in (
                        Path(__file__),
                        Path(rollout.__file__),
                        Path(bounded.__file__),
                    )
                },
                "versions": {p: version(p) for p in ("numpy", "scipy", "casadi")},
            },
        )
    with _lock(folder / arm):
        plan = sealed_write(
            folder / f"{arm}.json",
            {
                "common": common["artifact_sha256"],
                "arm": arm,
                "formulas": formulas,
            },
        )
        receipts = []
        for index, formula in enumerate(formulas):
            marker = {"plan": plan["artifact_sha256"], "slot": index}
            path = root / "fits" / panel_id / arm / f"{index}.json"
            if path.exists():
                receipt = sealed_read(path)
                if any(receipt[k] != v for k, v in marker.items()):
                    raise ValueError("Fit receipt belongs to a different arm or slot")
            else:
                reservation = folder / arm / f"start-{index}.json"
                interrupted = reservation.exists()
                sealed_write(reservation, marker)
                result = (
                    {"status": "interrupted", "training_nmse": None}
                    if interrupted
                    else _fit_one(graph, formula, train, bounds, index)
                )
                receipt = sealed_write(path, {**marker, "result": result})
            receipts.append(receipt)
        scores = [
            (r["result"]["training_nmse"], r["slot"])
            for r in receipts
            if r["result"]["status"] == "fitted"
        ]
        best = min(scores, default=(None, None))
        return sealed_write(
            root / "selections" / panel_id / f"{arm}.json",
            {
                "panel": panel_id,
                "arm": arm,
                "plan": plan["artifact_sha256"],
                "receipts": [r["artifact_sha256"] for r in receipts],
                "fit_statuses": [r["result"]["status"] for r in receipts],
                "selected_slot": best[1],
                "training_nmse": best[0],
            },
        )


def _peak(time: np.ndarray, channel: np.ndarray, scale: float) -> dict:
    """Identify the earliest sampled maximum above the event's initial value."""
    index = int(np.argmax(channel))
    if channel[index] - channel[0] <= 1e-8 * scale:
        return {"status": "no_rise", "time": None}
    if index == len(time) - 1:
        return {"status": "right_boundary", "time": None}
    return {"status": "interior", "time": float(time[index])}


def _diagnostics(curves: list[dict], scales: dict[str, float]) -> dict:
    """Calculate post-initial forecast error and generic timing/recovery metrics."""
    events = []
    for curve in curves:
        time = np.asarray(curve["time"])
        late = time >= time[0] + 0.75 * (time[-1] - time[0])
        targets = {}
        for name, scale in scales.items():
            y = np.asarray(curve["observed"][name])
            yp = np.asarray(curve["predicted"][name])
            observed, predicted = (_peak(time, v, scale) for v in (y, yp))
            timing = (
                abs(predicted["time"] - observed["time"])
                if observed["time"] is not None and predicted["time"] is not None
                else None
            )
            deviation_error = (yp[late] - yp[0]) - (y[late] - y[0])
            targets[name] = {
                "nmse": float(np.mean(((yp[1:] - y[1:]) / scale) ** 2)),
                "observed_peak": observed,
                "predicted_peak": predicted,
                "peak_absolute_time_error": timing,
                "late_recovery_mae_scaled": float(
                    np.mean(abs(deviation_error)) / scale
                ),
                "late_recovery_mean_bias": float(np.mean(deviation_error)),
                "minimum_prediction": float(yp.min()),
                "negative_samples": int(np.sum(yp < -bounded.TOLERANCE)),
            }
        events.append(
            {
                "trajectory_id": curve["trajectory_id"],
                "targets": targets,
                "samples": len(time) - 1,
                "late_samples": int(late.sum()),
            }
        )
    per_target = {}
    for name in scales:
        records = [e["targets"][name] for e in events]
        peaks = [
            r["peak_absolute_time_error"]
            for r in records
            if r["peak_absolute_time_error"] is not None
        ]
        per_target[name] = {
            "nmse": float(
                np.average(
                    [r["nmse"] for r in records], weights=[e["samples"] for e in events]
                )
            ),
            "observed_interior_peaks": sum(
                r["observed_peak"]["status"] == "interior" for r in records
            ),
            "comparable_peaks": len(peaks),
            "mean_peak_absolute_time_error": float(np.mean(peaks)) if peaks else None,
            "mean_event_late_recovery_mae_scaled": float(
                np.mean([r["late_recovery_mae_scaled"] for r in records])
            ),
            "negative_samples": sum(r["negative_samples"] for r in records),
        }
    return {
        "events": events,
        "targets": per_target,
        "pooled_nmse": float(np.mean([r["nmse"] for r in per_target.values()])),
    }


def score_arm(
    root: Path, *, panel_id: str, arm: str, graph: dict, held_rows: list
) -> dict:
    """Forecast only disjoint train_* events with the frozen training winner."""
    _safe_ids(panel_id, arm)
    folder = root / "fit_plans" / panel_id
    common = sealed_read(folder / "common.json")
    plan = sealed_read(folder / f"{arm}.json")
    selected = sealed_read(root / "selections" / panel_id / f"{arm}.json")
    if (plan["common"], selected["plan"], common["graph_sha256"]) != (
        common["artifact_sha256"],
        plan["artifact_sha256"],
        content_hash(graph),
    ):
        raise ValueError("Graph, fit plan and selected training receipt must agree")
    held = rollout._rows(held_rows, "train")
    ids = [r.trajectory_id for r in held]
    if set(ids) & set(common["training_ids"]):
        raise ValueError("Held events overlap the fitting events")
    marker = {
        "selection": selected["artifact_sha256"],
        "held_ids": ids,
        "held_sha256": content_hash([r.model_dump(mode="json") for r in held]),
    }
    with _lock(folder / arm):
        score_path = root / "scores" / panel_id / f"{arm}.json"
        if score_path.exists():
            saved = sealed_read(score_path)
            if any(saved[k] != v for k, v in marker.items()):
                raise ValueError("Different held events or selection on resume")
            return saved
        result = {"status": "no_training_winner", "pooled_nmse": None}
        slot = selected["selected_slot"]
        if slot is not None:
            fit = sealed_read(root / "fits" / panel_id / arm / f"{slot}.json")
            form = rollout.validate_formula(graph, plan["formulas"][slot])
            fitted = fit["result"]
            if (
                fit["artifact_sha256"] != selected["receipts"][slot]
                or fitted["graph_sha256"] != common["graph_sha256"]
                or fitted["formula_sha256"] != content_hash(form)
                or set(fitted["scales"]) != set(held[0].targets)
            ):
                raise ValueError(
                    "Held data, formula, or fit differs from the frozen selection"
                )
            names = sorted(fitted["scales"])
            scales = np.array([fitted["scales"][n] for n in names])
            checked = rollout._check_grid(
                graph, form, held, fitted["parameters"], names, scales, curves=True
            )
            result.update(status=checked["status"], grid=checked)
            if checked["status"] == "scored":
                result.update(_diagnostics(checked["curves"], fitted["scales"]))
        return sealed_write(
            score_path, {**marker, "panel": panel_id, "arm": arm, **result}
        )
