"""Estimate two public basin discharge laws from interval water balances."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import BASIN
from autoformalism.research.trajectory_function_fitting import _rows
from autoformalism.search.topology_compatibility import CompleteTrajectory
from scripts import basin_function_family as family
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash

STRIDES = (1, 2, 4)
PROFILE = tuple(float(p) for p in np.linspace(1, 2, 21))
PROTOCOL = "basin-public-balance-inference-1"


@dataclass(frozen=True)
class EventVolumes:
    """Observed depths and inlet volumes between retained observation times."""

    name: str
    t: np.ndarray
    h_down: np.ndarray
    up_in: np.ndarray
    down_in: np.ndarray
    isolated: np.ndarray
    survey: dict


def head_integral(a, b, dt, p: float) -> np.ndarray:
    """Integrate positive head to p along each linear signed-head segment."""
    if not 1 <= p <= 2:
        raise ValueError("Expected an exponent in [1,2]")
    a, b, dt = np.broadcast_arrays(a, b, dt)
    if np.any(dt <= 0) or not all(np.all(np.isfinite(v)) for v in (a, b, dt)):
        raise ValueError("Expected finite heads and positive interval lengths")
    low, high = np.minimum(a, b), np.maximum(a, b)
    mean = np.zeros_like(dt, dtype=float)
    flat = (high > 0) & (high == low)
    mean[flat] = high[flat] ** p
    active = (high > 0) & (high > low)
    # The antiderivative from zero is x**(p+1)/(p+1). expm1 avoids
    # cancellation when two above-crest endpoints are numerically close.
    ratio = (high[active] - low[active]) / high[active]
    factor = 1 / ((p + 1) * ratio)
    above = ratio < 1
    factor[above] *= -np.expm1((p + 1) * np.log1p(-ratio[above]))
    mean[active] = high[active] ** p * factor
    return dt * mean


def event_volumes(row: CompleteTrajectory, stride: int) -> EventVolumes:
    """Thin measured depths and integrate inputs at all original knots."""
    if stride not in STRIDES or not row.trajectory_id.startswith("train_"):
        raise ValueError("Expected a registered stride and training event")
    t = np.asarray(row.time)
    dt = np.diff(t)
    if len(t) < 2 or np.any(dt <= 0):
        raise ValueError("Need strictly increasing observation times")
    keep = np.unique(np.r_[np.arange(0, len(t), stride), len(t) - 1])
    volumes = {}
    for side in ("up", "down"):
        u = np.asarray(row.inputs[f"inflow_{side}"])
        if np.any(u < 0):
            raise ValueError("The isolation certificate needs nonnegative runoff")
        full = np.r_[0.0, np.cumsum((u[:-1] + u[1:]) / 2 * dt)]
        volumes[side] = full[keep]
    c = row.covariates
    if min(c["area_up"], c["area_down"]) <= 0:
        raise ValueError("Surveyed basin areas must be positive")
    room = (c["crest_up"] - c["initial_up"]) * c["area_up"]
    return EventVolumes(
        row.trajectory_id,
        t[keep],
        np.asarray(row.targets["h_down"])[keep],
        np.diff(volumes["up"]),
        np.diff(volumes["down"]),
        volumes["up"][1:] <= room,
        c,
    )


@dataclass(frozen=True)
class BalanceSample:
    """One or more intervals with a measured or reconstructed volume response."""

    start: np.ndarray
    end: np.ndarray
    dt: np.ndarray
    volume: np.ndarray


def fit_power(samples: list[BalanceSample]) -> dict:
    """Profile nonnegative gain for each power and refine grid minima."""
    count = sum(len(s.dt) for s in samples)
    if not count:
        return {"status": "no_intervals", "intervals": 0}
    a, b, dt, y = (
        np.concatenate([getattr(s, name) for s in samples])
        for name in ("start", "end", "dt", "volume")
    )
    active = int(np.sum(np.maximum(a, b) > 0))
    if not active:
        return {"status": "no_positive_head", "intervals": count, "active_intervals": 0}

    def profile(p):
        x = head_integral(a, b, dt, p)
        # Residuals are interval mean rates, weighted by interval duration.
        gain = max(0.0, float(np.dot(x / dt, y) / np.dot(x / dt, x)))
        residual = gain * x - y
        return {
            "power": float(p),
            "gain": gain,
            "flow_rmse_m3_per_min": float(
                np.sqrt(np.sum(residual**2 / dt) / np.sum(dt))
            ),
        }

    grid = [profile(p) for p in PROFILE]
    key = "flow_rmse_m3_per_min"
    baseline = np.sqrt(np.sum(y**2 / dt) / np.sum(dt))
    chosen = min(grid, key=lambda r: r[key])
    record = {
        "intervals": count,
        "active_intervals": active,
        "profile": grid,
        "zero_law_rmse_m3_per_min": float(baseline),
    }
    if max(r[key] for r in grid) - chosen[key] <= 1e-10 * max(baseline, 1e-12):
        return {**record, "status": "unidentified_power"}
    minima = [chosen]
    for left, center, right in zip(grid[:-2], grid[1:-1], grid[2:], strict=True):
        if center[key] <= min(left[key], right[key]):
            result = minimize_scalar(
                lambda p: profile(p)[key],
                method="bounded",
                bounds=(left["power"], right["power"]),
                options={"xatol": 1e-9},
            )
            if result.success:
                minima.append(profile(result.x))
    best = min(minima, key=lambda r: (r[key], r["power"]))
    return {**record, "status": "estimated", "best": best}


def estimate(rows: list[CompleteTrajectory], stride: int) -> dict:
    """Recover outlet, transfer and upstream depth from public training."""
    events = [event_volumes(r, stride) for r in rows]
    samples = []
    for e in events:
        c, mask = e.survey, e.isolated
        head = e.h_down - c["crest_down"]
        volume = e.down_in - c["area_down"] * np.diff(e.h_down)
        samples.append(
            BalanceSample(
                head[:-1][mask], head[1:][mask], np.diff(e.t)[mask], volume[mask]
            )
        )
    down = fit_power(samples)
    record = {
        "stride": stride,
        "downstream": down,
        "isolated_intervals": {e.name: int(e.isolated.sum()) for e in events},
    }
    if down["status"] != "estimated":
        return {**record, "upstream": {"status": "outlet_unidentified"}}

    upstream_samples, reconstructions = [], {}
    law = down["best"]
    for e in events:
        c, dt = e.survey, np.diff(e.t)
        head = e.h_down - c["crest_down"]
        outlet = law["gain"] * head_integral(head[:-1], head[1:], dt, law["power"])
        q = c["area_down"] * np.diff(e.h_down) - e.down_in + outlet
        h = c["initial_up"] + np.r_[0.0, np.cumsum(e.up_in - q)] / c["area_up"]
        head_up = h - c["crest_up"]
        # Isolated windows cannot identify an upstream shape. Keep their signed
        # reconstructed flux as a check and in the cumulative storage calculation.
        mask = ~e.isolated
        upstream_samples.append(
            BalanceSample(head_up[:-1][mask], head_up[1:][mask], dt[mask], q[mask])
        )
        reconstructions[e.name] = {
            "time": e.t.tolist(),
            "upstream_depth_m": h.tolist(),
            "transfer_volume_m3": q.tolist(),
            "negative_transfer_intervals": int(np.sum(q < -1e-9)),
            "minimum_transfer_volume_m3": float(np.min(q)),
            "minimum_upstream_depth_m": float(np.min(h)),
            "isolated_max_abs_transfer_m3": (
                float(np.max(np.abs(q[e.isolated]))) if np.any(e.isolated) else None
            ),
        }
    up = fit_power(upstream_samples)
    return {**record, "upstream": up, "reconstruction": reconstructions}


def _prediction(time, inputs, c, initial_down, laws, refinement):
    """Predict from event initials and linear inlets with an independent solver."""
    t = np.asarray(time)
    up, down = laws["upstream"], laws["downstream"]

    def rhs(when, h):
        q = up["gain"] * max(h[0] - c["crest_up"], 0.0) ** up["power"]
        d = down["gain"] * max(h[1] - c["crest_down"], 0.0) ** down["power"]
        inlet_u = np.interp(when, t, inputs["inflow_up"])
        inlet_d = np.interp(when, t, inputs["inflow_down"])
        return [(inlet_u - q) / c["area_up"], (inlet_d + q - d) / c["area_down"]]

    prediction = solve_ivp(
        rhs,
        (t[0], t[-1]),
        (c["initial_up"], initial_down),
        t_eval=t,
        method="DOP853",
        rtol=1e-8 / refinement**2,
        atol=1e-10 / refinement**2,
        max_step=float(np.min(np.diff(t))) / refinement,
    )
    if not prediction.success or not np.all(np.isfinite(prediction.y)):
        raise ArithmeticError("Open-loop integration failed")
    return prediction.y


def score(rows, frozen: dict) -> dict:
    """Score the previously frozen full-resolution training estimates after rollout."""
    if frozen["laws"] is None:
        return {"status": "fit_unavailable"}
    scale = frozen["training_depth_std_m"]
    rows_out, errors, diff = [], [], 0.0
    for row in rows:
        y = np.array(row.targets["h_down"])
        args = row.time, row.inputs, row.covariates, y[0], frozen["laws"]
        try:
            coarse, fine = (_prediction(*args, n) for n in (2, 4))
        except (ArithmeticError, ValueError) as exc:
            return {
                "status": "rollout_failed",
                "event": row.trajectory_id,
                "reason": str(exc),
                "pooled_nmse": None,
            }
        diff = max(diff, float(np.max(abs(coarse[1] - fine[1])) / scale))
        residual = (fine[1] - y) / scale
        errors.extend(residual.tolist())
        rows_out.append(
            {
                "event": row.trajectory_id,
                "nmse": float(np.mean(residual**2)),
                "minimum_downstream_m": float(min(fine[1])),
                "minimum_upstream_m": float(min(fine[0])),
                "time": list(row.time),
                "observed": y.tolist(),
                "predicted": fine[1].tolist(),
            }
        )
    passed = diff <= 0.01
    return {
        "status": "scored" if passed else "resolution_unverified",
        "pooled_nmse": float(np.mean(np.square(errors))) if passed else None,
        "max_scaled_grid_difference": diff,
        "events": rows_out,
    }


def _code() -> dict:
    return {
        "script_sha256": file_hash(Path(__file__)),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }


def train(campaign: Path, root: Path) -> dict:
    """Seal all training estimates before an optional validation command."""
    if root.resolve().is_relative_to(campaign.resolve()):
        raise ValueError("Use an output root separate from the returned campaign")
    intake = sealed_read(campaign / "intake.json")
    original = sealed_read(campaign / "plan.json")
    if (
        intake["artifact_sha256"] != family.INTAKE
        or original["artifact_sha256"] != family.SOURCE_PLAN
        or original["intake_sha256"] != family.INTAKE
        or intake["source"] != trial.TRAIN
    ):
        raise ValueError("Unregistered public basin training cohort")
    rows = _rows(intake["train"][BASIN], "train")
    with _lock(root):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "code": _code(),
                "training": family.INTAKE,
                "source_plan": family.SOURCE_PLAN,
                "validation": trial.VALIDATION,
                "strides": list(STRIDES),
                "profile_powers": list(PROFILE),
                "training_depth_std_m": max(
                    1e-8,
                    float(np.std(np.concatenate([r.targets["h_down"] for r in rows]))),
                ),
                "law": "flow = gain * max(h - surveyed crest, 0)**power; gain >= 0",
                "assumptions": [
                    "Open-loop depth balances with known areas, boundary flows "
                    "and measured initial depths.",
                    "No intervening storage, other source or loss, or backwater. "
                    "Law parameters are common to events.",
                    "Head integrals assume linear depth between retained observations. "
                    "Reconstruction inherits outlet and interpolation errors.",
                    "Profile minima describe this bounded law family. Statistical "
                    "identification needs assumptions about noise and excitation.",
                ],
                "selection": "Forecast at stride 1. Strides 2 and 4 test sensitivity.",
                "test_data_opened": False,
            },
        )
        estimates = []
        for stride in STRIDES:
            path = root / f"estimates/stride-{stride}.json"
            if path.exists():
                result = sealed_read(path)
                if (
                    result["plan"] != plan["artifact_sha256"]
                    or result["stride"] != stride
                ):
                    raise ValueError("Mismatched estimation receipt")
            else:
                result = sealed_write(
                    path, {"plan": plan["artifact_sha256"], **estimate(rows, stride)}
                )
            estimates.append(result)
        frozen = sealed_write(
            root / "training.json",
            {
                "plan": plan["artifact_sha256"],
                "estimates": [r["artifact_sha256"] for r in estimates],
                "training_depth_std_m": plan["training_depth_std_m"],
                "laws": _laws(estimates[0]),
                "validation_opened": False,
            },
        )
        return _score_once(root, rows, frozen, "train")


def _laws(estimate: dict) -> dict | None:
    sides = ("upstream", "downstream")
    if any(estimate[s]["status"] != "estimated" for s in sides):
        return None
    return {s: estimate[s]["best"] for s in sides}


def _score_once(root, rows, frozen, split):
    path = root / f"{split}-rollout.json"
    if path.exists():
        result = sealed_read(path)
        if result["freeze"] != frozen["artifact_sha256"] or result["split"] != split:
            raise ValueError("Mismatched forecast receipt")
        return result
    return sealed_write(
        path,
        {"freeze": frozen["artifact_sha256"], "split": split, **score(rows, frozen)},
    )


def validate(root: Path, validation: Path) -> dict:
    """Evaluate the frozen training law on the registered validation events once."""
    with _lock(root):
        plan, frozen = (sealed_read(root / n) for n in ("plan.json", "training.json"))
        if (
            plan["protocol"] != PROTOCOL
            or plan["code"] != _code()
            or frozen["plan"] != plan["artifact_sha256"]
            or plan["training"] != family.INTAKE
        ):
            raise ValueError("Invalid training freeze for this code or cohort")
        estimates = [sealed_read(root / f"estimates/stride-{s}.json") for s in STRIDES]
        if (
            [r["artifact_sha256"] for r in estimates] != frozen["estimates"]
            or frozen["laws"] != _laws(estimates[0])
            or frozen["training_depth_std_m"] != plan["training_depth_std_m"]
            or any(
                r["plan"] != plan["artifact_sha256"] or r["stride"] != s
                for r, s in zip(estimates, STRIDES, strict=True)
            )
        ):
            raise ValueError("Training estimation receipts changed")
        if validation.name != "validation.json":
            raise ValueError("Expected registered validation.json")
        hold = sealed_read(validation)
        if (
            hold["artifact_sha256"] != trial.VALIDATION
            or hold["training_intake_sha256"] != trial.TRAIN
            or hold["split"] != "val"
        ):
            raise ValueError("Unregistered validation cohort")
        return _score_once(root, _rows(hold["rows"][BASIN], "val"), frozen, "val")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "validate"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    if args.action == "train" and args.campaign:
        result = train(args.campaign, args.root)
    elif args.action == "validate" and args.validation:
        result = validate(args.root, args.validation)
    else:
        parser.error("train requires --campaign; validate requires --validation")
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
