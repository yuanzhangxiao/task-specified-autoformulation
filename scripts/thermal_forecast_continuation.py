"""Continue a public CSTR interval fit using progressively longer training forecasts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import thermal_balance_trial as trial
from scripts.construction_functions import file_hash

FAMILIES = ("concentration", "exponential_inverse_temperature")
STARTS = ("interval", "small_shared")
PARTITIONS = (15, 4, 1)  # 2, 7.5 and 30 minutes on the registered regular grid.
LIMIT = 24
UPPER = [8.0, 8.0, 320.0, 8.0]
SCALE = [1.0, 2.0, 60.0, 2.0]


@dataclass(frozen=True)
class Windows:
    """Time steps, supplied signals and temperature indexed by step and window."""

    dt: np.ndarray
    supplied: np.ndarray
    temperature: np.ndarray


def windows(rows, partitions: int) -> Windows:
    """Divide each event into equally sized windows, retaining all transitions."""
    dt, signals, temperatures = [], [], []
    for row in rows:
        n = len(row.time) - 1
        if partitions < 1 or n % partitions:
            raise ValueError("Window count must divide the number of intervals")
        size = n // partitions
        u = np.column_stack(
            (row.inputs["Tf"], row.auxiliaries["Tj"], row.auxiliaries["C"])
        )
        for start in range(0, n, size):
            sl = slice(start, start + size + 1)
            dt.append(np.diff(row.time[sl]))
            signals.append(u[sl])
            temperatures.append(row.targets["T"][sl])
    return Windows(
        np.stack(dt, axis=1), np.stack(signals, axis=1), np.stack(temperatures, axis=1)
    )


def _rates(state, sensitivity, signals, parameters, lo, hi):
    a, b, c = parameters[:3]
    feed, jacket, concentration = signals.T
    direct = [feed - state, jacket - state]
    clipped = False
    dz = np.full_like(state, -a - b)
    if len(parameters) == 4:
        beta = parameters[3]
        bounded = np.clip(state, max(1.0, lo - 2 * (hi - lo)), hi + 2 * (hi - lo))
        if not np.array_equal(bounded, state):
            clipped = True
        span = 1 / lo - 1 / hi
        coordinate = (1 / lo - 1 / bounded) / span
        heat = concentration * np.exp(beta * coordinate)
        dz += c * heat * beta / (span * bounded**2) * (bounded == state)
        direct.extend((heat, c * heat * coordinate))
    else:
        heat = concentration
        direct.append(heat)
    rate = a * direct[0] + b * direct[1] + c * heat
    return rate, dz[:, None] * sensitivity + np.column_stack(direct), clipped


def residual_jacobian(data, parameters, lo, hi, scale):
    """Integrate forecasts and their parameter sensitivities without target feedback."""
    z = data.temperature[0].copy()
    s = np.zeros((len(z), len(parameters)))
    residual, jacobian = [], []
    clipped = False
    for i, step in enumerate(data.dt):
        left, right = data.supplied[i : i + 2]
        mid = (left + right) / 2
        h = step[:, None]
        k1, j1, c1 = _rates(z, s, left, parameters, lo, hi)
        k2, j2, c2 = _rates(z + step * k1 / 2, s + h * j1 / 2, mid, parameters, lo, hi)
        k3, j3, c3 = _rates(z + step * k2 / 2, s + h * j2 / 2, mid, parameters, lo, hi)
        k4, j4, c4 = _rates(z + step * k3, s + h * j3, right, parameters, lo, hi)
        z = z + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        s = s + h * (j1 + 2 * j2 + 2 * j3 + j4) / 6
        residual.append((z - data.temperature[i + 1]) / scale)
        jacobian.append(s / scale)
        clipped |= c1 or c2 or c3 or c4
    return np.concatenate(residual), np.vstack(jacobian), bool(clipped)


def optimize(data, initial, lo, hi, scale, limit=LIMIT):
    """Fit a bounded stage and count calls using the optimizer-only extension."""
    previous = None
    result = None
    calls = 0
    capped_calls = 0

    def evaluate(x):
        nonlocal previous, result, calls, capped_calls
        if previous is None or not np.array_equal(x, previous):
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = residual_jacobian(data, x, lo, hi, scale)
            previous = x.copy()
            calls += 1
            capped_calls += int(result[2])
        return result

    n = len(initial)
    fitted = least_squares(
        lambda x: evaluate(x)[0],
        initial,
        jac=lambda x: evaluate(x)[1],
        bounds=(np.zeros(n), UPPER[:n]),
        x_scale=SCALE[:n],
        max_nfev=limit,
    )
    return {
        "vector": fitted.x.tolist(),
        "status": int(fitted.status),
        "message": fitted.message,
        "nfev": fitted.nfev,
        "njev": fitted.njev,
        "objective_evaluations": calls,
        "clipped_objective_evaluations": capped_calls,
        "clipping_at_solution": evaluate(fitted.x)[2],
        "window_rmse_K": float(np.sqrt(np.mean(fitted.fun**2)) * scale),
    }


def _vector(model):
    p = model["best"]
    return [p[k] for k in ("a", "b", "c", "beta")][
        : 3 if model["law"] == FAMILIES[0] else 4
    ]


def _model(template, vector):
    p = dict(zip(("a", "b", "c", "beta")[: len(vector)], vector, strict=True))
    return {**template, "best": {"beta": 0.0, **p}}


def _forecast(rows, model, scale):
    report = trial.score(rows, model, scale)
    events = report.pop("events", [])
    report["per_event"] = {
        e["event"]: {
            "rmse_K": e["rmse_K"],
            "max_error_K": float(
                np.max(
                    np.abs(np.asarray(e["predicted_T"]) - np.asarray(e["observed_T"]))
                )
            ),
        }
        for e in events
    }
    return report


def _source(public_root, baseline):
    inputs = [
        trial._read(public_root, ds, "specification") for ds in (trial.EASY, trial.HARD)
    ]
    plan = sealed_read(baseline / "plan.json")
    if any(plan[k] != v for k, v in trial._plan(*inputs).items()):
        raise ValueError(
            "Prior fit differs from the registered family and input design"
        )
    freeze = sealed_read(baseline / "training.json")
    full = sealed_read(baseline / "full/fit.json")
    if freeze["plan"] != plan["artifact_sha256"] or (
        freeze["full_fit"] != full["artifact_sha256"] or full["plan"] != freeze["plan"]
    ):
        raise ValueError("Original fitting receipt differs from its training freeze")
    return freeze, full


def design(baseline):
    return {
        "protocol": "cstr-training-forecast-continuation-1",
        "code": file_hash(Path(__file__)),
        "baseline_training": baseline["artifact_sha256"],
        "families": list(FAMILIES),
        "starts": list(STARTS),
        "partitions": list(PARTITIONS),
        "maximum_evaluations_per_stage": LIMIT,
        "parameter_upper_bounds": UPPER,
        "common_small_start": [0.3, 0.3, 1.0, 0.125],
        "training": (
            "Nonnegative a,b,c with optional beta. One classical RK4 step on each "
            "sample interval with exact forward sensitivities of the discrete step. "
            "Each training window starts from its measured temperature. During the "
            "optimization only, cap the temperature in the heat factor two full "
            "training spans beyond the observed range, with a positive Kelvin floor."
        ),
        "selection": (
            "Include original and all six training checkpoints per family. Choose "
            "the lowest pooled original-equation full-training forecast NMSE passing "
            "the registered two/four-substep convergence guard with a,b,c > 1e-8. "
            "Pick the earlier vector on a tie. Freeze one vector per family."
        ),
        "scoring": (
            "Run both selected vectors once on registered validation, given only "
            "measured T(0), supplied Tf, Tj and C. Use the original heat equation "
            "and training variance. Report event errors and retain failures."
        ),
        "test_data_opened": False,
    }


def _receipt(path, identity, compute):
    if path.exists():
        record = sealed_read(path)
        if any(record[k] != v for k, v in identity.items()):
            raise ValueError(f"Forecast continuation receipt differs: {path}")
        return record
    return sealed_write(path, {**identity, **compute()})


def train(public_root: Path, baseline: Path, root: Path):
    if any(root.resolve().is_relative_to(p.resolve()) for p in (baseline, public_root)):
        raise ValueError("Write outside the registered inputs and original fit")
    original, full = _source(public_root, baseline)
    rows = trial._read(public_root, trial.EASY, "train")
    scale = original["std_T_K"]
    data = {n: windows(rows, n) for n in PARTITIONS}
    with _lock(root):
        plan = sealed_write(root / "plan.json", design(original))
        selections, dependencies = {}, {}
        for family in FAMILIES:
            template = full["models"][family]
            first = _vector(template)
            path = root / family / "interval.json"
            identity = {"plan": plan["artifact_sha256"], "law": family}
            interval = _receipt(
                path,
                identity,
                lambda m=template, v=first: {
                    "vector": v,
                    "forecast": _forecast(rows, m, scale),
                },
            )
            available = [("interval", interval)]
            for start in STARTS:
                previous = interval
                v = (
                    first
                    if start == "interval"
                    else [0.3, 0.3, 1.0, 0.125][: len(first)]
                )
                for n in PARTITIONS:
                    key = f"{start}-{n}"
                    ident = {
                        **identity,
                        "previous": previous["artifact_sha256"],
                        "initial_vector": v,
                        "partitions": n,
                        "start": start,
                    }

                    def run_stage(initial=v, batch=data[n], m=template):
                        try:
                            opt = optimize(
                                batch, initial, m["T_min"], m["T_max"], scale
                            )
                        except (ArithmeticError, ValueError, RuntimeError) as exc:
                            return {
                                "status": "optimizer_failed",
                                "reason": str(exc),
                                "vector": initial,
                                "forecast": {"status": "optimizer_failed"},
                            }
                        return {
                            **opt,
                            "forecast": _forecast(
                                rows, _model(m, opt["vector"]), scale
                            ),
                        }

                    previous = _receipt(root / family / f"{key}.json", ident, run_stage)
                    v = previous["vector"]
                    available.append((key, previous))
            eligible = [
                (key, r)
                for key, r in available
                if r["forecast"]["status"] == "scored"
                and all(value > 1e-8 for value in r["vector"][:3])
            ]
            if eligible:
                key, best = min(
                    eligible, key=lambda pair: pair[1]["forecast"]["pooled_nmse"]
                )
                selections[family] = {
                    "checkpoint": key,
                    "vector": best["vector"],
                    "train_nmse": best["forecast"]["pooled_nmse"],
                }
            else:
                selections[family] = {
                    "checkpoint": None,
                    "status": "no_eligible_rollout",
                }
            dependencies[family] = {key: r["artifact_sha256"] for key, r in available}
        choices = [f for f in FAMILIES if selections[f]["checkpoint"] is not None]
        return sealed_write(
            root / "training.json",
            {
                "plan": plan["artifact_sha256"],
                "selections": selections,
                "preferred": min(
                    choices, key=lambda f: selections[f]["train_nmse"], default=None
                ),
                "checkpoints": dependencies,
                "validation_opened": False,
            },
        )


def validate(public_root: Path, baseline: Path, root: Path):
    """Verify training selections first and score both arms exactly once."""
    original, full = _source(public_root, baseline)
    with _lock(root):
        plan = sealed_read(root / "plan.json")
        freeze = sealed_read(root / "training.json")
        if any(plan[k] != v for k, v in design(original).items()) or (
            freeze["plan"] != plan["artifact_sha256"]
        ):
            raise ValueError("Training plan differs")
        for family, records in freeze["checkpoints"].items():
            for key, sha in records.items():
                if sealed_read(root / family / f"{key}.json")["artifact_sha256"] != sha:
                    raise ValueError("Candidate checkpoint differs")
        scoring = []
        for name in FAMILIES:
            selected = freeze["selections"][name]
            if selected["checkpoint"] is not None:
                scoring.append((name, _model(full["models"][name], selected["vector"])))

        def run():
            vals = trial._read(public_root, trial.EASY, "val")
            scores = {n: _forecast(vals, m, original["std_T_K"]) for n, m in scoring}
            return {
                "scores": {
                    n: scores.get(n, {"status": "no_eligible_rollout"})
                    for n in FAMILIES
                }
            }

        return _receipt(
            root / "val-scores.json", {"training": freeze["artifact_sha256"]}, run
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "validate"))
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run = train if args.action == "train" else validate
    print(json.dumps(run(args.public_root, args.baseline_root, args.root), indent=2))


if __name__ == "__main__":
    main()
