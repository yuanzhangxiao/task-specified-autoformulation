"""Identify four temperature-balance families on public CSTR-easy trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, nnls

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import _rows
from scripts.construction_functions import file_hash
from scripts.trajectory_function_intake import public_rows

EASY = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1"
HARD = EASY.replace("_easy_", "_hard_")
FILES = {
    (
        EASY,
        "specification",
    ): "7c0730280c097057408d853e05199f1514ce490681c8288aa173844ed79b7f49",
    (EASY, "train"): "6ef8c170b90d7059366838b55dd8b1e39d81cb2b7d2da9bf54efbcc38d30de9e",
    (EASY, "val"): "8e7a2511ad6a30c7fede08c0e716651357e345eb902be94bd2937e4fb78973fb",
    (
        HARD,
        "specification",
    ): "1555727a0f8a35efc0c6e9628dc12e26812e959c64a5753146ffc3f0bbb90586",
}
FAMILIES = (
    "concentration",
    "linear_temperature",
    "exponential_temperature",
    "exponential_inverse_temperature",
)
BETA = tuple(np.linspace(0.0, 8.0, 17).tolist())
# Three Gauss-Legendre nodes for linearly interpolated public channels.
_gl_nodes, _gl_weights = np.polynomial.legendre.leggauss(3)
NODES, WEIGHTS = (_gl_nodes + 1) / 2, _gl_weights / 2


def _read(public_root, dataset, name):
    path = public_root / dataset / f"{name}.json"
    if (dataset, name) not in FILES or file_hash(path) != FILES[dataset, name]:
        raise ValueError(f"Unregistered public {dataset}/{name}")
    return (
        read_seal(path) if name == "specification" else _rows(public_rows(path), name)
    )


def eligibility(spec):
    """Check the public measurement roles before constructing the balance."""
    missing = {"C", "Tj"} - set(spec["auxiliaries"])
    if missing:
        return {
            "status": "unavailable",
            "missing_supplied_auxiliaries": sorted(missing),
        }
    if spec["targets"] != ["T"] or "Tf" not in spec["external_inputs"]:
        return {
            "status": "unavailable",
            "reason": "Need measured T and feed temperature Tf",
        }
    return {"status": "applicable"}


def multiplier(name, beta, temperature, lo, hi):
    """Temperature factor in a fixed heat-release law; positive Kelvin is required."""
    if name not in FAMILIES:
        raise ValueError("Unknown heat family")
    if name == "concentration" or beta == 0:
        return np.ones_like(temperature)
    if lo <= 0 or hi <= lo or np.any(temperature <= 0):
        raise ValueError("Positive training span and absolute temperature required")
    x = (temperature - lo) / (hi - lo)
    if name == "linear_temperature":
        return np.maximum(1 + beta * x, 0)
    if name == "exponential_inverse_temperature":
        x = x * hi / temperature
    with np.errstate(over="raise", invalid="raise"):
        return np.exp(beta * x)


@dataclass(frozen=True)
class Intervals:
    """Interval mean reaction covariates and exact transport/exchange integrals."""

    t_nodes: np.ndarray
    c_nodes: np.ndarray
    deltas: np.ndarray
    dt: np.ndarray
    transport: np.ndarray
    jacket: np.ndarray

    def design(self, name, beta, lo, hi):
        heat = np.sum(
            WEIGHTS * self.c_nodes * multiplier(name, beta, self.t_nodes, lo, hi),
            axis=1,
        )
        # Minimize the squared interval-mean derivative residual, weighted by duration.
        return np.column_stack((self.transport, self.jacket, heat)) * np.sqrt(
            self.dt[:, None]
        )

    def response(self):
        return self.deltas / np.sqrt(self.dt)


def intervals(rows):
    records = []
    for row in rows:
        t = np.asarray(row.time)
        T = np.asarray(row.targets["T"])
        C = np.asarray(row.auxiliaries["C"])
        if len(t) < 2 or np.any(T <= 0):
            raise ValueError("Need at least two positive measured temperatures")
        samples = [a[:-1, None] + np.diff(a)[:, None] * NODES for a in (T, C)]
        diffs = [v - T for v in (row.inputs["Tf"], row.auxiliaries["Tj"])]
        records.append(
            (*samples, np.diff(T), np.diff(t), *((a[:-1] + a[1:]) / 2 for a in diffs))
        )
    return Intervals(*(np.concatenate(column) for column in zip(*records, strict=True)))


def fit(rows, families=FAMILIES):
    """Fit each heat law on this training subset, recomputing temperature scales."""
    if not rows or not all(row.trajectory_id.startswith("train_") for row in rows):
        raise ValueError("Expected public training rows")
    d = intervals(rows)
    y = d.response()
    lo = min(min(r.targets["T"]) for r in rows)
    hi = max(max(r.targets["T"]) for r in rows)
    fits = {}
    for name in families:
        if name != "concentration" and hi - lo < 1e-8:
            fits[name] = {"status": "temperature_range_uninformative"}
            continue

        def coefficients(beta, name=name):
            A = d.design(name, beta, lo, hi)
            scale = np.linalg.norm(A, axis=0)
            norm = A / np.where(scale > 0, scale, 1)
            coefs, error = nnls(norm, y)
            coefs = coefs / np.where(scale > 0, scale, 1)
            return {
                "beta": float(beta),
                "a": float(coefs[0]),
                "b": float(coefs[1]),
                "c": float(coefs[2]),
                "rate_rmse": float(error / np.sqrt(np.sum(d.dt))),
            }

        series = [
            coefficients(b) for b in ((0.0,) if name == "concentration" else BETA)
        ]
        refined = []
        for i, sampled in enumerate(series):
            lo_idx, hi_idx = max(0, i - 1), min(i + 1, len(series) - 1)
            neighbors = [
                series[j]["rate_rmse"] for j in range(lo_idx, hi_idx + 1) if j != i
            ]
            if neighbors and sampled["rate_rmse"] < min(neighbors):
                res = minimize_scalar(
                    lambda z: coefficients(z)["rate_rmse"],
                    bounds=(series[lo_idx]["beta"], series[hi_idx]["beta"]),
                    method="bounded",
                    options={"xatol": 1e-8},
                )
                if res.success:
                    refined.append(coefficients(res.x))
        best = min(series + refined, key=lambda r: (r["rate_rmse"], r["beta"]))
        A = d.design(name, best["beta"], lo, hi)
        scale = np.linalg.norm(A, axis=0)
        singular = np.linalg.svd(A / np.where(scale > 0, scale, 1), compute_uv=False)
        rank = int(np.linalg.matrix_rank(A))
        condition = float(singular[0] / singular[-1]) if rank == 3 else None
        fits[name] = {
            "status": "fitted",
            "law": name,
            "best": best,
            "T_min": lo,
            "T_max": hi,
            "intervals": len(y),
            "nonzero_terms": sum(best[k] > 0 for k in ("a", "b", "c")),
            "design_rank": rank,
            "scaled_design_condition": condition,
            "beta_profile": series,
        }
    return fits


def rate_error(rows, model):
    """Score held training intervals using the fitting subset's normalization."""
    if model["status"] != "fitted":
        return None
    d = intervals(rows)
    p = model["best"]
    A = d.design(model["law"], p["beta"], model["T_min"], model["T_max"])
    residual = A @ np.array([p["a"], p["b"], p["c"]]) - d.response()
    return float(np.dot(residual, residual)), float(np.sum(d.dt))


def predict(t, feed, jacket, concentration, initial, model, substeps):
    """Integrate from measured T0 with only supplied future C, Tj and Tf."""
    p = model["best"]
    y = float(initial)
    path = [y]
    u = np.asarray([feed, jacket, concentration])

    def rate(z, inputs):
        if z <= 0 or not np.isfinite(z) or z > 1e8:
            raise ArithmeticError("Temperature rollout left its numerical domain")
        heat = multiplier(model["law"], p["beta"], z, model["T_min"], model["T_max"])
        return (
            p["a"] * (inputs[0] - z)
            + p["b"] * (inputs[1] - z)
            + p["c"] * inputs[2] * heat
        )

    for i, step in enumerate(np.diff(t) / substeps):
        base = u[:, i]
        inc = (u[:, i + 1] - base) / substeps
        for j in range(substeps):
            left = base + j * inc
            mid = left + inc / 2
            k1 = rate(y, left)
            k2 = rate(y + step * k1 / 2, mid)
            k3 = rate(y + step * k2 / 2, mid)
            k4 = rate(y + step * k3, left + inc)
            y += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        path.append(float(y))
    if not np.all(np.isfinite(path)) or np.min(path) <= 0:
        raise ArithmeticError("Invalid temperature forecast")
    return np.array(path)


def score(rows, model, scale):
    if model["status"] != "fitted":
        return {"status": "fit_unavailable"}
    curves, errors, diff = [], [], 0.0
    for r in rows:
        observed = np.asarray(r.targets["T"])
        args = (
            r.time,
            r.inputs["Tf"],
            r.auxiliaries["Tj"],
            r.auxiliaries["C"],
            observed[0],
            model,
        )
        try:
            coarse, fine = (predict(*args, n) for n in (2, 4))
        except (ArithmeticError, ValueError) as exc:
            return {
                "status": "rollout_failed",
                "event": r.trajectory_id,
                "reason": str(exc),
            }
        diff = max(diff, float(np.max(abs(coarse - fine)) / scale))
        delta = fine - observed
        errors.extend(delta.tolist())
        curves.append(
            {
                "event": r.trajectory_id,
                "time": list(r.time),
                "observed_T": observed.tolist(),
                "predicted_T": fine.tolist(),
                "rmse_K": float(np.sqrt(np.mean(delta**2))),
            }
        )
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    verified = diff <= 0.01
    return {
        "status": "scored" if verified else "resolution_unverified",
        "pooled_nmse": (rmse / scale) ** 2 if verified else None,
        "rmse_K": rmse,
        "max_scaled_grid_difference": diff,
        "events": curves,
    }


def _plan(easy, hard):
    return {
        "protocol": "cstr-public-temperature-law-control-1",
        "families": list(FAMILIES),
        "beta_grid": list(BETA),
        "code_sha256": file_hash(Path(__file__)),
        "scipy": version("scipy"),
        "numpy": version("numpy"),
        "easy": eligibility(easy),
        "hard": eligibility(hard),
        "inputs": {f"{ds}/{s}": sha for (ds, s), sha in FILES.items()},
        "selection": (
            "Lowest pooled held-training interval-rate RMSE across whole-event folds "
            "among models with all three nonzero full-fit terms. "
            "Ties follow listed family order."
        ),
        "scope": (
            "Temperature equation with public terms supplied by design. "
            "Gauss-Legendre quadrature uses linear interpolation of measured "
            "temperature and supplied channels inside training intervals. "
            "Forecasts interpolate only supplied C, Tj and Tf."
        ),
        "rollout": (
            "Classical RK4 from each observed T(0). Report four-substep forecasts "
            "when each time point agrees with two-substep forecasts within 1% of "
            "pooled training T standard deviation. Normalize MSE by training variance."
        ),
        "test_data_opened": False,
    }


def _fit_once(folder, plan, rows, hold=None):
    path = folder / "fit.json"
    identity = {
        "plan": plan["artifact_sha256"],
        "held": hold.trajectory_id if hold else None,
        "training_events": [r.trajectory_id for r in rows],
    }
    if path.exists():
        r = sealed_read(path)
        if any(r[k] != v for k, v in identity.items()):
            raise ValueError("Fit receipt differs from training fold")
        return r
    fits = fit(rows)
    scores = {n: rate_error([hold], m) for n, m in fits.items()} if hold else {}
    return sealed_write(path, {**identity, "models": fits, "held_errors": scores})


def train(public_root: Path, root: Path):
    """Freeze full/fold fits and selection before opening validation."""
    if root.resolve().is_relative_to(public_root.resolve()):
        raise ValueError("Choose an experiment root outside public inputs")
    easy, hard = (_read(public_root, ds, "specification") for ds in (EASY, HARD))
    plan_value = _plan(easy, hard)
    if (
        plan_value["easy"]["status"] != "applicable"
        or plan_value["hard"]["status"] != "unavailable"
    ):
        raise ValueError("Public roles differ from the predeclared control")
    rows = _read(public_root, EASY, "train")
    with _lock(root):
        plan = sealed_write(root / "plan.json", plan_value)
        full = _fit_once(root / "full", plan, rows)
        folds = [
            _fit_once(
                root / "folds" / r.trajectory_id,
                plan,
                [s for s in rows if s.trajectory_id != r.trajectory_id],
                r,
            )
            for r in rows
        ]
        costs = {}
        for n in FAMILIES:
            terms = [f["held_errors"][n] for f in folds]
            if all(v is not None for v in terms):
                loss, weight = np.sum(terms, axis=0)
                costs[n] = float(np.sqrt(loss / weight))
        choices = [
            n
            for n in FAMILIES
            if n in costs and full["models"][n].get("nonzero_terms") == 3
        ]
        chosen = min(choices, key=costs.get, default=None)
        scale = max(float(np.std(np.concatenate([r.targets["T"] for r in rows]))), 1e-8)
        winners = []
        for f in folds:
            valid = [
                n
                for n in FAMILIES
                if f["held_errors"][n] and f["models"][n].get("nonzero_terms") == 3
            ]
            if valid:
                winners.append(min(valid, key=lambda n: f["held_errors"][n][0]))
        freeze = sealed_write(
            root / "training.json",
            {
                "plan": plan["artifact_sha256"],
                "full_fit": full["artifact_sha256"],
                "folds": {f["held"]: f["artifact_sha256"] for f in folds},
                "selection": chosen,
                "cv_rate_rmse": costs,
                "fold_win_counts": dict(Counter(winners)),
                "std_T_K": scale,
                "validation_opened": False,
            },
        )
        return _score_all(root, rows, freeze, full, "train")


def _score_all(root, rows, freeze, full, split):
    scores = {}
    for name in FAMILIES:
        path = root / split / f"{name}.json"
        identity = {
            "training": freeze["artifact_sha256"],
            "family": name,
            "split": split,
        }
        if path.exists():
            record = sealed_read(path)
            if any(record[k] != v for k, v in identity.items()):
                raise ValueError("Forecast receipt differs from training freeze")
        else:
            record = sealed_write(
                path,
                {**identity, **score(rows, full["models"][name], freeze["std_T_K"])},
            )
        scores[name] = {
            k: v
            for k, v in record.items()
            if k not in ("events", "training", "family", "split")
        }
    return sealed_write(
        root / f"{split}-scores.json",
        {"training": freeze["artifact_sha256"], "scores": scores},
    )


def validate(public_root: Path, root: Path):
    """Score every frozen law on validation without fitting or target resets."""
    with _lock(root):
        plan = sealed_read(root / "plan.json")
        freeze = sealed_read(root / "training.json")
        full = sealed_read(root / "full/fit.json")
        expected = _plan(
            *(_read(public_root, ds, "specification") for ds in (EASY, HARD))
        )
        if (
            any(plan[k] != v for k, v in expected.items())
            or freeze["plan"] != plan["artifact_sha256"]
            or freeze["full_fit"] != full["artifact_sha256"]
            or full["plan"] != plan["artifact_sha256"]
        ):
            raise ValueError("Current protocol differs from the training freeze")
        for event, expected_digest in freeze["folds"].items():
            if (
                sealed_read(root / "folds" / event / "fit.json")["artifact_sha256"]
                != expected_digest
            ):
                raise ValueError("Training fold changed")
        return _score_all(root, _read(public_root, EASY, "val"), freeze, full, "val")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "validate"))
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    run = train if args.action == "train" else validate
    print(json.dumps(run(args.public_root, args.root), indent=2))


if __name__ == "__main__":
    main()
