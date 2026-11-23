"""Training diagnostic for a fixed glucose balance with disposal as an input."""

from __future__ import annotations

import argparse
from pathlib import Path

import casadi as ca
import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.positive_insulin_refit_inputs import load_training
from scripts.screen_t2_disposal_revisions import MODEL_ARCHIVE, MODELS, TRAIN


def balance(model: dict, supplied: set[str]) -> tuple[list[str], ca.Function, float]:
    """Separate dGp/dt = S(supplied) + b*U only if b is a nonzero constant."""
    formula = model["formula"]
    expr = RestrictedParser().parse(formula["equations"]["Gp"], location="Gp")
    names = sorted(expr.symbols - formula["parameters"].keys() - {"U"})
    if "U" not in expr.symbols or not set(names) <= supplied:
        raise ValueError("Expected U and only supplied signals in the glucose balance")
    x = ca.SX.sym("forcing", len(names))
    u = ca.SX.sym("disposal")
    env = dict(zip(names, ca.vertsplit(x), strict=True)) | {"U": u}
    rhs = _translate(ca, expr, formula["parameters"] | env)
    slope = ca.jacobian(rhs, u)
    if not slope.is_constant() or not np.isfinite(float(slope)) or float(slope) == 0:
        raise ValueError("Expected a constant nonzero disposal coefficient")
    source = ca.Function("source", [x], [ca.substitute(rhs, u, 0)])
    return names, source, float(slope)


def source_integral(
    time: np.ndarray, channels: np.ndarray, source: ca.Function, order: int
) -> np.ndarray:
    """Integrate the nonlinear supplied balance between piecewise-linear knots."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    args = channels[:, :-1, None] + np.diff(channels)[:, :, None] * ((1 + nodes) / 2)
    rates = np.asarray(source(args.reshape(len(channels), -1))).reshape(-1, order)
    increments = np.diff(time) * (rates @ weights) / 2
    return np.r_[0.0, np.cumsum(increments)]


def integrated_disposal(time: np.ndarray) -> np.ndarray:
    """Map the sampled disposal values to their integral under linear interpolation."""
    dt = np.diff(time)
    if not dt.size or not np.all(np.isfinite(time)) or not np.all(dt > 0):
        raise ValueError("Expected finite increasing event times")
    intervals = np.zeros((len(time), len(time)))
    j = np.arange(len(dt))
    intervals[j + 1, j] = intervals[j + 1, j + 1] = dt / 2
    return np.cumsum(intervals, axis=0)


def event(row: dict, model: dict) -> dict:
    """Compare measured U and the best unconstrained U sequence at these knots."""
    supplied = row["auxiliaries"] | row["inputs"]
    names, source, b = balance(model, set(supplied))
    t = np.asarray(row["time"])
    channels = np.array([supplied[k] for k in names])
    g, u = (np.asarray(row["targets"][k]) for k in ("Gp", "U"))
    gscale, uscale = (model["scales"][k] for k in ("Gp", "U"))
    integral = source_integral(t, channels, source, 16)
    check = source_integral(t, channels, source, 32)
    if not np.allclose(check, integral, rtol=1e-10, atol=1e-8):
        raise ArithmeticError("Unresolved forcing quadrature")
    response = b * integrated_disposal(t)
    residual = (g[0] + check + response @ u - g) / gscale
    # A free correction at every observation time is more flexible than any
    # algebraic readout with a finite set of shared parameters on this grid.
    design = response * uscale / gscale
    delta = np.linalg.lstsq(
        np.vstack([design, np.eye(len(t))]),
        np.r_[-residual, np.zeros(len(t))],
        rcond=None,
    )[0]
    best_g = residual + design @ delta
    return {
        "event": row["trajectory_id"],
        "samples": len(t),
        "disposal_slope": b,
        "measured_U_glucose_scaled_sse": float(residual @ residual),
        "free_U_glucose_scaled_sse": float(best_g @ best_g),
        "free_U_disposal_scaled_sse": float(delta @ delta),
        "max_quadrature_difference": float(max(abs(check - integral))),
    }


def run(models: Path, training: Path, destination: Path) -> dict:
    """Use the authenticated training cohort and frozen Orion slot 015."""
    fitted = sealed_read(models)
    if fitted["artifact_sha256"] != MODEL_ARCHIVE:
        raise ValueError("Expected the matched Orion refits")
    rows = load_training(training)
    result = [event(r, fitted["models"]["015-positive_EGP"]["joint"]) for r in rows]
    n = sum(r["samples"] for r in result)
    scores = {
        k.removesuffix("_scaled_sse") + "_nmse": sum(r[k] for r in result) / n
        for k in result[0]
        if k.endswith("_scaled_sse")
    }
    return sealed_write(
        destination,
        {
            "protocol": "t2-disposal-fixed-balance-diagnostic-1",
            "source_models": MODEL_ARCHIVE,
            "model": "015-positive_EGP/joint",
            "training": sealed_read(training)["artifact_sha256"],
            "events": result,
            "scores": scores,
            "assumption": "U interpolated linearly between samples",
            "interpretation": "Training-only conditional calculation with known Gp(0). "
            "The optimized U sequence sees each whole training event. It is an error "
            "floor for this interpolation and frozen glucose equation, not a forecast "
            "or a bound on continuous paths between sampled observations.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--training", type=Path, default=TRAIN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.models, args.training, args.output)["scores"])
