"""Evaluate hidden-pathway identification and whole-event ambiguity decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import hidden_pathway_shape as pathway
from autoformalism.research import hidden_shape_selection as selection
from scripts.construction_functions import file_hash

DEFAULT = Path(__file__).resolve().parents[1] / "output/partial-shape-20261126/pathway"
SEEDS = tuple(range(8))
FULL_LEVELS = (0.0, 1.0, 2.0, 3.0)
TWO_LEVELS = (0.0, 1.0)
TIME = np.linspace(0, 8, 321)
PREFIX_END = 1.0
MARGIN = 0.015
# The two compact analytic hypotheses have equal numbers of adjustable coefficients.
POWERS = {"affine": 1, "offset_plus_square": 2}


def event(
    law: str, level: float, seed: int, index: int, split: int, noise: float
) -> dict:
    """Simulate two first-order states with event-specific hidden initial values."""
    rng = np.random.default_rng(np.random.SeedSequence([20261126, seed, index, split]))
    y0, z0 = rng.uniform(0.1, 2.0, 2)
    a, b = 0.4, 1.3
    q = 1 + level ** POWERS[law]
    ea, eb = np.exp(-a * TIME), np.exp(-b * TIME)
    y = ea * y0 + q / (b * a) * (1 - ea) + (z0 - q / b) * (eb - ea) / (a - b)
    y += rng.uniform(-noise, noise, len(TIME))
    return {
        "trajectory_id": f"{('train', 'development', 'evaluation')[split]}_{index:03}",
        "time": TIME.tolist(),
        "targets": {"y": y.tolist()},
        "inputs": {"u": [float(level)] * len(TIME)},
        "auxiliaries": {},
        "covariates": {},
    }


def prefix_forecast(
    time: np.ndarray, initial_observations: np.ndarray, s: float, p: float, q: float
) -> np.ndarray:
    """Estimate the initial slope on the prefix and forecast at constant input."""
    prefix = time <= PREFIX_END
    if len(initial_observations) != sum(prefix) or sum(prefix) < 2 or time[0] != 0:
        raise ValueError("Supply only the declared prefix, starting at time zero")
    if s <= 0 or p <= 0 or s * s < 4 * p - 1e-12:
        raise ValueError("The two-decay hypothesis requires positive real decay rates")
    root_gap = np.sqrt(max(0.0, s * s - 4 * p))
    r1 = (-s + root_gap) / 2
    h = (
        np.exp(r1 * time) * (-np.expm1(-root_gap * time)) / root_gap
        if root_gap > 0
        else time * np.exp(r1 * time)
    )
    base = q / p + (initial_observations[0] - q / p) * (np.exp(r1 * time) - r1 * h)
    velocity = np.linalg.lstsq(
        h[prefix, None], initial_observations - base[prefix], rcond=None
    )[0][0]
    return base + velocity * h


def forecast_errors(fit: dict, rows: list[dict]) -> dict:
    """Fit competing shapes to recovered levels and forecast separate events."""
    u = np.asarray(fit["levels"])
    errors = {}
    for law, power in POWERS.items():
        design = np.column_stack([np.ones(len(u)), u**power])
        if np.linalg.matrix_rank(design) < 2:
            raise ValueError("Insufficient distinct input levels")
        c0, c1 = np.linalg.lstsq(design, fit["q"], rcond=None)[0]
        result = {}
        for row in rows:
            t = np.array(row["time"])
            levels = np.unique(row["inputs"]["u"])
            if len(levels) != 1:
                raise ValueError("Constant-level control only")
            y = np.asarray(row["targets"]["y"])
            after = t > PREFIX_END
            pred = prefix_forecast(
                t,
                y[~after],
                fit["rate_sum"],
                fit["rate_product"],
                c0 + c1 * levels[0] ** power,
            )
            result[row["trajectory_id"]] = float(
                np.sqrt(np.mean((y[after] - pred[after]) ** 2))
            )
        errors[law] = result
    return errors


def trial(law: str, seed: int, noise: float, narrow: bool) -> dict:
    """Construct, select and evaluate on three separate sets of synthetic events."""
    levels = TWO_LEVELS if narrow else FULL_LEVELS
    rows = [event(law, u, seed, i, 0, noise) for i, u in enumerate(levels * 3)]
    # Public synthetic acquisition contract: |y''| <= 32 between samples,
    # sensor error <= noise. Chord interpolation adds at most 32*dt^2/8.
    error = noise + 32 * float(max(np.diff(TIME))) ** 2 / 8
    fitted = pathway.fit_hidden_pathway(
        rows,
        target="y",
        source="u",
        interpolation="previous_complete_knots",
        window_steps=80,
        output_error=error,
    )
    record = {"law": law, "seed": seed, "noise": noise, "narrow": narrow, "fit": fitted}
    if fitted["status"] != "estimated":
        return record
    development = {
        "same_levels": levels * 2,
        "intermediate_levels": (0.2, 0.4, 0.6, 0.8) * 2
        if narrow
        else (0.5, 1.5, 2.5) * 2,
    }
    decisions = {}
    for name, values in development.items():
        rows = [event(law, u, seed, i, 1, noise) for i, u in enumerate(values)]
        errors = forecast_errors(fitted, rows)
        decisions[name] = selection.compare_event_errors(errors, MARGIN)
    record["decisions"] = decisions
    max_u = max(levels)
    evaluate = [
        event(law, u * max_u, seed, i, 2, noise)
        for i, u in enumerate((0.13, 0.37, 0.56, 0.73, 0.88))
    ]
    record["evaluation_mean_rmse"] = {
        name: float(np.mean(list(errs.values())))
        for name, errs in forecast_errors(fitted, evaluate).items()
    }
    truth = np.array([1 + u ** POWERS[law] for u in fitted["levels"]])
    record["q_max_error"] = float(max(abs(truth - np.array(fitted["q"]))))
    return record


def equivalence_control() -> dict:
    """Compare two analytically equivalent hidden equations and a wrong decay."""
    t = np.linspace(0, 6, 61)
    b = Fraction(2, 5)
    results = []
    for other_b in (2 * b, 3 * b):
        errors = []
        for initial in (0.5, 1.0, 2.0):
            y1 = initial**2 * (-np.expm1(-2 * float(b) * t)) / (2 * float(b))
            y2 = initial**2 * (-np.expm1(-float(other_b) * t)) / float(other_b)
            errors.append(float(max(abs(y1 - y2))))
        results.append(
            {
                "other_decay": str(other_b),
                "proved_equivalent": selection.equivalent_power_readouts(
                    2, 1, b, other_b
                ),
                "largest_observed_difference": max(errors),
            }
        )
    return {"coordinate_map": "w=z**2 on z>0; w(0)=z(0)**2", "cases": results}


def run(root: Path) -> dict:
    """Seal the inputs, individually resumable trials and scores."""
    with _lock(root):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "partial-observation-shape-controls-1",
                "seeds": list(SEEDS),
                "noise_bounds": [0.0, 0.002],
                "prefix_end": PREFIX_END,
                "margin": MARGIN,
                "powers": POWERS,
                "code": {
                    Path(m).name: file_hash(Path(m))
                    for m in (__file__, pathway.__file__, selection.__file__)
                },
            },
        )
        summary = []
        for law in POWERS:
            for noise in plan["noise_bounds"]:
                for narrow in (False, True):
                    results = []
                    for seed in SEEDS:
                        path = (
                            root / "trials" / f"{law}-{noise}-{int(narrow)}-{seed}.json"
                        )
                        if path.exists():
                            item = sealed_read(path)
                            identity = (
                                item["plan"],
                                item["seed"],
                                item["law"],
                                item["noise"],
                                item["narrow"],
                            )
                            if identity != (
                                plan["artifact_sha256"],
                                seed,
                                law,
                                noise,
                                narrow,
                            ):
                                raise ValueError("Trial identity differs from its plan")
                        else:
                            item = sealed_write(
                                path,
                                {
                                    "plan": plan["artifact_sha256"],
                                    **trial(law, seed, noise, narrow),
                                },
                            )
                        results.append(item)
                    ok = [r for r in results if "decisions" in r]
                    summary.append(
                        {
                            "truth": law,
                            "noise": noise,
                            "narrow": narrow,
                            "runs": len(results),
                            "estimated": len(ok),
                            "max_q_error": max(
                                (r["q_max_error"] for r in ok), default=None
                            ),
                            "worst_condition": max(
                                (r["fit"]["condition"] for r in ok), default=None
                            ),
                            "decisions": {
                                setting: dict(
                                    Counter(
                                        r["decisions"][setting]["winner"]
                                        or "unresolved"
                                        for r in ok
                                    )
                                )
                                for setting in ("same_levels", "intermediate_levels")
                            },
                            "mean_evaluation_rmse": {
                                key: float(
                                    np.mean(
                                        [r["evaluation_mean_rmse"][key] for r in ok]
                                    )
                                )
                                for key in POWERS
                                if ok
                            },
                        }
                    )
        return sealed_write(
            root / "summary.json",
            {
                "plan": plan["artifact_sha256"],
                "rows": summary,
                "equivalence": equivalence_control(),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    print(json.dumps(run(parser.parse_args().root), indent=2))
