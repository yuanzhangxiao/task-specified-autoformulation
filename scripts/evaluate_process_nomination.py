"""Evaluate full scalar law nomination on synthetic controls and public basin events."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import basin_observed_process, observed_process_laws
from autoformalism.research.preproposal_cases import BASIN
from autoformalism.search.topology_compatibility import CompleteTrajectory
from scripts.construction_functions import file_hash

PUBLIC = Path("artifacts/phase-c-development-final-v2/public") / BASIN
TRUTH = ("square_root", "linear", "power", "saturating", "turnover")
NOISE = (0.0, 0.03)
SEEDS = 12
POWER_TOLERANCE = 0.10


def _integral(x: np.ndarray, family: str, exponent: float) -> np.ndarray:
    """Independent antiderivatives of the five synthetic rates above threshold."""
    if family == "square_root":
        return 2.2 * (2 / 3) * x**1.5
    if family == "linear":
        return 0.85 * x**2
    if family == "power":
        return 1.9 * x ** (exponent + 1) / (exponent + 1)
    if family == "saturating":
        return 3.5 * (x - 0.48 * np.log1p(x / 0.48))
    if family == "turnover":
        return 2 * np.log1p(x**2 / 0.28)
    raise ValueError("Unknown synthetic control")


def _excursion(rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """A signed excursion relative to the specified activation threshold."""
    phase = np.linspace(0, 2 * np.pi, 25)
    return amplitude * (0.36 + 0.66 * np.sin(phase + rng.uniform(-0.45, 0.45)))


def _mean_truth(path: np.ndarray, family: str, exponent: float) -> np.ndarray:
    """Integrate over the positive fraction of each linear source interval."""
    lo = np.minimum(path[1:], path[:-1])
    hi = np.maximum(path[1:], path[:-1])
    if np.any(hi == lo):
        raise ValueError("The synthetic excursion design needs moving intervals")
    return (
        _integral(np.maximum(hi, 0), family, exponent)
        - _integral(np.maximum(lo, 0), family, exponent)
    ) / (hi - lo)


def _recovers(candidate: dict, family: str, exponent: float) -> bool:
    """Count a nested power as a recovery when its estimated exponent agrees."""
    if family == "saturating":
        return candidate["family"] == family
    if family == "turnover" or candidate["family"] == "saturating":
        return False
    fixed = {"square_root": 0.5, "linear": 1.0}
    estimated = (
        candidate["parameters"]["exponent"]
        if candidate["family"] == "power"
        else fixed[candidate["family"]]
    )
    return abs(estimated - fixed.get(family, exponent)) <= POWER_TOLERANCE


def synthetic_case(family: str, noise: float, seed: int) -> dict:
    """Train on three excursions, then score the selected laws on a new excursion."""
    rng = np.random.default_rng(seed)
    exponent = rng.uniform(1.3, 1.95)
    events = []
    for index, amplitude in enumerate((0.65, 1.55, 2.7)):
        path = _excursion(rng, amplitude)
        signal = _mean_truth(path, family, exponent)
        error_bound = noise * np.ptp(signal)
        observed = signal + rng.uniform(-error_bound, error_bound, size=signal.size)
        events.append(
            {
                "trajectory_id": f"train_{index:03d}",
                "left": path[:-1].tolist(),
                "right": path[1:].tolist(),
                "mean_rate": observed.tolist(),
                "absolute_error_bound": float(error_bound),
            }
        )
    menu = observed_process_laws.fit_law_menu(events)
    selected = menu["candidates"][:2]
    path = _excursion(rng, 3.25)
    actual = _mean_truth(path, family, exponent)
    scores = []
    for candidate in selected:
        predicted = observed_process_laws.interval_mean_law(
            candidate, path[:-1], path[1:]
        )
        scores.append(
            {
                "family": candidate["family"],
                "parameters": candidate["parameters"],
                "recovers": _recovers(candidate, family, exponent),
                "new_event_rate_nmse": float(
                    np.mean((actual - predicted) ** 2) / np.var(actual)
                ),
            }
        )
    return {
        "truth": family,
        "noise": noise,
        "seed": seed,
        "status": menu["status"],
        "shape_check": menu["shape_check"],
        "true_power": float(exponent) if family == "power" else None,
        "shortlist": scores,
    }


def _training_rows() -> list[CompleteTrajectory]:
    """Only the registered public training split is opened."""
    split = read_seal(PUBLIC / "train.json")
    if split["name"] != "train":
        raise ValueError("Expected the public training split")
    rows = [
        CompleteTrajectory(
            trajectory_id=r["trajectory_id"],
            time=r["time"],
            targets=r["targets"],
            covariates=r["fixed_covariates"],
            inputs=r["external_inputs"],
            auxiliaries=r["auxiliaries"],
        )
        for r in split["rows"]
    ]
    names = [row.trajectory_id for row in rows]
    if (
        len(rows) < 3
        or len(set(names)) != len(names)
        or any(not n.startswith("train_") for n in names)
    ):
        raise ValueError("Expected at least three distinct training events")
    return rows


def _checkpoint(path: Path, plan: str, case: str, compute: Callable[[], dict]) -> dict:
    """Keep completed folds across interruptions and verify before reuse."""
    if path.exists():
        saved = sealed_read(path)
        if (saved["plan"], saved["case"]) != (plan, case):
            raise ValueError("Checkpoint belongs to another experiment or fold")
        return saved["result"]
    result = compute()
    sealed_write(path, {"plan": plan, "case": case, "result": result})
    return result


def synthetic_summary(cases: list[dict]) -> list[dict]:
    """Keep unseen families and missing nominations in their denominators."""
    rows = []
    for family in TRUTH:
        for noise in NOISE:
            cases_here = [
                c for c in cases if (c["truth"], c["noise"]) == (family, noise)
            ]
            lists = [c["shortlist"] for c in cases_here]
            first = [s[0] for s in lists if s]
            errors = [s["new_event_rate_nmse"] for s in first]
            rows.append(
                {
                    "truth": family,
                    "noise": noise,
                    "attempted": len(lists),
                    "returned": len(first),
                    "menus_ruled_out": sum(
                        c["status"] == "menu_contradicted" for c in cases_here
                    ),
                    "top1_recovered": sum(s["recovers"] for s in first),
                    "top2_recovered": sum(
                        any(s["recovers"] for s in pair) for pair in lists
                    ),
                    "median_new_event_rate_nmse": float(np.median(errors))
                    if errors
                    else None,
                }
            )
    return rows


def run(root: Path) -> dict:
    """Run the controls and hold out each basin training event exactly once."""
    rows = _training_rows()
    with _lock(root):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "complete-scalar-law-nomination-1",
                "basin": BASIN,
                "train_file_sha256": file_hash(PUBLIC / "train.json"),
                "source_hashes": {
                    p.name: file_hash(p)
                    for p in (
                        Path(__file__),
                        Path(basin_observed_process.__file__),
                        Path(observed_process_laws.__file__),
                    )
                },
                "versions": {key: version(key) for key in ("numpy", "scipy")},
                "basin_events": [r.trajectory_id for r in rows],
                "synthetic_families": list(TRUTH),
                "noise_fractions": list(NOISE),
                "seeds": SEEDS,
                "power_recovery_tolerance": POWER_TOLERANCE,
                "uses_benchmark_validation_or_test": False,
            },
        )
        key = plan["artifact_sha256"]
        cases = []
        for family in TRUTH:
            for noise in NOISE:
                for seed in range(SEEDS):
                    name = f"{family}-{noise}-{seed}"
                    cases.append(
                        _checkpoint(
                            root / "synthetic" / f"{name}.json",
                            key,
                            name,
                            lambda f=family, n=noise, s=seed: synthetic_case(f, n, s),
                        )
                    )
        full = _checkpoint(
            root / "basin" / "all.json",
            key,
            "all",
            lambda: basin_observed_process.fit_basin_laws(rows),
        )
        folds = []
        for row in rows:
            name = row.trajectory_id
            training = [r for r in rows if r.trajectory_id != name]
            folds.append(
                _checkpoint(
                    root / "basin" / f"{name}.json",
                    key,
                    name,
                    lambda t=training, h=row: basin_observed_process.run_fold(t, h),
                )
            )
        return sealed_write(
            root / "summary.json",
            {
                "plan": key,
                "synthetic": synthetic_summary(cases),
                "basin_training": full,
                "basin_folds": folds,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.root)
    print(
        json.dumps(
            {
                "summary": str(args.root / "summary.json"),
                "folds": len(result["basin_folds"]),
                "synthetic_controls": sum(r["attempted"] for r in result["synthetic"]),
            }
        )
    )


if __name__ == "__main__":
    main()
