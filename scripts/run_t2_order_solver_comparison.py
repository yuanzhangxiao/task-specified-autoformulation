"""Compare two treatments of the initial disposal constraint on one frozen T2 model."""

from __future__ import annotations

import argparse
import statistics
from importlib.metadata import version
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import fit_t2_initial_disposal_order as original
from scripts.construction_functions import file_hash

ROOT = Path("output/t2-order-solver-comparison-20261124")
ARCHIVED = "6e0e84e3c19fc54d9b548993f2afe6689a6d2402088ea035358a8f2887d0458e"
STARTS = ("warm", "displaced")
FRACTIONS = {
    "unrestricted": None,
    "one_percent": 0.01,
    "reference": 0.5,
    "feasible": 0.5,
}
HISTORICAL = {
    "unrestricted": "control",
    "one_percent": "small_gap",
    "reference": "half_gap",
}
BUDGET = 80
REPEATS = 3
EQUIVALENCE = 1e-6


def _inputs():
    """Only the existing full-training model and the registered training rows enter."""
    frozen = sealed_read(original.SOURCE)
    if (
        frozen["artifact_sha256"] != original.FREEZE
        or frozen["validation_opened_at_freeze"]
    ):
        raise ValueError("Incorrect training model intake")
    train_file = original.coupled.INPUTS / "train.json"
    rows = original.coupled.data.load_training(train_file)
    if [r["trajectory_id"] for r in rows] != [f"train_{i:03d}" for i in range(16)]:
        raise ValueError("Expected the complete sixteen-event fitting set")
    return frozen["models"]["delayed_refit"], rows


def _archived():
    summary = sealed_read(original.OUT / "summary.json")
    if (
        summary["artifact_sha256"] != ARCHIVED
        or summary["validation_opened"]
        or summary["test_opened"]
    ):
        raise ValueError("Unexpected historical reference or split")
    return summary


def _schedule():
    """Reverse primary arm order on alternate repeats; time the two controls once."""
    primary = [(a, s) for s in STARTS for a in ("reference", "feasible")]
    for i in range(REPEATS):
        for arm, start in primary[:: (-1 if i == 1 else 1)]:
            yield i, arm, start
    for arm in ("unrestricted", "one_percent"):
        for start in STARTS:
            yield 0, arm, start


def run_fits(root: Path) -> None:
    from scripts import t2_order_constraint_solvers as solver

    model, rows = _inputs()
    source = [
        Path(__file__),
        Path(solver.__file__),
        Path(original.__file__),
        Path(original.base.__file__),
        Path(original.coupled.orion.__file__),
        Path(original.coupled.orion.kernel.__file__),
    ]
    plan = sealed_write(
        root / "fit-plan.json",
        {
            "protocol": "fixed-delayed-disposal-solvers-1",
            "source": original.FREEZE,
            "model": content_hash(model),
            "initial_parameters": model["formula"]["parameters"],
            "training": content_hash(rows),
            "fit_ids": [r["trajectory_id"] for r in rows],
            "code": {p.name: file_hash(p) for p in source},
            "packages": {k: version(k) for k in ("numpy", "scipy", "casadi")},
            "variants": FRACTIONS,
            "timing_order": list(map(list, _schedule())),
            "budget": BUDGET,
            "within_start_equivalence_pooled_nmse": EQUIVALENCE,
            "minute15_used_in_fit": False,
            "official_validation_opened": False,
            "test_opened": False,
        },
    )
    for repeat, arm, start in _schedule():
        path = root / "fits" / arm / start / f"{repeat}.json"
        key = [plan["artifact_sha256"], repeat, arm, start]
        if path.exists():
            if sealed_read(path)["key"] != key:
                raise ValueError("Fit receipt belongs to another run")
            continue
        method = "feasible" if arm == "feasible" else "reference"
        result = solver.fit(
            model, rows, method, start, budget=BUDGET, fraction=FRACTIONS[arm]
        )
        sealed_write(path, {"key": key, "result": result})
        print(arm, start, repeat, result["training_nmse"], flush=True)


def _timing(fits):
    if len({content_hash(f["parameters"]) for f in fits}) != 1:
        raise ValueError("Timing replicates returned different parameters")
    counts = (
        "full_residual_batches",
        "full_jacobian_batches",
        "infeasible_residual_batches",
        "infeasible_jacobian_batches",
        "infeasible_setup_jacobians",
        "full_residual_trajectories",
        "full_jacobian_trajectories",
    )
    timing = {"repeats": len(fits)}
    for k in counts:
        numbers = {f["cost"][k] for f in fits}
        if len(numbers) != 1:
            raise ValueError(f"Nonreproducible simulation count for {k}")
        timing[k] = numbers.pop()
    for k in (
        "elapsed_seconds",
        "setup_seconds",
        "projection_seconds",
        "solver_seconds",
    ):
        values = [f["cost"][k] for f in fits]
        timing[k] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return timing


def _replay_changes(old: dict) -> dict:
    """Localize the verified historical half-gap tradeoff against its control."""
    replays = []
    for arm in ("control", "half_gap"):
        fit = old["arms"][arm]
        key = fit["replay"]
        record = sealed_read(original.OUT / "replays" / "train" / f"{key}.json")
        if record["key"] != key:
            raise ValueError("Wrong training replay")
        replays.append(record["result"]["events"])
    changes = []
    for a, b in zip(*replays, strict=True):
        if (a["event"], a["time"], a["observed"]) != (
            b["event"],
            b["time"],
            b["observed"],
        ):
            raise ValueError("Historical training replays are not aligned")
        metrics = {}
        for k in ("Gp", "U", "I"):
            y = np.array(a["observed"][k])
            ca, cb = np.array(a["predicted"][k]), np.array(b["predicted"][k])
            j = int(np.argmax(abs(cb - y) - abs(ca - y)))
            metrics[k] = {
                "delta_nmse": b["per_target_nmse"][k] - a["per_target_nmse"][k],
                "time_of_largest_absolute_error_increase": a["time"][j],
                "worst_increase": float(abs(cb[j] - y[j]) - abs(ca[j] - y[j])),
                "control": float(ca[j]),
                "half_gap": float(cb[j]),
                "observed": float(y[j]),
            }
        changes.append({"event": a["event"], "targets": metrics})
    return {
        "reference_endpoints": [
            old["arms"][k]["replay"] for k in ("control", "half_gap")
        ],
        "events": changes,
    }


def report(root: Path) -> dict:
    """Independently check every distinct endpoint after all fixed fits are recorded."""
    from scripts import t2_order_checkpoint_diagnostics as diagnostic

    model, rows = _inputs()
    archived = _archived()
    plan = sealed_read(root / "fit-plan.json")
    if plan["model"] != content_hash(model) or plan["training"] != content_hash(rows):
        raise ValueError("Different model or training data")
    receipts = {}
    for repeat, arm, start in _schedule():
        doc = sealed_read(root / "fits" / arm / start / f"{repeat}.json")
        if doc["key"] != [plan["artifact_sha256"], repeat, arm, start]:
            raise ValueError("Mismatched fit receipt")
        receipts.setdefault((arm, start), []).append(doc["result"])
    audit = sealed_write(
        root / "check-plan.json",
        {
            "fitting_plan": plan["artifact_sha256"],
            "report_code": file_hash(Path(__file__)),
            "diagnostic_code": file_hash(Path(diagnostic.__file__)),
            "minute": 15,
            "later_gap_tolerance_mg_per_kg_per_min": 0.01,
            "diagnostic_fraction": 0.5,
            "use_later_time_for_selection": False,
            "historical": ARCHIVED,
        },
    )
    table = []
    models = {}
    archive_bank = {
        f"{arm}-{start}": original.coupled.insulin.with_parameters(
            model, archived["attempts"][arm][start]["parameters"]
        )
        for arm in HISTORICAL.values()
        for start in STARTS
    }
    screen_path = root / "fixed-vector-screen.json"
    screen_key = [audit["artifact_sha256"], content_hash(archive_bank)]
    if not screen_path.exists():
        sealed_write(
            screen_path,
            {
                "key": screen_key,
                "result": diagnostic.screen_bank(archive_bank, rows),
            },
        )
    screen = sealed_read(screen_path)
    if screen["key"] != screen_key:
        raise ValueError("Wrong checkpoint for the fixed-vector screening comparison")
    for (arm, start), fits in receipts.items():
        f = fits[0]
        result = {
            "arm": arm,
            "start": start,
            "timing": _timing(fits),
            "solver_success": f["solver_success"],
            "termination": f["termination"],
            "training_nmse": f["training_nmse"],
            "qualified": False,
        }
        table.append(result)
        if f["parameters"] is None:
            continue
        fitted = original.coupled.insulin.with_parameters(model, f["parameters"])
        model_id = content_hash(fitted)
        models[model_id] = fitted
        path = root / "diagnostics" / f"{model_id}.json"
        key = [audit["artifact_sha256"], model_id]
        if not path.exists():
            d = diagnostic.diagnose(fitted, rows, rho=0.5)
            sealed_write(path, {"key": key, "diagnostics": d})
        record = sealed_read(path)
        if record["key"] != key:
            raise ValueError("Wrong diagnostic receipt")
        d = record["diagnostics"]
        result.update(
            {
                "model": model_id,
                "diagnostic": record["artifact_sha256"],
                "pooled_nmse": d["pooled_nmse"],
                "per_target_nmse": d["per_target_nmse"],
            }
        )
        if abs(f["training_nmse"] - d["pooled_nmse"]) > 1e-7:
            raise ValueError("Fitting and independent training replays disagree")
        if arm in HISTORICAL:
            prior = archived["attempts"][HISTORICAL[arm]][start]
            result["difference_from_archive"] = d["pooled_nmse"] - prior["pooled"]
            if abs(result["difference_from_archive"]) > 1e-7:
                raise ValueError("Historical training reference failed reproduction")
        rho = FRACTIONS[arm]
        pairs = d["initial"]["pair_details"]
        result["worst_enforced_margin"] = (
            None
            if rho is None
            else min(p["predicted_gap"] - rho * p["observed_gap"] for p in pairs)
        )
        margin = result["worst_enforced_margin"]
        result["qualified"] = d["domain"]["eligible"] and (
            margin is None or margin >= -1e-9
        )
        result["check_cost"] = d["cost"]
        for label in ("initial", "minute15", "conditional_minute15"):
            result[label] = {k: v for k, v in d[label].items() if k != "pair_details"}
    sealed_write(
        root / "fitted-models.json", {"plan": plan["artifact_sha256"], "models": models}
    )
    return sealed_write(
        root / "summary.json",
        {
            "plan": plan["artifact_sha256"],
            "audit": audit["artifact_sha256"],
            "rows": table,
            "tradeoff": _replay_changes(archived),
            "fixed_vector_screen": screen["artifact_sha256"],
            "scope": "Sixteen training events, delayed balance, fixed insulin law.",
            "official_validation_opened": False,
            "test_opened": False,
        },
    )


def run(root: Path):
    with _lock(root):
        run_fits(root)
        return report(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
    print(args.root / "summary.json")
