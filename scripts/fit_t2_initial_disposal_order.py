"""Continue Orion's delayed-model training fit with an initial-disposal order test."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initial_algebraic_order as initial_order
from autoformalism.staged_topology import content_hash
from scripts import fit_t2_disposal_constrained as base
from scripts import run_t2_disposal_coupled as coupled
from scripts.construction_functions import file_hash

SOURCE = coupled.OUT / "training-freeze.json"
FREEZE = "bbbbd572f688479dd08b6f0daa3a8d477b101e0805a16632f35e7dd29e2796b6"
OUT = Path("output/t2-initial-order-20261123/refit-matched-v2")
FRACTIONS = {"control": None, "small_gap": 0.01, "half_gap": 0.5}


def order_pairs(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the higher and lower event in every unequal measured initial pair."""
    if observed.ndim != 1 or not np.isfinite(observed).all():
        raise ValueError("Expected finite scalar initial observations")
    return np.where(observed[:, None] > observed)


def fit(model: dict, rows: list[dict], arm: str, budget: int, start: str) -> dict:
    """Apply the original slot-015 qualification before constructing the fit."""
    return optimize(model, base._Training(model, rows), arm, budget, start)


def optimize(
    model: dict, data: base._Training, arm: str, budget: int, start: str
) -> dict:
    """Fit a qualified training problem with affine initial-U differences.

    The caller verifies this property for the graph and active coordinates.
    Each candidate's actual initial differences govern retention.
    """
    if arm not in FRACTIONS or budget < 1 or start not in coupled.orion.STARTS:
        raise ValueError("Invalid comparison arm, start or budget")
    fraction = FRACTIONS[arm]
    ix = data.active
    p0 = data.reference
    jac0 = data.engine.jacobian(p0).reshape(-1, 3, len(p0))[:, :, ix]
    rms = np.sqrt(np.mean(jac0[data.samples] ** 2, axis=(0, 1)))
    scale = np.minimum(1 / np.maximum(rms, 1e-10), 100 * np.maximum(abs(p0[ix]), 1))
    lo = (data.lo[ix] - p0[ix]) / scale
    first = data.edges[:-1]
    obs = data.observed[first, 2]
    high, low = order_pairs(obs)
    required = (fraction or 0.0) * (obs[high] - obs[low])
    best_score, best = np.inf, None
    calls, derivative_calls = 0, 1
    zero = np.zeros_like(scale)

    def expand(x):
        vector = p0.copy()
        vector[ix] = np.maximum(p0[ix] + scale * x, data.lo[ix])
        return vector

    @lru_cache(maxsize=2)
    def evaluate(key):
        nonlocal best_score, best, calls
        if calls >= budget:
            raise StopIteration("Forward evaluation budget reached")
        calls += 1
        vector = expand(np.array(key))
        error = data.engine.residual(vector).reshape(-1, 3)
        margins, minima = data.margins(error)
        init = obs + error[first, 2] * data.scales[2]
        order_margin = init[high] - init[low] - required
        objective = float(np.mean(error[data.samples] ** 2))
        if (
            min(margins) >= -base.DOMAIN_TOL
            and (arm == "control" or np.all(order_margin >= -base.DOMAIN_TOL))
            and objective < best_score
        ):
            best_score, best = objective, vector
        return error, margins, minima, objective, order_margin

    @lru_cache(maxsize=2)
    def jac(key):
        nonlocal derivative_calls
        if key == tuple(zero):
            return jac0 * scale
        derivative_calls += 1
        whole = data.engine.jacobian(expand(np.array(key))).reshape(-1, 3, len(p0))
        return whole[:, :, ix] * scale

    def gradient(x):
        errors = evaluate(tuple(x))[0][data.samples]
        return (
            2
            * np.einsum("ti,tij->j", errors, jac(tuple(x))[data.samples])
            / errors.size
        )

    def domain_jac(x):
        indices = evaluate(tuple(x))[2]
        return jac(tuple(x))[indices, [0, 2], :].reshape(-1, len(ix))

    initial = evaluate(tuple(zero))
    constraints = [
        {"type": "ineq", "fun": lambda x: evaluate(tuple(x))[1], "jac": domain_jac}
    ]
    if fraction is not None and len(high):
        # In this pinned graph only afcoef13 multiplies the linear initializer.
        # That coefficient is fixed, making all initial-U differences affine in x.
        slope = jac(tuple(zero))[first, 2] * data.scales[2]
        matrix = slope[high] - slope[low]
        unique = np.unique(np.column_stack([matrix, -initial[4]]), axis=0)
        constraints.append(LinearConstraint(unique[:, :-1], unique[:, -1], np.inf))

    pstart = coupled.orion.optimizer_start(
        model, model["formula"]["parameters"], "fixed_insulin", start
    )
    x0 = np.maximum(
        [(pstart[data.names[i]] - p0[i]) / s for i, s in zip(ix, scale, strict=True)],
        lo,
    )
    try:
        solved = minimize(
            lambda x: evaluate(tuple(x))[3],
            x0,
            jac=gradient,
            bounds=Bounds(lo, np.inf),
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": budget, "ftol": 1e-11},
        )
        success, message = bool(solved.success), str(solved.message)
    except (StopIteration, ArithmeticError) as exc:
        success, message = False, str(exc)
    return {
        "parameters": None
        if best is None
        else dict(zip(data.names, best.tolist(), strict=True)),
        "training_nmse": best_score if best is not None else None,
        "source_nmse": initial[3],
        "forward_evaluations": calls,
        "jacobian_evaluations": derivative_calls,
        "solver_success": success,
        "termination": message,
    }


def qualify(
    model: dict, rows: list[dict], arm: str, record: dict, root: Path, key: str
) -> dict:
    """Check original equations, training order and dense trajectory domains."""
    if record["parameters"] is None:
        return record | {"eligible": False}
    changed = coupled.insulin.with_parameters(model, record["parameters"])
    result = coupled.cached_replay(root, key, "train", changed, rows)
    if result["summary"]["failed_events"]:
        return record | {"eligible": False, "reason": "Incomplete independent replay"}
    order = initial_order.check_initial_order(result["events"], "U", tolerance=0.0)
    observed = np.array([e["observed"]["U"][0] for e in result["events"]])
    predicted = np.array([e["predicted"]["U"][0] for e in result["events"]])
    high, low = order_pairs(observed)
    gap, difference = observed[high] - observed[low], predicted[high] - predicted[low]
    fraction = FRACTIONS[arm]
    margin = difference - (fraction or 0.0) * gap
    if order["comparable_pairs"] != len(high) or order["reversed_pairs"] != np.sum(
        difference < 0
    ):
        raise ValueError("Independent pair-order checks disagree")
    eligible = result["domain"]["eligible"] and (
        fraction is None or np.all(margin >= -base.DOMAIN_TOL)
    )
    if not np.isclose(
        result["summary"]["pooled_nmse"], record["training_nmse"], atol=1e-7, rtol=1e-7
    ):
        raise ValueError("Fitter and independent replay disagree")
    return record | {
        "eligible": bool(eligible),
        "order": order
        | {
            "minimum_observed_gap_fraction": float(min(difference / gap))
            if len(high)
            else None,
            "minimum_margin": float(min(margin)) if len(high) else None,
        },
        "domain": result["domain"],
        "pooled": result["summary"]["pooled_nmse"],
        "per_target": result["summary"]["available_per_target_nmse"],
        "observed_initials": observed.tolist(),
        "predicted_initials": predicted.tolist(),
        "event_ids": [e["event"] for e in result["events"]],
        "replay": content_hash([key, "train", changed]),
    }


def run(root: Path, budget: int = 80) -> dict:
    """Freeze the train-only comparison, checkpoint each start, and verify the fits."""
    source = sealed_read(SOURCE)
    if source["artifact_sha256"] != FREEZE or source["validation_opened_at_freeze"]:
        raise ValueError("Expected Orion's frozen training result")
    model = source["models"]["delayed_refit"]
    training = coupled.INPUTS / "train.json"
    rows = coupled.data.load_training(training)
    code = [
        Path(__file__),
        *(
            Path(m.__file__)
            for m in (
                base,
                coupled,
                coupled.orion,
                coupled.orion.kernel,
                coupled.data,
                coupled.insulin,
                initial_order,
            )
        ),
    ]
    with _lock(root / "lock"):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "t2-initial-disposal-order-fit-2",
                "source": FREEZE,
                "model": content_hash(model),
                "training": file_hash(training),
                "code": {p.name: file_hash(p) for p in code},
                "packages": {k: version(k) for k in ("numpy", "scipy", "casadi")},
                "required_fractions_of_each_nonzero_observed_gap": FRACTIONS,
                "budget_per_arm_per_start": budget,
                "starts": list(coupled.orion.STARTS),
                "selection": "Lowest verified training NMSE, first start within 1e-10",
                "reference": "Same archived delayed refit for all arms",
                "scope": "Exploratory constraints on exact public training initials",
            },
        )
        identity = plan["artifact_sha256"]
        arms, attempts = {}, {}
        for arm in FRACTIONS:
            attempts[arm] = {}
            for start in coupled.orion.STARTS:
                path = root / "fits" / arm / f"{start}.json"
                if not path.exists():
                    seeded = fit(model, rows, arm, budget, start)
                    sealed_write(
                        path, {"plan": identity, "arm": arm, "start": start, **seeded}
                    )
                record = sealed_read(path)
                if (record["plan"], record["arm"], record["start"]) != (
                    identity,
                    arm,
                    start,
                ):
                    raise ValueError(
                        "Fit checkpoint differs from the registered comparison"
                    )
                attempts[arm][start] = qualify(model, rows, arm, record, root, identity)
            eligible = [r for r in attempts[arm].values() if r["eligible"]]
            arms[arm] = {"eligible": False}
            if eligible:
                lowest = min(r["pooled"] for r in eligible)
                arms[arm] = next(r for r in eligible if r["pooled"] <= lowest + 1e-10)
        return sealed_write(
            root / "summary.json",
            {
                "plan": identity,
                "arms": arms,
                "attempts": attempts,
                "validation_opened": False,
                "test_opened": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--budget", type=int, default=80)
    args = parser.parse_args()
    outcome = run(args.root, args.budget)
    print(
        json.dumps(
            {
                k: {
                    label: a.get(label)
                    for label in ("eligible", "pooled", "start", "order")
                }
                for k, a in outcome["arms"].items()
            },
            indent=2,
        )
    )
