"""Training-event comparison of four prespecified reduced insulin equations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts.construction_functions import file_hash
from scripts.t2_challenge_folds import folds

DRUG = "insulin_pmol_per_kg_min"
MEAL = "meal_rate_g_per_min"
ARMS = ("reference", "meal", "EGP", "adaptation")
GRID = (5, 15, 45, 135)
SOURCE = "2721655575b70a6fc07f9b6244d13e995405ba595d5123c64d1e4b9986ff9e06"
TRAIN = "fb99154de838132b45701dda20e35a4a2ad29a4f711c34ba8eed7b5edbe886e7"
REPLAY_STEP = 0.125
ATOL, RTOL = 1e-6, 1e-7


def source(paired):
    """Read authenticated public training rows and declared interpolation semantics."""
    plan = sealed_read(paired / "generation" / "plan.json")
    intake = sealed_read(paired / "generation" / "intake.json")
    if (
        plan["artifact_sha256"] != SOURCE
        or intake["artifact_sha256"] != plan["intake_sha256"]
        or content_hash(intake["train"]) != TRAIN
    ):
        raise ValueError("Wrong paired training source")
    for case in intake["cases"].values():
        contract = case["evidence"]["contract"]
        if (
            case["evidence"]["split"] != "train"
            or contract["initial_rules"]["I"]["mode"] != "observed_initial"
            or any(
                contract["interpolation"].get(k) != "linear_complete_knots"
                for k in ("Gt", DRUG, MEAL, "EGP")
            )
        ):
            raise ValueError("Missing measured initial or piecewise linear forcing")
    return {r["trajectory_id"]: r for r in intake["train"]}


def prepare(paired, root):
    """Seal selection, numerical settings, outcomes and exact folds before scoring."""
    rows = source(paired)
    if sorted(rows) != [f"train_{k:03}" for k in range(16)]:
        raise ValueError("Missing an expected training event")
    splits = folds(rows.values())
    design = {
        "protocol": "t2-insulin-order-repair-fitting-1",
        "source_plan": SOURCE,
        "training_sha256": TRAIN,
        "code_sha256": file_hash(Path(__file__)),
        "folds": splits,
        "timescales_min": list(GRID),
        "arms": list(ARMS),
        "equations": {
            "reference": "I' = -I/tau + offset + b*Gt + c*u_ins",
            "meal": "reference + d*meal_rate",
            "EGP": "reference + d*EGP",
            "adaptation": "reference + d*(Gt-Z), Z'=(Gt-Z)/tauZ, Z(0)=Gt(0)",
        },
        "fit": (
            "Observed I(0). Shared unrestricted gains. Minimize total insulin "
            "SSE at all sampled fitting-event times by column-scaled SVD least "
            "squares at each fixed positive timescale. An absent input column "
            "has zero estimated gain. Break exact ties by smaller timescales."
        ),
        "checks": {
            "integrator": "piecewise linear input matrix exponential",
            "independent_replay": "RK4 split exactly at every supplied knot",
            "maximum_step_min": REPLAY_STEP,
            "atol": ATOL,
            "rtol": RTOL,
        },
        "metrics": (
            "Pooled held SSE and RMSE and paired per-event SSE wins versus "
            "reference, strict comparison; count min predicted I, negative "
            "samples and worst absolute I error at original sample times. "
            "Also fit each family on the full training cohort."
        ),
        "scope": (
            "Training-only exploratory subsystem. Equations were chosen after "
            "screening this same cohort. Gt and EGP are supplied observations; "
            "EGP is a diagnostic proxy. No inference of causal mechanisms."
        ),
        "llm_calls": 0,
        "validation_opened": False,
        "test_opened": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(root / "plan.json", design), rows


def drivers(row, arm):
    """The prediction function takes supplied histories and a single observed I(0)."""
    t = np.asarray(row["time"], dtype=float)
    cols = [np.ones(len(t)), row["auxiliaries"]["Gt"], row["inputs"][DRUG]]
    if arm == "meal":
        cols.append(row["inputs"][MEAL])
    elif arm == "EGP":
        cols.append(row["auxiliaries"]["EGP"])
    elif arm not in ("reference", "adaptation"):
        raise ValueError("Unknown family")
    x = np.asarray(cols, dtype=float).T
    i0 = float(row["targets"]["I"][0])
    if (
        len(t) < 2
        or np.any(np.diff(t) <= 0)
        or not np.isfinite(t).all()
        or not np.isfinite(x).all()
        or not np.isfinite(i0)
    ):
        raise ValueError("Nonfinite input, measured initial or non-increasing time")
    return t, x, i0


@lru_cache(maxsize=256)
def kernel(dt, tau, tau_z):
    """Exact first-order hold using a constant input derivative within each interval."""
    a = np.array([[-1 / tau, 1, 0], [0, 0, 1], [0, 0, 0]])
    k = expm(dt * a)[0]
    z = None
    if tau_z is not None:
        # States W, Z, G, G'. W is the unit gain response of G-Z in the insulin ODE.
        b = np.array(
            [
                [-1 / tau, -1, 1, 0],
                [0, -1 / tau_z, 1 / tau_z, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )
        z = expm(dt * b)[:2]
    return k, z


def design(row, arm, tau, tau_z=None):
    """Return fixed-initial contribution and exact responses to each unit gain."""
    if (
        tau <= 0
        or ((tau_z is not None) != (arm == "adaptation"))
        or (tau_z is not None and tau_z <= 0)
    ):
        raise ValueError("Invalid timescales for this family")
    t, x, i0 = drivers(row, arm)
    adapt = arm == "adaptation"
    basis = np.zeros((len(t), x.shape[1] + adapt))
    w, z = 0.0, x[0, 1]
    for j, dt in enumerate(np.diff(t)):
        k, m = kernel(dt, tau, tau_z)
        slope = (x[j + 1] - x[j]) / dt
        basis[j + 1, : x.shape[1]] = (
            k[0] * basis[j, : x.shape[1]] + k[1] * x[j] + k[2] * slope
        )
        if adapt:
            w, z = m @ [w, z, x[j, 1], slope[1]]
            basis[j + 1, -1] = w
    return i0 * np.exp(-(t - t[0]) / tau), basis


def fit(training, arm, grid=GRID):
    """Profile the shared linear gains using only events in this fitting split."""
    best, profiles = None, []
    for tau, tau_z in itertools.product(grid, grid if arm == "adaptation" else [None]):
        systems = [design(r, arm, tau, tau_z) for r in training]
        mat = np.vstack([d[1] for d in systems])
        rhs = np.concatenate(
            [
                np.asarray(r["targets"]["I"]) - d[0]
                for r, d in zip(training, systems, strict=True)
            ]
        )
        scale = np.linalg.norm(mat, axis=0)
        scaled = mat / np.where(scale > 0, scale, 1)
        solution, _, rank, _ = np.linalg.lstsq(scaled, rhs, rcond=None)
        coef = np.divide(solution, scale, out=np.zeros_like(solution), where=scale > 0)
        residual = mat @ coef - rhs
        opt = {
            "tau": tau,
            "tau_z": tau_z,
            "gains": coef.tolist(),
            "rank": int(rank),
            "parameters": len(coef),
            "sse": float(residual @ residual),
        }
        if not math.isfinite(opt["sse"]):
            raise ValueError("Non-finite least-squares result")
        profiles.append(opt)
        if best is None or opt["sse"] < best["sse"]:
            best = opt
    return best, profiles


def independent(row, arm, selected):
    """Integrate the original ODE using scalar RK4 and knotwise interpolation."""
    t, x, i = drivers(row, arm)
    gain = selected["gains"]
    tau, tz = selected["tau"], selected["tau_z"]
    adapt = arm == "adaptation"
    z = x[0, 1]
    measured_forcing = x @ np.array(gain[: x.shape[1]])

    def field(ii, zz, f, g):
        dev = g - zz
        return (
            f - ii / tau + (gain[-1] * dev if adapt else 0),
            dev / tz if adapt else 0,
        )

    out = [i]
    for j, dt in enumerate(np.diff(t)):
        n = max(1, math.ceil(dt / REPLAY_STEP))
        h = dt / n
        f0, g0 = measured_forcing[j], x[j, 1]
        df = (measured_forcing[j + 1] - f0) / n
        dg = (x[j + 1, 1] - g0) / n
        for _ in range(n):
            i1, z1 = field(i, z, f0, g0)
            i2, z2 = field(i + h * i1 / 2, z + h * z1 / 2, f0 + df / 2, g0 + dg / 2)
            i3, z3 = field(i + h * i2 / 2, z + h * z2 / 2, f0 + df / 2, g0 + dg / 2)
            i4, z4 = field(i + h * i3, z + h * z3, f0 + df, g0 + dg)
            i += h * (i1 + 2 * i2 + 2 * i3 + i4) / 6
            z += h * (z1 + 2 * z2 + 2 * z3 + z4) / 6
            f0 += df
            g0 += dg
        out.append(i)
    return np.asarray(out)


def score(row, arm, selected):
    off, mat = design(row, arm, selected["tau"], selected["tau_z"])
    pred = off + mat @ selected["gains"]
    replay = independent(row, arm, selected)
    verified = bool(np.allclose(replay, pred, atol=ATOL, rtol=RTOL))
    err = pred - row["targets"]["I"]
    return {
        "event": row["trajectory_id"],
        "verified": verified,
        "n": len(pred),
        "sse": float(err @ err),
        "min_I": float(min(pred)),
        "negative_samples": int(sum(pred < 0)),
        "max_error": float(max(abs(err))),
        "max_replay_difference": float(max(abs(pred - replay))),
    }


def combine(rows):
    """All required events must numerically verify before they contribute a score."""
    if not rows or not all(r["verified"] for r in rows):
        return {"status": "unverified"}
    sse = sum(r["sse"] for r in rows)
    n = sum(r["n"] for r in rows)
    return {
        "status": "verified",
        "n": n,
        "sse": sse,
        "rmse": math.sqrt(sse / n),
        "min_I": min(r["min_I"] for r in rows),
        "negative_samples": sum(r["negative_samples"] for r in rows),
        "max_error": max(r["max_error"] for r in rows),
        "within_one": all(r["max_error"] <= 1 for r in rows),
        "max_replay_difference": max(r["max_replay_difference"] for r in rows),
    }


def experiment(root, rows, plan):
    splits = dict(enumerate(plan["folds"], 1))
    splits["full"] = {"fit": sorted(rows), "held": []}
    results = {}
    for name, s in splits.items():
        results[str(name)] = {}
        training = [rows[n] for n in s["fit"]]
        for arm in ARMS:
            path = root / f"{name}-{arm}.json"
            if path.exists():
                r = sealed_read(path)
                if (r["arm"], r["split"], r["plan"]) != (
                    arm,
                    str(name),
                    plan["artifact_sha256"],
                ):
                    raise ValueError("Checkpoint identity differs")
            else:
                best, profiles = fit(training, arm)
                r = sealed_write(
                    path,
                    {
                        "plan": plan["artifact_sha256"],
                        "arm": arm,
                        "split": str(name),
                        "selected": best,
                        "profiles": profiles,
                        "fit": [score(row, arm, best) for row in training],
                        "held": [score(rows[n], arm, best) for n in s["held"]],
                    },
                )
            results[str(name)][arm] = r
    all_held = {
        arm: [row for k in (1, 2, 3) for row in results[str(k)][arm]["held"]]
        for arm in ARMS
    }
    summary = {}
    for arm in ARMS:
        compared = list(zip(all_held[arm], all_held["reference"], strict=True))
        if any(a["event"] != b["event"] for a, b in compared):
            raise ValueError("Mismatched event order in comparator")
        summary[arm] = {
            "held": combine(all_held[arm]),
            "verified_comparisons": sum(
                a["verified"] and b["verified"] for a, b in compared
            ),
            "event_wins": sum(
                a["verified"] and b["verified"] and a["sse"] < b["sse"]
                for a, b in compared
            ),
            "full_training": combine(results["full"][arm]["fit"]),
            "folds": {str(k): combine(results[str(k)][arm]["held"]) for k in (1, 2, 3)},
            "fits": {k: r[arm]["artifact_sha256"] for k, r in results.items()},
        }
    return sealed_write(
        root / "summary.json", {"plan": plan["artifact_sha256"], "arms": summary}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.root.resolve().is_relative_to(args.paired.resolve()):
        raise ValueError("Keep result files outside the source archive")
    args.root.mkdir(parents=True, exist_ok=True)
    with _lock(args.root):
        plan, data = prepare(args.paired, args.root)
        if args.action == "prepare":
            result = {"plan": plan["artifact_sha256"]}
        else:
            result = experiment(args.root, data, plan)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
