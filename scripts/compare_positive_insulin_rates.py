"""Compare signed and nonnegative production on held insulin training events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear, nnls

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import compare_insulin_order_repairs as prior
from scripts.construction_functions import file_hash

CHANNEL_ORDER = ("Gt", prior.DRUG, "EGP")
SOURCES = {"base": CHANNEL_ORDER[:2], "EGP": CHANNEL_ORDER}
WEIGHTS = ("signed", "nonnegative")
REPLAY_STEP = 0.125
ATOL, RTOL = 1e-6, 1e-7


def channel_order(channels):
    """Use the same coefficient order before and after a sorted JSON round-trip."""
    order = tuple(k for k in CHANNEL_ORDER if k in channels)
    if not order or len(order) != len(channels):
        raise ValueError("Unexpected or duplicate production input")
    return order


def coefficient_hats(knots):
    return [
        {"channel": name, "hat_index": i, "knot": float(value)}
        for name in channel_order(knots)
        for i, value in enumerate(knots[name])
    ]


def checked_weights(knots, model):
    hats = coefficient_hats(knots)
    w = np.asarray(model["weights"], dtype=float)
    if w.shape != (len(hats),) or not np.isfinite(w).all():
        raise ValueError("Expected one finite coefficient per input hat")
    if "coefficient_hats" in model and model["coefficient_hats"] != hats:
        raise ValueError("Saved coefficient hats differ from the declared input knots")
    return w


def inputs(row, channels):
    """Read permitted forcing, times and initial insulin only."""
    t = np.asarray(row["time"], dtype=float)
    x = np.stack(
        [
            (row["inputs"] if name == prior.DRUG else row["auxiliaries"])[name]
            for name in channels
        ],
        axis=1,
    )
    i0 = float(row["targets"]["I"][0])
    if (
        not row["trajectory_id"].startswith("train_")
        or len(t) < 2
        or x.shape != (len(t), len(channels))
        or not np.all(np.diff(t) > 0)
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(x))
        or not np.isfinite(i0)
    ):
        raise ValueError(
            "Expected a complete finite training event and its recorded initial"
        )
    return t, x, i0


def calibrate(rows, channels):
    """Freeze distinct min, median and max from the fitting subset for each channel."""
    channels = channel_order(channels)
    x = np.vstack([inputs(r, channels)[1] for r in rows])
    return {
        k: np.unique(np.quantile(x[:, i], [0, 0.5, 1])).tolist()
        for i, k in enumerate(channels)
    }


def hats(x, knots):
    """Saturated cardinal hats; a constant training channel supplies one flat basis."""
    knots = np.array(knots, dtype=float)
    if (
        not len(knots)
        or not np.all(np.diff(knots) > 0)
        or not np.all(np.isfinite(knots))
    ):
        raise ValueError("Finite distinct knots must increase")
    return np.column_stack([np.interp(x, knots, col) for col in np.eye(len(knots))])


def grid(row, knots):
    """Split each forcing interval where any supplied signal crosses a basis knot."""
    channels = channel_order(knots)
    t, x, i0 = inputs(row, channels)
    parts = [t]
    for i, name in enumerate(channels):
        ks = knots[name]
        x0, x1 = x[:-1, i], x[1:, i]
        for k in ks:
            cross = ((x0 < k) & (k < x1)) | ((x1 < k) & (k < x0))
            f = (k - x0[cross]) / (x1[cross] - x0[cross])
            parts.append(t[:-1][cross] + f * np.diff(t)[cross])
    tt = np.unique(np.concatenate(parts))
    h = np.column_stack(
        [
            hats(np.interp(tt, t, x[:, i]), knots[name])
            for i, name in enumerate(channels)
        ]
    )
    return t, tt, h, i0


def design(row, knots, tau):
    """Compute exact first-order-hold responses for every basis weight."""
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("A positive finite decay time is required")
    t, tt, h, i0 = grid(row, knots)
    dt = np.diff(tt)
    a = dt / tau
    decay = np.exp(-a)
    total = -tau * np.expm1(-a)
    # Integral of the linearly increasing unit hat over a segment. The series
    # avoids subtracting nearly equal numbers near a knot crossing.
    right = np.empty_like(dt)
    tiny = a < 1e-3
    z = a[tiny]
    right[tiny] = dt[tiny] * (0.5 - z / 6 + z * z / 24 - z**3 / 120 + z**4 / 720)
    right[~tiny] = tau * (1 + np.expm1(-a[~tiny]) / a[~tiny])
    increments = (total - right)[:, None] * h[:-1] + right[:, None] * h[1:]
    m = np.zeros_like(h)
    for j in range(len(dt)):
        m[j + 1] = decay[j] * m[j] + increments[j]
    return i0 * np.exp(-(t - t[0]) / tau), m[np.searchsorted(tt, t)]


def predict(row, knots, model):
    initial, m = design(row, knots, model["tau"])
    return initial + m @ checked_weights(knots, model)


def independent(row, knots, model):
    """Classical RK4 on the raw interpolated inputs, at most 1/8 minute per step."""
    channels = channel_order(knots)
    t, x, i0 = inputs(row, channels)
    t0, parts, _, _ = grid(row, knots)
    counts = np.maximum(1, np.ceil(np.diff(parts) / REPLAY_STEP).astype(int))
    ts = np.r_[
        np.concatenate(
            [
                np.linspace(lo, hi, n, endpoint=False)
                for lo, hi, n in zip(parts[:-1], parts[1:], counts, strict=True)
            ]
        ),
        parts[-1],
    ]
    dt = np.diff(ts)
    mids = ts[:-1] + dt / 2

    weights = checked_weights(knots, model)

    def production(times):
        q = np.zeros(len(times))
        j = 0
        for i, name in enumerate(channels):
            ks = knots[name]
            vals = np.interp(times, t, x[:, i])
            q += np.interp(vals, ks, weights[j : j + len(ks)])
            j += len(ks)
        return q

    left = production(ts)
    middle = production(mids)
    insulin = np.empty_like(ts)
    insulin[0] = i0
    tau = model["tau"]
    for j, h in enumerate(dt):
        y = insulin[j]
        k1 = left[j] - y / tau
        k2 = middle[j] - (y + h * k1 / 2) / tau
        k3 = middle[j] - (y + h * k2 / 2) / tau
        k4 = left[j + 1] - (y + h * k3) / tau
        insulin[j + 1] = y + h * (k1 + 2 * (k2 + k3) + k4) / 6
    return insulin[np.searchsorted(ts, t0)]


def solve(m, y, constraint):
    """Column-scaled fits, independent solver cross-check and scaled KKT residual."""
    norms = np.linalg.norm(m, axis=0)
    div = np.where(norms > 0, norms, 1)
    mat = m / div
    rcond = np.finfo(float).eps * max(mat.shape)
    w, _, rank, sing = lstsq(mat, y, cond=rcond, lapack_driver="gelsd")
    if constraint == "signed":
        alt = lstsq(mat, y, cond=rcond, lapack_driver="gelsy")[0]
    elif constraint == "nonnegative":
        q, r = np.linalg.qr(mat, mode="reduced")
        z = q.T @ y
        bv = lsq_linear(
            r, z, bounds=(0, np.inf), method="bvls", tol=1e-12, max_iter=200
        )
        if not bv.success:
            raise ValueError("BVLS did not converge")
        alt = bv.x
        w, _ = nnls(r, z, maxiter=100 * mat.shape[1])
    else:
        raise ValueError("Unknown coefficient constraint")
    residual = mat @ w - y
    sse = float(residual @ residual)
    alt_r = mat @ alt - y
    gap = abs(float(alt_r @ alt_r) - sse) / max(1, sse)
    grad = mat.T @ residual
    if constraint == "nonnegative":
        if np.any(w < 0):
            raise ValueError("Negative production weight")
        kkt = np.where(w == 0, np.minimum(grad, 0), grad)
    else:
        kkt = grad
    stationarity = float(np.max(abs(kkt))) / max(1, float(np.linalg.norm(y)))
    if not (gap < 1e-8 and stationarity < 1e-8):
        raise ValueError(f"Unverified least-squares optimum: {gap=} {stationarity=}")
    return {
        "weights": (w / div).tolist(),
        "sse": sse,
        "rank": int(rank),
        "columns": len(w),
        "singular_values": sing.tolist(),
        "relative_objective_disagreement": gap,
        "normalized_kkt_residual": stationarity,
    }


def fit(rows, channels, grid_tau=prior.GRID):
    knots = calibrate(rows, channels)
    profiles = {k: [] for k in WEIGHTS}
    for tau in grid_tau:
        systems = [design(r, knots, tau) for r in rows]
        m = np.vstack([v for _, v in systems])
        y = np.concatenate(
            [r["targets"]["I"] - a for r, (a, _) in zip(rows, systems, strict=True)]
        )
        for k in WEIGHTS:
            profiles[k].append(
                {
                    "tau": tau,
                    "coefficient_hats": coefficient_hats(knots),
                    **solve(m, y, k),
                }
            )
    return knots, profiles


def score(row, knots, model):
    pred = predict(row, knots, model)
    check = independent(row, knots, model)
    err = pred - row["targets"]["I"]
    return {
        "event": row["trajectory_id"],
        "forecast_sha256": content_hash(pred.tolist()),
        "independent_forecast_sha256": content_hash(check.tolist()),
        "verified": bool(np.allclose(pred, check, atol=ATOL, rtol=RTOL)),
        "n": len(pred),
        "sse": float(err @ err),
        "min_I": float(min(pred)),
        "negative_samples": int(sum(pred < 0)),
        "max_error": float(max(abs(err))),
        "max_replay_difference": float(max(abs(pred - check))),
    }


def prepare(paired, root):
    rows = prior.source(paired)
    if sorted(rows) != [f"train_{i:03}" for i in range(16)]:
        raise ValueError("Incomplete frozen training cohort")
    folds = prior.folds(rows.values())
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "positive-insulin-production-2",
            "coefficient_channel_order": list(CHANNEL_ORDER),
            "source": prior.SOURCE,
            "train": prior.TRAIN,
            "code": file_hash(Path(__file__)),
            "source_loader": file_hash(Path(prior.__file__)),
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "folds": folds,
            "sources": {k: list(v) for k, v in SOURCES.items()},
            "weights": list(WEIGHTS),
            "tau": list(prior.GRID),
            "basis": (
                "Per source, fit-split min/median/max, deduplicated saturating "
                "cardinal hats. A constant channel contributes a flat unit column."
            ),
            "ode": "I'=-I/tau+sum(weights*input_hats), I(0) observed.",
            "select": "Lowest fitting-event sample SSE. Exact ties choose smaller tau.",
            "numerical": (
                "Split at supplied knots and basis crossings. Exact convolution, "
                "independent RK4 at most 0.125 min, atol=1e-6, rtol=1e-7."
            ),
            "metric": (
                "Pooled insulin RMSE, minimum, negative count, maximum error and "
                "paired SSE wins on held challenges. Full cohort is descriptive."
            ),
            "validation_opened": False,
            "test_opened": False,
            "llm_calls": 0,
        },
    ), rows


def experiment(root, plan, rows):
    results = {}
    splits = {str(i): s for i, s in enumerate(plan["folds"], 1)}
    splits["full"] = {"fit": sorted(rows), "held": []}
    for k, split in splits.items():
        results[k] = {}
        for name, channels in SOURCES.items():
            path = root / f"{k}-{name}.json"
            identity = {"plan": plan["artifact_sha256"], "split": k, "source": name}
            if path.exists():
                out = sealed_read(path)
                if any(out[n] != v for n, v in identity.items()):
                    raise ValueError("Checkpoint belongs to a different experiment")
            else:
                train = [rows[n] for n in split["fit"]]
                knots, profiles = fit(train, channels)
                arms = {}
                for constraint, profile in profiles.items():
                    selected = min(profile, key=lambda p: (p["sse"], p["tau"]))
                    ids = split["fit"] if k == "full" else split["held"]
                    arms[constraint] = {
                        "profile": profile,
                        "selected": selected,
                        "scores": [score(rows[n], knots, selected) for n in ids],
                    }
                out = sealed_write(path, identity | {"knots": knots, "arms": arms})
            results[k][name] = out
    by = {}
    for name in SOURCES:
        for constraint in WEIGHTS:
            rec = {k: r[name]["arms"][constraint] for k, r in results.items()}
            held = [v for k in ("1", "2", "3") for v in rec[k]["scores"]]
            by[f"{name}-{constraint}"] = {
                "held": prior.combine(held),
                "full": prior.combine(rec["full"]["scores"]),
                "events": {r["event"]: r for r in held},
                "folds": {k: prior.combine(rec[k]["scores"]) for k in ("1", "2", "3")},
                "tau": {k: r["selected"]["tau"] for k, r in rec.items()},
                "ranks": {
                    k: [r["selected"]["rank"], r["selected"]["columns"]]
                    for k, r in rec.items()
                },
                "max_solver_disagreement": max(
                    p["relative_objective_disagreement"]
                    for r in rec.values()
                    for p in r["profile"]
                ),
                "max_kkt": max(
                    p["normalized_kkt_residual"]
                    for r in rec.values()
                    for p in r["profile"]
                ),
            }
    for v in by.values():
        v["wins"] = {}
        for other, base in by.items():
            if base["events"].keys() != v["events"].keys():
                raise ValueError("Mismatched challenge cohorts")
            v["wins"][other] = sum(
                a["verified"]
                and base["events"][n]["verified"]
                and a["sse"] < base["events"][n]["sse"]
                for n, a in v["events"].items()
            )
    return sealed_write(
        root / "summary.json",
        {
            "plan": plan["artifact_sha256"],
            "arms": by,
            "checkpoints": {
                k: {n: r["artifact_sha256"] for n, r in data.items()}
                for k, data in results.items()
            },
        },
    )


def replay_saved(root, rows):
    """Reopen every selected model and reproduce unrounded forecasts without fitting."""
    summary = sealed_read(root / "summary.json")
    plan = sealed_read(root / "plan.json")
    if summary["plan"] != plan["artifact_sha256"]:
        raise ValueError("Summary belongs to another plan")
    verified = []
    for split, checkpoint in summary["checkpoints"].items():
        events = (
            sorted(rows) if split == "full" else plan["folds"][int(split) - 1]["held"]
        )
        for name, digest in checkpoint.items():
            result = sealed_read(root / f"{split}-{name}.json")
            if result["artifact_sha256"] != digest:
                raise ValueError("Stored fit differs from the frozen summary")
            for label, arm in result["arms"].items():
                model, knots = arm["selected"], result["knots"]
                if model.get("coefficient_hats") != coefficient_hats(knots):
                    raise ValueError("Cannot bind fitted coefficients to inputs")
                scores = [score(rows[event], knots, model) for event in events]
                if scores != arm["scores"]:
                    raise ValueError("Restored forecasts or metrics differ")
                verified.append(
                    {
                        "split": split,
                        "sources": name,
                        "weights": label,
                        "events": len(scores),
                        "scores_sha256": content_hash(scores),
                    }
                )
    return sealed_write(
        root / "saved-replay.json",
        {
            "summary": summary["artifact_sha256"],
            "models": len(verified),
            "events": sum(r["events"] for r in verified),
            "verified": verified,
            "equality": "Exact hashes of both unrounded forecasts and exact metrics.",
        },
    )


def report(root, summary):
    rows = []
    for arm, v in summary["arms"].items():
        h = v["held"]
        if h["status"] != "verified":
            rows.append(f"<tr><td>{arm}</td><td colspan='7'>Unverified</td></tr>")
            continue
        cells = [
            arm,
            f"{h['rmse']:.2f}",
            f"{h['min_I']:.2f}",
            h["negative_samples"],
            f"{h['max_error']:.2f}",
            v["wins"]["base-signed"],
            ", ".join(str(v["tau"][k]) for k in ("1", "2", "3")),
            f"{v['full']['rmse']:.2f}",
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    table = "".join(rows)
    details = []
    for arm, v in summary["arms"].items():
        fold_rmse = ", ".join(
            f"{r['rmse']:.2f}" if r["status"] == "verified" else "unverified"
            for r in v["folds"].values()
        )
        details.append(
            f"<li>{arm} had fold RMSEs of {fold_rmse}. Full-sample rank and "
            f"column count were {v['ranks']['full']}. Maximum relative objective "
            f"disagreement was {v['max_solver_disagreement']:.2g} and maximum "
            f"scaled KKT residual was {v['max_kkt']:.2g}.</li>"
        )
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Does nonnegative insulin production help?</title><style>
body{{max-width:960px;margin:3rem auto;padding:0 1rem;font:16px/1.5 system-ui}}
table{{border-collapse:collapse;font-size:.9rem}}td,th{{padding:.45rem .6rem;
border-bottom:1px solid #ccd8d5;text-align:left}}th{{background:#ebf2ef}}
p{{max-width:88ch}}</style>
<h1>Nonnegative insulin production on held training events</h1>
<p>In the scalar equation dI/dt = -I/τ + q, each supplied input contributes a
saturated piecewise-linear function. Both fits use the same basis. With
nonnegative weights the production q stays nonnegative, so a nonnegative
observed initial insulin remains nonnegative in continuous time. The signed
fit can give outward flux at zero.</p><table><thead><tr>
<th>Sources and weights</th><th>Held RMSE</th><th>Minimum I</th>
<th>Negative samples</th><th>Worst error</th><th>Wins vs base signed (15)</th>
<th>Fold τ (min)</th><th>Full training RMSE</th></tr></thead><tbody>
{table}</tbody></table>
<p>Insulin measurements and errors are in pmol/L. Base uses supplied tissue
glucose Gt and administered insulin. EGP adds supplied endogenous glucose
production. Each blocked fold holds out five complete training challenges.
The fasting reference remains in every fitting set. Each forecast starts from
its single measured I(0) and uses only supplied inputs thereafter.</p>
<h2>Numerical checks</h2><ul>{"".join(details)}</ul>
<p>Knots are the distinct minimum, median and maximum on the fitting fold. End
hats saturate beyond this range. The hats for each input sum to one, so fitted
weight vectors can be nonunique. SVD and pivoted QR check the signed optimum.
NNLS and bounded-variable least squares check the constrained optimum. A fine
RK4 integration checks each scored forecast against the exact convolution.
Each stored weight names its input and hat knot. Reloaded fits reproduce the
unrounded forecasts and metrics.</p>
<p>The four arms and timescales were fixed before this comparison. Earlier
inspection of these training events motivated the experiment. EGP serves as a
supplied predictor. Predictive association alone cannot identify its biological
role. All scores in this report use the training cohort.</p>
<p><a href="summary.json">Event scores and paired wins</a> ·
<a href="plan.json">Frozen protocol</a> ·
<a href="saved-replay.json">Exact saved-parameter replay</a></p></html>"""
    (root / "REPORT.html").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "replay"))
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.root.resolve().is_relative_to(args.paired.resolve()):
        raise ValueError("Keep results separate from the source")
    args.root.mkdir(parents=True, exist_ok=True)
    with _lock(args.root):
        plan, rows = prepare(args.paired, args.root)
        if args.action == "run":
            result = experiment(args.root, plan, rows)
            replay_saved(args.root, rows)
            report(args.root, result)
        elif args.action == "replay":
            replay_saved(args.root, rows)
    print(json.dumps({"plan": plan["artifact_sha256"], "action": args.action}))


if __name__ == "__main__":
    main()
