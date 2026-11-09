"""Development-only perturbation of the privileged canonical T1 reference fit bounds."""

from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_reference_fits as reference
from scripts.iclr_submission_evidence import REPO, digest, table

PUBLIC = REPO / "artifacts/phase-c-development-final-v2/public"
REFERENCES = REPO / "output/iclr-2027-new-controls/reference-final"
HISTORICAL = REPO / "output/iclr-local-completion/historical"
PROFILES = ("wide_same_starts", "wide_distributed_starts")


def limits(cell: str, expanded: bool) -> tuple[np.ndarray, np.ndarray]:
    """Set positive log-multiplier intervals with a valid gastric transition law."""
    centers = reference.coefficients(cell)
    names = list(centers)
    if expanded and cell != reference.T1P:
        raise ValueError("the wider experiment is defined only for perturbed T1")
    width = np.log(4.0) if expanded else 0.5
    lo = np.full(len(centers), -width)
    hi = np.full(len(centers), width)
    if "b" in names:
        hi[names.index("b")] = np.log(0.98 / centers["b"])
    if expanded:
        mid = np.sqrt(centers["kmin"] * centers["kmax"])
        lo[names.index("kmax")] = np.log(mid / centers["kmax"])
        hi[names.index("kmin")] = np.log(mid / centers["kmin"])
        # Do not introduce a larger upper limit for the effective absorbed mass gain.
        hi[names.index("ra_gain")] = 0.5
    return lo, hi


def conditioning(jac: np.ndarray) -> dict:
    """Describe the conditioning of the normalized training residual Jacobian."""
    v = np.linalg.svd(jac, compute_uv=False)
    tol = max(jac.shape) * np.finfo(float).eps * v[0]
    rank = int(np.sum(v > tol))
    return {
        "rows": jac.shape[0],
        "columns": jac.shape[1],
        "jacobian_rank": rank,
        "largest_singular_value": float(v[0]),
        "smallest_singular_value": float(v[-1]),
        "condition_number": float(v[0] / v[-1]) if rank == jac.shape[1] else None,
        "rank_tolerance": float(tol),
    }


def parameter_rows(cell: str, theta, profile: str, start: int) -> list[dict]:
    """Expand fitted multipliers into named physical coefficients and boundary flags."""
    lo, hi = limits(cell, profile != "submitted_bounds")
    centers = reference.coefficients(cell)
    if len(centers) != len(theta):
        raise ValueError("incomplete coefficient vector")
    return [
        {
            "cell": cell,
            "profile": profile,
            "optimizer_start": start,
            "parameter": n,
            "scale": scale,
            "log_multiplier": v,
            "value": scale * np.exp(v),
            "lower": scale * np.exp(a),
            "upper": scale * np.exp(b),
            "bound_hit": "lower"
            if abs(v - a) <= 1e-6
            else "upper"
            if abs(v - b) <= 1e-6
            else "",
        }
        for (n, scale), v, a, b in zip(centers.items(), theta, lo, hi, strict=True)
    ]


def checkpoint_jacobian(root, cell, start, theta) -> dict:
    """Check the saved Jacobian at the exact fitted coefficient vector."""
    key = hashlib.sha256(theta.tobytes()).hexdigest()
    path = root / "fits" / cell / str(start) / (key + ".npz")
    if not path.is_file():
        return {"diagnostic_status": "final_jacobian_unavailable"}
    with np.load(path, allow_pickle=False) as a:
        expected = hashlib.sha256(a["residual"].tobytes() + a["jacobian"].tobytes())
        if expected.hexdigest() != a["digest"].item() or not np.array_equal(
            theta, a["theta"]
        ):
            raise ValueError("corrupt final Jacobian cache")
        return {"diagnostic_status": "available", **conditioning(a["jacobian"])}


def expanded_fit(train, val, start, profile, cell, steps, directory):
    """Fit the canonical family and independently refine its integration grid."""
    tr, va = (reference.public_data(cell, d) for d in (train, val))
    scale = np.std(np.hstack([d[2] for d in tr]), axis=1)
    count = tr[0][1].shape[1]
    fn = reference.family(cell, count, 2, jacobian=True)
    lo, hi = limits(cell, True)
    rng = np.random.default_rng(800 + start)
    if profile == "wide_same_starts":
        initial = rng.uniform(-0.1, 0.1, len(lo))
    else:
        # Independent points in the inner 80% of each allowed interval.
        initial = lo + (hi - lo) * rng.uniform(0.1, 0.9, len(lo))

    @lru_cache(maxsize=1)
    def residual_and_jac(theta_bytes):
        theta = np.frombuffer(theta_bytes, dtype=np.float64)
        path = directory / (hashlib.sha256(theta_bytes).hexdigest() + ".npz")
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                r, j = data["residual"], data["jacobian"]
                if not np.array_equal(theta, data["theta"]) or data[
                    "digest"
                ].item() != (hashlib.sha256(r.tobytes() + j.tobytes()).hexdigest()):
                    raise ValueError("corrupt saved residual")
                return r, j
        err, jac = [], []
        for first, knots, actual in tr:
            pred, gradient = fn(first, knots, theta)
            err.append(
                ((np.asarray(pred)[: len(scale)] - actual) / scale[:, None]).T.ravel()
            )
            jac.append(np.asarray(gradient) / np.tile(scale, actual.shape[1])[:, None])
        r, j = np.concatenate(err), np.vstack(jac)
        if not np.isfinite(r).all() or not np.isfinite(j).all():
            raise ValueError("nonfinite rollout or sensitivity")
        temp = path.with_suffix(".partial")
        with temp.open("wb") as f:
            np.savez_compressed(
                f,
                residual=r,
                jacobian=j,
                theta=theta,
                digest=hashlib.sha256(r.tobytes() + j.tobytes()).hexdigest(),
            )
        temp.replace(path)
        return r, j

    opt = least_squares(
        lambda t: residual_and_jac(t.tobytes())[0],
        initial,
        jac=lambda t: residual_and_jac(t.tobytes())[1],
        bounds=(lo, hi),
        max_nfev=steps,
        x_scale="jac",
        ftol=1e-9,
    )
    coarse, fine = (reference.family(cell, count, n, jacobian=False) for n in (2, 4))
    scores = {}
    gaps = []
    for split, data in (("training", tr), ("validation", va)):
        errs = []
        for first, knots, obs in data:
            a, b = (
                np.asarray(f(first, knots, opt.x))[: len(scale)] for f in (fine, coarse)
            )
            gaps.append(float(np.sqrt(np.mean(((a - b) / scale[:, None]) ** 2))))
            errs.append(np.mean(((a - obs) / scale[:, None]) ** 2, axis=1))
        scores[split] = np.mean(errs, axis=0).tolist()
    return {
        "status": "ready" if max(gaps) < 0.001 else "grid_disagreement",
        "optimizer_converged": bool(opt.success),
        "optimizer_message": opt.message,
        "initial": initial.tolist(),
        "theta": opt.x.tolist(),
        "nfev": opt.nfev,
        "scores": scores,
        "max_grid_discrepancy": max(gaps),
        "optimality": float(opt.optimality),
        **conditioning(opt.jac),
    }


def study(public_root: Path, references: Path, out: Path) -> dict:
    """Seal the development design before fitting and summarizing both profiles."""
    cell = reference.T1P
    source = sealed_read(references / "plan.json")
    if digest(Path(reference.__file__)) != source["source"]:
        raise ValueError("original numerical model implementation changed")
    files = {k: digest(public_root / cell / k) for k in source["files"][cell]}
    if files != source["files"][cell] or source["starts"] != 3:
        raise ValueError("original evaluation data or starts differ")
    with public._lock(out):
        lo, hi = limits(cell, True)
        plan = sealed_write(
            out / "transfer_plan.json",
            {
                "protocol": "iclr-privileged-transfer-bounds-1",
                "development_only": True,
                "original_plan": source["artifact_sha256"],
                "scripts": [digest(Path(__file__)), source["source"]],
                "cell": cell,
                "runtime": public._runtime(),
                "files": files,
                "starts": [0, 1, 2],
                "profiles": list(PROFILES),
                "seeds": [800, 801, 802],
                "max_nfev": source["max_evals"],
                "lower": lo.tolist(),
                "upper": hi.tolist(),
                "source_structure": "canonical",
                "selection": "minimum validation NMSE among ready starts per profile",
            },
        )
        _, tr, va = reference.load_public_cell(public_root / cell)
        parameters, fits = [], []
        for c in source["files"]:
            for i in range(source["starts"]):
                old = sealed_read(references / "fits" / c / f"{i}.json")
                if old["plan_sha256"] != source["artifact_sha256"]:
                    raise ValueError("changed historical fit")
                theta = np.array(old["theta"], dtype=float)
                p = parameter_rows(c, theta, "submitted_bounds", i)
                parameters.extend(p)
                fits.append(
                    fit_row(c, "submitted_bounds", i, old, p)
                    | checkpoint_jacobian(references, c, i, theta)
                )
        for profile in PROFILES:
            for i in range(3):
                dest = out / "transfer-fits" / profile / str(i)
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / "fit.json"
                if not path.exists():
                    payload = expanded_fit(
                        tr, va, i, profile, cell, source["max_evals"], dest
                    )
                    sealed_write(path, {"plan": plan["artifact_sha256"], **payload})
                fit = sealed_read(path)
                if fit["plan"] != plan["artifact_sha256"]:
                    raise ValueError("expanded fit does not match the sealed study")
                p = parameter_rows(cell, fit["theta"], profile, i)
                parameters.extend(p)
                fits.append(
                    fit_row(cell, profile, i, fit, p)
                    | {
                        "diagnostic_status": "available",
                        **{
                            k: fit[k]
                            for k in (
                                "jacobian_rank",
                                "condition_number",
                                "rows",
                                "columns",
                                "largest_singular_value",
                                "smallest_singular_value",
                                "rank_tolerance",
                                "optimality",
                            )
                        },
                    }
                )
        best = {}
        for r in fits:
            if r["status"] == "ready":
                key = r["cell"], r["profile"]
                best[key] = min(
                    best.get(key, (float("inf"), -1)),
                    (r["validation_nmse"], r["optimizer_start"]),
                )
        for r in fits:
            r["selected"] = best.get((r["cell"], r["profile"])) == (
                r["validation_nmse"],
                r["optimizer_start"],
            )
        table(out / "reference_parameters.csv", parameters)
        table(out / "canonical_structure_transfer_and_conditioning.csv", fits)
        return sealed_write(
            out / "transfer_summary.json",
            {
                "plan": plan["artifact_sha256"],
                "rows": fits,
                "parameters": len(parameters),
            },
        )


def fit_row(cell, profile, i, fit, parameters):
    return {
        "cell": cell,
        "profile": profile,
        "optimizer_start": i,
        "status": fit["status"],
        "training_nmse": float(np.mean(fit["scores"]["training"])),
        "validation_nmse": float(np.mean(fit["scores"]["validation"])),
        "max_grid_discrepancy": fit["max_grid_discrepancy"],
        "nfev": fit["nfev"],
        "optimizer_converged": fit["optimizer_converged"],
        "reason": fit["optimizer_message"],
        "bound_hits": sum(bool(p["bound_hit"]) for p in parameters),
        "fit_seal": fit["artifact_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--output", type=Path, default=HISTORICAL)
    args = parser.parse_args()
    study(args.public, args.references, args.output)
