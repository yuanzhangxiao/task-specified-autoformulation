"""Fit privileged simulator equation families using only public Phase C observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.benchmarks.phase_c_release import load_public_cell
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_man as dm
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

CSTR = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1"
T2 = "phase_c_dalla_man_t2_canonical_named_hard_rates_v1"
T1 = "phase_c_dalla_man_t1_canonical_named_easy_rates_v1"
T1P = "phase_c_dalla_man_t1_perturbed_named_easy_rates_v1"
CELLS = (CSTR, T2, T1, T1P)
CHANNELS = {
    CSTR: ("Cf", "Tf", "Tjf"),
    T2: ("meal_rate_g_per_min", "insulin_pmol_per_kg_min", "Uii"),
    T1: ("meal_rate_g_per_min", "EGP", "Uii", "E", "Gt"),
}
CHANNELS[T1P] = CHANNELS[T1]
TARGETS = {CSTR: ["T"], T2: ["Gp", "I"], T1: ["Gp"], T1P: ["Gp"]}
HIDDEN = {
    CSTR: ["C", "Tj", "reaction_heat", "jacket_heat"],
    T2: ["Ra", "Uid", "EGP", "X"],
    T1: ["Ra"],
    T1P: ["Ra"],
}
VERSION = "iclr-privileged-reference-fit-1"


def coefficients(cell: str) -> dict[str, float]:
    """Centers define disclosed physical scale priors; all listed coordinates fit."""
    if cell == CSTR:
        return {
            "flow": 1.1,
            "rate": 1.1,
            "activation": 24.0,
            "heat_rate": 80.0,
            "exchange": 2.0,
            "jacket_flow": 1.2,
            "jacket_exchange": 1.2,
            "C0": 0.35,
            "Tj0": 345.0,
        }
    nominal = dm.DallaManParameters()
    names = ["k1", "k2", "kgri", "kmax", "kmin", "kabs", "b", "d"]
    if cell == T2:
        names += [
            "VI",
            "VG",
            "m1",
            "m2",
            "m4",
            "m5",
            "m6",
            "kp1",
            "kp2",
            "kp3",
            "kp4",
            "ki",
            "Vm0",
            "Vmx",
            "Km0",
            "p2U",
            "K",
            "alpha",
            "beta",
            "gamma",
            "ke1",
            "ke2",
        ]
    coeff = {k: getattr(nominal, k) for k in names}
    coeff["ra_gain"] = nominal.f / nominal.BW
    if cell == T2:
        coeff.update(
            Gt0=130.0,
            Gp_b=172.0,
            I_b=25.0,
            Il0=4.5,
            Ipo0=3.0,
            secretion_b=1.5,
            Gt_init_gain=1.0,
            Il_init_gain=0.18,
        )
    return coeff


def public_inputs(cell, tr) -> np.ndarray:
    """Recover public forcing knots and commanded meal masses."""
    channels = {**tr.external_inputs, **tr.auxiliaries}
    values = np.array([channels[n] for n in CHANNELS[cell]])
    t = np.asarray(tr.time)
    if len(t) < 2 or not np.allclose(np.diff(t), 0.1 if cell == CSTR else 1.0):
        raise ValueError("expected the Phase C anchor time grid")
    dose = np.zeros(len(t) - 1)
    if cell != CSTR:
        rate = values[0]
        positive = np.flatnonzero(rate > 0)
        for indices in np.split(positive, np.where(np.diff(positive) > 1)[0] + 1):
            if not len(indices):
                continue
            left, right = int(indices[0] - 1), int(indices[-1] + 1)
            if left < 0 or right >= len(t):
                raise ValueError("truncated public ingestion command")
            # Each Phase C meal is a non-overlapping ten-minute triangle.
            if not np.isclose(t[right] - t[left], 10.0) or np.any(
                rate[left : right + 1] < 0
            ):
                raise ValueError("unrecognized ingestion schedule")
            mass = np.trapezoid(rate[left : right + 1], t[left : right + 1])
            expected = (
                np.maximum(0, 1 - np.abs((t[left : right + 1] - t[left] - 5) / 5))
                * mass
                / 5
            )
            if not np.allclose(rate[left : right + 1], expected):
                raise ValueError(
                    "ingestion does not match the declared public triangle"
                )
            dose[left] = 1000 * mass
    return np.vstack((values[:, :-1], values[:, 1:], dose))


def family(cell: str, count: int, substeps: int, *, jacobian: bool):
    """Build a trusted RK4 map and exact parameter sensitivities on the public grid."""
    import casadi as ca

    centers = coefficients(cell)
    log_scale = ca.SX.sym("theta", len(centers))
    p = {n: v * ca.exp(log_scale[i]) for i, (n, v) in enumerate(centers.items())}
    n = 3 if cell == CSTR else (13 if cell == T2 else 5)
    nu = len(CHANNELS[cell])
    x = ca.SX.sym("x", n)
    u = ca.SX.sym("u", nu)
    init = ca.SX.sym("initial", len(TARGETS[cell]))
    if cell == CSTR:
        c, temp, jacket = ca.vertsplit(x)
        cf, tf, tjf = ca.vertsplit(u)
        shape = ca.exp(-p["activation"] * (350 / ca.fmax(temp, 250) - 1)) * ca.fmax(
            c, 0
        )
        heat = p["heat_rate"] * shape
        exchange = p["exchange"] * (jacket - temp)
        dx = [
            p["flow"] * (cf - c) - p["rate"] * shape,
            p["flow"] * (tf - temp) + heat + exchange,
            p["jacket_flow"] * (tjf - jacket) + p["jacket_exchange"] * (temp - jacket),
        ]
        y = [temp, c, jacket, heat, exchange]
        x0 = [p["C0"], init[0], p["Tj0"]]
    else:
        gp, q1, q2, qgut, reference_mass = ca.vertsplit(x[:5])
        q1, q2, qgut = (ca.fmax(q, 0) for q in (q1, q2, qgut))
        safe_mass = ca.fmax(reference_mass, 1e-10)
        # b is bounded strictly below one by the optimizer bounds.
        fall = ca.tanh(
            2.5 * (q1 + q2 - p["b"] * safe_mass) / (safe_mass * (1 - p["b"]))
        )
        rise = ca.tanh(2.5 * (q1 + q2 - p["d"] * safe_mass) / (safe_mass * p["d"]))
        kg = ca.if_else(
            reference_mass > 0,
            p["kmin"] + (p["kmax"] - p["kmin"]) * (fall - rise + 2) / 2,
            p["kmax"],
        )
        gut = [
            1000 * u[0] - p["kgri"] * q1,
            p["kgri"] * q1 - kg * q2,
            kg * q2 - p["kabs"] * qgut,
            0,
        ]
        ra = p["ra_gain"] * p["kabs"] * qgut
        if cell == T2:
            gt, ip, il, ipo, ys, i1, delay, action = ca.vertsplit(x[5:])
            gt = ca.fmax(gt, 0)
            ip = ca.fmax(ip, 0)
            ipo = ca.fmax(ipo, 0)
            production = ca.fmax(
                0,
                p["kp1"]
                - p["kp2"] * ca.fmax(gp, 0)
                - p["kp3"] * delay
                - p["kp4"] * ipo,
            )
            disposal = ca.fmax((p["Vm0"] + p["Vmx"] * action) * gt / (p["Km0"] + gt), 0)
            dgp = (
                production
                + ra
                - u[2]
                - p["ke1"] * ca.fmax(0, gp - p["ke2"])
                - p["k1"] * ca.fmax(gp, 0)
                + p["k2"] * gt
            )
            secretion = p["gamma"] * ipo
            he = ca.fmin(0.99, ca.fmax(1e-6, p["m6"] - p["m5"] * secretion))
            dy = p["alpha"] * (
                ca.fmax(p["beta"] * (gp - p["Gp_b"]) / p["VG"], -p["secretion_b"]) - ys
            )
            dx = [
                dgp,
                *gut,
                p["k1"] * ca.fmax(gp, 0) - p["k2"] * gt - disposal,
                -(p["m2"] + p["m4"]) * ip + p["m1"] * il + u[1],
                -(p["m1"] + he * p["m1"] / (1 - he)) * il + p["m2"] * ip + secretion,
                -p["gamma"] * ipo
                + ys
                + p["K"] * ca.fmax(0, dgp / p["VG"])
                + p["secretion_b"],
                dy,
                p["ki"] * (ip / p["VI"] - i1),
                p["ki"] * (i1 - delay),
                p["p2U"] * (ip / p["VI"] - p["I_b"] - action),
            ]
            x0 = [
                init[0],
                0,
                0,
                0,
                0,
                p["Gt0"] - p["Gt_init_gain"] * (init[0] - p["Gp_b"]),
                init[1] * p["VI"],
                p["Il0"] + p["Il_init_gain"] * (init[1] - p["I_b"]),
                p["Ipo0"],
                0,
                p["I_b"],
                p["I_b"],
                0,
            ]
            y = [gp, ip / p["VI"], ra, disposal, production, action]
        else:
            dx = [
                u[1] + ra - u[2] - u[3] - p["k1"] * ca.fmax(gp, 0) + p["k2"] * u[4],
                *gut,
            ]
            x0 = [init[0], 0, 0, 0, 0]
            y = [gp, ra]
    rhs = ca.Function("rates", [x, u, log_scale], [ca.vertcat(*dx)])
    output = ca.Function("output", [x, log_scale], [ca.vertcat(*y)])
    w = ca.SX.sym("knots", 2 * nu + 1)
    z = ca.vertcat(*ca.vertsplit(x))
    if cell != CSTR:
        z[4] = ca.if_else(w[-1] > 0, w[-1] + ca.fmax(z[1] + z[2], 0), z[4])
    h = (0.1 if cell == CSTR else 1.0) / substeps
    for i in range(substeps):
        left = w[:nu] + i / substeps * (w[nu : 2 * nu] - w[:nu])
        mid = left + (w[nu : 2 * nu] - w[:nu]) / (2 * substeps)
        right = 2 * mid - left
        a = rhs(z, left, log_scale)
        b = rhs(z + h * a / 2, mid, log_scale)
        c = rhs(z + h * b / 2, mid, log_scale)
        d = rhs(z + h * c, right, log_scale)
        z = z + h * (a + 2 * b + 2 * c + d) / 6
    step = ca.Function("step", [x, w, log_scale], [z]).mapaccum(count)
    theta = ca.MX.sym("theta", len(centers))
    knots = ca.MX.sym("knots", 2 * nu + 1, count)
    start = ca.MX.sym("initial", len(TARGETS[cell]))
    initial = ca.Function("initial", [init, log_scale], [ca.vertcat(*x0)])(start, theta)
    all_states = ca.horzcat(initial, step(initial, knots, ca.repmat(theta, 1, count)))
    pred = output.map(count + 1)(all_states, ca.repmat(theta, 1, count + 1))
    outputs = [pred]
    if jacobian:
        outputs.append(ca.jacobian(ca.vec(pred[: len(TARGETS[cell]), :]), theta))
    return ca.Function("predict", [start, knots, theta], outputs)


def public_data(cell, split):
    """Extract input and targets without using trajectory IDs."""
    return [
        (
            np.array([r.targets[k][0] for k in TARGETS[cell]]),
            public_inputs(cell, r),
            np.array([r.targets[k] for k in TARGETS[cell]]),
        )
        for r in split.rows
    ]


def fit_start(cell, train, val, start, max_evals, checkpoint):
    """Fit one prespecified start to all training samples with equal channel weight."""
    import casadi as ca
    from scipy.linalg import norm

    tr, va = public_data(cell, train), public_data(cell, val)
    names = list(coefficients(cell))
    scales = np.std(np.hstack([d[2] for d in tr]), axis=1)
    if np.any(scales <= 1e-12):
        raise ValueError("constant training channel")
    fn = family(cell, tr[0][1].shape[1], 2, jacobian=True)
    last = None

    def calculate(theta):
        nonlocal last
        if last is not None and np.array_equal(last[0], theta):
            return last[1:]
        cache = checkpoint / (hashlib.sha256(theta.tobytes()).hexdigest() + ".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as record:
                if not np.array_equal(record["theta"], theta):
                    raise ValueError("wrong cached parameter point")
                last = (theta.copy(), record["residual"], record["jacobian"])
                digest = hashlib.sha256(
                    last[1].tobytes() + last[2].tobytes()
                ).hexdigest()
                if record["digest"].item() != digest:
                    raise ValueError("corrupt numerical checkpoint")
            return last[1:]
        residuals = []
        js = []
        for initial, knots, actual in tr:
            pred, jac = fn(initial, knots, theta)
            err = (np.asarray(pred)[: len(scales)] - actual) / scales[:, None]
            residuals.append(err.T.ravel())
            js.append(np.asarray(jac) / np.tile(scales, actual.shape[1])[:, None])
        last = (theta.copy(), np.concatenate(residuals), np.vstack(js))
        if not np.isfinite(last[1]).all() or not np.isfinite(last[2]).all():
            raise ValueError("nonfinite rollout or sensitivity")
        temporary = cache.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                theta=last[0],
                residual=last[1],
                jacobian=last[2],
                digest=hashlib.sha256(
                    last[1].tobytes() + last[2].tobytes()
                ).hexdigest(),
            )
        temporary.replace(cache)
        return last[1:]

    lb = np.full(len(names), -0.5)
    ub = np.full(len(names), 0.5)
    if "b" in names:
        ub[names.index("b")] = np.log(0.98 / coefficients(cell)["b"])
    rng = np.random.default_rng(800 + start)
    initial = rng.uniform(-0.10, 0.10, len(names))
    t0 = monotonic()
    result = least_squares(
        lambda z: calculate(z)[0],
        initial,
        jac=lambda z: calculate(z)[1],
        bounds=(lb, ub),
        max_nfev=max_evals,
        x_scale="jac",
        ftol=1e-9,
    )
    fine = family(cell, tr[0][1].shape[1], 4, jacobian=False)
    coarse = family(cell, tr[0][1].shape[1], 2, jacobian=False)
    gaps = []
    scores = {}
    for label, data in (("training", tr), ("validation", va)):
        errors = []
        for initial, knots, target in data:
            a = np.asarray(fine(initial, knots, result.x))[: len(scales)]
            b = np.asarray(coarse(initial, knots, result.x))[: len(scales)]
            gaps.append(float(norm((a - b) / scales[:, None]) / np.sqrt(a.size)))
            errors.append(np.mean(((a - target) / scales[:, None]) ** 2, axis=1))
        scores[label] = np.mean(errors, axis=0).tolist()
    return {
        "status": "ready" if max(gaps) < 1e-3 else "grid_disagreement",
        "cell": cell,
        "start": start,
        "nfev": result.nfev,
        "optimizer_converged": result.success,
        "optimizer_message": result.message,
        "fit_seconds": monotonic() - t0,
        "theta": result.x.tolist(),
        "coefficient_names": names,
        "scales": scales.tolist(),
        "scores": scores,
        "max_grid_discrepancy": max(gaps),
        "casadi": ca.__version__,
    }


def prepare(public_root: Path, root: Path, starts: int, max_evals: int, cells=CELLS):
    if starts < 1 or max_evals < 1:
        raise ValueError("positive starts and evaluations required")
    cells = list(cells)
    if len(set(cells)) != len(cells) or set(cells) - set(CELLS):
        raise ValueError("unknown or repeated reference task")
    files = {
        cell: {
            k: hashlib.sha256((public_root / cell / k).read_bytes()).hexdigest()
            for k in ("train.json", "val.json", "specification.json")
        }
        for cell in cells
    }
    import casadi as ca

    return sealed_write(
        root / "plan.json",
        {
            "protocol": VERSION,
            "files": files,
            "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": public._runtime(),
            "casadi": ca.__version__,
            "starts": starts,
            "max_evals": max_evals,
            "access": (
                "Supplied structure and parameter scales. Fit listed coefficients "
                "and shared initializer parameters to observed training targets."
            ),
        },
    )


def run_one(public_root, root, cell, start):
    plan = sealed_read(root / "plan.json")
    if cell not in plan["files"] or start not in range(plan["starts"]):
        raise ValueError("task outside plan")
    if (
        prepare(public_root, root, plan["starts"], plan["max_evals"], plan["files"])
        != plan
    ):
        raise ValueError("changed run identity")
    path = root / "fits" / cell / f"{start}.json"
    with public._lock(path.parent / str(start)):
        if path.exists():
            out = sealed_read(path)
            if out["plan_sha256"] != plan["artifact_sha256"]:
                raise ValueError("fit belongs to another plan")
            return out
        spec, tr, va = load_public_cell(public_root / cell)
        if spec["targets"] != TARGETS[cell]:
            raise ValueError("public targets differ")
        try:
            out = fit_start(
                cell, tr, va, start, plan["max_evals"], path.parent / str(start)
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            out = {
                "cell": cell,
                "start": start,
                "status": "fit_failed",
                "reason": str(exc),
            }
        return sealed_write(path, {"plan_sha256": plan["artifact_sha256"], **out})


def freeze(root):
    plan = sealed_read(root / "plan.json")
    rows = []
    fits = []
    for cell in plan["files"]:
        choices = []
        for start in range(plan["starts"]):
            r = sealed_read(root / "fits" / cell / f"{start}.json")
            if r["plan_sha256"] != plan["artifact_sha256"]:
                raise ValueError("wrong fit")
            fits.append(r)
            if r["status"] == "ready":
                choices.append(r)
        best = (
            min(choices, key=lambda r: (np.mean(r["scores"]["validation"]), r["start"]))
            if choices
            else {}
        )
        rows.append(
            {
                "method": "fitted_reference_structure",
                "benchmark_id": cell,
                "repetition": 0,
                "status": best.get("status", "fit_failed"),
                "fit": best,
                "source_id": cell,
            }
        )
    bundle = sealed_write(
        root / "frozen_models.json",
        {"protocol": VERSION, "plan_sha256": plan["artifact_sha256"], "rows": rows},
    )
    for name, subset in (
        ("reference_structure_results", fits),
        ("literature_transfer_results", [r for r in fits if r["cell"] in (T1, T1P)]),
    ):
        with (root / (name + ".csv")).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "cell",
                    "start",
                    "status",
                    "training_nmse",
                    "validation_nmse",
                    "grid_discrepancy",
                    "nfev",
                    "seconds",
                ]
            )
            for r in subset:
                scores = r.get("scores", {})
                w.writerow(
                    [
                        r["cell"],
                        r["start"],
                        r["status"],
                        *[
                            np.mean(scores[k]) if k in scores else None
                            for k in ("training", "validation")
                        ],
                        r.get("max_grid_discrepancy"),
                        r.get("nfev"),
                        r.get("fit_seconds"),
                    ]
                )
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fit", "freeze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=CELLS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-evals", type=int, default=50)
    a = parser.parse_args()
    if a.stage != "freeze" and not a.public_root:
        parser.error("--public-root required")
    if a.stage == "prepare":
        prepare(a.public_root, a.root, a.starts, a.max_evals, a.cells)
    elif a.stage == "fit":
        if not a.cell:
            parser.error("fit needs --cell")
        print(run_one(a.public_root, a.root, a.cell, a.start)["status"])
    else:
        print(freeze(a.root)["artifact_sha256"])
