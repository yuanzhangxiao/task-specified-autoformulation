"""Evaluator-owned M18 linear controls; no benchmark release is modified."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-larger-coupled-inputs-1"
CASES = {
    "linear3": (0.4, 0.7, 1.0),
    "linear3_fast_slow": (0.15, 1.5, 12.0),
    "linear6": (0.3, 0.45, 0.65, 0.85, 1.05, 1.25),
    "linear6_fast_slow": (0.15, 0.35, 0.8, 2.0, 5.0, 12.0),
}
COUPLINGS = (0.9, 1.1, 0.8, 1.2, 1.0)
INITIALS = (0.35, -0.25, 0.2, -0.15, 0.1)
RANK_RATIO = 1e-8


def request(case: str, seed: int) -> PublicFitRequest:
    """Known skew couplings anchor units; all positive decays and gain are unknown.

    Same dimension implies the same generic starts, regardless of timescales.
    No reference values are used to construct guesses or bounds.
    """
    n = len(CASES[case])
    rng = np.random.default_rng(20261008 + seed)
    names = [f"a{i}" for i in range(n)] + ["gain"]
    guesses = {name: float(np.exp(rng.uniform(-1, 1))) for name in names}
    equations = []
    for i in range(n):
        rhs = f"-a{i}*x{i}"
        if i:
            rhs += f"-{COUPLINGS[i - 1]}*x{i - 1}"
        if i < n - 1:
            rhs += f"+{COUPLINGS[i]}*x{i + 1}"
        if i == 0:
            rhs += "+gain*u"
        equations.append({"state": f"x{i}", "rhs": rhs})
    return PublicFitRequest.model_validate(
        {
            "base_candidate": {
                "candidate_id": f"larger_{case}",
                "parent_candidate_id": None,
                "states": [
                    {"name": f"x{i}", "kind": "observed" if i == 0 else "latent"}
                    for i in range(n)
                ],
                "state_equations": equations,
                "observation_mappings": [{"channel": "y", "expression": "x0"}],
                "parameters": [
                    {
                        "name": name,
                        "scope": "global",
                        "role": "nonnegative_coefficient",
                        "bounds": {"lower": 0.01, "upper": 30.0},
                    }
                    for name in names
                ],
                "initial_conditions": [
                    {"state": f"x{i}", "scope": "global", "fixed_value": 0.0}
                    for i in range(n)
                ],
            },
            "context": {"targets": ["y"], "external_inputs": ["u"]},
            "initialization_plan": {
                "rules": {
                    f"x{i}": {
                        "initial": {"mode": "value", "guess": float(rng.uniform(-1, 1))}
                    }
                    for i in range(1, n)
                }
            },
            "parameter_guesses": guesses,
            "profile": "collocation-feasible-v1",
            "source": {
                "stage": "synthetic_control",
                "task_id": case,
                "artifact_sha256": public.content_sha256(
                    {"protocol": PROTOCOL, "case": case}
                ),
            },
        }
    )


def matrix(n: int, parameters: dict[str, float]) -> np.ndarray:
    """Stable for every permitted positive diagonal; fixed skew off-diagonals."""
    a = -np.diag([parameters[f"a{i}"] for i in range(n)])
    for i, weight in enumerate(COUPLINGS[: n - 1]):
        a[i, i + 1], a[i + 1, i] = weight, -weight
    return a


def reference(row: dict, parameters: dict[str, float], n: int) -> np.ndarray:
    """Matrix exponential with affine forcing, independent of optimizer integrators.

    Augment [x,u,slope] so each sampled input interval is integrated exactly up
    to floating-point matrix-exponential error. No hidden values leave this module.
    """
    t = np.asarray(row["time"])
    u = np.asarray(row["external_inputs"]["u"])
    generator = np.zeros((n + 2, n + 2))
    generator[:n, :n] = matrix(n, parameters)
    generator[0, n], generator[n, n + 1] = parameters["gain"], 1.0
    state = np.array(
        [row["targets"]["y"][0]] + [parameters[f"init_x{i}_value"] for i in range(1, n)]
    )
    result, transitions = [state.copy()], {}
    for i, dt in enumerate(np.diff(t)):
        key = round(float(dt), 12)
        if key not in transitions:
            transitions[key] = expm(generator * key)
        state = (transitions[key] @ np.r_[state, u[i], (u[i + 1] - u[i]) / dt])[:n]
        result.append(state.copy())
    return np.asarray(result)


def rank_audit(jacobian: np.ndarray, names: list[str]) -> dict:
    """Column-normalized local rank, with raw sensitivity norms retained."""
    norms = np.linalg.norm(jacobian, axis=0)
    singular = np.linalg.svd(jacobian / np.maximum(norms, 1e-30), compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] else 0.0
    return {
        "parameters": names,
        "column_norms": norms.tolist(),
        "column_normalized_singular_values": singular.tolist(),
        "singular_ratio": ratio,
        "threshold": RANK_RATIO,
        "passed": bool(np.all(norms > 0) and ratio > RANK_RATIO),
        "scope": "local sampled-training sensitivities; not global identifiability",
    }


def make_case(name: str) -> dict:
    """Freeze one noiseless control and qualify it before any fitting outcome."""
    n = len(CASES[name])
    truth = {
        **{f"a{i}": value for i, value in enumerate(CASES[name])},
        "gain": 1.4,
        **{f"init_x{i}_value": INITIALS[i - 1] for i in range(1, n)},
    }
    splits = {}
    for split, initials in (("train", (0.2, -0.4, 0.7)), ("val", (0.5, -0.3))):
        rows = []
        for j, initial in enumerate(initials):
            # Fine initial samples resolve short transients; all forcing corners
            # occur on both the public table and observation grid.
            # Binary-exact grids/levels avoid artificial interpolation corners
            # from rounding. The production forcing-boundary policy is unchanged.
            time = np.unique(np.r_[np.arange(193) / 8, np.arange(33) / 32])
            knots = np.arange(25, dtype=float)
            phase = j * 0.9 + (0 if split == "train" else 0.47)
            values = 1.4 * np.sin(1.1 * knots + phase) + 0.7 * np.cos(
                2.3 * knots - phase
            )
            values = np.round(8 * values) / 8
            row = {
                "trajectory_id": f"{split}_{j}",
                "time": time.tolist(),
                "targets": {"y": np.full(len(time), initial).tolist()},
                "external_inputs": {"u": np.interp(time, knots, values).tolist()},
                "auxiliaries": {},
            }
            row["targets"]["y"] = reference(row, truth, n)[:, 0].tolist()
            rows.append(row)
        splits[split] = {
            "name": split,
            "rows": rows,
            "fingerprint": public.content_sha256(rows),
        }
    columns = []
    for parameter, value in truth.items():
        step = 1e-4 * max(abs(value), 1.0)
        columns.append(
            np.concatenate(
                [
                    (
                        reference(r, truth | {parameter: value + step}, n)[:, 0]
                        - reference(r, truth | {parameter: value - step}, n)[:, 0]
                    )
                    / (2 * step)
                    for r in splits["train"]["rows"]
                ]
            )
        )
    rank = rank_audit(np.column_stack(columns), list(truth))
    if not rank["passed"]:
        raise ValueError(f"unqualified control {name}: {rank}")
    return {
        "training": splits["train"],
        "validation": splits["val"],
        "reference_parameters": truth,
        "identifiability": rank,
        "states": n,
        "eigenvalues_real": np.linalg.eigvals(matrix(n, truth)).real.tolist(),
        "public_contract": (
            "Known skew-tridiagonal couplings fix latent coordinates. Positive "
            "diagonal decays and input gain are fitted. Only x0=y and u are observed. "
            "Signed hidden initials are unknown and shared across all trajectories."
        ),
    }


def export_inputs(output: Path) -> dict:
    """Export once or reuse exactly; never regenerate a completed input file."""
    if output.exists():
        data = read_seal(output)
        bases(data)
        return {"identity": public.content_sha256(data), "starts": 12}
    cases = {name: make_case(name) for name in CASES}
    commons = {}
    for name, case in cases.items():
        train = public.unpack_split(PublicSplit.model_validate(case["training"]))
        for seed in range(3):
            req = request(name, seed)
            model, start, _ = public._lower(req)
            commons[f"{name}_s{seed}"] = {
                "case": name,
                "seed": seed,
                "request": req.model_dump(mode="json"),
                "coordinates": training_coordinates(model, train, start).model_dump(
                    mode="json"
                ),
                "nodes": {},
                "start": start,
            }
    data = {
        "protocol": PROTOCOL,
        "cases": cases,
        "commons": commons,
        "config": {
            "cstr_nmse": 1e-6,
            "parameter_relative": 0.01,
            "initial_absolute": 0.001,
        },
        "test_data_opened": False,
        "benchmark_release_modified": False,
        "noise": "none",
    }
    bases(data)
    seal(output, data)
    return {"identity": public.content_sha256(data), "starts": 12}


def bases(data: dict) -> dict:
    """Allowlist only training observations, generic starts and their coordinates."""
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["cases"]) != set(CASES)
        or set(data["commons"]) != {f"{case}_s{i}" for case in CASES for i in range(3)}
    ):
        raise ValueError("expected all twelve larger coupled starts")
    result = {}
    for key, entry in data["commons"].items():
        case, seed = entry["case"], entry["seed"]
        if case not in CASES or key != f"{case}_s{seed}":
            raise ValueError("larger coupled case/seed identity differs")
        req = PublicFitRequest.model_validate(entry["request"])
        if req != request(case, seed) or entry["start"] != public._lower(req)[1]:
            raise ValueError("frozen generic request/start differs")
        training = data["cases"][case]["training"]
        TrainingOnlySplit.model_validate(training)
        result[key] = {
            k: deepcopy(entry[k]) for k in ("request", "coordinates", "nodes", "start")
        }
        result[key]["training"] = deepcopy(training)
    return result
