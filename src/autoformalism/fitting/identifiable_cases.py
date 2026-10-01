"""Evaluator-owned identifiable controls, separate from benchmark releases.

Only sampled y and public input u enter fitting. Derivatives, z and true
coefficients stay in the evaluator, including the excitation qualification.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest

PROTOCOL = "identifiable-fitting-inputs-1"
TRUTHS = {
    "linear": {"a": 0.8, "b": 1.3, "c": 1.6},
    "nonlinear": {"a": 0.6, "b": 0.8, "c": 1.4, "d": 0.7},
    "fast_slow": {"a": 18.0, "b": 2.0, "c": 2.0, "d": 0.5},
}
INITIAL_Z = 0.4


def reference(row: dict, theta: dict, *, method: str = "DOP853") -> np.ndarray:
    """Independent handwritten reference, segmented at every public input knot."""
    time = np.asarray(row["time"])
    u = np.asarray(row["external_inputs"]["u"])
    x = np.array([row["targets"]["y"][0], theta["init_z_value"]])
    out = [x.copy()]
    # Collinear sample intervals may be merged; all slope changes remain boundaries.
    slopes = np.diff(u) / np.diff(time)
    edges = [0, *(np.flatnonzero(abs(np.diff(slopes)) > 1e-8) + 1), len(time) - 1]
    for left, right in pairwise(edges):

        def rhs(t, state):
            y, z = state
            return [
                z,
                -theta["a"] * z
                - theta["b"] * y
                - theta.get("d", 0) * y**3
                + theta["c"] * np.interp(t, time, u),
            ]

        solved = solve_ivp(
            rhs,
            (time[left], time[right]),
            x,
            method=method,
            t_eval=time[left + 1 : right + 1],
            rtol=2e-10,
            atol=2e-12,
        )
        if not solved.success or solved.y.shape[1] != right - left:
            raise ValueError("reference integration unavailable")
        out.extend(solved.y.T)
        x = solved.y[:, -1]
    return np.asarray(out)


def request(case: str, seed: int) -> PublicFitRequest:
    """Generic reproducible starts independent of the reference coefficient vector."""
    nonlinear = case != "linear"
    names = ["a", "b", "c"] + (["d"] if nonlinear else [])
    rng = np.random.default_rng(20260930 + seed)
    guesses = {
        n: float(v * np.exp(rng.uniform(-1, 1)))
        for n, v in zip(names, [1.0, 1.0, 1.0, 0.3], strict=False)
    }
    z_guess = float(rng.uniform(-0.8, 0.8))
    return PublicFitRequest.model_validate(
        {
            "base_candidate": {
                "candidate_id": f"identifiable_{case}",
                "parent_candidate_id": None,
                "states": [
                    {"name": "y", "kind": "observed"},
                    {"name": "z", "kind": "latent"},
                ],
                "state_equations": [
                    {"state": "y", "rhs": "z"},
                    {
                        "state": "z",
                        "rhs": "-a*z-b*y+c*u" + ("-d*y**3" if nonlinear else ""),
                    },
                ],
                "observation_mappings": [{"channel": "y", "expression": "y"}],
                "parameters": [
                    {
                        "name": n,
                        "role": "nonnegative_coefficient",
                        "scope": "global",
                        "bounds": {"lower": 0.001, "upper": 100.0},
                    }
                    for n in names
                ],
                "initial_conditions": [
                    {"state": n, "scope": "global", "fixed_value": 0.0}
                    for n in ("y", "z")
                ],
            },
            "context": {"targets": ["y"], "external_inputs": ["u"]},
            "initialization_plan": {
                "rules": {"z": {"initial": {"mode": "value", "guess": z_guess}}}
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


def excitation(rows: list[dict], states: list[np.ndarray], *, nonlinear: bool) -> dict:
    """Check the sufficient global-identifiability condition on the ideal witness.

    y'=z fixes the latent scale; identical y implies identical z. Subtraction of
    y''=-a*y'-b*y-d*y^3+c*u then gives X*(theta1-theta2)=0. Full column rank
    forces equality, and z0=y'(0). Numerical rank is witness evidence, not a
    general symbolic test for arbitrary constructed models.
    """
    matrices = []
    for row, x in zip(rows, states, strict=True):
        y, z = x.T
        columns = [-z, -y, np.asarray(row["external_inputs"]["u"])]
        if nonlinear:
            columns.append(-(y**3))
        matrices.append(np.column_stack(columns))
    design = np.vstack(matrices)
    norms = np.linalg.norm(design, axis=0)
    singular = np.linalg.svd(design / np.maximum(norms, 1e-30), compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    return {
        "argument": "fixed_latent_scale_and_full_rank_derivative_regression",
        "columns": ["-y_prime", "-y", "u"] + (["-y_cubed"] if nonlinear else []),
        "column_normalized_singular_values": singular.tolist(),
        "singular_ratio": ratio,
        "passed": bool(np.all(norms > 0) and ratio > 1e-4),
        "rank_scope": "ideal training witness; conditional analytic uniqueness",
        "derivatives_or_hidden_labels_given_to_fitter": False,
    }


def make_inputs() -> dict:
    """Create standalone noiseless controls; never touch the 28-cell release."""
    cases = {}
    for name, coefficients in TRUTHS.items():
        truth = {**coefficients, "init_z_value": INITIAL_Z}
        splits, private = {}, {}
        for split, initials in (("train", [0.2, -0.4, 0.7]), ("val", [0.5, -0.3])):
            rows, states = [], []
            for j, y0 in enumerate(initials):
                time = np.linspace(0, 12, 121)
                knots = np.arange(13, dtype=float)
                phase = j * 0.9 + (0 if split == "train" else 0.47)
                values = 1.4 * np.sin(1.1 * knots + phase) + 0.7 * np.cos(
                    2.3 * knots - phase
                )
                row = {
                    "trajectory_id": f"{split}_{j}",
                    "time": time.tolist(),
                    "targets": {"y": np.full(len(time), y0).tolist()},
                    "external_inputs": {"u": np.interp(time, knots, values).tolist()},
                    "auxiliaries": {},
                }
                state = reference(row, truth)
                row["targets"]["y"] = state[:, 0].tolist()
                rows.append(row)
                states.append(state)
            splits[split] = {
                "name": split,
                "rows": rows,
                "fingerprint": public.content_sha256(rows),
            }
            private[split] = states
        gate = excitation(
            splits["train"]["rows"], private["train"], nonlinear=name != "linear"
        )
        if not gate["passed"]:
            raise ValueError(f"insufficient excitation for {name}")
        cases[name] = {
            "training": splits["train"],
            "validation": splits["val"],
            "reference_parameters": truth,
            "identifiability": gate,
            "public_contract": (
                "Generate y from public continuous piecewise-linear u; y'=z fixes the "
                "latent units. Initial y is observed; initial z is unknown, signed, "
                "and shared across all runs. Fit coefficients and z0 on training only. "
                "No z trajectories or derivatives are supplied to fitting."
            ),
        }
    return {
        "protocol": PROTOCOL,
        "cases": cases,
        "test_data_opened": False,
        "benchmark_release_modified": False,
        "noise": "none",
    }
