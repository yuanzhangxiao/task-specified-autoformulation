"""Evaluator-only saturating oscillator with a sufficient identifiability witness."""

from __future__ import annotations

from copy import deepcopy
from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.fitting import identifiable_cases as base
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest

TRUTH = {"a": 0.45, "b": 1.4, "c": 1.2, "q": 0.8, "init_z_value": 0.4}


def request(seed: int) -> PublicFitRequest:
    """Generic starts; q changes the shape of the law, not just its amplitude."""
    payload = base.request("nonlinear", seed).model_dump(mode="json")
    candidate = payload["base_candidate"]
    candidate["candidate_id"] = "identifiable_saturating_stiffness"
    candidate["state_equations"][1]["rhs"] = "-a*z-b*y/(1+q*y**2)+c*u"
    for parameter in candidate["parameters"]:
        if parameter["name"] == "d":
            parameter["name"] = "q"
            parameter["bounds"]["upper"] = 10.0
    payload["parameter_guesses"]["q"] = payload["parameter_guesses"].pop("d")
    payload["source"]["task_id"] = "shape"
    payload["source"]["artifact_sha256"] = public.content_sha256(
        {"protocol": "nonlinear-shape-1"}
    )
    return PublicFitRequest.model_validate(payload)


def reference(row, theta):
    """Independent high-accuracy reference; exact forcing knots are all retained."""
    t = np.asarray(row["time"])
    u = row["external_inputs"]["u"]
    x = np.array([row["targets"]["y"][0], theta["init_z_value"]])
    states = [x.copy()]
    for a, b in pairwise(t):

        def rhs(time, state):
            y, z = state
            return [
                z,
                -theta["a"] * z
                - theta["b"] * y / (1 + theta["q"] * y * y)
                + theta["c"] * np.interp(time, t, u),
            ]

        solved = solve_ivp(rhs, (a, b), x, method="DOP853", rtol=2e-10, atol=2e-12)
        if not solved.success:
            raise ValueError("nonlinear reference unavailable")
        x = solved.y[:, -1]
        states.append(x.copy())
    return np.asarray(states)


def make_case(template: dict) -> dict:
    """Produce separate synthetic data; do not modify a released benchmark.

    Multiplying y''=-a*y'+c*u-b*y/(1+q*y^2) by (1+q*y^2) gives a
    linear regression in the lifted coefficients (a,c,b,q,a*q,c*q).
    Full column rank uniquely identifies these, hence (a,b,c,q); y'=z
    fixes the latent scale and z0. Ideal derivatives appear only in this audit.
    """
    splits = {key: deepcopy(template[key]) for key in ("training", "validation")}
    design = []
    for key, split in splits.items():
        for row in split["rows"]:
            states = reference(row, TRUTH)
            row["targets"]["y"] = states[:, 0].tolist()
            if key == "training":
                y, z = states.T
                u = np.array(row["external_inputs"]["u"])
                ydd = (
                    -TRUTH["a"] * z
                    - TRUTH["b"] * y / (1 + TRUTH["q"] * y * y)
                    + TRUTH["c"] * u
                )
                design.append(
                    np.column_stack([-z, u, -y, -y * y * ydd, -y * y * z, y * y * u])
                )
        split["fingerprint"] = public.content_sha256(split["rows"])
    matrix = np.vstack(design)
    norms = np.linalg.norm(matrix, axis=0)
    singular = np.linalg.svd(matrix / np.maximum(norms, 1e-30), compute_uv=False)
    ratio = float(singular[-1] / singular[0])
    return {
        **splits,
        "reference_parameters": dict(TRUTH),
        "public_contract": (
            "y'=z; z'=-a*z-b*y/(1+q*y^2)+c*u; a,b,c,q positive, "
            "z0 signed shared; observe y only."
        ),
        "identifiability": {
            "passed": bool(np.all(norms > 0) and ratio > 1e-4),
            "argument": "full_rank_lifted_regression_implies_unique_a_b_c_q_and_z0",
            "columns": [
                "-y_prime",
                "u",
                "-y",
                "-y_squared_y_second",
                "-y_squared_y_prime",
                "y_squared_u",
            ],
            "column_normalized_singular_values": singular.tolist(),
            "singular_ratio": ratio,
            "scope": (
                "conditional ideal-trajectory uniqueness; "
                "finite noisy data need separate qualification"
            ),
            "derivatives_or_hidden_labels_given_to_fitter": False,
        },
    }
