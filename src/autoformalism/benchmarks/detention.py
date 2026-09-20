"""Engineering-grounded development generator, never a proposer input module.

Two level-pool basins with free overflow represented by a surveyed-style rating
table based on Q=K*max(h-crest, 0)**1.5. Linear interpolation of that table is the
exact v1 law (not an undisclosed smoothing of the weir). Units are m and minutes.
This module creates train/val only; a held-out release is deliberately absent.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

Case = Literal["coupled", "independent"]
TRUTH = {"k_transfer": 72.0, "k_outlet": 54.0}
GEOMETRY = {
    "area_up": 800.0,
    "area_down": 1200.0,
    "crest_up": 0.35,
    "crest_down": 0.12,
    "warning_depth": 1.0,
}
# The upstream floor is five metres above the downstream floor. Downstream
# water remains below the upper crest; receiving-channel stage is below the
# lower crest. Vertical-sided lined detention cells are the level-pool idealization.
BANK_DEPTH = 4.0
RATING_HEADS = np.array([0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4])
RATING_VALUES = RATING_HEADS**1.5


def rating(head: float) -> float:
    """Free-discharge rating, zero below crest; no extrapolation above the table."""
    if head > RATING_HEADS[-1]:
        raise ValueError("head exceeds rating table")
    return float(np.interp(max(head, 0), RATING_HEADS, RATING_VALUES))


def rating_expression(head: str) -> str:
    """Exactly equivalent hinge expansion in the frozen expression grammar."""
    slopes = np.diff(RATING_VALUES) / np.diff(RATING_HEADS)
    weights = np.diff(np.r_[0, slopes])
    return " + ".join(
        f"{weight:.17g} * max(({head}) - {knot:.17g}, 0)"
        for knot, weight in zip(RATING_HEADS[:-1], weights, strict=True)
    )


class Start(StrictSchema):
    """Frozen broad guesses, not generated from reference coefficients."""

    k_transfer: float = Field(gt=0, allow_inf_nan=False)
    k_outlet: float = Field(gt=0, allow_inf_nan=False)


class DetentionConfig(StrictSchema):
    """A small fixed release design; altered experiments need a new version."""

    protocol: Literal["detention-development-1"] = "detention-development-1"
    seed: int = Field(default=20260919, ge=0, le=2**32 - 1)
    duration_minutes: Literal[240] = 240
    sample_minutes: Literal[4] = 4
    training_trajectories: Literal[6] = 6
    validation_trajectories: Literal[3] = 3
    noise_fractions: tuple[float, ...] = (0.0, 0.01)
    starts: tuple[Start, ...] = (
        Start(k_transfer=30, k_outlet=30),
        Start(k_transfer=120, k_outlet=100),
    )
    fit_profile: Literal["collocation-single-target-v2"] = (
        "collocation-single-target-v2"
    )

    @model_validator(mode="after")
    def frozen_matrix(self):
        if self.noise_fractions != (0.0, 0.01) or len(self.starts) != 2:
            raise ValueError("v1 requires two noise levels and two starts")
        return self


def forcing(config: DetentionConfig, split: str, index: int) -> dict:
    """Deterministic, independently excited measured storm hydrographs."""
    if split not in {"train", "val"}:
        raise ValueError("only development train/val generation is permitted")
    count = config.training_trajectories if split == "train" else 3
    if index not in range(count):
        raise ValueError("trajectory index outside release")
    time = np.arange(0, config.duration_minutes + 1, config.sample_minutes, dtype=float)

    def pulse(center, width, peak):
        return peak * np.maximum(1 - np.abs(time - center) / width, 0)

    # Training: isolated/paired upstream storms, local-only forcing, coincident
    # storms and a zero-input recession. Validation uses unseen timing/overlap.
    designs = (
        [
            (36, 28, 70, 100, 24, 18),
            (52, 36, 100, 24, 20, 35),
            (28, 20, 55, 92, 36, 50),
            (72, 48, 0, 56, 40, 75),
            (40, 32, 95, 40, 32, 45),
            (40, 32, 0, 80, 24, 0),
        ]
        if split == "train"
        else [
            (64, 40, 85, 112, 36, 25),
            (100, 48, 105, 44, 28, 55),
            (36, 28, 65, 72, 48, 40),
        ]
    )
    a, b, c, d, e, f = designs[index]
    up, down = pulse(a, b, c), pulse(d, e, f)
    if index == 2:
        up = up + pulse(112 if split == "train" else 100, 24, 65)
    initial_up = [0.2, 0.55, 0.4, 0.35, 0.7, 1.1][index]
    initial_down = [0.15, 0.25, 0.18, 0.2, 0.3, 0.9][index]
    return {
        "time": time,
        "inflow_up": up,
        "inflow_down": down,
        "initial_up": initial_up,
        "initial_down": initial_down,
    }


def reference_rollout(
    row: dict,
    case: Case,
    *,
    method: str = "DOP853",
    parameters: dict | None = None,
) -> dict:
    """Independent reference integration with flow integrals and knot restarts."""
    if case not in {"coupled", "independent"}:
        raise ValueError("unknown basin case")
    parameters = TRUTH if parameters is None else parameters
    time = row["time"]
    state = np.array([row["initial_up"], row["initial_down"], 0.0, 0.0])
    columns = [state.copy()]

    def rhs(t, x):
        q = parameters["k_transfer"] * rating(x[0] - GEOMETRY["crest_up"])
        out = parameters["k_outlet"] * rating(x[1] - GEOMETRY["crest_down"])
        i1 = np.interp(t, time, row["inflow_up"])
        i2 = np.interp(t, time, row["inflow_down"])
        return [
            (i1 - q) / GEOMETRY["area_up"],
            (i2 + (q if case == "coupled" else 0) - out) / GEOMETRY["area_down"],
            q,
            out,
        ]

    for left, right in pairwise(time):
        solution = solve_ivp(
            rhs,
            (left, right),
            state,
            method=method,
            rtol=1e-11,
            atol=1e-13,
            max_step=1.0,
            t_eval=[right],
        )
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError(f"reference integration failed: {solution.message}")
        state = solution.y[:, -1]
        columns.append(state.copy())
    values = np.asarray(columns).T
    depths = values[:2]
    if depths.min() < -1e-9 or depths.max() >= BANK_DEPTH:
        raise ValueError("reference left nonnegative, non-overtopping operating range")
    inflow = row["inflow_up"] + row["inflow_down"]
    integral = np.r_[0, np.cumsum(np.diff(time) * (inflow[:-1] + inflow[1:]) / 2)]
    storage = GEOMETRY["area_up"] * depths[0] + GEOMETRY["area_down"] * depths[1]
    release = values[3] + (values[2] if case == "independent" else 0)
    error = storage - storage[0] - integral + release
    return {
        "states": depths.tolist(),
        "transfer_integral": values[2].tolist(),
        "outlet_integral": values[3].tolist(),
        "maximum_balance_error_m3": float(np.max(np.abs(error))),
        "maximum_relative_balance_error": float(
            np.max(np.abs(error)) / max(1, integral[-1])
        ),
    }


def generate_case(config: DetentionConfig, case: Case) -> tuple[dict, dict]:
    """Separate proposer-visible development arrays from evaluator-only labels."""
    public, private = {}, {"case": case, "truth": TRUTH, "rows": {}}
    clean_rows = {}
    for split in ("train", "val"):
        rows = []
        count = config.training_trajectories if split == "train" else 3
        for index in range(count):
            data = forcing(config, split, index)
            ref = reference_rollout(data, case)
            check = reference_rollout(data, case, method="Radau")
            agreement = float(
                np.max(np.abs(np.asarray(ref["states"]) - np.asarray(check["states"])))
            )
            name = f"{split}_{index:03d}"
            private["rows"][name] = {**ref, "solver_depth_difference_m": agreement}
            if agreement > 1e-7 or ref["maximum_relative_balance_error"] > 1e-8:
                raise ValueError("reference solver/conservation audit failed")
            rows.append(
                {
                    "trajectory_id": name,
                    "time": data["time"].tolist(),
                    "targets": {"h_down": ref["states"][1]},
                    "external_inputs": {
                        key: data[key].tolist() for key in ("inflow_up", "inflow_down")
                    },
                    "fixed_covariates": {**GEOMETRY, "initial_up": data["initial_up"]},
                }
            )
        clean_rows[split] = rows
    scale = float(np.std([r["targets"]["h_down"] for r in clean_rows["train"]]))
    private["clean_training_sd_m"] = scale
    for noise_index, fraction in enumerate(config.noise_fractions):
        public[f"noise{noise_index}"] = {}
        for split_index, split in enumerate(("train", "val")):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        config.seed,
                        0 if case == "coupled" else 1,
                        noise_index,
                        split_index,
                    ]
                )
            )
            rows = []
            for row in clean_rows[split]:
                clean = np.asarray(row["targets"]["h_down"])
                noise = rng.normal(0, fraction * scale, clean.size)
                noise[0] = 0  # calibrated initial reading, stated in release metadata
                observed = clean + noise
                rows.append({**row, "targets": {"h_down": observed.tolist()}})
            public[f"noise{noise_index}"][split] = PublicSplit.model_validate(
                {
                    "name": split,
                    "fingerprint": content_sha256(rows),
                    "rows": rows,
                }
            ).model_dump(mode="json")
    return public, private


def reference_request(
    case: Case, start: Start, *, inline: bool = False
) -> PublicFitRequest:
    """Oracle skeleton for fitter diagnostics ONLY; never a discovery prompt."""
    transfer = f"k_transfer * ({rating_expression('h_up - crest_up')})"
    outlet = f"k_outlet * ({rating_expression('h_down - crest_down')})"
    coupled = case == "coupled"
    if case not in {"coupled", "independent"}:
        raise ValueError("unknown basin case")
    names = ["h_up", "h_down"] if coupled else ["h_down"]
    processes = [{"name": "q_outlet", "expression": outlet}]
    equations = [{"state": "h_down", "rhs": "(inflow_down - q_outlet)/area_down"}]
    if coupled:
        processes.append({"name": "q_transfer", "expression": transfer})
        equations = [
            {"state": "h_up", "rhs": "(inflow_up - q_transfer)/area_up"},
            {
                "state": "h_down",
                "rhs": "(inflow_down + q_transfer - q_outlet)/area_down",
            },
        ]
    if inline:
        for equation in equations:
            equation["rhs"] = equation["rhs"].replace("q_transfer", f"({transfer})")
            equation["rhs"] = equation["rhs"].replace("q_outlet", f"({outlet})")
        processes = []
    parameters = ["k_transfer", "k_outlet"] if coupled else ["k_outlet"]
    candidate = {
        "candidate_id": f"detention_{case}_{'inline' if inline else 'shared'}",
        "parent_candidate_id": None,
        "states": [
            {"name": n, "kind": "observed" if n == "h_down" else "latent", "unit": "m"}
            for n in names
        ],
        "state_equations": equations,
        "processes": processes,
        "parameters": [
            {"name": n, "scope": "global", "role": "nonnegative_coefficient"}
            for n in parameters
        ],
        "observation_mappings": [{"channel": "h_down", "expression": "h_down"}],
        "initial_conditions": [
            {"state": n, "scope": "global", "fixed_value": 0} for n in names
        ],
    }
    return PublicFitRequest.model_validate(
        {
            "base_candidate": candidate,
            "context": {
                "targets": ["h_down"],
                "external_inputs": ["inflow_up", "inflow_down"],
                "fixed_covariates": [*GEOMETRY, "initial_up"],
            },
            "initialization_plan": {
                "rules": {
                    "h_up": {
                        "initial": {
                            "mode": "map",
                            "expression": "initial_up",
                            "parameters": [],
                        }
                    }
                }
                if coupled
                else {}
            },
            "parameter_guesses": {n: getattr(start, n) for n in parameters},
            "profile": "collocation-single-target-v2",
            "source": {
                "stage": "synthetic_control",
                "task_id": candidate["candidate_id"],
                "artifact_sha256": content_sha256(candidate),
            },
        }
    )
