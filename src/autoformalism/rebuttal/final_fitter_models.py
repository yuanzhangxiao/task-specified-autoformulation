"""Frozen public-information model alternative and ordinary parameter starts."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.attainability_controls import system_for
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.staged_topology import content_hash


def smaller_problem(splits: dict) -> dict:
    """Three signed states: input memory, recovery feedback and observed output.

    Only training observations determine centering and scale. The structure is
    hand specified before fitting; no reference equations, parameters or hidden
    boundaries enter this builder. Initial maps use the initial observed output.
    """
    public = deepcopy(splits)
    for split in public.values():
        for row in split["rows"]:
            row["fixed_covariates"] = {}
        split["fingerprint"] = content_hash(["public-boundaries", split["rows"]])
    training = unpack_split(public["train"])
    values = np.concatenate([r.targets["v01"] for r in training.trajectories])
    center = float(np.mean(values))
    scale = TrainingScaler().fit(training).scales["target:v01"].standard_deviation
    inputs = np.concatenate([r.external_inputs["u01"] for r in training.trajectories])
    input_scale = max(float(np.std(inputs)), 1.0)
    z = f"((v01-({center!r}))/{scale!r})"
    u = f"(u01/{input_scale!r})"
    equations = {
        "memory": f"-rate_m*memory+gain*{u}",
        "feedback": f"omega*{z}-rate_f*feedback",
        "v01": (f"{scale!r}*(a*{z}-b*{z}**3-omega*feedback+h*memory+d*{u}+c)"),
    }
    roles = {
        "rate_m": "rate",
        "gain": "coefficient",
        "omega": "rate",
        "rate_f": "rate",
        "a": "coefficient",
        "b": "rate",
        "h": "coefficient",
        "d": "coefficient",
        "c": "coefficient",
    }
    # A data-derived frequency is a starting guess, never a fixed hidden label.
    frequencies = []
    for row in training.trajectories:
        dt = np.diff(row.time)
        y = row.targets["v01"] - np.mean(row.targets["v01"])
        if len(y) >= 8 and np.std(y) > 1e-8 and np.allclose(dt, dt[0]):
            spectrum = abs(np.fft.rfft(y))
            k = 1 + int(np.argmax(spectrum[1:]))
            frequencies.append(float(np.fft.rfftfreq(len(y), dt[0])[k]))
    span = float(np.median([r.time[-1] - r.time[0] for r in training.trajectories]))
    omega = 2 * np.pi * (float(np.median(frequencies)) if frequencies else 1 / span)
    start = {
        "rate_m": omega / 2,
        "gain": omega,
        "omega": omega,
        "rate_f": omega / 10,
        "a": omega / 2,
        "b": omega / 3,
        "h": omega / 2,
        "d": 0.0,
        "c": 0.0,
    }
    p = {
        "candidate": {
            "candidate_id": "final_three_state_memory_oscillator",
            "parent_candidate_id": None,
            "states": [
                {"name": n, "kind": "observed" if n == "v01" else "latent"}
                for n in equations
            ],
            "state_equations": [
                {"state": n, "rhs": rhs} for n, rhs in equations.items()
            ],
            "observation_mappings": [{"channel": "v01", "expression": "v01"}],
            "parameters": [
                {"name": n, "scope": "global", "role": role}
                for n, role in roles.items()
            ],
            "initial_conditions": [
                {
                    "state": n,
                    "scope": "global",
                    **({"expression": "v01"} if n == "v01" else {"fixed_value": 0.0}),
                }
                for n in equations
            ],
        },
        "context": ValidationContext(
            targets=("v01",), external_inputs=("u01",)
        ).model_dump(mode="json"),
        "splits": public,
        "initialization_plan": {
            "rules": {
                n: {
                    "initial": {
                        "mode": "map",
                        "expression": f"offset+slope*{z}",
                        "parameters": [
                            {
                                "name": "offset",
                                "guess": 0.1 if n == "feedback" else 0.0,
                            },
                            {"name": "slope", "guess": 0.0},
                        ],
                    }
                }
                for n in ("memory", "feedback")
            }
        },
        "start": start,
        "design": {
            "center": center,
            "output_scale": scale,
            "input_scale": input_scale,
            "angular_frequency_guess": omega,
            "training_only": True,
            "scientific_status": "hand-specified hypothesis; no judge certification",
        },
    }
    system, guesses = system_for(p)
    p["start"].update(guesses)
    assert len(p["start"]) == len(system.names) == 13
    return p


def alternative_starts(problem: dict, seed: int) -> list[dict]:
    """Three deterministic, reference-independent points in the physical domain."""
    base = problem["start"]
    system, _ = system_for(problem)
    roles = {p.name: p.role.value for p in system.model.validated.candidate.parameters}
    rng = np.random.default_rng(seed)
    random = {}
    for name in sorted(base):
        value = base[name]
        random[name] = (
            float(max(abs(value), 0.1) * np.exp(rng.normal(0, 1)))
            if roles[name] in {"rate", "nonnegative_coefficient"}
            else float(value + max(abs(value), 0.3) * rng.normal())
        )
    slower = {n: float(v * 0.1) for n, v in base.items()}
    return [dict(base), random, slower]


def training_prefix(problem: dict, fraction: float) -> dict:
    """Keep an original-time prefix of every training row, with unchanged x(0)."""
    if not 0 < fraction <= 1:
        raise ValueError("prefix fraction must be in (0, 1]")
    payload = deepcopy(problem["splits"]["train"])
    for row in payload["rows"]:
        times = np.asarray(row["time"])
        end = times[0] + fraction * (times[-1] - times[0])
        count = max(2, int(np.searchsorted(times, end, side="right")))
        row["time"] = row["time"][:count]
        for key in ("targets", "auxiliaries", "external_inputs"):
            row[key] = {n: v[:count] for n, v in row[key].items()}
    payload["fingerprint"] = content_hash(
        ["training-prefix", fraction, payload["rows"]]
    )
    return payload
