"""Small M34 analytic basin controls; no benchmark or production-data dependency.

The public problem contains noisy training readings and known acquisition facts.
Generating coefficients, clean outputs and analytic optima live only in a separate
post-fit evaluator record. These fixtures cannot calibrate scientific confidence.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.measurement_fitting import MeasurementProblem

CASES = (
    "exact_varied",
    "shared_initial",
    "active_gain_bound",
    "missing_readings",
    "nonlinear_decay",
    "multi_output",
)
SEEDS = (0, 1, 2)
INPUT_PROTOCOL = "phase-c-measurement-fitting-inputs-1"


def response(time: float, decay: float, forcing: float) -> tuple[float, float]:
    """Return exact initial and gain coefficients for x' = g*u - k*x."""
    initial = float(np.exp(-decay * time))
    gain = float(forcing * -np.expm1(-decay * time) / decay)
    return initial, gain


def _parameter(name: str, lower: float, upper: float) -> dict:
    """Supply predeclared operational domains, independent of noisy endpoints."""
    return {
        "name": name,
        "scope": "global",
        "role": "nonnegative_coefficient",
        "bounds": {"lower": lower, "upper": upper},
    }


def _build(case: str, seed: int) -> tuple[MeasurementProblem, dict, dict]:
    """Create one deterministic acquisition, its evaluator and exact source bytes."""
    if case not in CASES or seed not in SEEDS:
        raise ValueError("unknown measurement fixture case/noise seed")
    rng = np.random.default_rng(20261014 + 100 * CASES.index(case) + seed)
    shared, nonlinear = case == "shared_initial", case == "nonlinear_decay"
    multi = case == "multi_output"
    targets = ("y", "z") if multi else ("y",)
    true_gain, true_decay = (2.4 if case == "active_gain_bound" else 1.3), 0.65
    gain_upper = 0.8 if case == "active_gain_bound" else 3.0
    parameters = [_parameter("g", 0.05, gain_upper)]
    start = {"g": 0.45}
    if shared:
        parameters.append(_parameter("x0", 0.01, 2.0))
        start["x0"] = 0.35
    if nonlinear:
        parameters.append(_parameter("k", 0.1, 1.5))
        start["k"] = 1.05
    rows, acquisition, observations, noise_ids, noise_sigma = [], [], [], [], []
    clocks, forcings, boundaries, truth_initials = [], [], [], {}
    times = np.linspace(0.0, 6.0, 25)
    for j, forcing in enumerate((0.4, 1.1)):
        trajectory = f"train_{j}"
        initial = 0.7 if shared else (0.3, 1.1)[j]
        truth_initials[trajectory] = initial
        raw_clocks = [float(100 + 60 * time) for time in times]
        readings = []
        for i, (time, clock) in enumerate(zip(times, raw_clocks, strict=True)):
            a, b = response(float(time), true_decay, forcing)
            clean = a * initial + b * true_gain
            for channel in (*targets, "u"):
                reading_id = f"{trajectory}_{channel}_{i}"
                missing = (
                    case == "missing_readings"
                    and channel in targets
                    and i in (0, 4 + j, 10 + j, 21)
                )
                multiplier = 2.0 if channel == "z" else 1.0
                sigma = 0.03 * (1 + 0.2 * j + (i % 4) / 10) * multiplier
                value = (
                    forcing
                    if channel == "u"
                    else multiplier * clean + float(rng.normal(0, sigma))
                )
                raw_value = (
                    None
                    if missing
                    else value
                    if channel == "u"
                    else (value - 0.03) / 0.01
                )
                source_row = j * len(times) + i
                record = {
                    "id": reading_id,
                    "trajectory": trajectory,
                    "source_row": source_row,
                    "channel": channel,
                    "clock_value": clock,
                    "raw_value": raw_value,
                    "condition": "missing" if missing else "present",
                }
                acquisition.append(record)
                readings.append({k: v for k, v in record.items() if k != "trajectory"})
                if channel in targets and not missing:
                    noise_ids.append(reading_id)
                    noise_sigma.append(sigma)
                    observations.append(
                        {
                            "reading_id": reading_id,
                            "trajectory": trajectory,
                            "time": float(time),
                            "value": 0.01 * raw_value + 0.03,
                            "sigma": sigma,
                            "multiplier": multiplier,
                            "forcing": forcing,
                            "initial": initial,
                            "clean": multiplier * clean,
                        }
                    )
        rows.append(
            {
                "trajectory_id": trajectory,
                "clock_values": raw_clocks,
                "readings": readings,
            }
        )
        clocks.append(
            {
                "trajectory": trajectory,
                "raw_unit": "s",
                "analysis_unit": "min",
                "origin": 100,
                "factor": 1 / 60,
                "assumption": "clock_known",
            }
        )
        forcings.append(
            {
                "trajectory": trajectory,
                "channel": "u",
                "knot_reading_ids": [f"{trajectory}_u_{i}" for i in range(len(times))],
                "assumption": "input_known",
            }
        )
        boundaries.append(
            {
                "trajectory": trajectory,
                "value": initial,
                "preparation_source": "preparation",
            }
        )
    raw_artifact = {"parser_protocol": "m34_acquisition_json_1", "records": acquisition}
    raw_sha = public.content_sha256(raw_artifact)
    for row in rows:
        for reading in row["readings"]:
            reading["artifact_sha256"] = raw_sha
    preparation = (
        {"shared_initial_protocol": "same unknown preparation for both trajectories"}
        if shared
        else {"exact_initial_boundaries": boundaries}
    )
    calibration = {
        "target_gain": 0.01,
        "target_offset": 0.03,
        "sigma": noise_sigma,
        "reading_ids": noise_ids,
        "noise_protocol": "independent known synthetic Gaussian in analysis units",
    }
    sources = [
        {"id": "acquisition", "artifact_sha256": raw_sha, "kind": "acquisition"},
        {
            "id": "preparation",
            "artifact_sha256": public.content_sha256(preparation),
            "kind": "public_preparation",
        },
        {
            "id": "calibration",
            "artifact_sha256": public.content_sha256(calibration),
            "kind": "calibration",
        },
    ]
    assumptions = [
        {
            "id": f"{topic}_known",
            "topic": actual,
            "status": "known",
            "sources": [source],
            "reason": reason,
        }
        for topic, actual, source, reason in (
            ("clock", "time", "acquisition", "Exact synthetic acquisition clock."),
            ("input", "input", "acquisition", "Exact constant forcing at every knot."),
            (
                "calibration",
                "calibration",
                "calibration",
                "Declared exact affine units.",
            ),
            (
                "noise",
                "noise",
                "calibration",
                "Known generating SD, never residual estimated.",
            ),
            (
                "independence",
                "independence",
                "calibration",
                "Independent draws per reading.",
            ),
            (
                "initial",
                "initial",
                "preparation",
                "Independent exact/shared preparation.",
            ),
        )
    ]
    channels = [
        {
            "id": channel,
            "role": "target",
            "raw_unit": "cm",
            "analysis_unit": "m",
            "gain": 0.01,
            "offset": 0.03,
            "calibration_assumption": "calibration_known",
        }
        for channel in targets
    ]
    channels.append(
        {
            "id": "u",
            "role": "external_input",
            "raw_unit": "m/min",
            "analysis_unit": "m/min",
            "gain": 1,
            "offset": 0,
            "calibration_assumption": "calibration_known",
            "exact_input_assumption": "input_known",
        }
    )
    initial_rule = {
        "state": "x",
        "mode": "shared_value" if shared else "exact_boundary",
        "assumption": "initial_known",
        **({"shared_coordinate": "x0"} if shared else {"exact_boundaries": boundaries}),
    }
    request = {
        "base_candidate": {
            "candidate_id": f"m34_{case}",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "observed", "unit": "m"}],
            "state_equations": [
                {"state": "x", "rhs": "g*u-k*x" if nonlinear else "g*u-0.65*x"}
            ],
            "observation_mappings": [
                {"channel": channel, "expression": "2*x" if channel == "z" else "x"}
                for channel in targets
            ],
            "parameters": parameters,
        },
        "context": {"targets": targets, "external_inputs": ["u"]},
        "initialization_plan": {"rules": {}},
        "parameter_guesses": start,
        "profile": "general-rollout-v1",
        "random_seed": 20261014,
        "source": {
            "stage": "synthetic_control",
            "task_id": f"m34_{case}",
            "artifact_sha256": public.content_sha256(
                {"equations": "x'=g*u-k*x", "case": case, "version": 1}
            ),
        },
    }
    problem = MeasurementProblem.model_validate(
        {
            "request": request,
            "manifest": {
                "sources": sources,
                "assumptions": assumptions,
                "artifacts": [
                    {
                        "sha256": raw_sha,
                        "media_type": "application/json",
                        "acquisition_source": "acquisition",
                        "parser_protocol": "m34_acquisition_json_1",
                    }
                ],
                "channels": channels,
                "time_bases": clocks,
                "forcing_rules": forcings,
                "noise": {
                    "reading_ids": noise_ids,
                    "sigma": noise_sigma,
                    "scale_assumption": "noise_known",
                    "independence_assumption": "independence_known",
                },
            },
            "training": {"rows": rows},
            "initial_rules": [initial_rule],
            "coordinates": {
                "parameters": {name: {"center": 0, "scale": 1} for name in start},
                "states": {"x": {"center": 0, "scale": 1}},
                "provenance": {"policy": "fixed experimental metre/minute coordinates"},
            },
            "start": start,
            "incumbent": start,
        }
    )
    evaluator = {
        "case": case,
        "noise_seed": seed,
        "truth": {"g": true_gain, "k": true_decay, "initials": truth_initials},
        "observations": observations,
        "parameter_names": list(start),
        "gain_bounds": [0.05, gain_upper],
        "shared_initial": shared,
        "nonlinear_decay": nonlinear,
        "analytic_objective": None,
        "analytic_parameters": None,
    }
    if not nonlinear:
        optimum = analytic_optimum(evaluator)
        evaluator.update(optimum)
    artifacts = {
        raw_sha: raw_artifact,
        public.content_sha256(preparation): preparation,
        public.content_sha256(calibration): calibration,
    }
    return problem, evaluator, artifacts


def make(case: str = "exact_varied", seed: int = 0) -> tuple[MeasurementProblem, dict]:
    """Return training-only problem and separate evaluator for one tiny control."""
    problem, evaluator, _ = _build(case, seed)
    return problem, evaluator


def make_inputs() -> dict:
    """Generate all 18 deterministic acquisitions without fitting or held-out data."""
    problems, evaluation, artifacts = {}, {}, {}
    for case in CASES:
        for seed in SEEDS:
            problem, evaluator, raw = _build(case, seed)
            name = f"{case}_s{seed}"
            problems[name] = problem.model_dump(mode="json")
            evaluation[name] = evaluator
            artifacts.update(raw)
    return {
        "protocol": INPUT_PROTOCOL,
        "problems": problems,
        "evaluation": evaluation,
        "raw_artifacts": artifacts,
        "test_data_opened": False,
    }


def verify_raw_artifacts(data: dict) -> None:
    """Verify retained source content and every acquisition record, without fitting."""
    artifacts = data["raw_artifacts"]
    if any(public.content_sha256(value) != key for key, value in artifacts.items()):
        raise ValueError("measurement raw artifact content differs")
    for payload in data["problems"].values():
        for source in payload["manifest"]["sources"]:
            if source["artifact_sha256"] not in artifacts:
                raise ValueError("measurement source artifact is missing")
        acquisition = {}
        for artifact in payload["manifest"]["artifacts"]:
            digest = artifact["sha256"]
            acquisition[digest] = {
                row["id"]: row for row in artifacts[digest]["records"]
            }
        for row in payload["training"]["rows"]:
            for reading in row["readings"]:
                original = acquisition[reading["artifact_sha256"]].get(reading["id"])
                expected = {
                    key: value
                    for key, value in reading.items()
                    if key != "artifact_sha256"
                } | {"trajectory": row["trajectory_id"]}
                if original != expected:
                    raise ValueError("measurement reading differs from raw artifact")


def analytic_optimum(evaluator: dict) -> dict:
    """Solve the known-decay weighted linear controls independently of ODE code."""
    design, target = [], []
    shared = evaluator["shared_initial"]
    for row in evaluator["observations"]:
        a, b = response(row["time"], evaluator["truth"]["k"], row["forcing"])
        scale = row["multiplier"] / row["sigma"]
        design.append([scale * b, scale * a] if shared else [scale * b])
        target.append(
            row["value"] / row["sigma"] - (0 if shared else scale * a * row["initial"])
        )
    matrix, rhs = np.array(design), np.array(target)
    optimum = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    optimum[0] = np.clip(optimum[0], *evaluator["gain_bounds"])
    if shared and not (0.01 < optimum[1] < 2 and 0.05 < optimum[0] < 3):
        raise ValueError("shared analytic control must have an interior optimum")
    return {
        "analytic_parameters": {
            name: float(value)
            for name, value in zip(evaluator["parameter_names"], optimum, strict=True)
        },
        "analytic_objective": float(0.5 * np.sum((matrix @ optimum - rhs) ** 2)),
        "analytic_design_rank": int(np.linalg.matrix_rank(matrix)),
        "analytic_active_gain_bound": bool(optimum[0] in evaluator["gain_bounds"]),
    }


def evaluate(evaluator: dict, parameters: Mapping[str, float] | None) -> dict:
    """Evaluate only after endpoint selection; noisy truth is never a fit gate."""
    if parameters is None:
        return {"status": "no_parameters", "analytic_objective_passed": None}
    if set(parameters) != set(evaluator["parameter_names"]):
        raise ValueError("evaluator parameter names differ")
    residuals, errors = [], []
    for row in evaluator["observations"]:
        decay = parameters.get("k", evaluator["truth"]["k"])
        a, b = response(row["time"], decay, row["forcing"])
        initial = parameters.get("x0", row["initial"])
        prediction = row["multiplier"] * (a * initial + b * parameters["g"])
        residuals.append((prediction - row["value"]) / row["sigma"])
        errors.append(prediction - row["clean"])
    objective = float(0.5 * np.dot(residuals, residuals))
    analytic = evaluator["analytic_objective"]
    gap = objective - analytic if analytic is not None else None
    return {
        "status": "evaluated",
        "independent_analytic_endpoint_objective": objective,
        "analytic_optimum_objective": analytic,
        "analytic_objective_gap": gap,
        "analytic_objective_passed": (
            abs(gap) <= 1e-5 * (1 + abs(analytic)) if gap is not None else None
        ),
        "analytic_parameters": evaluator["analytic_parameters"],
        "clean_prediction_rmse_evaluator_only": float(
            np.sqrt(np.mean(np.square(errors)))
        ),
        "parameter_truth_error_evaluator_only": {
            name: float(parameters[name] - evaluator["truth"][name])
            for name in ("g", "k")
            if name in parameters
        },
        "noise_free_recovery_assessed": False,
        "confidence_probability": None,
    }
