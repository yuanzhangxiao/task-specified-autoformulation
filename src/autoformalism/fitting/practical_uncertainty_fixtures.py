"""M39 paired known-scale acquisitions with evaluator-only generating truth."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.fitting import measurement_fixtures as templates
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.measurement_fitting import analysis_layout
from autoformalism.schemas.sensor_drift import DriftProblem

PARAMETERS = ("g", "x0", "sensor_drift")
SEPARATIONS = (0.5, 0.1, 0.02, 0.005, 0.0)
SIGMAS = (0.01, 0.05, 0.2)
SEEDS = (0, 1, 2, 3, 4)
INPUT_PROTOCOL = "phase-c-practical-uncertainty-fitting-inputs-1"


def case_key(separation: float, sigma: float, seed: int) -> str:
    """Give each frozen acquisition a portable, deterministic identifier."""
    return f"d{separation:g}_sigma{sigma:g}_s{seed}".replace(".", "p")


def _build(
    separation: float, sigma: float, seed: int
) -> tuple[DriftProblem, dict, dict]:
    """Build three acquisitions from a public template, with matched noise draws."""
    if separation not in SEPARATIONS or sigma not in SIGMAS or seed not in SEEDS:
        raise ValueError("unknown practical uncertainty acquisition")
    template, _, old_artifacts = templates._build("shared_initial", 0)
    payload = template.model_dump(mode="json")
    truth = {"g": 1.3, "x0": 0.4, "sensor_drift": 0.06}
    noise = np.random.default_rng(20261015 + seed).standard_normal((3, 17))
    times = np.linspace(0, 4, 17)
    rows, records, observations, ids = [], [], [], []
    clocks, forcings = [], []
    for j, multiplier in enumerate((-1, 0, 1)):
        trajectory = f"train_{j}"
        forcing = 1.0 + separation * multiplier
        readings, raw_clocks = [], [float(100 + 60 * t) for t in times]
        for i, (time, clock) in enumerate(zip(times, raw_clocks, strict=True)):
            physical = truth["x0"] + truth["g"] * forcing * time
            measured = physical + truth["sensor_drift"] * time
            value = float(measured + sigma * noise[j, i])
            reading_id = f"{trajectory}_y_{i}"
            ids.append(reading_id)
            raw_value = (value - 0.03) / 0.01
            observations.append(
                {
                    "reading_id": reading_id,
                    "trajectory": trajectory,
                    "time": float(time),
                    "forcing": forcing,
                    "value": 0.01 * raw_value + 0.03,
                    "sigma": sigma,
                    "clean_physical": float(physical),
                    "clean_measured": float(measured),
                }
            )
            for channel, observed in (("y", raw_value), ("u", forcing)):
                record = {
                    "id": f"{trajectory}_{channel}_{i}",
                    "trajectory": trajectory,
                    "source_row": j * len(times) + i,
                    "channel": channel,
                    "clock_value": clock,
                    "raw_value": observed,
                    "condition": "present",
                }
                records.append(record)
                readings.append(
                    {key: value for key, value in record.items() if key != "trajectory"}
                )
        rows.append(
            {
                "trajectory_id": trajectory,
                "clock_values": raw_clocks,
                "readings": readings,
            }
        )
        clocks.append(
            deepcopy(payload["manifest"]["time_bases"][0]) | {"trajectory": trajectory}
        )
        forcings.append(
            deepcopy(payload["manifest"]["forcing_rules"][0])
            | {
                "trajectory": trajectory,
                "knot_reading_ids": [f"{trajectory}_u_{i}" for i in range(len(times))],
            }
        )
    raw = {"parser_protocol": "m39_acquisition_json_1", "records": records}
    raw_sha = public.content_sha256(raw)
    for row in rows:
        for reading in row["readings"]:
            reading["artifact_sha256"] = raw_sha
    calibration = {
        "target_gain": 0.01,
        "target_offset": 0.03,
        "sigma": [sigma] * len(ids),
        "reading_ids": ids,
        "noise_protocol": "Independent known synthetic Gaussian in analysis units.",
    }
    calibration_sha = public.content_sha256(calibration)
    drift_source = {
        "name": "sensor_drift",
        "channel": "y",
        "lower": -2.0,
        "upper": 2.0,
        "time_origin": 0.0,
        "unit": "m/min",
        "rationale": (
            "Shared linear observation drift with a predeclared broad "
            "operational domain."
        ),
    }
    drift_sha = public.content_sha256(drift_source)
    artifacts = {raw_sha: raw, calibration_sha: calibration, drift_sha: drift_source}
    for source in payload["manifest"]["sources"]:
        if source["id"] == "acquisition":
            source["artifact_sha256"] = raw_sha
        elif source["id"] == "calibration":
            source["artifact_sha256"] = calibration_sha
        elif source["id"] == "preparation":
            preparation = {
                "shared_initial_protocol": (
                    "same unknown preparation for all three trajectories"
                )
            }
            source["artifact_sha256"] = public.content_sha256(preparation)
            artifacts[source["artifact_sha256"]] = preparation
        else:
            artifacts[source["artifact_sha256"]] = old_artifacts[
                source["artifact_sha256"]
            ]
    payload["manifest"]["sources"].append(
        {"id": "drift_protocol", "artifact_sha256": drift_sha, "kind": "calibration"}
    )
    payload["manifest"]["artifacts"] = [
        {
            "sha256": raw_sha,
            "media_type": "application/json",
            "acquisition_source": "acquisition",
            "parser_protocol": "m39_acquisition_json_1",
        }
    ]
    payload["manifest"]["noise"]["sigma"] = [sigma] * len(ids)
    payload["manifest"]["noise"]["reading_ids"] = ids
    payload["manifest"]["time_bases"], payload["manifest"]["forcing_rules"] = (
        clocks,
        forcings,
    )
    payload["training"]["rows"] = rows
    candidate = payload["request"]["base_candidate"]
    candidate["candidate_id"] = "m39_fixed_physical_contract"
    candidate["state_equations"][0]["rhs"] = "g*u"
    payload["request"]["source"] = {
        "stage": "synthetic_control",
        "task_id": "m39_fixed_physical_contract",
        "artifact_sha256": public.content_sha256(
            {"protocol": "m39_fixed_candidate_1", "equation": "g*u"}
        ),
    }
    problem = DriftProblem.model_validate(
        {
            "base": payload,
            "drift": drift_source
            | {
                "start": 0.0,
                "incumbent": 0.0,
                "scale": 1.0,
                "sources": ["drift_protocol"],
            },
        }
    )
    evaluator = {
        "separation": separation,
        "sigma": sigma,
        "seed": seed,
        "acquisition_sha256": raw_sha,
        "observations": observations,
        "truth": truth,
        "parameter_names": list(PARAMETERS),
        "parameter_bounds": [[0.05, 3.0], [0.01, 2.0], [-2.0, 2.0]],
    }
    return problem, evaluator, artifacts


def make(
    separation: float = 0.5, sigma: float = 0.01, seed: int = 0
) -> tuple[DriftProblem, dict]:
    """Return one public contract and its isolated post-fit evaluator."""
    problem, evaluator, _ = _build(separation, sigma, seed)
    return problem, evaluator


def make_inputs() -> dict:
    """Freeze all 75 exact raw acquisitions once, without fitting or held-out data."""
    problems, evaluation, artifacts = {}, {}, {}
    for separation in SEPARATIONS:
        for sigma in SIGMAS:
            for seed in SEEDS:
                problem, evaluator, raw = _build(separation, sigma, seed)
                key = case_key(separation, sigma, seed)
                problems[key], evaluation[key] = (
                    problem.model_dump(mode="json"),
                    evaluator,
                )
                artifacts.update(raw)
    return {
        "protocol": INPUT_PROTOCOL,
        "problems": problems,
        "evaluation": evaluation,
        "raw_artifacts": artifacts,
        "test_data_opened": False,
    }


def verify_raw_artifacts(data: dict) -> None:
    """Bind the factorial design and evaluator observations to actual public bytes."""
    templates.verify_raw_artifacts(
        data
        | {"problems": {key: value["base"] for key, value in data["problems"].items()}}
    )
    for key, evaluator in data["evaluation"].items():
        separation, sigma, seed = (
            evaluator["separation"],
            evaluator["sigma"],
            evaluator["seed"],
        )
        if (
            separation not in SEPARATIONS
            or sigma not in SIGMAS
            or type(seed) is not int
            or seed not in SEEDS
            or key != case_key(separation, sigma, seed)
        ):
            raise ValueError("practical uncertainty declared acquisition differs")
        problem = DriftProblem.model_validate(data["problems"][key])
        acquisition = data["problems"][key]["base"]["manifest"]["artifacts"][0][
            "sha256"
        ]
        if evaluator["acquisition_sha256"] != acquisition:
            raise ValueError("practical uncertainty evaluator acquisition differs")
        _verify_public_design(problem, separation, sigma, data["raw_artifacts"])
        layout = analysis_layout(problem.base)
        expected = [
            {
                "reading_id": reading,
                "trajectory": trajectory,
                "time": time,
                "value": value,
                "sigma": scale,
                "forcing": 1 + separation * (int(trajectory[-1]) - 1),
            }
            for reading, trajectory, time, value, scale in zip(
                layout.reading_ids,
                layout.trajectory_ids,
                layout.analysis_times,
                layout.values,
                layout.sigma,
                strict=True,
            )
        ]
        if len(evaluator["observations"]) != len(expected) or any(
            {name: actual.get(name) for name in source} != source
            for actual, source in zip(evaluator["observations"], expected, strict=True)
        ):
            raise ValueError("practical uncertainty evaluator observation differs")
        if evaluator["parameter_names"] != list(PARAMETERS) or evaluator[
            "parameter_bounds"
        ] != [[0.05, 3.0], [0.01, 2.0], [-2.0, 2.0]]:
            raise ValueError("practical uncertainty evaluator parameter domain differs")


def _verify_public_design(
    problem: DriftProblem, separation: float, sigma: float, artifacts: dict
) -> None:
    """Check imported public design without redrawing data or consulting truth."""
    base, candidate = problem.base, problem.request.base_candidate
    domains = {
        parameter.name: (parameter.bounds.lower, parameter.bounds.upper)
        for parameter in candidate.parameters
    }
    if (
        domains != {"g": (0.05, 3.0), "x0": (0.01, 2.0)}
        or len(candidate.state_equations) != 1
        or candidate.state_equations[0].state != "x"
        or candidate.state_equations[0].rhs != "g*u"
        or len(candidate.observation_mappings) != 1
        or candidate.observation_mappings[0].channel != "y"
        or candidate.observation_mappings[0].expression != "x"
        or tuple(base.request.context.targets) != ("y",)
        or tuple(base.request.context.external_inputs) != ("u",)
        or dict(base.start) != {"g": 0.45, "x0": 0.35}
        or dict(base.incumbent) != dict(base.start)
    ):
        raise ValueError("practical uncertainty physical domain/start differs")
    drift = problem.drift
    if (
        drift.name,
        drift.channel,
        drift.lower,
        drift.upper,
        drift.start,
        drift.incumbent,
        drift.scale,
        drift.time_origin,
        drift.unit,
    ) != ("sensor_drift", "y", -2.0, 2.0, 0.0, 0.0, 1.0, 0.0, "m/min"):
        raise ValueError("practical uncertainty drift domain differs")
    if len(base.initial_rules) != 1 or (
        base.initial_rules[0].state,
        base.initial_rules[0].mode,
        base.initial_rules[0].shared_coordinate,
    ) != ("x", "shared_value", "x0"):
        raise ValueError("practical uncertainty shared initial contract differs")
    if any(
        axis.center != 0 or axis.scale != 1
        for axis in (
            *base.coordinates.parameters.values(),
            *base.coordinates.states.values(),
        )
    ):
        raise ValueError("practical uncertainty numerical coordinates differ")
    channels = {channel.id: channel for channel in base.manifest.channels}
    if (
        set(channels) != {"y", "u"}
        or (
            channels["y"].gain,
            channels["y"].offset,
            channels["y"].raw_unit,
            channels["y"].analysis_unit,
        )
        != (0.01, 0.03, "cm", "m")
        or (
            channels["u"].gain,
            channels["u"].offset,
            channels["u"].raw_unit,
            channels["u"].analysis_unit,
        )
        != (1.0, 0.0, "m/min", "m/min")
    ):
        raise ValueError("practical uncertainty calibration differs")
    times = tuple(float(value) for value in np.linspace(0, 4, 17))
    trajectories = tuple(f"train_{index}" for index in range(3))
    if tuple(row.trajectory_id for row in base.training.rows) != trajectories:
        raise ValueError("practical uncertainty trajectory design differs")
    for index, row in enumerate(base.training.rows):
        clock = next(
            clock
            for clock in base.manifest.time_bases
            if clock.trajectory == row.trajectory_id
        )
        if (clock.raw_unit, clock.analysis_unit, clock.origin, clock.factor) != (
            "s",
            "min",
            100,
            1 / 60,
        ) or tuple(
            clock.factor * (value - clock.origin) for value in row.clock_values
        ) != times:
            raise ValueError("practical uncertainty time design differs")
        input_readings = [reading for reading in row.readings if reading.channel == "u"]
        if len(input_readings) != 17 or any(
            reading.raw_value != 1 + separation * (index - 1)
            for reading in input_readings
        ):
            raise ValueError("practical uncertainty forcing design differs")
    layout = analysis_layout(base)
    if (
        not all(layout.included)
        or len(layout.reading_ids) != 51
        or set(layout.sigma) != {sigma}
    ):
        raise ValueError("practical uncertainty known sigma/mask differs")
    sources = {
        source.id: artifacts[source.artifact_sha256] for source in base.manifest.sources
    }
    if tuple(drift.sources) != ("drift_protocol",) or any(
        sources["drift_protocol"].get(field) != getattr(drift, field)
        for field in ("name", "channel", "lower", "upper", "time_origin", "unit")
    ):
        raise ValueError("practical uncertainty drift source differs")
    calibration = sources["calibration"]
    if (
        calibration["target_gain"] != channels["y"].gain
        or calibration["target_offset"] != channels["y"].offset
        or calibration["reading_ids"] != list(base.manifest.noise.reading_ids)
        or calibration["sigma"] != list(base.manifest.noise.sigma)
    ):
        raise ValueError("practical uncertainty calibration source differs")
    if sources["preparation"] != {
        "shared_initial_protocol": "same unknown preparation for all three trajectories"
    }:
        raise ValueError("practical uncertainty preparation source differs")


def _design(evaluator: dict) -> tuple[np.ndarray, np.ndarray]:
    """Derive the exact observation model using declared inputs and known sigma."""
    design = [
        [row["forcing"] * row["time"], 1.0, row["time"]]
        for row in evaluator["observations"]
    ]
    return np.array(design), np.array(
        [row["value"] for row in evaluator["observations"]]
    )


def analytic_optimum(evaluator: dict) -> dict:
    """Independently solve the bounded Gaussian optimum, without uniqueness claims."""
    matrix, target = _design(evaluator)
    sigma = np.array([row["sigma"] for row in evaluator["observations"]])
    matrix, target = matrix / sigma[:, None], target / sigma
    units = np.linalg.norm(matrix, axis=0)
    design = matrix / units
    bounds = np.array(evaluator["parameter_bounds"])
    lower, upper = bounds[:, 0] * units, bounds[:, 1] * units
    solution = lsq_linear(
        design, target, bounds=(lower, upper), method="bvls", tol=1e-12
    )
    residual = design @ solution.x - target
    gradient = design.T @ residual
    projected = solution.x - np.clip(solution.x - gradient, lower, upper)
    if not np.isfinite(solution.x).all() or np.max(abs(projected)) > 1e-5:
        raise ValueError("independent practical reference failed bounded KKT")
    rank = int(np.linalg.matrix_rank(design))
    return {
        "analytic_parameters": dict(
            zip(PARAMETERS, map(float, solution.x / units), strict=True)
        ),
        "analytic_objective": float(residual @ residual / 2),
        "analytic_design_rank": rank,
        "rank_deficient_optimum": rank < 3,
        "analytic_projected_gradient_maximum": float(np.max(abs(projected))),
        "unique_parameters_assessed": False,
        "analytic_method": "independent_exact_affine_BVLS_bounded_KKT",
    }


def evaluate(evaluator: dict, parameters: dict | None) -> dict:
    """Replay a sealed endpoint; generating values never rank fits or profiles."""
    if parameters is None:
        return {"status": "no_parameters", "analytic_objective_passed": None}
    if (
        set(parameters) != set(PARAMETERS)
        or not np.isfinite(list(parameters.values())).all()
    ):
        raise ValueError("practical uncertainty parameter names/values differ")
    matrix, values = _design(evaluator)
    prediction = matrix @ np.array([parameters[name] for name in PARAMETERS])
    rows = evaluator["observations"]
    residual = prediction - values
    standardized = residual / np.array([row["sigma"] for row in rows])
    objective = float(standardized @ standardized / 2)
    reference = analytic_optimum(evaluator)
    gap = objective - reference["analytic_objective"]
    physical = prediction - parameters["sensor_drift"] * np.array(
        [row["time"] for row in rows]
    )
    errors = {
        name: float(parameters[name] - evaluator["truth"][name]) for name in PARAMETERS
    }
    return {
        "status": "evaluated",
        "independent_analytic_endpoint_objective": objective,
        "analytic_optimum_objective": reference["analytic_objective"],
        "analytic_objective_gap": gap,
        "analytic_objective_passed": abs(gap)
        <= 1e-5 * (1 + abs(reference["analytic_objective"])),
        "analytic_reference": reference,
        "analytic_conditional_information": 187
        * evaluator["separation"] ** 2
        / evaluator["sigma"] ** 2,
        "analytic_frozen_information": 93.5
        * (3 + 2 * evaluator["separation"] ** 2)
        / evaluator["sigma"] ** 2,
        "parameter_truth_error_evaluator_only": errors,
        "absolute_parameter_truth_error_evaluator_only": {
            name: abs(value) for name, value in errors.items()
        },
        "clean_physical_prediction_rmse_evaluator_only": float(
            np.sqrt(np.mean((physical - [row["clean_physical"] for row in rows]) ** 2))
        ),
        "clean_measured_prediction_rmse_evaluator_only": float(
            np.sqrt(
                np.mean((prediction - [row["clean_measured"] for row in rows]) ** 2)
            )
        ),
        "analysis_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "analysis_residual_unit": "m",
        "standardized_residuals": standardized.tolist(),
        "physical_recovery_identified_by_objective": False,
        "confidence_probability": None,
        "data_fault_attribution": None,
    }
