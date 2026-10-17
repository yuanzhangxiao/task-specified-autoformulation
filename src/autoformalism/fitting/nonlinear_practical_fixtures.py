"""M41 nonlinear decay acquisitions with isolated evaluator-only generating truth."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from autoformalism.fitting import measurement_fixtures as templates
from autoformalism.fitting import practical_uncertainty_fixtures as acquisition
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.sensor_drift import DriftProblem

PARAMETERS = ("g", "k", "x0", "sensor_drift")
SEPARATIONS = (0.5, 0.02, 0.0)
SIGMAS = (0.01, 0.2)
SEEDS = (0, 1, 2)
BOUNDS = {
    "g": [0.05, 3.0],
    "k": [0.02, 1.5],
    "x0": [0.01, 2.0],
    "sensor_drift": [-2.0, 2.0],
}
START = {"g": 0.45, "k": 1.1, "x0": 0.35}
INPUT_PROTOCOL = "phase-c-nonlinear-practical-uncertainty-inputs-1"
case_key = acquisition.case_key


def response(time: float, forcing: float, parameters: dict) -> float:
    """Evaluate the exact physical response of x'=-k*x+g*u at constant input."""
    initial, gain = templates.response(time, parameters["k"], forcing)
    return initial * parameters["x0"] + gain * parameters["g"]


def _build(
    separation: float, sigma: float, seed: int
) -> tuple[DriftProblem, dict, dict]:
    """Reuse reviewed acquisition metadata with a distinct nonlinear response."""
    if (
        separation not in SEPARATIONS
        or sigma not in SIGMAS
        or type(seed) is not int
        or seed not in SEEDS
    ):
        raise ValueError("unknown nonlinear practical acquisition")
    original, evaluator, artifacts = acquisition._build(separation, sigma, seed)
    payload = original.model_dump(mode="json")
    base = payload["base"]
    truth = {"g": 1.3, "k": 0.65, "x0": 0.4, "sensor_drift": 0.06}
    old_sha = evaluator["acquisition_sha256"]
    raw = artifacts.pop(old_sha)
    raw["parser_protocol"] = "m41_acquisition_json_1"
    records = {record["id"]: record for record in raw["records"]}
    for observation in evaluator["observations"]:
        noise = observation["value"] - observation["clean_measured"]
        physical = response(observation["time"], observation["forcing"], truth)
        measured = physical + truth["sensor_drift"] * observation["time"]
        raw_value = (measured + noise - 0.03) / 0.01
        observation.update(
            value=0.01 * raw_value + 0.03,
            clean_physical=physical,
            clean_measured=measured,
        )
        records[observation["reading_id"]]["raw_value"] = raw_value
    raw_sha = public.content_sha256(raw)
    artifacts[raw_sha] = raw
    for row in base["training"]["rows"]:
        for reading in row["readings"]:
            reading["artifact_sha256"] = raw_sha
            reading["raw_value"] = records[reading["id"]]["raw_value"]
    for source in base["manifest"]["sources"]:
        if source["id"] == "acquisition":
            source["artifact_sha256"] = raw_sha
    base["manifest"]["artifacts"][0].update(
        sha256=raw_sha, parser_protocol=raw["parser_protocol"]
    )
    candidate = base["request"]["base_candidate"]
    candidate["candidate_id"] = "m41_fixed_physical_contract"
    candidate["state_equations"][0]["rhs"] = "g*u-k*x"
    candidate["parameters"] = [
        templates._parameter(name, *BOUNDS[name]) for name in START
    ]
    base["request"]["source"] = {
        "stage": "synthetic_control",
        "task_id": "m41_fixed_physical_contract",
        "artifact_sha256": public.content_sha256(
            {"protocol": "m41_fixed_candidate_1", "equation": "g*u-k*x"}
        ),
    }
    base["request"]["parameter_guesses"] = dict(START)
    base["start"] = dict(START)
    base["incumbent"] = dict(START)
    base["coordinates"]["parameters"] = {
        name: {"center": 0, "scale": 1} for name in START
    }
    evaluator.update(
        truth=truth,
        acquisition_sha256=raw_sha,
        parameter_names=list(PARAMETERS),
        parameter_bounds=[BOUNDS[name] for name in PARAMETERS],
    )
    return DriftProblem.model_validate(payload), evaluator, artifacts


def make(
    separation: float = 0.5, sigma: float = 0.01, seed: int = 0
) -> tuple[DriftProblem, dict]:
    """Return the training-only fitting contract and a separate truth evaluator."""
    problem, evaluator, _ = _build(separation, sigma, seed)
    return problem, evaluator


def make_inputs() -> dict:
    """Freeze 18 acquisitions; direct and profiled fits share each exact payload."""
    problems, evaluation, artifacts = {}, {}, {}
    for separation in SEPARATIONS:
        for sigma in SIGMAS:
            for seed in SEEDS:
                problem, evaluator, raw = _build(separation, sigma, seed)
                key = case_key(separation, sigma, seed)
                problems[key] = problem.model_dump(mode="json")
                evaluation[key] = evaluator
                artifacts.update(raw)
    return {
        "protocol": INPUT_PROTOCOL,
        "problems": problems,
        "evaluation": evaluation,
        "raw_artifacts": artifacts,
        "test_data_opened": False,
    }


def verify_raw_artifacts(data: dict) -> None:
    """Verify the nonlinear declaration and the unchanged reviewed raw contract.

    The M39 verifier checks calibration, clocks, forcing, sigma, raw bytes and
    observation provenance on a projection with the extra nonlinear coordinate
    removed. The full nonlinear declaration is checked before that projection;
    neither verifier consults generating values or clean companions.
    """
    projected = deepcopy(data)
    if set(data["problems"]) != set(data["evaluation"]):
        raise ValueError("nonlinear practical evaluation roster differs")
    for key, payload in projected["problems"].items():
        problem = DriftProblem.model_validate(data["problems"][key])
        candidate = problem.request.base_candidate
        evaluator = projected["evaluation"][key]
        domains = {
            item.name: [item.bounds.lower, item.bounds.upper]
            for item in candidate.parameters
        }
        if (
            domains != {name: BOUNDS[name] for name in START}
            or len(candidate.state_equations) != 1
            or candidate.state_equations[0].rhs != "g*u-k*x"
            or dict(problem.base.start) != START
            or dict(problem.base.incumbent) != START
            or dict(problem.request.parameter_guesses) != START
            or evaluator["separation"] not in SEPARATIONS
            or evaluator["sigma"] not in SIGMAS
            or type(evaluator["seed"]) is not int
            or evaluator["seed"] not in SEEDS
            or evaluator["parameter_names"] != list(PARAMETERS)
            or evaluator["parameter_bounds"] != [BOUNDS[name] for name in PARAMETERS]
        ):
            raise ValueError("nonlinear practical physical domain/start differs")
        axis = problem.base.coordinates.parameters["k"]
        if axis.center != 0 or axis.scale != 1:
            raise ValueError("nonlinear practical numerical coordinates differ")
        base = payload["base"]
        base["request"]["base_candidate"]["parameters"] = [
            parameter
            for parameter in base["request"]["base_candidate"]["parameters"]
            if parameter["name"] != "k"
        ]
        base["request"]["base_candidate"]["state_equations"][0]["rhs"] = "g*u"
        for values in (
            base["start"],
            base["incumbent"],
            base["request"]["parameter_guesses"],
            base["coordinates"]["parameters"],
        ):
            values.pop("k")
        evaluator["parameter_names"] = list(acquisition.PARAMETERS)
        evaluator["parameter_bounds"] = [
            BOUNDS[name] for name in acquisition.PARAMETERS
        ]
    acquisition.verify_raw_artifacts(projected)


def evaluate(evaluator: dict, parameters: dict | None) -> dict:
    """Replay a frozen endpoint exactly; do no fitting or truth-guided selection."""
    if parameters is None:
        return {"status": "no_parameters"}
    if (
        set(parameters) != set(PARAMETERS)
        or not np.isfinite(list(parameters.values())).all()
        or parameters["k"] <= 0
    ):
        raise ValueError("nonlinear practical parameter names/values differ")
    rows = evaluator["observations"]
    physical = np.array(
        [response(row["time"], row["forcing"], parameters) for row in rows]
    )
    measured = physical + parameters["sensor_drift"] * np.array(
        [row["time"] for row in rows]
    )
    residual = measured - [row["value"] for row in rows]
    standardized = residual / [row["sigma"] for row in rows]
    errors = {
        name: float(parameters[name] - evaluator["truth"][name]) for name in PARAMETERS
    }
    return {
        "status": "evaluated",
        "independent_analytic_endpoint_objective": float(
            standardized @ standardized / 2
        ),
        "parameter_truth_error_evaluator_only": errors,
        "absolute_parameter_truth_error_evaluator_only": {
            name: abs(value) for name, value in errors.items()
        },
        "clean_physical_prediction_rmse_evaluator_only": float(
            np.sqrt(np.mean((physical - [row["clean_physical"] for row in rows]) ** 2))
        ),
        "clean_measured_prediction_rmse_evaluator_only": float(
            np.sqrt(np.mean((measured - [row["clean_measured"] for row in rows]) ** 2))
        ),
        "analysis_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "analysis_residual_unit": "m",
        "physical_recovery_identified_by_objective": False,
        "confidence_probability": None,
        "data_fault_attribution": None,
    }
