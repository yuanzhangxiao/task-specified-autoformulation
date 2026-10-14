"""M35 frozen paired acquisitions and independent loss-specific evaluation.

M34 fixtures are read-only source acquisitions. This module retains those source
artifacts, transforms raw readings explicitly, and gives the fitter only the
ordinary public measurement contract. Corruption labels and truth are evaluator
records; they cannot select endpoints or tune the declared Huber threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from scipy.optimize import brentq, minimize

from autoformalism.fitting import measurement_fixtures as original
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.measurement_fitting import MeasurementProblem

CASES = (
    "clean_exact",
    "sparse_outliers",
    "shared_clean",
    "shared_outliers",
    "nonlinear_clean",
    "nonlinear_outliers",
    "wrong_gain_bound",
    "wrong_dynamics",
)
SEEDS = original.SEEDS
INPUT_PROTOCOL = "phase-c-robust-measurement-fitting-inputs-1"
HUBER_DELTA = 1.345
PAIRS = (
    ("clean_exact", "sparse_outliers"),
    ("shared_clean", "shared_outliers"),
    ("nonlinear_clean", "nonlinear_outliers"),
)


def _build(case: str, seed: int) -> tuple[MeasurementProblem, dict, dict]:
    """Transform a retained acquisition, without placing truth in fitting inputs."""
    if case not in CASES or seed not in SEEDS:
        raise ValueError("unknown robust measurement fixture case/noise seed")
    shared = case in ("shared_clean", "shared_outliers")
    nonlinear = case in ("nonlinear_clean", "nonlinear_outliers")
    source_case = (
        "shared_initial"
        if shared
        else "nonlinear_decay"
        if nonlinear
        else "exact_varied"
    )
    source, evaluator, artifacts = original._build(source_case, seed)
    payload = source.model_dump(mode="json")
    evaluator = deepcopy(evaluator)
    observations = {row["reading_id"]: row for row in evaluator["observations"]}
    original_sha = payload["manifest"]["artifacts"][0]["sha256"]
    raw = deepcopy(artifacts[original_sha])
    corruption = []
    # Four prespecified positive perturbations (8% of target readings); the
    # positions/magnitudes are fixed before fitting, not selected from outcomes.
    corrupt_ids = (
        {"train_0_y_9", "train_0_y_18", "train_1_y_11", "train_1_y_21"}
        if case in ("sparse_outliers", "shared_outliers", "nonlinear_outliers")
        else set()
    )
    for record in raw["records"]:
        if record["id"] in corrupt_ids:
            observation = observations[record["id"]]
            before = record["raw_value"]
            delta = 18.0 * observation["sigma"]
            record["raw_value"] += delta / 0.01
            observation["value"] = record["raw_value"] * 0.01 + 0.03
            corruption.append(
                {
                    "reading_id": record["id"],
                    "original_raw_value": before,
                    "corrupted_raw_value": record["raw_value"],
                    "analysis_increment": delta,
                    "standardized_increment": 18.0,
                }
            )
    raw_sha = public.content_sha256(raw)
    artifacts[raw_sha] = raw
    indexed = {r["id"]: r for r in raw["records"]}
    for row in payload["training"]["rows"]:
        for reading in row["readings"]:
            reading["raw_value"] = indexed[reading["id"]]["raw_value"]
            reading["artifact_sha256"] = raw_sha
    for item in payload["manifest"]["sources"]:
        if item["id"] == "acquisition":
            item["artifact_sha256"] = raw_sha
    payload["manifest"]["artifacts"][0]["sha256"] = raw_sha
    candidate = payload["request"]["base_candidate"]
    # Public candidate identity has no clean/corruption condition labels.
    candidate["candidate_id"] = "m35_fixed_contract"
    payload["request"]["source"] = {
        "stage": "synthetic_control",
        "task_id": "m35_fixed_contract",
        "artifact_sha256": public.content_sha256(
            {
                "protocol": "m35_fixed_candidate_1",
                "shared_initial": shared,
                "nonlinear": nonlinear,
            }
        ),
    }
    candidate_decay = 0.25 if case == "wrong_dynamics" else 0.65
    candidate["state_equations"][0]["rhs"] = (
        "g*u-k*x" if nonlinear else f"g*u-{candidate_decay}*x"
    )
    gain_upper = 0.8 if case == "wrong_gain_bound" else 3.0
    candidate["parameters"][0]["bounds"]["upper"] = gain_upper
    evaluator.update(
        {
            "case": case,
            "source_case": source_case,
            "original_acquisition_sha256": original_sha,
            "fitting_acquisition_sha256": raw_sha,
            "corruption_evaluator_only": corruption,
            "candidate_decay": candidate_decay,
            "gain_bounds": [0.05, gain_upper],
            "parameter_bounds": [[0.05, gain_upper]]
            + ([[0.01, 2.0]] if shared else [[0.1, 1.5]] if nonlinear else []),
            "model_mismatch_control_evaluator_only": case
            if case in ("wrong_gain_bound", "wrong_dynamics")
            else None,
        }
    )
    # M34's Gaussian optimum refers to the original data and candidate. Recompute
    # both declared losses independently only in the post-fit evaluator.
    for key in list(evaluator):
        if key.startswith("analytic_"):
            evaluator.pop(key)
    return MeasurementProblem.model_validate(payload), evaluator, artifacts


def make(case: str = "clean_exact", seed: int = 0) -> tuple[MeasurementProblem, dict]:
    """Return one public training contract and its separate post-fit reference."""
    problem, evaluator, _ = _build(case, seed)
    return problem, evaluator


def make_inputs() -> dict:
    """Freeze 24 acquisitions once; resume reads these exact stored inputs."""
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
    """Check source bytes, transformed readings and retained original provenance."""
    original.verify_raw_artifacts(data)
    for key, evaluator in data["evaluation"].items():
        acquisition = data["problems"][key]["manifest"]["artifacts"][0]["sha256"]
        if evaluator["fitting_acquisition_sha256"] != acquisition:
            raise ValueError("robust evaluator acquisition differs")
        if evaluator["original_acquisition_sha256"] not in data["raw_artifacts"]:
            raise ValueError("original robust acquisition artifact is missing")


def contributions(residual: np.ndarray, kind: str, delta: float) -> np.ndarray:
    """Independent explicit objective arithmetic, without importing the fitter."""
    if kind == "gaussian":
        return 0.5 * residual**2
    if kind != "huber" or not np.isfinite(delta) or delta <= 0:
        raise ValueError("invalid evaluator loss")
    magnitude = np.abs(residual)
    return np.where(
        magnitude <= delta, 0.5 * residual**2, delta * (magnitude - 0.5 * delta)
    )


def _design(
    evaluator: dict, decay: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the candidate's exact affine prediction in analysis units."""
    design, target, offset = [], [], []
    for row in evaluator["observations"]:
        a, b = original.response(
            row["time"],
            evaluator["candidate_decay"] if decay is None else decay,
            row["forcing"],
        )
        design.append(
            [row["multiplier"] * b, row["multiplier"] * a]
            if evaluator["shared_initial"]
            else [row["multiplier"] * b]
        )
        base = (
            0.0
            if evaluator["shared_initial"]
            else row["multiplier"] * a * row["initial"]
        )
        offset.append(base)
        target.append(row["value"] - base)
    return np.array(design), np.array(target), np.array(offset)


def analytic_optimum(evaluator: dict, kind: str, delta: float = HUBER_DELTA) -> dict:
    """Solve a separate bounded convex problem with exact analytic predictions."""
    if evaluator["nonlinear_decay"]:
        return {
            "analytic_parameters": None,
            "analytic_objective": None,
            "analytic_method": "unavailable_for_nonlinear_decay",
        }
    matrix, rhs, _ = _design(evaluator)
    sigma = np.array([r["sigma"] for r in evaluator["observations"]])
    matrix, rhs = matrix / sigma[:, None], rhs / sigma
    bounds = np.array(evaluator["parameter_bounds"])

    def objective(vector):
        residual = matrix @ vector - rhs
        score = float(np.sum(contributions(residual, kind, delta)))
        psi = residual if kind == "gaussian" else np.clip(residual, -delta, delta)
        return score, matrix.T @ psi

    if len(bounds) == 1:
        lower, upper = bounds[0]
        if objective(np.array([lower]))[1][0] >= 0:
            solution = lower
        elif objective(np.array([upper]))[1][0] <= 0:
            solution = upper
        else:
            solution = brentq(
                lambda x: objective(np.array([x]))[1][0],
                lower,
                upper,
                xtol=1e-13,
            )
        vector = np.array([solution])
    else:
        start = np.clip(np.linalg.lstsq(matrix, rhs, rcond=None)[0], *bounds.T)
        result = minimize(
            objective,
            start,
            jac=True,
            bounds=bounds.tolist(),
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
        )
        vector = result.x
    value, gradient = objective(vector)
    projected = vector - np.clip(vector - gradient, *bounds.T)
    if not np.isfinite(value) or np.max(np.abs(projected)) > 1e-5:
        raise ValueError("independent affine reference did not satisfy bounded KKT")
    return {
        "analytic_parameters": dict(
            zip(evaluator["parameter_names"], map(float, vector), strict=True)
        ),
        "analytic_objective": value,
        "analytic_projected_gradient_maximum": float(np.max(np.abs(projected))),
        "analytic_active_gain_bound": bool(np.min(abs(bounds[0] - vector[0])) < 1e-8),
        "analytic_method": "analytic_affine_brentq_or_bounded_L-BFGS-B",
    }


def evaluate(
    evaluator: dict,
    parameters: Mapping[str, float] | None,
    kind: str,
    delta: float = HUBER_DELTA,
) -> dict:
    """Score a sealed endpoint without turning downweighting into a fault verdict."""
    if parameters is None:
        return {"status": "no_parameters", "analytic_objective_passed": None}
    if set(parameters) != set(evaluator["parameter_names"]):
        raise ValueError("robust evaluator parameter names differ")
    vector = np.array([parameters[name] for name in evaluator["parameter_names"]])
    if not np.isfinite(vector).all():
        raise ValueError("robust evaluator parameters must be finite")
    matrix, rhs, offset = _design(evaluator, parameters.get("k"))
    if evaluator["nonlinear_decay"]:
        vector = vector[:1]
    analysis_residual = matrix @ vector - rhs
    sigma = np.array([r["sigma"] for r in evaluator["observations"]])
    standardized = analysis_residual / sigma
    prediction = matrix @ vector + offset
    values = np.array([r["value"] for r in evaluator["observations"]])
    clean = np.array([r["clean"] for r in evaluator["observations"]])
    objective = float(np.sum(contributions(standardized, kind, delta)))
    reference = analytic_optimum(evaluator, kind, delta)
    optimum = reference["analytic_objective"]
    gap = objective - optimum if optimum is not None else None
    truth = {"g": evaluator["truth"]["g"]}
    if evaluator["shared_initial"]:
        truth["x0"] = next(iter(evaluator["truth"]["initials"].values()))
    if evaluator["nonlinear_decay"]:
        truth["k"] = evaluator["truth"]["k"]
    weights = np.ones_like(standardized)
    if kind == "huber":
        outside = abs(standardized) > delta
        weights[outside] = delta / abs(standardized[outside])
    return {
        "status": "evaluated",
        "loss_kind": kind,
        "huber_delta": delta if kind == "huber" else None,
        "independent_analytic_endpoint_objective": objective,
        "independent_standardized_half_sum_squares": float(
            0.5 * standardized @ standardized
        ),
        "analytic_optimum_objective": reference["analytic_objective"],
        "analytic_objective_gap": gap,
        "analytic_objective_passed": abs(gap) <= 1e-5 * (1 + abs(optimum))
        if gap is not None
        else None,
        "analytic_reference": reference,
        "clean_prediction_rmse_evaluator_only": float(
            np.sqrt(np.mean((prediction - clean) ** 2))
        ),
        "parameter_truth_error_evaluator_only": {
            name: float(parameters[name] - value) for name, value in truth.items()
        },
        "analysis_residual_rmse": float(np.sqrt(np.mean(analysis_residual**2))),
        "analysis_observed_nmse_descriptive_only": float(
            np.mean(analysis_residual**2) / np.var(values)
        ),
        "analysis_residual_unit": "m",
        "analysis_residuals": analysis_residual.tolist(),
        "standardized_residuals": standardized.tolist(),
        "robust_weights": weights.tolist(),
        "downweighted_reading_ids": [
            row["reading_id"]
            for row, weight in zip(evaluator["observations"], weights, strict=True)
            if weight < 1
        ],
        "model_mismatch_control_evaluator_only": evaluator[
            "model_mismatch_control_evaluator_only"
        ],
        "data_fault_attribution": None,
        "confidence_probability": None,
        "interpretation": (
            "Downweighting limits influence; it does not identify a faulty reading."
        ),
    }
