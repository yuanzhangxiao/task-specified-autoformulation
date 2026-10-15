"""M36 frozen sensor-drift acquisitions and independent post-fit arithmetic.

The same raw measurements are fitted with two observation hypotheses. Generating
drift, clean physical outputs and ambiguity witnesses belong to the evaluator;
the public contract supplies only bounded nuisance hypotheses and provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from scipy.optimize import lsq_linear, minimize

from autoformalism.fitting import measurement_fixtures as original
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.robust_measurement_fixtures import contributions
from autoformalism.schemas.sensor_drift import DriftProblem

CASES = (
    "clean_exact",
    "drift_exact",
    "drift_shared_initial",
    "drift_nonlinear_decay",
    "confounded_constant_input",
    "resolved_varied_input",
)
SEEDS = original.SEEDS
INPUT_PROTOCOL = "phase-c-sensor-drift-fitting-inputs-1"
HUBER_DELTA = 1.345


def response(time: float, decay: float, forcing: float) -> tuple[float, float]:
    """Return analytic initial/gain effects, including the zero-decay limit."""
    if decay == 0:
        return 1.0, forcing * time
    return original.response(time, decay, forcing)


def _build(case: str, seed: int) -> tuple[DriftProblem, dict, dict]:
    """Retain source bytes and transform only explicit synthetic acquisitions."""
    if case not in CASES or seed not in SEEDS:
        raise ValueError("unknown sensor drift fixture case/noise seed")
    shared, nonlinear = case == "drift_shared_initial", case == "drift_nonlinear_decay"
    source_case = (
        "shared_initial"
        if shared
        else "nonlinear_decay"
        if nonlinear
        else "exact_varied"
    )
    source, reference, artifacts = original._build(source_case, seed)
    payload = source.model_dump(mode="json")
    old_digest = payload["manifest"]["artifacts"][0]["sha256"]
    raw = deepcopy(artifacts[old_digest])
    records = {record["id"]: record for record in raw["records"]}
    zero_decay = case in ("confounded_constant_input", "resolved_varied_input")
    decay, drift = (
        (0.0 if zero_decay else 0.65),
        (0.0 if case == "clean_exact" else 0.06),
    )
    observations = []
    for row in reference["observations"]:
        forcing = 0.7 if case == "confounded_constant_input" else row["forcing"]
        a, b = response(row["time"], decay, forcing)
        physical = a * row["initial"] + b * reference["truth"]["g"]
        measured = physical + drift * row["time"]
        noise = row["value"] - row["clean"]
        records[row["reading_id"]]["raw_value"] = (measured + noise - 0.03) / 0.01
        input_id = row["reading_id"].replace("_y_", "_u_")
        records[input_id]["raw_value"] = forcing
        observations.append(
            {
                "reading_id": row["reading_id"],
                "trajectory": row["trajectory"],
                "time": row["time"],
                "value": 0.01 * records[row["reading_id"]]["raw_value"] + 0.03,
                "sigma": row["sigma"],
                "forcing": forcing,
                "initial": row["initial"],
                "clean_physical": physical,
                "clean_measured": measured,
            }
        )
    digest = public.content_sha256(raw)
    artifacts[digest] = raw
    for row in payload["training"]["rows"]:
        for reading in row["readings"]:
            reading["raw_value"] = records[reading["id"]]["raw_value"]
            reading["artifact_sha256"] = digest
    for item in payload["manifest"]["sources"]:
        if item["id"] == "acquisition":
            item["artifact_sha256"] = digest
    payload["manifest"]["artifacts"][0]["sha256"] = digest
    candidate = payload["request"]["base_candidate"]
    candidate["candidate_id"] = "m36_fixed_physical_contract"
    candidate["state_equations"][0]["rhs"] = (
        "g*u" if zero_decay else "g*u-k*x" if nonlinear else "g*u-0.65*x"
    )
    payload["request"]["source"] = {
        "stage": "synthetic_control",
        "task_id": "m36_fixed_physical_contract",
        "artifact_sha256": public.content_sha256(
            {
                "protocol": "m36_fixed_candidate_1",
                "equations": candidate["state_equations"],
            }
        ),
    }
    protocol = {
        "name": "sensor_drift",
        "channel": "y",
        "lower": -0.2,
        "upper": 0.2,
        "time_origin": 0.0,
        "unit": "m/min",
        "rationale": (
            "Optional sensor-specific linear drift shared across acquisitions; "
            "bounds fixed by the declared acquisition protocol."
        ),
    }
    protocol_digest = public.content_sha256(protocol)
    artifacts[protocol_digest] = protocol
    payload["manifest"]["sources"].append(
        {
            "id": "drift_protocol",
            "artifact_sha256": protocol_digest,
            "kind": "calibration",
        }
    )
    problem = DriftProblem.model_validate(
        {
            "base": payload,
            "drift": protocol
            | {
                "start": 0.0,
                "incumbent": 0.0,
                "scale": 0.1,
                "sources": ["drift_protocol"],
            },
        }
    )
    truth = reference["truth"] | {"k": decay, "sensor_drift": drift}
    evaluator = {
        "case": case,
        "seed": seed,
        "original_acquisition_sha256": old_digest,
        "fitting_acquisition_sha256": digest,
        "observations": observations,
        "truth": truth,
        "physical_parameter_names": list(source.start),
        "physical_parameter_bounds": [[0.05, 3.0]]
        + ([[0.01, 2.0]] if shared else [[0.1, 1.5]] if nonlinear else []),
        "drift_bounds": [-0.2, 0.2],
        "candidate_decay": decay,
        "shared_initial": shared,
        "nonlinear_decay": nonlinear,
        "ambiguity_witness_evaluator_only": (
            [{"g": 1.3, "sensor_drift": 0.06}, {"g": 1.5, "sensor_drift": -0.08}]
            if case == "confounded_constant_input"
            else None
        ),
    }
    return problem, evaluator, artifacts


def make(case: str = "clean_exact", seed: int = 0) -> tuple[DriftProblem, dict]:
    """Return one frozen public hypothesis and separate synthetic evaluator."""
    problem, evaluator, _ = _build(case, seed)
    return problem, evaluator


def make_inputs() -> dict:
    """Materialize exact acquisition bytes once, before any fitting or selection."""
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
    """Verify every transformed reading and both original and fitted source bytes."""
    original.verify_raw_artifacts(
        data
        | {"problems": {key: value["base"] for key, value in data["problems"].items()}}
    )
    for key, evaluator in data["evaluation"].items():
        acquisition = data["problems"][key]["base"]["manifest"]["artifacts"][0][
            "sha256"
        ]
        if evaluator["fitting_acquisition_sha256"] != acquisition:
            raise ValueError("sensor drift evaluator acquisition differs")
        if evaluator["original_acquisition_sha256"] not in data["raw_artifacts"]:
            raise ValueError("original sensor drift artifact is missing")


def _design(evaluator: dict, hypothesis: str, decay: float | None = None) -> tuple:
    """Build exact candidate predictions without consulting generating truth."""
    if hypothesis not in ("no_drift", "shared_drift"):
        raise ValueError("unknown sensor drift hypothesis")
    matrix, offsets, target = [], [], []
    for row in evaluator["observations"]:
        a, b = response(
            row["time"],
            evaluator["candidate_decay"] if decay is None else decay,
            row["forcing"],
        )
        columns = [b] + ([a] if evaluator["shared_initial"] else [])
        if hypothesis == "shared_drift":
            columns.append(row["time"])
        offset = 0.0 if evaluator["shared_initial"] else a * row["initial"]
        matrix.append(columns)
        offsets.append(offset)
        target.append(row["value"] - offset)
    return np.array(matrix), np.array(offsets), np.array(target)


def analytic_optimum(
    evaluator: dict, hypothesis: str, kind: str, delta: float = HUBER_DELTA
) -> dict:
    """Certify only objective optimality for bounded affine observation models."""
    if evaluator["nonlinear_decay"]:
        return {
            "analytic_parameters": None,
            "analytic_objective": None,
            "unique_parameters_assessed": False,
        }
    matrix, _, target = _design(evaluator, hypothesis)
    sigma = np.array([row["sigma"] for row in evaluator["observations"]])
    matrix, target = matrix / sigma[:, None], target / sigma
    units = np.linalg.norm(matrix, axis=0)
    design = matrix / units
    bounds = np.array(
        evaluator["physical_parameter_bounds"]
        + ([evaluator["drift_bounds"]] if hypothesis == "shared_drift" else [])
    )
    low, high = bounds[:, 0] * units, bounds[:, 1] * units
    initial = lsq_linear(design, target, bounds=(low, high), method="bvls", tol=1e-12)

    def objective(vector):
        residual = design @ vector - target
        psi = residual if kind == "gaussian" else np.clip(residual, -delta, delta)
        return float(contributions(residual, kind, delta).sum()), design.T @ psi

    vector = initial.x
    if kind == "huber":
        solved = minimize(
            objective,
            vector,
            jac=True,
            bounds=list(zip(low, high, strict=True)),
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        vector = solved.x
    value, gradient = objective(vector)
    projected = vector - np.clip(vector - gradient, low, high)
    if not np.isfinite(value) or np.max(abs(projected)) > 1e-5:
        raise ValueError("independent sensor drift reference failed bounded KKT")
    names = evaluator["physical_parameter_names"] + (
        ["sensor_drift"] if hypothesis == "shared_drift" else []
    )
    rank = int(np.linalg.matrix_rank(design))
    return {
        "analytic_parameters": dict(
            zip(names, map(float, vector / units), strict=True)
        ),
        "analytic_objective": value,
        "analytic_projected_gradient_maximum": float(np.max(abs(projected))),
        "analytic_design_rank": rank,
        "analytic_coordinate_count": len(names),
        "unique_parameters_assessed": False,
        "rank_deficient_optimum": rank < len(names),
        "analytic_method": "exact_affine_BVLS_or_SLSQP_with_bounded_KKT",
    }


def predictions(
    evaluator: dict, parameters: Mapping[str, float], hypothesis: str
) -> tuple:
    """Separate latent physical output from the stated observation hypothesis."""
    matrix, offsets, _ = _design(evaluator, hypothesis, parameters.get("k"))
    names = [name for name in evaluator["physical_parameter_names"] if name != "k"]
    names += ["sensor_drift"] if hypothesis == "shared_drift" else []
    vector = np.array([parameters[name] for name in names])
    measured = matrix @ vector + offsets
    drift = parameters.get("sensor_drift", 0.0)
    physical = measured - drift * np.array(
        [row["time"] for row in evaluator["observations"]]
    )
    return physical, measured


def evaluate(
    evaluator: dict,
    parameters: Mapping[str, float] | None,
    hypothesis: str,
    kind: str,
    delta: float = HUBER_DELTA,
) -> dict:
    """Evaluate a sealed endpoint without selecting hypotheses or assigning faults."""
    if parameters is None:
        return {"status": "no_parameters", "analytic_objective_passed": None}
    expected = set(evaluator["physical_parameter_names"]) | (
        {"sensor_drift"} if hypothesis == "shared_drift" else set()
    )
    if set(parameters) != expected or not np.isfinite(list(parameters.values())).all():
        raise ValueError("sensor drift evaluator parameter names/values differ")
    physical, measured = predictions(evaluator, parameters, hypothesis)
    observations = evaluator["observations"]
    values = np.array([row["value"] for row in observations])
    sigma = np.array([row["sigma"] for row in observations])
    residual = measured - values
    standardized = residual / sigma
    objective = float(contributions(standardized, kind, delta).sum())
    reference = analytic_optimum(evaluator, hypothesis, kind, delta)
    optimum = reference["analytic_objective"]
    gap = objective - optimum if optimum is not None else None
    truth = evaluator["truth"] | {
        "x0": next(iter(evaluator["truth"]["initials"].values()))
    }
    witness = evaluator["ambiguity_witness_evaluator_only"]
    witness_difference = None
    if witness is not None:
        first = predictions(evaluator, witness[0], "shared_drift")[1]
        second = predictions(evaluator, witness[1], "shared_drift")[1]
        witness_difference = float(np.max(abs(first - second)))
    return {
        "status": "evaluated",
        "hypothesis": hypothesis,
        "loss_kind": kind,
        "independent_analytic_endpoint_objective": objective,
        "independent_standardized_half_sum_squares": float(
            0.5 * standardized @ standardized
        ),
        "analytic_optimum_objective": optimum,
        "analytic_objective_gap": gap,
        "analytic_objective_passed": abs(gap) <= 1e-5 * (1 + abs(optimum))
        if gap is not None
        else None,
        "analytic_reference": reference,
        "clean_physical_prediction_rmse_evaluator_only": float(
            np.sqrt(
                np.mean(
                    (physical - [row["clean_physical"] for row in observations]) ** 2
                )
            )
        ),
        "clean_measured_prediction_rmse_evaluator_only": float(
            np.sqrt(
                np.mean(
                    (measured - [row["clean_measured"] for row in observations]) ** 2
                )
            )
        ),
        "physical_parameter_truth_error_evaluator_only": {
            name: float(parameters[name] - truth[name])
            for name in evaluator["physical_parameter_names"]
        },
        "sensor_drift_truth_error_evaluator_only": float(
            parameters.get("sensor_drift", 0.0) - truth["sensor_drift"]
        ),
        "analysis_residuals": residual.tolist(),
        "analysis_residual_unit": "m",
        "analysis_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "analysis_observed_nmse_descriptive_only": float(
            np.mean(residual**2) / np.var(values)
        ),
        "standardized_residuals": standardized.tolist(),
        "ambiguity_witness_measured_difference_evaluator_only": witness_difference,
        "physical_recovery_identified_by_objective": False,
        "observation_hypothesis_selected": None,
        "data_fault_attribution": None,
        "confidence_probability": None,
    }
