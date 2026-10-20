"""M46 synthetic independent-root controls and conditional prefix diagnostics.

These are newly generated qualification acquisitions, never benchmark test data.
The evaluator payload is separate from the observed arrays and numerical APIs.
Closed-form diagnostics apply only to the declared controls, hold shared theta
fixed, and make no confidence or global root-identification claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import product
from time import monotonic

import numpy as np

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting.initialization_qualification import response
from autoformalism.fitting.prefix_window_diagnostics import _inference_identity
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

SCENARIOS = ("separated", "near_confounded", "exact_confounded")
RATES = dict(zip(SCENARIOS, ((1.0, 2.0), (1.0, 1.05), (1.0, 1.0)), strict=True))
ROOT_NAMES = ("init_z_value", "init_r_value")
TRAIN_ROOTS = ((0.4, 0.9), (1.2, 0.3), (1.6, 1.4))
VALIDATION_ROOTS = ((0.65, 1.2), (0.95, 0.55), (1.45, 1.7), (1.8, 0.4))
FOLLOWUP_ROOTS = ((0.25, 1.5), (0.85, 0.8), (1.35, 0.2), (1.95, 1.8))


def problem(scenario: str, seed: int = 0) -> PrefixFitProblem:
    """Declare known equations and two independent, nonnegative trajectory roots."""
    if scenario not in SCENARIOS or type(seed) is not int or seed not in (0, 1):
        raise ValueError("unknown multiple-root scenario or training start")
    rate_z, rate_r = RATES[scenario]
    theta, z, r = ((0.2, 0.3, 1.3), (0.6, 1.3, 0.3))[seed]
    return PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": f"prefix_multiroot_{scenario}",
                "parent_candidate_id": None,
                "states": [
                    {"name": "x", "kind": "observed"},
                    {"name": "z", "kind": "latent"},
                    {"name": "r", "kind": "latent"},
                ],
                "parameters": [
                    {
                        "name": "theta",
                        "scope": "global",
                        "role": "rate",
                        "bounds": {"lower": 0.1, "upper": 0.8},
                    }
                ],
                "state_equations": [
                    {"state": "x", "rhs": "-theta*x+z+r"},
                    {"state": "z", "rhs": f"-{rate_z:g}*z"},
                    {"state": "r", "rhs": f"-{rate_r:g}*r"},
                ],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "z", "r")
                ],
            },
            "context": {"targets": ["y"]},
            "parameter_guesses": {"theta": theta},
            "graph": {
                "rules": {
                    name: {
                        "initial": {
                            "mode": "value",
                            "guess": guess,
                            "role": "nonnegative_coefficient",
                            "scope": "trajectory",
                        }
                    }
                    for name, guess in (("z", z), ("r", r))
                }
            },
        }
    )


def analytic(
    time: Sequence[float] | np.ndarray,
    y0: float,
    theta: float,
    z0: float,
    r0: float,
    rate_z: float,
    rate_r: float,
) -> np.ndarray:
    """Evaluate the measured state, including equal decay-pole limits."""
    return (
        y0 * np.exp(-theta * np.asarray(time, float))
        + z0 * response(time, rate_z, theta)
        + r0 * response(time, rate_r, theta)
    )


def fingerprint(split: Mapping) -> dict:
    """Validate and hash a public development split's actual observed arrays."""
    payload = PublicSplit.model_validate(split).model_dump(mode="json")
    payload["fingerprint"] = content_sha256(
        {"name": payload["name"], "rows": payload["rows"]}
    )
    return payload


def _split(label: str, roots: Sequence[tuple[float, float]], rates: tuple) -> dict:
    """Generate measurements; only the caller retains the latent root labels."""
    time = np.linspace(0, 4, 41).tolist()
    return fingerprint(
        {
            "name": "train" if label == "train" else "val",
            "fingerprint": "pending",
            "rows": [
                {
                    "trajectory_id": f"{label}_{index}",
                    "time": time,
                    "targets": {
                        "y": analytic(time, 0.4, 0.35, z, r, *rates).tolist()
                    },
                }
                for index, (z, r) in enumerate(roots)
            ],
        }
    )


def _noise_sigma(value: float) -> float:
    """Reject booleans, nonfinite scales, and negative measurement noise."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("known noise sigma must be a finite nonnegative scalar")
    return float(value)


def fixture(
    scenario: str, seed: int = 0, noise_sigma: float = 0.0, repeat: int = 0
) -> dict:
    """Return reproducible observations and a separately labeled truth payload.

    Training starts share exactly the same acquisition. Follow-up trajectories
    are new synthetic qualification data, represented by the development ``val``
    schema with distinct IDs. Their policy must be frozen before they are scored.
    Time-zero readings remain exactly 0.4, including in noisy acquisitions.
    """
    model = problem(scenario, seed)
    sigma = _noise_sigma(noise_sigma)
    if type(repeat) is not int or repeat < 0 or repeat > 2**32 - 1:
        raise ValueError("acquisition repeat must be a nonnegative 32-bit integer")
    rates = RATES[scenario]
    definitions = (
        ("training", "train", TRAIN_ROOTS),
        ("validation", "val", VALIDATION_ROOTS),
        ("followup", "followup", FOLLOWUP_ROOTS),
    )
    clean = {key: _split(label, roots, rates) for key, label, roots in definitions}
    measured = deepcopy(clean)
    rng = np.random.default_rng(
        np.random.SeedSequence([20261019, SCENARIOS.index(scenario), repeat])
    )
    for split in measured.values():
        for row in split["rows"]:
            values = np.asarray(row["targets"]["y"], float)
            values[1:] += rng.normal(0, sigma, len(values) - 1)
            row["targets"]["y"] = values.tolist()
    measured = {key: fingerprint(split) for key, split in measured.items()}
    return {
        "scenario": scenario,
        "seed": seed,
        "repeat": repeat,
        "noise_sigma": sigma,
        "problem": model.model_dump(mode="json"),
        **measured,
        "evaluation": {
            "theta": 0.35,
            "rate_z": rates[0],
            "rate_r": rates[1],
            "roots": {
                f"{label}_{index}": dict(zip(ROOT_NAMES, pair, strict=True))
                for _, label, roots in definitions
                for index, pair in enumerate(roots)
            },
            "clean": clean,
        },
        "followup_scope": "new_synthetic_qualification_trajectories",
        "test_data_opened": False,
    }


def _control_rates(model: PrefixFitProblem) -> tuple[float, float]:
    """Allow closed-form diagnostics only for the exact predeclared models."""
    for scenario in SCENARIOS:
        if model.base_candidate.candidate_id == f"prefix_multiroot_{scenario}":
            if any(model == problem(scenario, seed) for seed in (0, 1)):
                return RATES[scenario]
            break
    raise ValueError("analytic diagnostic requires a declared multiple-root control")


def _design(
    model: PrefixFitProblem,
    fit_result: Mapping,
    prefix: kernel.PrefixView,
    inference: Mapping,
    numerical: kernel.NumericalPolicy,
) -> tuple[kernel.PrefixOracle, np.ndarray, np.ndarray, np.ndarray]:
    """Build the conditional linear prefix objective without evaluator access."""
    if not isinstance(prefix, kernel.PrefixView) or not isinstance(
        numerical, kernel.NumericalPolicy
    ):
        raise TypeError("typed prefix and numerical policy required")
    _inference_identity(model, fit_result, prefix, inference)
    rates = _control_rates(model)
    oracle = kernel.PrefixOracle(
        model, fit_result, prefix, numerical, root_guesses=inference["roots"]
    )
    if oracle.names != ROOT_NAMES or set(oracle.frozen) != {"theta"}:
        raise ValueError("multiple-root control coordinate declaration differs")
    elapsed = np.asarray(prefix.time) - prefix.time[0]
    theta = oracle.frozen["theta"]
    physical = np.column_stack([response(elapsed, rate, theta) for rate in rates])
    baseline = prefix.targets["y"][0] * np.exp(-theta * elapsed)
    scales = oracle.scales["y"]
    design = physical / scales
    target = (np.asarray(prefix.targets["y"]) - baseline) / scales
    return oracle, design, target, physical


def _spectrum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return two physical-root axes, including null axes for short prefixes."""
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    singular = np.pad(singular, (0, 2 - len(singular)))
    tolerance = max(matrix.shape) * np.finfo(float).eps * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    for row in right:
        nonzero = np.flatnonzero(np.abs(row) > 1e-14)
        if len(nonzero) and row[nonzero[0]] < 0:
            row *= -1
    return singular, right, rank, tolerance


def prefix_information(
    model: PrefixFitProblem,
    fit_result: Mapping,
    prefix: kernel.PrefixView,
    inference: Mapping,
    numerical: kernel.NumericalPolicy,
    known_noise_sigma: float,
    checkpoint: kernel.Checkpoint | None = None,
) -> dict:
    """Report conditional two-root SVD information, without uncertainty claims.

    Noise scales are along right-singular directions in native root units and
    exclude the exact initial reading. A null direction has no finite scale;
    domain boundaries and uncertainty in the frozen shared fit are not included.
    The single diagnostic call is analytic and performs no integration or refit.
    """
    if not isinstance(prefix, kernel.PrefixView) or not isinstance(
        numerical, kernel.NumericalPolicy
    ):
        raise TypeError("typed prefix and numerical policy required")
    sigma = _noise_sigma(known_noise_sigma)
    _inference_identity(model, fit_result, prefix, inference)
    result = {
        "protocol": "prefix-multiroot-information-1",
        "available": False,
        "status": "inference_unavailable",
        "diagnostic_calls_started": 0,
        "diagnostic_calls_completed": 0,
        "integration_calls": 0,
        "problem_sha256": content_sha256(model),
        "fit_sha256": content_sha256(fit_result),
        "prefix_sha256": content_sha256(prefix),
        "inference_sha256": content_sha256(inference),
        "numerical_policy_sha256": content_sha256(numerical),
        "budget_seconds": numerical.verification_seconds,
        "prefix_cutoff": prefix.cutoff,
        "known_noise_sigma": sigma,
        "root_names": list(ROOT_NAMES),
        "root_values": None,
        "singular_values": None,
        "physical_output_singular_values": None,
        "right_singular_vectors": None,
        "weak_direction": None,
        "rank": None,
        "condition_number": None,
        "one_noise_unit_root_scales": None,
        "weakest_one_noise_unit_root_scale": None,
        "scale_status": "unavailable",
        "coordinate_system": "native_physical_roots",
        "output_normalization": "frozen_training_channel_scales",
        "rank_observation_scope": "complete_prefix_including_exact_initial_boundary",
        "noise_scale_observation_scope": "prefix_samples_strictly_after_initial_time",
        "conditional_on_frozen_shared_coefficients": True,
        "domain_bounds_in_noise_scales": False,
        "confidence_interval": False,
        "scope": "conditional local sensitivity; no uniqueness or confidence claim",
    }
    save = checkpoint or (lambda value: None)
    if not inference.get("fit_success"):
        save(deepcopy(result))
        return result
    started = monotonic()
    result["diagnostic_calls_started"] = 1
    save(deepcopy(result))
    try:
        oracle, design, _, physical = _design(
            model, fit_result, prefix, inference, numerical
        )
        singular, right, rank, tolerance = _spectrum(design)
        physical_singular, _, _, _ = _spectrum(physical)
        noise_singular, noise_right, noise_rank, _ = _spectrum(physical[1:])
        scales = [
            sigma / float(value) if sigma > 0 and index < noise_rank else None
            for index, value in enumerate(noise_singular)
        ]
        if monotonic() - started >= numerical.verification_seconds:
            raise TimeoutError("multiple-root diagnostic allowance exhausted")
        result.update(
            available=True,
            status="complete",
            diagnostic_calls_completed=1,
            root_values=oracle.parameters(oracle.vector(inference["roots"])),
            singular_values=singular.tolist(),
            physical_output_singular_values=physical_singular.tolist(),
            right_singular_vectors=right.tolist(),
            weak_direction=dict(zip(ROOT_NAMES, right[-1].tolist(), strict=True)),
            weak_direction_unique=bool(singular[0] - singular[1] > tolerance),
            rank=rank,
            rank_tolerance=tolerance,
            full_column_rank=rank == 2,
            condition_number=float(singular[0] / singular[1]) if rank == 2 else None,
            residual_count=len(prefix.time),
            observation_times=len(prefix.time),
            noise_scale_residual_count=len(prefix.time) - 1,
            noise_right_singular_vectors=noise_right.tolist(),
            one_noise_unit_root_scales=scales,
            weakest_one_noise_unit_root_scale=scales[-1],
            scale_status=(
                "uninformative"
                if noise_rank == 0
                else "noiseless"
                if sigma == 0
                else "available"
                if noise_rank == 2
                else "partially_uninformative"
            ),
        )
    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        TimeoutError,
        np.linalg.LinAlgError,
    ) as error:
        result.update(
            status="budget_exhausted"
            if isinstance(error, TimeoutError)
            else "numerical_failure",
            error=str(error),
        )
    result["diagnostic_wall_seconds"] = monotonic() - started
    save(deepcopy(result))
    return result


def _box_optimum(
    design: np.ndarray, target: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve a two-coordinate convex box problem by enumerating its nine faces."""
    best: tuple[np.ndarray, float] | None = None
    for status in product((0, -1, 1), repeat=2):
        free = np.asarray(status) == 0
        fixed = ~free
        vector = np.zeros(2)
        vector[fixed] = np.where(np.asarray(status) == -1, lower, upper)[fixed]
        if not np.isfinite(vector).all():
            continue
        if free.any():
            vector[free] = np.linalg.lstsq(
                design[:, free], target - design[:, fixed] @ vector[fixed], rcond=None
            )[0]
        if np.any(vector < lower - 1e-10) or np.any(vector > upper + 1e-10):
            continue
        vector = np.clip(vector, lower, upper)
        loss = float(np.mean((design @ vector - target) ** 2))
        if best is None or loss < best[1]:
            best = vector, loss
    if best is None:
        raise ValueError("no feasible conditional linear reference")
    return best


def conditional_linear_optimum(
    model: PrefixFitProblem,
    fit_result: Mapping,
    prefix: kernel.PrefixView,
    inference: Mapping,
    numerical: kernel.NumericalPolicy,
    checkpoint: kernel.Checkpoint | None = None,
    *,
    objective_tolerance: float = 1e-8,
) -> dict:
    """Verify the retained prefix objective against an analytic bounded optimum.

    This reference uses observed prefix values, frozen theta and existing root
    bounds only. It does not replace fitted roots or use generating values.
    At deficient rank a reference endpoint is one minimizer, not a true root.
    """
    if not isinstance(prefix, kernel.PrefixView) or not isinstance(
        numerical, kernel.NumericalPolicy
    ):
        raise TypeError("typed prefix and numerical policy required")
    if (
        isinstance(objective_tolerance, bool)
        or not isinstance(objective_tolerance, (int, float))
        or not np.isfinite(objective_tolerance)
        or objective_tolerance <= 0
    ):
        raise ValueError("objective tolerance must be finite and positive")
    _inference_identity(model, fit_result, prefix, inference)
    result = {
        "protocol": "prefix-multiroot-linear-reference-1",
        "available": False,
        "status": "inference_unavailable",
        "diagnostic_calls_started": 0,
        "diagnostic_calls_completed": 0,
        "integration_calls": 0,
        "problem_sha256": content_sha256(model),
        "fit_sha256": content_sha256(fit_result),
        "prefix_sha256": content_sha256(prefix),
        "inference_sha256": content_sha256(inference),
        "objective_tolerance": float(objective_tolerance),
        "reference_roots": None,
        "reference_normalized_mse": None,
        "retained_normalized_mse": None,
        "objective_gap": None,
        "objective_optimal": False,
        "conditional_on_frozen_shared_coefficients": True,
        "roots_replaced": False,
        "truth_used": False,
        "confidence_interval": False,
        "scope": "prefix-only conditional convex reference; no unique-root claim",
    }
    save = checkpoint or (lambda value: None)
    if not inference.get("fit_success"):
        save(deepcopy(result))
        return result
    started = monotonic()
    result["diagnostic_calls_started"] = 1
    save(deepcopy(result))
    try:
        oracle, design, target, _ = _design(
            model, fit_result, prefix, inference, numerical
        )
        reference, minimum = _box_optimum(design, target, oracle.lower, oracle.upper)
        retained = oracle.vector(inference["roots"])
        loss = float(np.mean((design @ retained - target) ** 2))
        gap = loss - minimum
        if monotonic() - started >= numerical.verification_seconds:
            raise TimeoutError("conditional linear reference allowance exhausted")
        result.update(
            available=True,
            status="complete",
            diagnostic_calls_completed=1,
            root_names=list(oracle.names),
            root_bounds={
                name: {
                    "lower": float(lower) if np.isfinite(lower) else None,
                    "upper": float(upper) if np.isfinite(upper) else None,
                }
                for name, lower, upper in zip(
                    oracle.names, oracle.lower, oracle.upper, strict=True
                )
            },
            unbounded_bound_encoding=None,
            reference_roots=oracle.parameters(reference),
            reference_normalized_mse=minimum,
            retained_normalized_mse=loss,
            objective_gap=gap,
            objective_optimal=abs(gap) <= objective_tolerance,
            rank=_spectrum(design)[2],
            residual_count=len(target),
            bounded_faces_considered=9,
        )
    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        TimeoutError,
        np.linalg.LinAlgError,
    ) as error:
        result.update(
            status="budget_exhausted"
            if isinstance(error, TimeoutError)
            else "numerical_failure",
            error=str(error),
        )
    result["diagnostic_wall_seconds"] = monotonic() - started
    save(deepcopy(result))
    return result


def _root_errors(roots: Mapping[str, float], truth: Mapping[str, float]) -> dict:
    """Evaluate coordinates and their observable sum without choosing a root."""
    difference = np.asarray([roots[name] - truth[name] for name in ROOT_NAMES])
    return {
        "per_root_error": dict(zip(ROOT_NAMES, difference.tolist(), strict=True)),
        "root_l2_error": float(np.linalg.norm(difference)),
        "maximum_absolute_root_error": float(np.max(np.abs(difference))),
        "sum_root_error": float(np.sum(difference)),
    }


def _analytic_difference(
    attempt: Mapping, row: Mapping, theta: float, rates: tuple[float, float]
) -> float | None:
    """Compare a retained forecast with its equations at that retained endpoint."""
    inference, forecast = attempt.get("inference"), attempt.get("forecast")
    if (
        not inference
        or not inference.get("fit_success")
        or not forecast
        or not forecast.get("forecast_success")
    ):
        return None
    if (
        forecast["trajectory_id"] != row["trajectory_id"]
        or forecast["time"] != row["time"]
    ):
        raise ValueError("evaluator forecast trajectory/time differs")
    roots = inference["roots"]
    exact = analytic(
        row["time"],
        row["targets"]["y"][0],
        theta,
        roots[ROOT_NAMES[0]],
        roots[ROOT_NAMES[1]],
        *rates,
    )
    predicted = np.asarray(forecast["predictions"]["y"], float)
    if predicted.shape != exact.shape or not np.isfinite(predicted).all():
        raise ValueError("evaluator forecast values differ from declared grid")
    return float(np.max(np.abs(predicted - exact)))


def _evaluate_windows(
    record: Mapping,
    fit: Mapping,
    windows: Sequence[Mapping],
    selection: Mapping,
    split_key: str,
) -> list[dict]:
    """Evaluate already chosen endpoints, retaining missing rows and horizons."""
    evaluator = record["evaluation"]
    clean = {
        row["trajectory_id"]: row for row in evaluator["clean"][split_key]["rows"]
    }
    theta = fit["shared_parameters"]["theta"]
    rates = (evaluator["rate_z"], evaluator["rate_r"])
    result = []
    for window in windows:
        duration = window["duration"]
        entries = []
        for entry in window["trajectories"]:
            row_id = entry["trajectory_id"]
            row = clean[row_id]
            selected = entry["chosen_start"]
            values = {
                "trajectory_id": row_id,
                "available": False,
                "chosen_start": selected,
                "root_errors": None,
                "clean_scores": None,
                "analytic_max_absolute_difference": None,
                "all_start_analytic_max_absolute_differences": [
                    _analytic_difference(attempt, row, theta, rates)
                    for attempt in entry["attempts"]
                ],
            }
            if selected is not None:
                attempt = entry["attempts"][selected]
                inference, forecast = attempt["inference"], attempt["forecast"]
                if inference and inference.get("fit_success"):
                    values["root_errors"] = _root_errors(
                        inference["roots"], evaluator["roots"][row_id]
                    )
                if values["root_errors"] is not None and forecast and forecast.get(
                    "forecast_success"
                ):
                    time = np.asarray(row["time"])
                    masks = {
                        "common": time > selection["common_start"],
                        "immediate": (time > duration)
                        & (time <= duration + selection["immediate_duration"] + 1e-12),
                        "own_suffix": time > duration,
                    }
                    error = (
                        np.asarray(forecast["predictions"]["y"])
                        - np.asarray(row["targets"]["y"])
                    ) / fit["training_scales"]["y"]
                    values.update(
                        available=True,
                        clean_scores={
                            name: {
                                "normalized_mse": float(np.mean(error[mask] ** 2))
                                if mask.any()
                                else None,
                                "sample_count": int(mask.sum()),
                            }
                            for name, mask in masks.items()
                        },
                        analytic_max_absolute_difference=values[
                            "all_start_analytic_max_absolute_differences"
                        ][selected],
                    )
            entries.append(values)
        ids = [entry["trajectory_id"] for entry in entries]
        complete = set(ids) == set(clean) and len(ids) == len(clean) and all(
            entry["available"] for entry in entries
        )
        result.append(
            {
                "duration": duration,
                "selected_duration": selection.get("duration") == duration,
                "rows": entries,
                "rows_expected": len(clean),
                "rows_available": sum(entry["available"] for entry in entries),
                "complete": complete,
                "maximum_root_l2_error": max(
                    entry["root_errors"]["root_l2_error"] for entry in entries
                )
                if complete
                else None,
                "maximum_absolute_sum_root_error": max(
                    abs(entry["root_errors"]["sum_root_error"]) for entry in entries
                )
                if complete
                else None,
                "mean_clean_scores": {
                    horizon: float(
                        np.mean(
                            [
                                entry["clean_scores"][horizon]["normalized_mse"]
                                for entry in entries
                            ]
                        )
                    )
                    for horizon in ("common", "immediate", "own_suffix")
                }
                if complete
                else None,
            }
        )
    return result


def evaluate(
    record: Mapping,
    fit: Mapping,
    windows: Sequence[Mapping],
    followup: Sequence[Mapping],
    selection: Mapping,
) -> dict:
    """Evaluate truth only after the caller seals the validation-only decision.

    The caller supplies the retained start choices and selected duration; this
    function never chooses either. Results are descriptive synthetic qualification
    errors. Follow-up errors are independent of tuning only when the caller kept
    the supplied complete policy fixed before inference on those trajectories.
    """
    result = {
        "protocol": "prefix-multiroot-postselection-evaluation-1",
        "available": False,
        "selection_sha256": content_sha256(selection),
        "truth_used_for_selection": False,
        "test_data_opened": False,
        "followup_scope": "new_synthetic_qualification_trajectories",
    }
    if not fit or not fit.get("fit_success"):
        return result
    evaluator = record["evaluation"]
    return result | {
        "available": True,
        "shared_theta_error": float(
            fit["shared_parameters"]["theta"] - evaluator["theta"]
        ),
        "training_root_errors": {
            name: _root_errors(roots, evaluator["roots"][name])
            for name, roots in fit["training_roots"].items()
        },
        "validation": _evaluate_windows(record, fit, windows, selection, "validation"),
        "followup": _evaluate_windows(record, fit, followup, selection, "followup"),
        "followup_status": (
            "not_run_without_selected_duration"
            if selection["status"] != "selected"
            else "recorded"
            if followup
            else "missing"
        ),
        "scope": "retrospective generated-root and clean-signal qualification only",
    }
