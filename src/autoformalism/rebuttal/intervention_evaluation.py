"""Adapters and metrics for evaluating frozen models on private interventions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.optimize import linear_sum_assignment

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal.dalla_man import dalla_man_hidden_trajectory
from autoformalism.rebuttal.interventions import (
    InterventionCase,
    ReferenceTrajectory,
)
from autoformalism.schemas import CandidateModel


@dataclass(frozen=True)
class FrozenModel:
    """Canonical frozen structure and fitted global quantities."""

    method: str
    source: Path
    candidate: CandidateModel
    parameters: dict[str, float]
    initial_conditions: dict[str, float]
    in_distribution_nmse: float | None
    target_scales: dict[str, float]


class InterventionEvaluation(BaseModel):
    """Failure-aware metrics for one frozen model and intervention case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    source: str
    case_id: str
    benchmark_id: str
    success: bool
    target_mse: float | None
    target_nmse: float | None
    in_distribution_nmse: float | None
    nmse_degradation_ratio: float | None
    response_direction_correct: bool | None = None
    response_shape_correlation: float | None = None
    peak_timing_error_fraction: float | None = None
    hidden_alignment_nmse: float | None = None
    hidden_state_coverage: float | None = None
    hidden_matched_states: int | None = None
    hidden_reference_states: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class HiddenAlignment:
    """Permutation/sign/scale-invariant alignment of proposed latent states."""

    nmse: float | None
    coverage: float
    matched_states: int
    reference_states: int


def load_frozen_model(path: Path, *, target: str) -> FrozenModel:
    """Load a full checkpoint or adapt a baseline result to canonical form."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "frozen" in payload and "final_fit" in payload:
        fit = payload["final_fit"]
        test_metrics = payload.get("test_metrics") or {}
        in_distribution_nmse = _optional_float(test_metrics.get("normalized_mse"))
        if in_distribution_nmse is None:
            in_distribution_nmse = _optional_float(payload.get("test_normalized_mse"))
        return FrozenModel(
            method="autoformalism",
            source=path,
            candidate=CandidateModel.model_validate(payload["frozen"]["candidate"]),
            parameters={
                name: float(value)
                for name, value in fit.get("global_parameters", {}).items()
            },
            initial_conditions={
                name: float(value)
                for name, value in fit.get("global_initial_conditions", {}).items()
            },
            in_distribution_nmse=in_distribution_nmse,
            target_scales={
                name: float(value)
                for name, value in fit.get("target_scales", {}).items()
            },
        )
    if str(payload.get("method", "")).startswith("d3_"):
        return _load_d3_checkpoint_model(path, payload)
    equations = payload.get("equations")
    if not isinstance(equations, dict) or target not in equations:
        raise ValueError(f"artifact has no frozen equation for target {target}")
    parameters = _baseline_parameters(payload)
    candidate = _equation_candidate(
        target=target,
        rhs=str(equations[target]),
        parameters=parameters,
        candidate_id=f"{payload.get('method', 'baseline')}_{target}",
    )
    return FrozenModel(
        method=str(payload.get("method", "baseline")),
        source=path,
        candidate=candidate,
        parameters=parameters,
        initial_conditions={},
        in_distribution_nmse=_optional_float(payload.get("test_normalized_mse")),
        target_scales={},
    )


def evaluate_frozen_model(
    model: FrozenModel,
    *,
    case: InterventionCase,
    reference: ReferenceTrajectory,
    context: ValidationContext,
    tier: str,
    system_spec: dict[str, Any],
    fallback_target_scale: float,
    reset_observed_states: bool = True,
) -> InterventionEvaluation:
    """Evaluate a frozen model without parameter or structural refitting."""

    target = context.targets[0]
    trajectory, clean_target = public_intervention_trajectory(
        case=case,
        reference=reference,
        context=context,
        tier=tier,
        system_spec=system_spec,
    )
    try:
        compiled = compile_candidate(model.candidate, context)
    except Exception as exc:  # deterministic validator provides the exact cause
        return _failed(model, case, f"candidate compilation failed: {exc}")
    if model.method.startswith("d3_") and reset_observed_states:
        try:
            prediction, model_states = _native_d3_prediction(
                compiled, trajectory, model.parameters, target
            )
        except Exception as exc:
            return _failed(model, case, f"native D3 prediction failed: {exc}")
    else:
        simulation = simulate_trajectory(
            compiled,
            trajectory,
            model.parameters,
            model.initial_conditions,
            FitConfig(
                integration_backend="solve_ivp",
                relative_tolerance=1e-7,
                absolute_tolerance=1e-9,
            ),
            reset_observed_states=reset_observed_states,
        )
        if not simulation.success:
            return _failed(model, case, simulation.message or "simulation failed")
        prediction = simulation.predictions[target]
        model_states = simulation.states
    mse = float(np.mean(np.square(prediction - clean_target)))
    scale = model.target_scales.get(target, fallback_target_scale)
    nmse = mse / max(float(scale) ** 2, 1e-16)
    baseline = model.in_distribution_nmse
    degradation = None if baseline is None or baseline <= 0.0 else nmse / baseline
    direction, shape, timing = qualitative_response_metrics(
        trajectory.time, prediction, clean_target
    )
    hidden = align_hidden_states(
        model=model,
        model_states=model_states,
        reference=_private_hidden_states(case, reference, system_spec),
    )
    return InterventionEvaluation(
        method=model.method,
        source=str(model.source),
        case_id=case.case_id,
        benchmark_id=case.benchmark_id,
        success=True,
        target_mse=mse,
        target_nmse=nmse,
        in_distribution_nmse=baseline,
        nmse_degradation_ratio=degradation,
        response_direction_correct=direction,
        response_shape_correlation=shape,
        peak_timing_error_fraction=timing,
        hidden_alignment_nmse=hidden.nmse,
        hidden_state_coverage=hidden.coverage,
        hidden_matched_states=hidden.matched_states,
        hidden_reference_states=hidden.reference_states,
    )


def qualitative_response_metrics(
    time: np.ndarray, prediction: np.ndarray, reference: np.ndarray
) -> tuple[bool | None, float | None, float | None]:
    """Score response direction, centered shape, and peak timing.

    The response is displacement from the first sample. Direction is evaluated
    at the private reference's largest absolute displacement. Timing error is
    normalized by the evaluated duration. Degenerate flat references return
    ``None`` rather than manufacturing a successful qualitative response.
    """

    reference_delta = np.asarray(reference, dtype=float) - float(reference[0])
    prediction_delta = np.asarray(prediction, dtype=float) - float(prediction[0])
    peak = int(np.argmax(np.abs(reference_delta)))
    scale = float(np.max(np.abs(reference_delta)))
    if scale <= 1e-12:
        return None, None, None
    predicted_at_peak = float(prediction_delta[peak])
    direction = bool(
        abs(predicted_at_peak) > 1e-12
        and np.sign(predicted_at_peak) == np.sign(reference_delta[peak])
    )
    if np.std(reference_delta) <= 1e-12 or np.std(prediction_delta) <= 1e-12:
        correlation = None
    else:
        correlation = float(np.corrcoef(reference_delta, prediction_delta)[0, 1])
    predicted_peak = int(np.argmax(np.abs(prediction_delta)))
    duration = max(float(time[-1] - time[0]), 1e-12)
    timing = abs(float(time[predicted_peak] - time[peak])) / duration
    return direction, correlation, timing


def align_hidden_states(
    *,
    model: FrozenModel,
    model_states: np.ndarray | None,
    reference: np.ndarray,
) -> HiddenAlignment:
    """Optimally align latent trajectories to private hidden coordinates.

    Pair cost is the residual NMSE of the best affine map, equal to
    ``1 - correlation**2``. Hungarian matching handles arbitrary latent-state
    names and permutations. The reported NMSE penalizes each missing private
    state by one variance unit; it is therefore comparable across models with
    different latent-state counts. Models with no persistent latent state have
    no hidden-alignment NMSE and zero coverage.
    """

    reference = np.asarray(reference, dtype=float)
    reference_count = int(reference.shape[1])
    latent_names = [
        state.name for state in model.candidate.states if state.kind.value == "latent"
    ]
    if model_states is None or not latent_names:
        return HiddenAlignment(None, 0.0, 0, reference_count)
    candidate_state_names = [state.name for state in model.candidate.states]
    state_indices = [
        index
        for index, name in enumerate(candidate_state_names)
        if name in latent_names
    ]
    proposed = np.asarray(model_states, dtype=float)[state_indices].T
    costs = np.ones((proposed.shape[1], reference_count), dtype=float)
    for left in range(proposed.shape[1]):
        for right in range(reference_count):
            if (
                np.std(proposed[:, left]) <= 1e-12
                or np.std(reference[:, right]) <= 1e-12
            ):
                continue
            correlation = float(
                np.corrcoef(proposed[:, left], reference[:, right])[0, 1]
            )
            if np.isfinite(correlation):
                costs[left, right] = max(0.0, 1.0 - correlation**2)
    rows, columns = linear_sum_assignment(costs)
    matched = len(rows)
    penalized_error = float(costs[rows, columns].sum()) + (reference_count - matched)
    return HiddenAlignment(
        nmse=penalized_error / max(reference_count, 1),
        coverage=matched / max(reference_count, 1),
        matched_states=matched,
        reference_states=reference_count,
    )


def _private_hidden_states(
    case: InterventionCase,
    reference: ReferenceTrajectory,
    system_spec: dict[str, Any],
) -> np.ndarray:
    clean = np.asarray(reference.states_clean, dtype=float)
    if case.benchmark_id == "benchmark5":
        centers = system_spec["state_centers"]
        scales = system_spec["state_scales"]
        # T is the hard-tier target; C and Tj are private hidden coordinates.
        return np.column_stack(
            (
                (clean[:, 0] - float(centers["C"])) / float(scales["C"]),
                (clean[:, 2] - float(centers["Tj"])) / float(scales["Tj"]),
            )
        )
    if case.benchmark_id == "benchmark6":
        return clean[:, : int(system_spec["n_latent"])]
    perturbed = case.benchmark_id in {
        "perturbed_b1",
        "obfuscated_perturbed_case01",
    }
    return dalla_man_hidden_trajectory(
        clean,
        variant="perturbed_b1" if perturbed else "original",
        parameter_multipliers=case.parameter_multipliers,
    )


def public_intervention_trajectory(
    *,
    case: InterventionCase,
    reference: ReferenceTrajectory,
    context: ValidationContext,
    tier: str,
    system_spec: dict[str, Any],
) -> tuple[Trajectory, np.ndarray]:
    """Convert a private reference to the public, tier-specific representation."""

    time = np.asarray(reference.time, dtype=float)
    clean = np.asarray(reference.states_clean, dtype=float)
    observed = np.asarray(reference.states_observed, dtype=float)
    forcing = np.asarray(reference.forcing, dtype=float)
    if case.benchmark_id == "benchmark5":
        centers = system_spec["state_centers"]
        scales = system_spec["state_scales"]
        state_names = ("C", "T", "Tj")
        public_states = system_spec["public_state_mapping"]
        clean_channels = {
            public_states[name]: (clean[:, index] - float(centers[name]))
            / float(scales[name])
            for index, name in enumerate(state_names)
        }
        observed_channels = {
            public_states[name]: (observed[:, index] - float(centers[name]))
            / float(scales[name])
            for index, name in enumerate(state_names)
        }
        input_names = ("Cf", "Tf", "Tjf")
        input_centers = system_spec["input_centers"]
        input_scales = system_spec["input_scales"]
        public_inputs = system_spec["public_input_mapping"]
        external_inputs = {
            public_inputs[name]: (forcing[:, index] - float(input_centers[name]))
            / float(input_scales[name])
            for index, name in enumerate(input_names)
        }
    elif case.benchmark_id == "benchmark6":
        semantic = ("z1", "z2", "z3", "z4", "z5", "y")
        mapping = system_spec["secret_mapping"]["semantic_to_opaque"]
        clean_channels = {
            mapping[name]: clean[:, index] for index, name in enumerate(semantic)
        }
        observed_channels = {
            mapping[name]: observed[:, index] for index, name in enumerate(semantic)
        }
        external_inputs = {"u01": forcing[:, 0]}
    else:
        target_name = {
            "original_b1": "Gp",
            "perturbed_b1": "Gp",
            "obfuscated_original_case01": "v009",
            "obfuscated_perturbed_case01": "v015",
        }[case.benchmark_id]
        clean_channels = {target_name: clean[:, 0]}
        observed_channels = {target_name: observed[:, 0]}
        pulse_name = (
            "meal_event_g"
            if case.benchmark_id in {"original_b1", "perturbed_b1"}
            else "u01"
        )
        external_inputs = {pulse_name: forcing[:, 0]}
        if case.benchmark_id == "obfuscated_original_case01":
            external_inputs["input_schedule"] = forcing[:, 0]

    target = context.targets[0]
    targets = {target: np.asarray(observed_channels[target], dtype=float).copy()}
    auxiliaries = {
        name: np.asarray(observed_channels[name], dtype=float).copy()
        for name in context.auxiliaries
    }
    trajectory = Trajectory(
        trajectory_id=case.case_id,
        time=time.copy(),
        targets=targets,
        auxiliaries=auxiliaries,
        external_inputs={
            name: np.asarray(external_inputs[name], dtype=float).copy()
            for name in context.external_inputs
        },
        fixed_covariates={
            name: 78.0 * float(case.parameter_multipliers.get("BW", 1.0))
            for name in context.fixed_covariates
        },
        derivatives={},
    )
    return trajectory, np.asarray(clean_channels[target], dtype=float)


def _equation_candidate(
    *, target: str, rhs: str, parameters: dict[str, float], candidate_id: str
) -> CandidateModel:
    specs = []
    for name, value in parameters.items():
        width = max(abs(value), 1.0)
        specs.append(
            {
                "name": name,
                "scope": "global",
                "bounds": {"lower": value - 2.0 * width, "upper": value + 2.0 * width},
                "initialization_range": {"lower": value, "upper": value},
            }
        )
    return CandidateModel.model_validate(
        {
            "candidate_id": candidate_id,
            "parent_candidate_id": None,
            "states": [{"name": target, "kind": "observed"}],
            "processes": [],
            "state_equations": [{"state": target, "rhs": rhs}],
            "observation_mappings": [{"channel": target, "expression": target}],
            "parameters": specs,
            "initial_conditions": [
                {"state": target, "scope": "global", "expression": target}
            ],
        }
    )


def _native_d3_prediction(
    compiled: Any,
    trajectory: Trajectory,
    parameters: dict[str, float],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce D3's teacher-forced one-slot update without a dt multiplier."""

    observed = compiled.observed_state_channels
    predictions = np.empty(len(trajectory.time), dtype=float)
    state_history = np.empty((len(compiled.state_names), len(trajectory.time)))
    predictions[0] = float(trajectory.targets[target][0])
    for index in range(len(trajectory.time) - 1):
        state = np.asarray(
            [
                (
                    trajectory.targets[channel][index]
                    if channel in trajectory.targets
                    else trajectory.auxiliaries[channel][index]
                )
                for channel in (observed[name] for name in compiled.state_names)
            ],
            dtype=float,
        )
        state_history[:, index] = state
        forcing = trajectory_forcing(compiled, trajectory, causal_index=index)
        next_state = state + compiled.rhs(
            float(trajectory.time[index]), state, parameters, forcing
        )
        target_state = next(
            name for name, channel in observed.items() if channel == target
        )
        predictions[index + 1] = float(
            next_state[compiled.state_names.index(target_state)]
        )
        state_history[:, index + 1] = next_state
    if not np.all(np.isfinite(predictions)):
        raise ValueError("native D3 prediction became nonfinite")
    return predictions, state_history


def _baseline_parameters(payload: dict[str, Any]) -> dict[str, float]:
    selected = payload.get("selected_hyperparameters") or {}
    encoded = selected.get("selected_parameters", "{}")
    if isinstance(encoded, str):
        decoded = json.loads(encoded)
    elif isinstance(encoded, dict):
        decoded = encoded
    else:
        raise ValueError("selected_parameters must be JSON text or an object")
    return {name: float(value) for name, value in decoded.items()}


def _load_d3_checkpoint_model(path: Path, payload: dict[str, Any]) -> FrozenModel:
    checkpoint_path = path.with_name("d3_checkpoint.json")
    if not checkpoint_path.is_file():
        raise ValueError("D3 intervention evaluation requires d3_checkpoint.json")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_hyperparameters") or {}
    generation = int(selected["selected_generation"])
    record = next(
        (
            item
            for item in checkpoint.get("records", [])
            if int(item.get("generation", -1)) == generation
        ),
        None,
    )
    if record is None or record.get("candidate") is None:
        raise ValueError(f"D3 checkpoint has no candidate for generation {generation}")
    parameters = {
        name: float(value) for name, value in record.get("parameters", {}).items()
    }
    candidate_payload = json.loads(json.dumps(record["candidate"]))
    # Native D3 treats declared ranges as initialization metadata and its Adam
    # updates can leave them. The historical operational model therefore may
    # contain a fitted value outside the proposal range. Widening the adapter's
    # metadata admits that already-frozen value; it does not alter the equation
    # or refit/clamp the parameter.
    for parameter in candidate_payload.get("parameters", []):
        value = parameters.get(parameter["name"])
        if value is None:
            continue
        parameter["bounds"]["lower"] = min(parameter["bounds"]["lower"], value)
        parameter["bounds"]["upper"] = max(parameter["bounds"]["upper"], value)
        initialization = parameter.get("initialization_range")
        if initialization is not None:
            initialization["lower"] = min(initialization["lower"], value)
            initialization["upper"] = max(initialization["upper"], value)
    return FrozenModel(
        method=str(payload["method"]),
        source=path,
        candidate=CandidateModel.model_validate(candidate_payload),
        parameters=parameters,
        initial_conditions={},
        in_distribution_nmse=_optional_float(payload.get("test_normalized_mse")),
        target_scales={
            name: float(value)
            for name, value in record.get("target_scales", {}).items()
        },
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _failed(
    model: FrozenModel, case: InterventionCase, message: str
) -> InterventionEvaluation:
    return InterventionEvaluation(
        method=model.method,
        source=str(model.source),
        case_id=case.case_id,
        benchmark_id=case.benchmark_id,
        success=False,
        target_mse=None,
        target_nmse=None,
        in_distribution_nmse=model.in_distribution_nmse,
        nmse_degradation_ratio=None,
        message=message,
    )
