"""Frozen-parameter public-channel probes and validation response assessments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import monotonic

import numpy as np
from pydantic import Field

from autoformalism.baselines.d3_rollout import NativeMap, finite_mean, predict
from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal.intervention_evaluation import qualitative_response_metrics
from autoformalism.rebuttal.mechanism_checks import EquationRequirement, counts
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class ProbeSettings(StrictSchema):
    """Predeclared numerical limits and descriptive response thresholds."""

    trajectory_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    rtol: float = Field(default=1e-7, gt=0, allow_inf_nan=False)
    atol: float = Field(default=1e-9, gt=0, allow_inf_nan=False)
    replay_scaled_tolerance: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    probe_range_fraction: float = Field(default=0.05, gt=0, le=0.25)
    activity_floor: float = Field(default=1e-5, gt=0)
    response_nmse_limit: float = Field(default=0.1, gt=0)
    response_shape_minimum: float = Field(default=0.9, ge=-1, le=1)
    response_timing_limit: float = Field(default=0.1, ge=0, le=1)


def _native(row: dict) -> NativeMap:
    """Restore the original saved auxiliary laws for the native replay adapter."""
    candidate = CandidateModel.model_validate(row["candidate"])
    payload = candidate.model_dump(mode="json")
    payload["state_equations"] += row.get("native_auxiliary_equations_ignored", [])
    context = ValidationContext.model_validate(row["context"])
    return NativeMap.build(
        CandidateModel.model_validate(payload),
        row["parameters"],
        (*context.targets, *context.auxiliaries),
        (*context.external_inputs, *context.fixed_covariates),
    )


def rollout(
    row: dict, trajectory: Trajectory, solver: str, settings: ProbeSettings
) -> dict:
    """Use the existing production/native interpreter; never fit or reset targets."""
    context = ValidationContext.model_validate(row["context"])
    if row["semantics"] == "native_increment":
        native = _native(row)
        values = predict(
            native,
            trajectory,
            teacher_forced=False,
            seconds=settings.trajectory_seconds,
        )
        return {t: values[:, native.states.index(t)].tolist() for t in context.targets}
    model = compile_candidate(CandidateModel.model_validate(row["candidate"]), context)
    result = simulate_trajectory(
        model,
        trajectory,
        row["parameters"],
        row["initials"],
        FitConfig(
            integration_backend="solve_ivp",
            integration_method=solver,
            relative_tolerance=settings.rtol,
            absolute_tolerance=settings.atol,
            maximum_wall_time_seconds=settings.trajectory_seconds,
            allow_derivative_regression=False,
        ),
        deadline=monotonic() + settings.trajectory_seconds,
        reset_observed_states=False,
    )
    if not result.success:
        raise ValueError(result.message or "production rollout failed")
    return {t: result.predictions[t].tolist() for t in context.targets}


def _cached_rollout(row, trajectory, solver, settings, directory, key):
    """Checkpoint every numerical call; interrupted calls are not silently retried."""
    identity = content_hash([row, key, solver, settings.model_dump()])
    path = directory / f"{identity}.json"
    if path.exists():
        saved = sealed_read(path)
        if saved["identity"] != identity:
            raise ValueError("cached numerical call identity differs")
        return saved
    started = path.with_suffix(".started.json")
    if started.exists():
        if sealed_read(started)["identity"] != identity:
            raise ValueError("started call identity differs")
        return sealed_write(
            path,
            {
                "identity": identity,
                "status": "interrupted",
                "error": "prior attempt interrupted; no automatic fresh budget",
            },
        )
    sealed_write(started, {"identity": identity, "key": key, "solver": solver})
    begin = monotonic()
    try:
        values = rollout(row, trajectory, solver, settings)
        if any(
            len(v) != len(trajectory.time) or not np.isfinite(v).all()
            for v in values.values()
        ):
            raise ValueError("nonfinite or incomplete prediction")
        result = {"status": "complete", "predictions": values}
    except Exception as exc:
        result = {
            "status": "numerical_failure",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return sealed_write(
        path,
        {
            "identity": identity,
            "key": key,
            "solver": solver,
            "seconds": monotonic() - begin,
            **result,
        },
    )


def _pair(row, trajectory, settings, directory, key, scales):
    solvers = (
        ("native",) if row["semantics"] == "native_increment" else ("Radau", "BDF")
    )
    runs = [
        _cached_rollout(row, trajectory, s, settings, directory, key) for s in solvers
    ]
    if any(r["status"] != "complete" for r in runs):
        return {
            "status": "unresolved",
            "replays": [
                {k: v for k, v in r.items() if k != "predictions"} for r in runs
            ],
        }
    primary = {k: np.asarray(v) for k, v in runs[0]["predictions"].items()}
    error = {
        k: float(
            np.max(np.abs(primary[k] - np.asarray(runs[-1]["predictions"][k]))) / s
        )
        for k, s in scales.items()
    }
    return {
        "status": "pass"
        if max(error.values()) <= settings.replay_scaled_tolerance
        else "unresolved",
        "predictions": primary,
        "scaled_replay_difference": error,
        "replay_kind": "native_deterministic_execution_only"
        if len(runs) == 1
        else "Radau_BDF_agreement",
    }


def _channel(t: Trajectory, name: str) -> np.ndarray:
    return np.asarray({**t.external_inputs, **t.auxiliaries}[name])


def _rms(values: np.ndarray) -> float:
    """Avoid squaring large finite responses before normalization."""
    if not np.isfinite(values).all():
        raise ValueError("nonfinite response difference")
    maximum = float(np.max(np.abs(values)))
    return (
        maximum * float(np.sqrt(np.mean((values / maximum) ** 2))) if maximum else 0.0
    )


def probe_trajectory(train: DatasetSplit, driver: str, settings: ProbeSettings):
    """Choose one training trajectory by public excitation, independent of model fit."""
    values = np.concatenate([_channel(t, driver) for t in train.trajectories])
    lower, upper = float(np.min(values)), float(np.max(values))
    candidates = sorted(
        train.trajectories,
        key=lambda t: (-float(np.ptp(_channel(t, driver))), t.trajectory_id),
    )
    t = candidates[0]
    duration = t.time[-1] - t.time[0]
    mask = (t.time >= t.time[0] + 0.25 * duration) & (
        t.time <= t.time[0] + 0.5 * duration
    )
    after = t.time > t.time[0] + 0.5 * duration
    if upper <= lower or not mask.any() or not after.any():
        return t, [], after
    trials = []
    original = _channel(t, driver)
    for sign in (-1, 1):
        changed = original.copy()
        changed[mask] = np.clip(
            changed[mask] + sign * settings.probe_range_fraction * (upper - lower),
            lower,
            upper,
        )
        if np.array_equal(changed, original):
            continue
        field = "external_inputs" if driver in t.external_inputs else "auxiliaries"
        trial = replace(t, **{field: {**getattr(t, field), driver: changed}})
        trials.append((sign, trial))
    return t, trials, after


def assess_activity(row, train, rules, scales, settings, directory):
    """Report conditional influence and post-pulse persistence, not causal truth."""
    results = []
    for raw in rules:
        rule = EquationRequirement.model_validate(raw)
        item = {
            "requirement_id": rule.id,
            "driver": rule.driver,
            "target": rule.target,
            "status": "unresolved",
            "probes": [],
        }
        base, trials, after = probe_trajectory(train, rule.driver, settings)
        item["trajectory_id"] = base.trajectory_id
        if not trials:
            item["reason"] = "no nonzero interior probe within training channel range"
            results.append(item)
            continue
        key = ["train", train.fingerprint, base.trajectory_id]
        normal = _pair(row, base, settings, directory, key, scales)
        if normal["status"] != "pass":
            item["reason"] = "baseline rollout failed or solvers disagree"
            results.append(item)
            continue
        for sign, trial in trials:
            comparison = _pair(
                row, trial, settings, directory, [*key, rule.driver, sign], scales
            )
            part = {"sign": sign, "status": "unresolved"}
            if comparison["status"] == "pass":
                target = rule.target
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        delta = (
                            comparison["predictions"][target]
                            - normal["predictions"][target]
                        ) / scales[target]
                        effect, tail = _rms(delta), _rms(delta[after])
                except (ArithmeticError, ValueError):
                    part["reason"] = "response difference cannot be represented"
                    item["probes"].append(part)
                    continue
                threshold = max(
                    settings.activity_floor,
                    10
                    * (
                        normal["scaled_replay_difference"][target]
                        + comparison["scaled_replay_difference"][target]
                    ),
                )
                measured = tail if rule.kind == "dynamic_memory" else effect
                part.update(
                    status="pass" if measured > threshold else "unresolved",
                    scaled_output_rms=effect,
                    scaled_post_pulse_rms=tail,
                    detection_threshold=threshold,
                    measured_input_change_max=float(
                        np.max(
                            np.abs(
                                _channel(trial, rule.driver)
                                - _channel(base, rule.driver)
                            )
                        )
                    ),
                )
                if measured <= threshold:
                    part["reason"] = (
                        "no effect resolved by this bounded probe; "
                        "not proof of global inactivity"
                    )
            item["probes"].append(part)
        item["status"] = (
            "pass"
            if any(p["status"] == "pass" for p in item["probes"])
            else "unresolved"
        )
        if rule.kind == "nonlinear_feedback":
            item["conditional_input_effect_status"] = item["status"]
            item["status"] = "unresolved"
            item["reason"] = (
                "input influence alone cannot isolate feedback-loop activity"
            )
        results.append(item)
    return {
        "counts": counts(results),
        "requirements": results,
        "scope": "training_range_conditional_public_channel_influence",
        "auxiliaries_held_fixed_except_probed_channel": True,
        "initial_public_values_unchanged": True,
        "physical_intervention_correctness_certified": False,
    }


def assess_responses(row, validation, scales, settings, directory):
    """Compare complete free rollouts with observed validation response shapes."""
    rows = []
    for t in validation.trajectories:
        result = _pair(
            row,
            t,
            settings,
            directory,
            ["val", validation.fingerprint, t.trajectory_id],
            scales,
        )
        for target, scale in scales.items():
            record = {
                "trajectory_id": t.trajectory_id,
                "target": target,
                "status": "unresolved",
                "samples": len(t.time),
                "public_input_ranges": {
                    k: [float(np.min(v)), float(np.max(v))]
                    for k, v in t.external_inputs.items()
                },
            }
            if result["status"] != "pass":
                record["reason"] = "rollout failed or numerical replay disagreement"
            else:
                pred, observed = result["predictions"][target], t.targets[target]
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        nmse = finite_mean(((pred - observed) / scale) ** 2)
                        direction, shape, timing = qualitative_response_metrics(
                            t.time, pred, observed
                        )
                    if any(
                        x is not None and not np.isfinite(x)
                        for x in (nmse, shape, timing)
                    ):
                        raise ValueError("nonfinite response metric")
                except (ArithmeticError, ValueError):
                    record["reason"] = "response metrics cannot be represented"
                    rows.append(record)
                    continue
                record.update(
                    nmse=nmse,
                    response_direction_correct=direction,
                    response_shape_correlation=shape,
                    relative_peak_timing_error=timing,
                    scaled_response_amplitude=float(
                        np.max(np.abs(observed - observed[0])) / scale
                    ),
                    scaled_replay_difference=result["scaled_replay_difference"][target],
                )
                if record["scaled_response_amplitude"] <= settings.activity_floor:
                    record["reason"] = (
                        "observed response is flat at the declared resolution"
                    )
                    record.update(
                        response_direction_correct=None,
                        response_shape_correlation=None,
                        relative_peak_timing_error=None,
                    )
                else:
                    good = (
                        nmse <= settings.response_nmse_limit
                        and direction is True
                        and shape is not None
                        and shape >= settings.response_shape_minimum
                        and timing is not None
                        and timing <= settings.response_timing_limit
                    )
                    record["status"] = "pass" if good else "fail"
            rows.append(record)
    per_target = {}
    if all("nmse" in r for r in rows):
        for target in scales:
            selected = [r for r in rows if r["target"] == target]
            denominator = sum(r["samples"] for r in selected)
            per_target[target] = sum(
                r["nmse"] * (r["samples"] / denominator) for r in selected
            )
    return {
        "counts": counts(rows),
        "trajectories": rows,
        "normalized_mse": finite_mean(list(per_target.values()))
        if per_target
        else None,
        "per_target_normalized_mse": per_target,
        "scope": "observed_validation_response_agreement",
        "prior_validation_use": "see_source_provenance; may include model selection",
        "pristine_test_set": False,
        "mechanism_specific_causal_attribution": False,
        "limitation": "Validation may already have selected these models. Supplied "
        "auxiliaries condition predictions. No counterfactual reference is invented.",
    }


def numerical_assessment(
    row: dict,
    train: DatasetSplit,
    validation: DatasetSplit,
    settings: ProbeSettings,
    directory: Path,
) -> dict:
    """Evaluate layers 2/3 with saved parameters and train-derived normalization."""
    if train.name is not SplitName.TRAIN or validation.name is not SplitName.VALIDATION:
        raise ValueError("only TRAIN and VALIDATION allowed; never TEST")
    if not train.trajectories or not validation.trajectories:
        raise ValueError("nonempty splits required")
    scaling = TrainingScaler().fit(train).scales
    context = ValidationContext.model_validate(row["context"])
    scales = {
        t: float(scaling[f"target:{t}"].standard_deviation) for t in context.targets
    }
    return {
        "normalization_scales": scales,
        "fitted_activity": assess_activity(
            row, train, row["requirements"], scales, settings, directory
        ),
        "response_behavior": assess_responses(
            row, validation, scales, settings, directory
        ),
        "parameter_refit_applied": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
