"""Opt-in fixed-noise fitting with bounded searches and independent retention.

The legacy rollout optimizer supplies guarded coordinates and bounded TRF.
Its historical internal metric names are translated at this boundary: weighted
measurement residuals are not NMSE, and no noiseless target ends a noisy fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import measurement_rollout as rollout
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.optimizer_scaling import RelativeScaling
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.measurement_fitting import MeasurementProblem


class Policy(StrictSchema):
    """Total search/verification allowances, shared by all frozen starts."""

    protocol: Literal["measurement-fitting-policy-1"] = "measurement-fitting-policy-1"
    search_seconds: FiniteFloat = Field(default=30, ge=1, le=300)
    verification_seconds: FiniteFloat = Field(default=15, ge=1, le=120)
    maximum_calls: int = Field(default=100, ge=2, le=1000)
    restart_count: int = Field(default=1, ge=0, le=3)
    random_seed: int = Field(default=20261014, ge=0, le=2**32 - 1)
    scaling: RelativeScaling = Field(default_factory=RelativeScaling)
    maximum_solver_difference: FiniteFloat = Field(default=1e-6, gt=0, le=1e-3)

    @model_validator(mode="after")
    def enough_calls(self):
        if self.maximum_calls < self.restart_count + 1:
            raise ValueError("each frozen start needs at least one call")
        return self


def _measurement_names(value):
    """Translate private optimizer telemetry without relabeling it as NMSE."""
    names = {
        "training_nmse": "mean_squared_standardized_residual",
        "best_training_nmse": "best_mean_squared_standardized_residual",
    }
    if isinstance(value, dict):
        return {names.get(k, k): _measurement_names(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_measurement_names(v) for v in value]
    return value


def _search(oracle, profile, start, policy, calls, deadline, save):
    """A failed profile may fall back, but cannot renew the block's allowance."""
    failures = []
    subattempts = []
    used = completed = 0
    best = None
    if profile is not None and not profile.outer:
        save({"calls": 1, "completed_calls": 0})
        used = 1
        try:
            vector, residual, _, audit = profile.evaluate(
                oracle.vector(start), deadline
            )
            best = {
                "parameters": oracle.parameters(vector),
                "mean_squared_standardized_residual": float(np.mean(residual**2)),
            }
            result = {
                "calls": 1,
                "completed_calls": 1,
                "best": best,
                "stop_reason": "bounded_affine_problem_solved",
                "projection": audit,
                "solution_scope": "fixed_weight_affine_output_family",
            }
            save(result)
            return result
        except (ValueError, RuntimeError, ArithmeticError) as error:
            failures.append(str(error)[-1000:])
            subattempts.append(
                {
                    "kernel": "fully_affine_profile",
                    "calls": 1,
                    "completed_calls": 0,
                    "stop_reason": "numerical_failure",
                    "message": failures[-1],
                }
            )
            profile = None

    def better(first, second):
        if first is None:
            return second
        if second is None:
            return first
        key = "mean_squared_standardized_residual"
        return first if first[key] <= second[key] else second

    def search(kernel, initial, remaining):
        def checkpoint(value):
            converted = _measurement_names(value)
            converted["calls"] = used + value["calls"]
            converted["completed_calls"] = completed + value.get("completed_calls", 0)
            converted["best"] = better(best, converted.get("best"))
            converted["subattempts"] = list(subattempts)
            converted["profile_failures"] = list(failures)
            save(converted)

        return _measurement_names(
            numerical.search(
                oracle,
                initial,
                deadline,
                checkpoint,
                calls=remaining,
                target=-1.0,  # Disable the historical noiseless accuracy stop.
                profile=kernel,
                telemetry=True,
                optimizer_scaling="guarded_relative",
                scaling_policy=policy.scaling,
                normalized_objective=False,
                objective_multiplier=1.0,
            )
        )

    if used >= calls or monotonic() >= deadline:
        return {
            "calls": used,
            "completed_calls": completed,
            "best": best,
            "stop_reason": "budget_limited",
            "profile_failures": failures,
            "subattempts": subattempts,
        }
    result = search(profile, start, calls - used)
    subattempts.append(result | {"kernel": "profile" if profile else "joint"})
    if profile is not None and result["stop_reason"] == "numerical_failure":
        failures.append(result.get("message", "profile unavailable"))
        used += result["calls"]
        completed += result["completed_calls"]
        best = result["best"]
        save(
            {
                "calls": used,
                "completed_calls": completed,
                "best": best,
                "subattempts": subattempts,
                "profile_failures": failures,
            }
        )
        if used < calls and monotonic() < deadline:
            result = search(None, best["parameters"] if best else start, calls - used)
            subattempts.append(result | {"kernel": "joint_fallback"})
            result["best"] = better(best, result["best"])
        else:
            result = {
                "calls": 0,
                "completed_calls": 0,
                "best": best,
                "stop_reason": "budget_limited",
            }
    result["calls"] += used
    result["completed_calls"] += completed
    result["profile_failures"] = failures
    result["subattempts"] = subattempts
    save(result)
    return result


def _verify(oracle, parameters, policy, deadline, save):
    """Compare original-equation solvers, including the complete initial rule."""
    vector = oracle.vector(parameters)
    save({"calls": 1, "completed_calls": 0})
    primary, jacobian = oracle.evaluate(vector, deadline, method="DOP853")
    save({"calls": 2, "completed_calls": 1})
    alternate, _ = oracle.evaluate(vector, deadline, method="Radau", jacobian=False)
    save({"calls": 2, "completed_calls": 2})
    difference = float(np.max(np.abs(primary - alternate)))
    first, second = float(primary @ primary / 2), float(alternate @ alternate / 2)
    passed = bool(
        monotonic() < deadline
        and np.isfinite(np.r_[primary, alternate, jacobian.ravel()]).all()
        and difference <= policy.maximum_solver_difference
    )
    return {
        "parameters": parameters,
        "passed": passed,
        "maximum_standardized_solver_difference": difference,
        "primary_half_sum_squares": first,
        "alternate_half_sum_squares": second,
        "standardized_residuals": primary.tolist(),
        "joint_scaled_singular_values": np.linalg.svd(
            jacobian * oracle.units, compute_uv=False
        ).tolist(),
        "parameter_order": list(oracle.names),
        "sensitivity_scope": "local_joint_whitened_J; no interval or coverage claim",
        "derivatives_independently_qualified": False,
        "method": "original_equations_DOP853_vs_Radau",
        "calls": 2,
        "completed_calls": 2,
    }


def _progress_point(folder, oracle):
    """Recover only a validated vector; a progress-file loss is never trusted."""
    path = folder / "progress.json"
    if not path.exists():
        return None
    try:
        values = json.loads(path.read_text()).get("best", {}).get("parameters")
        if values is not None:
            return oracle.parameters(oracle.vector(values))
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def _cost(operations, setup_seconds):
    """Aggregate actual stages; interrupted usage and allowance stay visible."""
    search = [op for op in operations if op["stage"] == "search"]
    known_stages = all(op["accounting_complete"] for op in operations)

    def count(field):
        values = [
            (op.get("value") or {}).get(field, op.get(field))
            if op["accounting_complete"]
            else None
            for op in search
        ]
        return sum(values) if all(v is not None for v in values) else None

    return {
        "wall_seconds": sum(op["wall_seconds"] or 0 for op in operations)
        + setup_seconds["wall"],
        "cpu_seconds": sum(op["cpu_seconds"] or 0 for op in operations)
        + setup_seconds["cpu"],
        "timing_scope": "known completed stages plus recorded setup; unknown omitted",
        "search_calls_started": count("calls"),
        "search_calls_completed": count("completed_calls"),
        "verification_calls_started": sum(
            op.get("calls") or 0 for op in operations if op["stage"] == "verify"
        )
        if known_stages
        else None,
        "budget_charge_seconds": sum(op["budget_charge_seconds"] for op in operations),
        "accounting_complete": known_stages
        and setup_seconds["complete"]
        and count("calls") is not None
        and count("completed_calls") is not None,
        "setup_wall_seconds": setup_seconds["wall"],
        "setup_cpu_seconds": setup_seconds["cpu"],
        "interrupted_setup_attempts": setup_seconds["interrupted"],
    }


def run(
    problem: MeasurementProblem,
    policy: Policy,
    folder: Path,
    *,
    strategy: Literal["direct", "profiled"] = "profiled",
) -> dict:
    """Fit only training readings; resume stages without resetting their budgets."""
    if not isinstance(problem, MeasurementProblem) or not isinstance(policy, Policy):
        raise TypeError("typed measurement problem and policy required")
    problem = MeasurementProblem.model_validate(problem.model_dump(mode="json"))
    policy = Policy.model_validate(policy.model_dump(mode="json"))
    if strategy not in {"direct", "profiled"}:
        raise ValueError("unsupported measurement fitting strategy")
    identity = {
        "protocol": "measurement-fitting-execution-1",
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "base_request_sha256": public.content_sha256(
            problem.request.model_dump(mode="json")
        ),
        "policy": policy.model_dump(mode="json"),
        "strategy": strategy,
    }
    with public._lock(folder):
        seal(folder / "identity.json", identity)
        output = folder / "result.json"
        if output.exists():
            saved = read_seal(output)
            if saved["identity"] != identity:
                raise ValueError("measurement fit resume identity differs")
            return saved
        # Compilation can be interrupted too. Preserve its unknown cost even
        # though reconstructing the numerical object is necessary for resume.
        setup_folder = folder / "setup"
        setup_folder.mkdir(exist_ok=True)
        setup_attempt = setup_folder / f"{len(list(setup_folder.iterdir())):04d}"
        seal(
            setup_attempt / "started.json",
            {"fit_identity": public.content_sha256(identity)},
        )
        begun, cpu = monotonic(), process_time()
        oracle, profile, route = rollout.build(problem, checks.ConfidencePolicy())
        if strategy == "direct":
            profile = None
            route = route | {"selected": "joint_rollout", "requested": "direct"}
        starts = [
            dict(problem.start),
            *numerical.diverse_starts(
                oracle, problem, policy.restart_count, policy.random_seed
            ),
        ]
        seal(folder / "starts.json", starts)
        setup = {"wall": monotonic() - begun, "cpu": process_time() - cpu}
        seal(setup_attempt / "result.json", setup)
        setups = [read_seal(p) for p in sorted(setup_folder.glob("*/result.json"))]
        setup_total = {key: sum(s[key] for s in setups) for key in ("wall", "cpu")}
        interrupted = len(list(setup_folder.glob("*/started.json"))) - len(setups)
        setup_total.update(complete=not interrupted, interrupted=interrupted)
        operations, candidates, selected = [], [], None
        verify_seconds = policy.verification_seconds / (len(starts) + 1)

        def operation(stage, name, seconds, point, work):
            op = checks.operation(
                folder / name,
                {"fit_identity": public.content_sha256(identity), "point": point},
                seconds,
                work,
            ) | {"stage": stage, "name": name}
            operations.append(op)
            return op

        def verify(name, parameters):
            nonlocal selected
            op = operation(
                "verify",
                name,
                verify_seconds,
                parameters,
                lambda deadline, save: _verify(
                    oracle, parameters, policy, deadline, save
                ),
            )
            result = op.get("value")
            candidates.append({"origin": name, "status": op["status"], "value": result})
            if (
                result
                and result["passed"]
                and (
                    selected is None
                    or max(
                        result["primary_half_sum_squares"],
                        result["alternate_half_sum_squares"],
                    )
                    < min(
                        selected["primary_half_sum_squares"],
                        selected["alternate_half_sum_squares"],
                    )
                )
            ):
                selected = result | {"origin": name}

        verify("incumbent", dict(problem.incumbent))
        attempts = []
        for index, start in enumerate(starts):
            calls = policy.maximum_calls // len(starts) + (
                index < policy.maximum_calls % len(starts)
            )
            name = f"search-{index}"
            op = operation(
                "search",
                name,
                policy.search_seconds / len(starts),
                start,
                lambda deadline, save, start=start, calls=calls: _search(
                    oracle, profile, start, policy, calls, deadline, save
                ),
            )
            attempt = op.get("value") or {"stop_reason": op["status"]}
            attempts.append(attempt)
            point = (attempt.get("best") or {}).get("parameters")
            if point is None:
                point = _progress_point(folder / name, oracle)
            if point is not None:
                verify(f"verify-{index}", point)
            if attempt["stop_reason"] == "bounded_affine_problem_solved":
                break  # No nonlinear outer coordinates need a different start.

        evidence = None
        if selected:
            evidence = oracle.residual_evidence(
                np.asarray(selected["standardized_residuals"])
            )
        cost = _cost(operations, setup_total)
        result = {
            "protocol": "measurement-fit-result-1",
            "identity": identity,
            "status": "complete" if selected else "retained_unverified",
            "parameters": selected["parameters"] if selected else None,
            "objective_half_sum_squares": selected["primary_half_sum_squares"]
            if selected
            else None,
            "negative_log_likelihood": (
                selected["primary_half_sum_squares"] + oracle.likelihood_constant
                if selected
                else None
            ),
            "selected_origin": selected["origin"] if selected else None,
            "search": {
                "stop_reason": attempts[-1]["stop_reason"],
                "calls_started": cost["search_calls_started"],
                "calls_completed": cost["search_calls_completed"],
                "attempts": attempts,
                "noise_adequacy_used_as_stopping_rule": False,
            },
            "verification": selected or {"passed": False},
            "residual_evidence": evidence,
            "route": route,
            "cost": cost,
            "operations": operations,
            "candidate_checks": candidates,
            "confidence_probability": None,
            "specification_verdict": "not_assessed",
            "data_fault_verdict": "not_assessed",
            "reference_values_used": False,
            "validation_used_for_selection": False,
            "test_data_opened": False,
        }
        seal(output, result)
        return result
