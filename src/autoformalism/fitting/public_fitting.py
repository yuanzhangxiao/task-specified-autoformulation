"""Frozen public fit requests with explicit backends and no automatic retry.

The numerical algorithms are unchanged. Preparation lowers shared initializers
once; execution pins that lowering, public arrays, source and profile settings.
An interrupted attempt retains its files and never receives a fresh budget.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import time

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.fitter import fit_candidate
from autoformalism.fitting.initialization import apply_initialization_plan
from autoformalism.fitting.models import FitConfig
from autoformalism.schemas.public_fitting import (
    PublicFitMetrics,
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)


def _jsonable(value):
    """Serialize diagnostic dataclasses, including immutable mapping proxies."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def content_sha256(value) -> str:
    """Canonical JSON digest, also used by exporters to join exact lineage."""
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value) -> None:
    """Atomically publish a JSON checkpoint; callers hold the attempt lock."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_jsonable(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("public fit directory is in use") from error
        yield


def _source_identity() -> str:
    """Bind execution to this Python source tree, independent of checkout path."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "pydantic", "casadi"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def profile_settings(request: PublicFitRequest) -> dict:
    """Materialize the two existing profiles, without promoting diagnostic budgets."""
    if request.profile == "general-rollout-v1":
        return FitConfig(
            number_of_starts=1,
            random_seed=request.random_seed,
            allow_derivative_regression=False,
            parameter_fit_strategy="bounded_nonlinear",
            nonlinear_initializer="none",
            maximum_function_evaluations=240,
            maximum_wall_time_seconds=300,
            integration_backend="solve_ivp",
            integration_method="Radau",
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
        ).model_dump(mode="json")
    # Stored explicitly so prepare/inspect do not require optional CasADi.
    return {
        "protocol": "collocation-forward-sensitivity-1",
        "initializer_seconds": 120.0,
        "refinement_seconds": 180.0,
        "maximum_function_evaluations": 240,
        "integration_method": "Radau",
        "relative_tolerance": 1e-7,
        "absolute_tolerance": 1e-9,
        "failure_penalty": 1e6,
        "collocation_node_start": "rollout_or_observed",
        "node_warmup_seconds": 5.0,
        "piecewise_policy": "allow",
        "piecewise_refinement": "auto",
        "collocation_mesh_substeps": 1,
        "least_squares_ftol": None,
        "recovery_policy": "feasible",
        "collocation_diagnostics": True,
        "recovery_max_starts": 10,
        "recovery_probe_seconds": 10.0,
    }


def pack_split(split: DatasetSplit) -> PublicSplit:
    """Export development observations only; omit all derivative labels."""
    return PublicSplit.model_validate(
        {
            "name": split.name.value,
            "fingerprint": split.fingerprint,
            "rows": [
                {
                    "trajectory_id": row.trajectory_id,
                    "time": row.time.tolist(),
                    **{
                        role: {
                            key: values.tolist()
                            for key, values in getattr(row, role).items()
                        }
                        for role in ("targets", "auxiliaries", "external_inputs")
                    },
                    "fixed_covariates": dict(row.fixed_covariates),
                }
                for row in split.trajectories
            ],
        }
    )


def unpack_split(split: PublicSplit) -> DatasetSplit:
    """Reconstruct immutable arrays; backend identity uses their content digest."""
    return DatasetSplit(
        SplitName(split.name),
        tuple(
            Trajectory(
                trajectory_id=row.trajectory_id,
                time=np.asarray(row.time, dtype=float),
                **{
                    role: {
                        key: np.asarray(values, dtype=float)
                        for key, values in getattr(row, role).items()
                    }
                    for role in ("targets", "auxiliaries", "external_inputs")
                },
                fixed_covariates=dict(row.fixed_covariates),
                derivatives={},
            )
            for row in split.rows
        ),
        content_sha256(split),
    )


def _lower(request: PublicFitRequest):
    if request.context.lagged_targets or request.context.fitted_initialization:
        raise ValueError("request requires an open-rollout context before lowering")
    candidate = request.base_candidate
    if any(p.scope.value != "global" for p in candidate.parameters):
        raise ValueError("only shared global parameters are allowed")
    if any(
        i.scope.value != "global" or i.initialization_range is not None
        for i in candidate.initial_conditions
    ):
        raise ValueError(
            "trajectory-specific or ranged initial conditions are forbidden"
        )
    model = compile_candidate(candidate, request.context)
    latent = set(model.state_names) - set(model.direct_state_observation_channels)
    if set(request.initialization_plan.rules) != latent:
        raise ValueError("initialization plan must cover exactly the latent states")
    if set(request.parameter_guesses) - set(model.parameter_names):
        raise ValueError("parameter_guesses may name only base equation parameters")
    lowered, guesses, audit = apply_initialization_plan(
        model, request.initialization_plan
    )
    return lowered, {**dict(request.parameter_guesses), **guesses}, audit


def _check_data(request, training, validation):
    if training.name != "train" or validation.name != "val":
        raise ValueError("requires train then val; test data are forbidden")
    if training.fingerprint == validation.fingerprint:
        raise ValueError("training and validation fingerprints must differ")
    for split in (training, validation):
        for row in split.rows:
            for role in (
                "targets",
                "auxiliaries",
                "external_inputs",
                "fixed_covariates",
            ):
                if set(getattr(row, role)) != set(getattr(request.context, role)):
                    raise ValueError(f"{split.name}/{row.trajectory_id}: {role} differ")


def _bundle(request, training, validation):
    _check_data(request, training, validation)
    model, guesses, audit = _lower(request)
    payload = {
        "protocol": "public-fit-freeze-1",
        "request": _jsonable(request),
        "training": _jsonable(training),
        "validation": _jsonable(validation),
        "lowered_candidate": model.validated.candidate.model_dump(mode="json"),
        "lowered_context": model.validated.context.model_dump(mode="json"),
        "initial_parameters": guesses,
        "initialization_audit": audit,
        "settings": profile_settings(request),
        "source_sha256": _source_identity(),
    }
    return {**payload, "identity": content_sha256(payload)}


def prepare_fit(
    request: PublicFitRequest,
    training: DatasetSplit | PublicSplit,
    validation: DatasetSplit | PublicSplit,
    directory: Path,
) -> dict:
    """Seal an exact public handoff; identical preparation is a read-only resume."""
    training = pack_split(training) if isinstance(training, DatasetSplit) else training
    validation = (
        pack_split(validation) if isinstance(validation, DatasetSplit) else validation
    )
    frozen = _bundle(request, training, validation)
    with _lock(directory):
        path = directory / "freeze.json"
        if path.exists():
            if _read(path) != frozen:
                raise ValueError(
                    "existing public fit request, data, settings or source differ"
                )
        else:
            if any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError(
                    "cannot prepare in a nonempty directory without a freeze"
                )
            _write(path, frozen)
    return {
        "identity": frozen["identity"],
        "profile": request.profile,
        "directory": str(directory),
        "status": "prepared",
    }


def _load(directory):
    frozen = _read(directory / "freeze.json")
    request = PublicFitRequest.model_validate(frozen["request"])
    training = PublicSplit.model_validate(frozen["training"])
    validation = PublicSplit.model_validate(frozen["validation"])
    if _bundle(request, training, validation) != frozen:
        raise ValueError("frozen content, lowered contract, settings or source changed")
    return frozen, request, training, validation


def _capability(request, model) -> str | None:
    if request.profile == "general-rollout-v1":
        return None
    if request.profile == "collocation-feasible-v1" and tuple(
        request.context.targets
    ) != ("v01",):
        return "collocation-feasible-v1 currently requires the single target v01"
    if len(request.context.targets) != 1:
        return "collocation-single-target-v2 requires exactly one public target"
    try:
        from autoformalism.fitting.sensitivity_probe import (
            SensitivityContractError,
            SymbolicODE,
        )
    except ImportError as error:
        return f"collocation dependency unavailable: {error}"
    try:
        SymbolicODE(model, allow_piecewise=True)
    except SensitivityContractError as error:
        return str(error)
    return None


def inspect_fit(directory: Path) -> dict:
    """Verify identity and backend capability, without an optimizer attempt."""
    frozen, request, _, _ = _load(directory)
    model, _, _ = _lower(request)
    issue = _capability(request, model)
    return {
        "identity": frozen["identity"],
        "profile": request.profile,
        "capability_supported": issue is None,
        "capability_message": issue,
        "settings": frozen["settings"],
        "attempt_started": (directory / "started.json").exists(),
        "result_exists": (directory / "result.json").exists(),
    }


def _metrics(raw, targets) -> PublicFitMetrics:
    raw = raw or {}
    failed = raw.get("failed_trajectories", ())
    score = raw.get("normalized_mse")
    per_target = raw.get("per_target_normalized_mse", {})
    complete = (
        not failed
        and isinstance(score, (float, int))
        and math.isfinite(score)
        and score >= 0
        and set(per_target) == set(targets)
        and all(
            isinstance(v, (float, int)) and math.isfinite(v) and v >= 0
            for v in per_target.values()
        )
    )
    return PublicFitMetrics(
        available=complete,
        normalized_mse=score if complete else None,
        per_target_normalized_mse=per_target if complete else {},
        failed_trajectories=tuple(failed),
    )


def _run_backend(request, model, training, validation, guesses, settings, directory):
    if request.profile == "general-rollout-v1":
        return _jsonable(
            fit_candidate(
                model,
                training,
                validation,
                FitConfig.model_validate(settings),
                initial_global_parameters=guesses,
            )
        )
    from autoformalism.fitting.collocation_sensitivity import (
        CollocationSensitivityConfig,
        _role_start,
        fit_collocation_forward_sensitivity,
    )

    return _jsonable(
        fit_collocation_forward_sensitivity(
            model,
            training,
            validation,
            CollocationSensitivityConfig.model_validate(settings),
            directory / "backend",
            initial_parameters={
                **_role_start(model.validated.candidate, training),
                **guesses,
            },
        )
    )


def _evidence(request, raw):
    if request.profile == "general-rollout-v1":
        diagnostics = raw.get("diagnostics", [])
        selected = next(
            (d for d in diagnostics if d["start_index"] == raw.get("best_start_index")),
            {},
        )
        counts = [d.get("actual_residual_evaluations") for d in diagnostics]
        return {
            "parameters": raw.get("global_parameters"),
            "training": _metrics(raw.get("training_metrics"), request.context.targets),
            "validation": _metrics(
                raw.get("validation_metrics"), request.context.targets
            ),
            "native_optimizer_converged": selected.get("success"),
            "budget_exhausted": any(
                d.get("status") in (0, -2) or d.get("retained_best_on_timeout", False)
                for d in diagnostics
            ),
            "actual_residual_calls": sum(counts)
            if counts and None not in counts
            else None,
        }
    refinement = raw.get("refinement") or {}
    return {
        "parameters": raw.get("parameters"),
        "training": _metrics(raw.get("training"), request.context.targets),
        "validation": _metrics(raw.get("validation"), request.context.targets),
        # Feasibility recovery retains the best rollout across stages. Its wrapper
        # native-success placeholder does not identify a converged selected point.
        "native_optimizer_converged": None,
        "budget_exhausted": refinement.get("budget_exhausted"),
        "actual_residual_calls": refinement.get("actual_residual_calls"),
    }


def _result_base(frozen, request):
    return {
        "identity": frozen["identity"],
        "profile": request.profile,
        "request_sha256": content_sha256(request),
        "lowered_candidate_sha256": content_sha256(frozen["lowered_candidate"]),
        "initialization_plan_sha256": content_sha256(request.initialization_plan),
        "training_content_sha256": content_sha256(frozen["training"]),
        "validation_content_sha256": content_sha256(frozen["validation"]),
        "source": request.source,
    }


def execute_fit(directory: Path) -> PublicFitResult:
    """Execute once or return the sealed result without replenishing budgets."""
    with _lock(directory):
        frozen, request, train, val = _load(directory)
        base = _result_base(frozen, request)
        result_path = directory / "result.json"
        if result_path.exists():
            saved = _read(result_path)
            if content_sha256(saved["result"]) != saved["sha256"]:
                raise ValueError("public fit result digest differs")
            result = PublicFitResult.model_validate(saved["result"])
            if any(
                _jsonable(getattr(result, key)) != _jsonable(value)
                for key, value in base.items()
            ):
                raise ValueError("public fit result lineage differs")
            if (
                result.backend_result_sha256 is not None
                and content_sha256(_read(directory / "backend_result.json"))
                != result.backend_result_sha256
            ):
                raise ValueError("backend result digest differs")
            return result
        started = directory / "started.json"
        if started.exists():
            if _read(started)["identity"] != frozen["identity"]:
                raise ValueError("started marker identity differs")
            result = PublicFitResult(
                **base,
                status="interrupted",
                message=(
                    "Previous attempt has no terminal result; files retained, "
                    "no fresh budget."
                ),
            )
        else:
            model, guesses, _ = _lower(request)
            issue = _capability(request, model)
            if issue:
                result = PublicFitResult(
                    **base, status="capability_unsupported", message=issue
                )
            else:
                _write(
                    started,
                    {
                        "identity": frozen["identity"],
                        "utc_seconds": time(),
                        "runtime": _runtime(),
                    },
                )
                try:
                    raw = _run_backend(
                        request,
                        model,
                        unpack_split(train),
                        unpack_split(val),
                        guesses,
                        frozen["settings"],
                        directory,
                    )
                    _write(directory / "backend_result.json", raw)
                    evidence = _evidence(request, raw)
                    complete = (
                        evidence["parameters"] is not None
                        and evidence["training"].available
                        and evidence["validation"].available
                    )
                    result = PublicFitResult(
                        **base,
                        **evidence,
                        status="complete" if complete else "fit_failed",
                        backend_result_sha256=content_sha256(raw),
                        message=(
                            "Finite complete train/validation rollouts; accuracy is "
                            "reported separately."
                            if complete
                            else "No complete finite train/validation fit; "
                            "see backend evidence."
                        ),
                    )
                except Exception as error:
                    result = PublicFitResult(
                        **base,
                        status="fit_failed",
                        message=f"{type(error).__name__}: {error}",
                    )
        payload = result.model_dump(mode="json")
        _write(result_path, {"result": payload, "sha256": content_sha256(payload)})
        return result
