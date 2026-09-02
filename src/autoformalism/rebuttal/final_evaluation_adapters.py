"""Method-specific adapters for the common frozen evaluation subject."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.baselines.models import BaselineResult
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.baseline_postfreeze import FrozenBaselineModel
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    FrozenParameterization,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
)
from autoformalism.schemas import CandidateModel, ParameterScope

SourceKind = Literal[
    "autoformalism",
    "raw_data_agent",
    "sindy",
    "pysr",
    "d3",
]


class SourceAdapterRequest(BaseModel):
    """One prespecified source artifact to adapt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    source_kind: SourceKind
    source_path: Path
    method_label: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )
    expected_benchmark_id: str | None = None
    expected_tier: str | None = None
    expected_repetition: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def expected_identity_is_complete(self) -> SourceAdapterRequest:
        """Require either a complete prespecified identity or no identity."""
        if self.method_label is not None and self.source_kind != "autoformalism":
            raise ValueError(
                "method_label is supported only for Autoformalism ablations"
            )
        values = (
            self.expected_benchmark_id,
            self.expected_tier,
            self.expected_repetition,
        )
        if any(value is not None for value in values) and any(
            value is None for value in values
        ):
            raise ValueError("expected source identity must be complete")
        return self


class SourceAdapterOutcome(BaseModel):
    """Explicit success or failure for one requested source artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    source_kind: SourceKind
    source_path: str
    status: Literal["adapted", "failed"]
    subject_id: str | None = None
    error_type: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def fields_match_status(self) -> SourceAdapterOutcome:
        """Keep adapted identifiers and failure diagnostics mutually exclusive."""
        if self.status == "adapted":
            if (
                self.subject_id is None
                or self.error_type is not None
                or self.error is not None
            ):
                raise ValueError("adapted source requires only a subject identifier")
        elif self.subject_id is not None or not self.error_type or not self.error:
            raise ValueError("failed source requires error diagnostics only")
        return self


def adapt_source(
    request: SourceAdapterRequest,
    context: ValidationContext,
) -> FrozenEvaluationSubject:
    """Adapt one frozen method artifact without opening or copying test metrics."""
    if request.source_kind == "autoformalism":
        return _adapt_autoformalism(request, context)
    if request.source_kind == "raw_data_agent":
        return _adapt_raw_agent(request, context)
    if request.source_kind in {"sindy", "pysr"}:
        return _adapt_symbolic_baseline(request, context)
    return _adapt_d3(request, context)


def source_identity(request: SourceAdapterRequest) -> tuple[str, str, int]:
    """Read only public run identity needed to construct the validation context."""
    path = request.source_path.expanduser().resolve()
    expected: tuple[str, str, int] | None = None
    if request.expected_benchmark_id is not None:
        assert request.expected_tier is not None
        assert request.expected_repetition is not None
        expected = (
            request.expected_benchmark_id,
            request.expected_tier,
            request.expected_repetition,
        )
    identity_path = (
        path / "run_config.json" if request.source_kind == "raw_data_agent" else path
    )
    if not identity_path.is_file():
        if expected is None:
            raise ValueError(f"required source artifact is missing: {identity_path}")
        return expected
    if request.source_kind == "raw_data_agent":
        config = _read_object(identity_path)
        actual = (
            str(config["benchmark_id"]),
            str(config["tier"]),
            int(config["repetition"]),
        )
    else:
        payload = _read_object(identity_path)
        actual = (
            str(payload["benchmark_id"]),
            str(payload["tier"]),
            int(payload.get("seed", 0)),
        )
    if expected is not None and actual != expected:
        raise ValueError(
            "source identity differs from the prespecified request; "
            f"expected={expected}, actual={actual}"
        )
    return actual


def _adapt_autoformalism(
    request: SourceAdapterRequest,
    context: ValidationContext,
) -> FrozenEvaluationSubject:
    path = _required_file(request.source_path)
    payload = _read_object(path)
    if payload.get("status") != "complete":
        raise ValueError("Autoformalism summary is not complete")
    raw_candidate = payload.get("selected_candidate")
    if not isinstance(raw_candidate, dict):
        raise ValueError("Autoformalism summary has no selected candidate")
    candidate = CandidateModel.model_validate(raw_candidate)
    parameters = _numeric_mapping(payload.get("final_global_parameters", {}))
    initials = _numeric_mapping(payload.get("final_global_initial_conditions", {}))
    parameterization = _parameterization(candidate, parameters, initials)
    return _subject(
        request=request,
        method=request.method_label or "autoformalism",
        benchmark_id=str(payload["benchmark_id"]),
        tier=str(payload["tier"]),
        repetition=int(payload.get("seed", 0)),
        candidate=candidate,
        parameterization=parameterization,
        context=context,
        source_path=path,
        source_hash=_sha256(path),
    )


def _adapt_raw_agent(
    request: SourceAdapterRequest,
    context: ValidationContext,
) -> FrozenEvaluationSubject:
    run = request.source_path.expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"raw-agent source is not a directory: {run}")
    required = {
        name: run / name
        for name in ("run_config.json", "candidate.json", "evaluation.json")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f"raw-agent source files are missing: {missing}")
    config = _read_object(required["run_config.json"])
    evaluation = _read_object(required["evaluation.json"])
    candidate = CandidateModel.model_validate_json(
        required["candidate.json"].read_text(encoding="utf-8")
    )
    if evaluation.get("schema_version") == "raw-data-agent-fitted-evaluation-1":
        parameters = _numeric_mapping(evaluation.get("fitted_parameter_values", {}))
        initials: dict[str, float] = {}
    elif evaluation.get("schema_version") == "raw-data-agent-evaluation-1":
        fit = evaluation.get("fit")
        if not isinstance(fit, dict):
            raise ValueError("structure-only raw-agent evaluation has no fit")
        parameters = _numeric_mapping(fit.get("global_parameters", {}))
        initials = _numeric_mapping(fit.get("global_initial_conditions", {}))
    else:
        raise ValueError("unsupported raw-agent evaluation schema")
    parameterization = _parameterization(candidate, parameters, initials)
    source_files = tuple(sorted(required.values()))
    auxiliary = {path.name: _sha256(path) for path in source_files}
    return _subject(
        request=request,
        method=f"raw_data_agent:{config['provider']}:{config['model']}",
        benchmark_id=str(config["benchmark_id"]),
        tier=str(config["tier"]),
        repetition=int(config["repetition"]),
        candidate=candidate,
        parameterization=parameterization,
        context=context,
        source_path=run,
        source_hash=_combined_sha256(source_files),
        auxiliary_hashes=auxiliary,
    )


def _adapt_symbolic_baseline(
    request: SourceAdapterRequest,
    context: ValidationContext,
) -> FrozenEvaluationSubject:
    path = _required_file(request.source_path)
    raw = path.read_text(encoding="utf-8")
    payload = _read_object(path)
    if payload.get("schema_version") == "phase-b-frozen-baseline-model-1":
        result: BaselineResult | FrozenBaselineModel = (
            FrozenBaselineModel.model_validate_json(raw)
        )
        if result.test_data_opened is not False:
            raise ValueError("frozen symbolic source has opened test data")
    else:
        result = BaselineResult.model_validate_json(raw)
    if result.method != request.source_kind:
        raise ValueError(
            f"baseline method {result.method!r} does not match {request.source_kind!r}"
        )
    candidate = _equation_candidate(result.method, result.equations, context)
    return _subject(
        request=request,
        method=result.method,
        benchmark_id=result.benchmark_id,
        tier=result.tier,
        repetition=result.seed,
        candidate=candidate,
        parameterization=FrozenParameterization(status="not_required"),
        context=context,
        source_path=path,
        source_hash=_sha256(path),
    )


def _adapt_d3(
    request: SourceAdapterRequest,
    context: ValidationContext,
) -> FrozenEvaluationSubject:
    path = _required_file(request.source_path)
    result = BaselineResult.model_validate_json(path.read_text(encoding="utf-8"))
    if not result.method.startswith("d3_"):
        raise ValueError(f"baseline method {result.method!r} is not D3")
    checkpoint_path = path.with_name("d3_checkpoint.json")
    checkpoint = _read_object(checkpoint_path)
    generation = int(result.selected_hyperparameters["selected_generation"])
    record = next(
        (
            item
            for item in checkpoint.get("records", [])
            if isinstance(item, dict) and int(item.get("generation", -1)) == generation
        ),
        None,
    )
    if record is None or not isinstance(record.get("candidate"), dict):
        raise ValueError(f"D3 checkpoint has no selected generation {generation}")
    candidate = CandidateModel.model_validate(record["candidate"])
    parameters = _numeric_mapping(record.get("parameters", {}))
    parameterization = _parameterization(candidate, parameters, {})
    return _subject(
        request=request,
        method=result.method,
        benchmark_id=result.benchmark_id,
        tier=result.tier,
        repetition=result.seed,
        candidate=candidate,
        parameterization=parameterization,
        context=context,
        source_path=path,
        source_hash=_sha256(path),
        auxiliary_hashes={checkpoint_path.name: _sha256(checkpoint_path)},
    )


def _equation_candidate(
    method: str,
    equations: dict[str, str],
    context: ValidationContext,
) -> CandidateModel:
    if set(equations) != set(context.targets):
        raise ValueError(
            "symbolic baseline equations differ from public targets; "
            f"expected={sorted(context.targets)}, actual={sorted(equations)}"
        )
    suffix = hashlib.sha256(
        json.dumps(equations, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return CandidateModel.model_validate(
        {
            "candidate_id": f"{method}_{suffix}",
            "parent_candidate_id": None,
            "states": [
                {"name": target, "kind": "observed"} for target in context.targets
            ],
            "state_equations": [
                {"state": target, "rhs": equations[target]}
                for target in context.targets
            ],
            "observation_mappings": [
                {"channel": target, "expression": target} for target in context.targets
            ],
            "initial_conditions": [
                {
                    "state": target,
                    "scope": "global",
                    "expression": target,
                }
                for target in context.targets
            ],
        }
    )


def _subject(
    *,
    request: SourceAdapterRequest,
    method: str,
    benchmark_id: str,
    tier: str,
    repetition: int,
    candidate: CandidateModel,
    parameterization: FrozenParameterization,
    context: ValidationContext,
    source_path: Path,
    source_hash: str,
    auxiliary_hashes: dict[str, str] | None = None,
) -> FrozenEvaluationSubject:
    identity = {
        "request_id": request.request_id,
        "method": method,
        "benchmark_id": benchmark_id,
        "tier": tier,
        "repetition": repetition,
        "candidate": _candidate_sha256(candidate),
    }
    subject_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return FrozenEvaluationSubject(
        subject_id=subject_id,
        method=method,
        benchmark_id=benchmark_id,
        tier=tier,
        repetition=repetition,
        private_metrics_opened_after_freeze=False,
        source_provenance=SourceArtifactProvenance(
            adapter={
                "autoformalism": "autoformalism_summary",
                "raw_data_agent": "raw_data_agent_run",
                "sindy": "sindy_result",
                "pysr": "pysr_result",
                "d3": "d3_result",
            }[request.source_kind],
            request_id=request.request_id,
            source_path=str(source_path),
            source_sha256=source_hash,
            candidate_sha256=_candidate_sha256(candidate),
            auxiliary_sha256=auxiliary_hashes or {},
        ),
        candidate=candidate,
        parameterization=parameterization,
        validation_context=context,
        target_prediction=TargetPredictionEndpoint(
            status="missing",
            message="private test metrics are populated only after source adaptation",
        ),
    )


def _parameterization(
    candidate: CandidateModel,
    parameters: dict[str, float],
    initials: dict[str, float],
) -> FrozenParameterization:
    expected_parameters = {
        item.name
        for item in candidate.parameters
        if item.scope is ParameterScope.GLOBAL
    }
    expected_initials = {
        item.state
        for item in candidate.initial_conditions
        if item.scope is ParameterScope.GLOBAL and item.initialization_range is not None
    }
    trajectory_specific = any(
        item.scope is ParameterScope.TRAJECTORY_SPECIFIC
        for item in (*candidate.parameters, *candidate.initial_conditions)
    )
    required = bool(expected_parameters or expected_initials or trajectory_specific)
    complete = (
        set(parameters) == expected_parameters
        and set(initials) == expected_initials
        and not trajectory_specific
    )
    if complete and not required:
        status = "not_required"
    elif complete:
        status = "available"
    elif parameters or initials:
        status = "partial"
    else:
        status = "missing"
    return FrozenParameterization(
        status=status,
        global_parameters=parameters,
        global_initial_conditions=initials,
        message=(
            "trajectory-specific fitted scalars are not portable"
            if trajectory_specific
            else None
        ),
    )


def _candidate_sha256(candidate: CandidateModel) -> str:
    payload = json.dumps(
        candidate.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"source artifact is not a file: {resolved}")
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required source artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _numeric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("fitted scalar mapping must be an object")
    output: dict[str, float] = {}
    for name, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"fitted scalar {name!r} is not numeric")
        output[str(name)] = float(raw)
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
