"""Frozen external-baseline sources for the sealed Phase-B test evaluation.

External baselines enter the common final evaluator as train-fitted,
validation-selected models. This module resolves their saved artifacts and
keeps three facts separate for every planned method/cell/repetition:

* **source availability** -- was the artifact ever produced;
* **evaluator readiness** -- can a frozen evaluator score it under its declared
  execution semantics;
* **adaptation outcome** -- what the shared adapter did with it.

Every planned identity yields exactly one roster row. Missing artifacts and
unsupported evaluators are recorded with their reason; neither is a zero score
and neither is evidence of compliance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterOutcome,
    SourceAdapterRequest,
)
from autoformalism.rebuttal.final_evaluation_pilot import (
    FinalEvaluationPilotCell,
    HiddenAuditRequirement,
)

DEVELOPMENT_RESULT_SCHEMA = "phase-b-baseline-development-result-1"
REFIT_DERIVED_SCHEMA = "phase-b-frozen-baseline-model-1"
REFIT_PROTOCOL = "selected_threshold_refit_on_train_plus_validation"

#: Sealed artifacts every native D3 campaign task must have written.
D3_REQUIRED_ARTIFACTS = ("result.json", "native-selection.json")
#: Written only when the generational checkpoint survived; never assumed.
D3_OPTIONAL_ARTIFACTS = ("d3_checkpoint.json",)

#: How a symbolic source root stores its selected development results.
SymbolicLayout = Literal["readiness_freeze", "development_runs", "absent"]


class ExternalBaselineMethodPlan(BaseModel):
    """One external baseline and the execution semantics its models require."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: str = Field(min_length=1)
    source_kind: Literal["sindy", "pysr", "d3", "raw_data_agent"]
    source_layout: Literal[
        "symbolic_development_freeze",
        "d3_native_campaign",
        "reused_external_freeze",
    ]
    execution_semantics: Literal[
        "continuous_ode_free_rollout",
        "discrete_increment_recursive_rollout",
    ]
    parameter_provenance: Literal[
        "equation_embedded_train_only",
        "checkpoint_fitted_train_only",
        "provider_returned_no_refit",
    ]
    fitting_data_access: Literal["train_only"]
    prompt_sensitive: bool
    implementation_status: Literal["ready", "blocked"]
    blocking_gap: str | None = None

    @model_validator(mode="after")
    def blocking_gap_matches_status(self) -> ExternalBaselineMethodPlan:
        """Require exactly one recorded gap for every blocked method."""
        blocked = self.implementation_status == "blocked"
        if blocked != (self.blocking_gap is not None):
            raise ValueError(f"blocked/gap mismatch for method {self.method_id!r}")
        return self

    @property
    def evaluator_status(self) -> Literal["ready", "evaluator_unsupported"]:
        """Report whether a frozen evaluator exists for these semantics."""
        return (
            "ready"
            if self.implementation_status == "ready"
            else "evaluator_unsupported"
        )


class ExternalBaselineFreezePlan(BaseModel):
    """Predeclared sealed test evaluation of the existing external baselines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-external-baseline-frozen-test-evaluation-plan-1"]
    status: Literal[
        "proposed_pending_review",
        "frozen_before_test_or_private_evaluation",
    ]
    cells: tuple[FinalEvaluationPilotCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    methods: tuple[ExternalBaselineMethodPlan, ...] = Field(min_length=1)
    hidden_contract_audit: HiddenAuditRequirement
    paired_numeric_data_across_semantic_variants: Literal[True]
    reporting_roster: Literal["all_40_conditions"]
    train_plus_validation_refit_permitted: Literal[False]
    parameter_refit_applied: Literal[False]
    model_selection_uses_test_results: Literal[False]
    weighted_overall_score_defined: Literal[False]
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]

    @model_validator(mode="after")
    def identities_are_unique(self) -> ExternalBaselineFreezePlan:
        """Reject duplicate cells, repetitions, or method identifiers."""
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cells) != len(set(cells)):
            raise ValueError("benchmark cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        identifiers = [item.method_id for item in self.methods]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("method identifiers must be unique")
        return self

    @property
    def expected_source_count(self) -> int:
        """Return the complete planned method/cell/repetition cross-product."""
        return len(self.methods) * len(self.cells) * len(self.repetitions)


class ExternalBaselineSource(BaseModel):
    """One planned identity with its availability and evaluator readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    repetition: int = Field(ge=0)
    source_kind: Literal["sindy", "pysr", "d3", "raw_data_agent"]
    source_path: str
    artifact_status: Literal["available", "missing"]
    evaluator_status: Literal["ready", "evaluator_unsupported"]
    adapter_requested: bool
    reason: str | None = None
    missing_artifacts: tuple[str, ...] = ()
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    #: Terminal status of an absent run, when the producer recorded one.
    #: A wall-clock timeout is a budget limit, not a method failure.
    terminal_status: Literal['failed', 'timed_out'] | None = None

    @model_validator(mode="after")
    def requested_only_when_available_and_ready(self) -> ExternalBaselineSource:
        """Bind request emission to both availability and evaluator readiness."""
        eligible = (
            self.artifact_status == "available" and self.evaluator_status == "ready"
        )
        if self.adapter_requested != eligible:
            raise ValueError(
                f"{self.request_id}: adapter_requested must equal availability "
                "and evaluator readiness"
            )
        if self.adapter_requested and self.reason is not None:
            raise ValueError(f"{self.request_id}: a requested source has no reason")
        if not self.adapter_requested and not self.reason:
            raise ValueError(f"{self.request_id}: a withheld source requires a reason")
        if self.artifact_status == "missing" and not self.missing_artifacts:
            raise ValueError(f"{self.request_id}: missing source must name artifacts")
        return self

    def outcome(self) -> SourceAdapterOutcome | None:
        """Return the non-adapted outcome this row contributes to assembly."""
        if self.adapter_requested:
            return None
        status = (
            "missing" if self.artifact_status == "missing" else "evaluator_unsupported"
        )
        return SourceAdapterOutcome(
            request_id=self.request_id,
            source_kind=self.source_kind,
            source_path=self.source_path,
            status=status,
            error_type=(
                "MissingSourceArtifact"
                if status == "missing"
                else "EvaluatorUnsupported"
            ),
            error=self.reason,
        )


def load_external_baseline_plan(path: Path) -> ExternalBaselineFreezePlan:
    """Load the frozen external-baseline evaluation plan."""
    return ExternalBaselineFreezePlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def require_executable_freeze(plan: ExternalBaselineFreezePlan) -> None:
    """Refuse to authorize test execution from a draft plan."""
    if plan.status != "frozen_before_test_or_private_evaluation":
        raise ValueError(
            "a draft plan may produce an inventory but cannot authorize execution; "
            f"status={plan.status!r}"
        )


def development_result(path: Path, *, nested: str | None = None) -> dict:
    """The saved development selection a guard should inspect.

    A campaign seals its selection beside its own result wrapper, so the file
    named as the source is not itself a development result. Reading the wrong
    one makes every guard below fail on a schema that was never meant to be
    there.
    """
    payload = _read_object(path)
    if nested is None:
        return payload
    inner = payload.get(nested)
    if not isinstance(inner, dict):
        raise ValueError(f"source has no {nested!r} development selection: {path}")
    return inner


def reject_refit_derived_source(path: Path, *, nested: str | None = None) -> None:
    """Fail closed on a source refit with validation data, for this campaign.

    Fitting on train plus validation does not invalidate an independent test
    estimate in general, and this guard is not a judgement on that workflow.
    It is scoped to this comparison, where three specific problems apply: the
    refit model's validation score is in-sample, its inherited
    `development_validation_normalized_mse` describes the train-only equations
    rather than the refit ones, and including it would break the chosen
    train-only equal-data contract.
    """
    payload = development_result(path, nested=nested)
    schema = payload.get("schema_version")
    if schema == REFIT_DERIVED_SCHEMA:
        raise ValueError(
            f"refit-derived source is not eligible for this campaign: {path}"
        )
    if payload.get("finalization_protocol") == REFIT_PROTOCOL:
        raise ValueError(f"source declares a train-plus-validation refit: {path}")
    if schema != DEVELOPMENT_RESULT_SCHEMA:
        raise ValueError(
            f"unexpected external-baseline source schema {schema!r}: {path}"
        )
    if payload.get("status") != "development_complete":
        raise ValueError(f"source is not a complete development selection: {path}")
    if payload.get("test_data_opened") is not False:
        raise ValueError(f"source already opened test data: {path}")


def require_complete_run_status(path: Path) -> None:
    """Apply the readiness freeze's terminal-status rule to one unsealed run."""
    payload = _read_object(path)
    if payload.get("status") != "complete":
        raise ValueError(
            f"development run did not terminate complete: {path} "
            f"(status={payload.get('status')!r})"
        )


def symbolic_layout(root: Path) -> SymbolicLayout:
    """Classify a symbolic source root.

    The readiness freeze requires all 360 development tasks to have completed,
    so it does not exist while any cell is unfinished. The raw development
    experiment root holds the same `BaselineDevelopmentResult` artifacts and is
    accepted instead, with each run's terminal status checked individually. Its
    provenance is weaker: the runs are not sealed behind one content hash.
    """
    resolved = root.expanduser().resolve()
    if (resolved / "inputs" / "task_plan.jsonl").is_file():
        return "readiness_freeze"
    if (resolved / "runs").is_dir():
        return "development_runs"
    return "absent"


def symbolic_source_index(
    development_freeze_root: Path,
) -> dict[tuple[str, str, str, int], int] | None:
    """Map frozen symbolic task identities to indices, or None if absent."""
    plan_path = development_freeze_root / "inputs" / "task_plan.jsonl"
    if not plan_path.is_file():
        return None
    index: dict[tuple[str, str, str, int], int] = {}
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        identity = (
            str(task["method"]),
            str(task["benchmark_id"]),
            str(task["tier"]),
            int(task["repetition"]),
        )
        if identity in index:
            raise ValueError(f"duplicate development task identity: {identity}")
        index[identity] = int(task["task_index"])
    return index


def d3_source_index(campaign_root: Path) -> dict[tuple[str, str, int], int] | None:
    """Map native D3 campaign row identities to indices, or None if absent."""
    plan_path = campaign_root / "plan.json"
    if not plan_path.is_file():
        return None
    plan = _read_object(plan_path)
    index: dict[tuple[str, str, int], int] = {}
    for position, row in enumerate(plan.get("rows", [])):
        identity = (
            str(row["benchmark_id"]),
            str(row["tier"]),
            int(row["repetition"]),
        )
        if identity in index:
            raise ValueError(f"duplicate D3 campaign identity: {identity}")
        index[identity] = position
    return index


def resolve_external_baseline_sources(
    plan: ExternalBaselineFreezePlan,
    *,
    roots: dict[str, Path],
) -> tuple[tuple[SourceAdapterRequest, ...], tuple[ExternalBaselineSource, ...]]:
    """Resolve locally produced sources without opening test or private data."""
    requests: list[SourceAdapterRequest] = []
    sources: list[ExternalBaselineSource] = []
    for method in plan.methods:
        if method.source_layout == "reused_external_freeze":
            continue
        root = roots.get(method.method_id)
        if root is None:
            raise ValueError(f"no source root supplied for {method.method_id!r}")
        resolved = root.expanduser().resolve()
        layout: SymbolicLayout = (
            symbolic_layout(resolved)
            if method.source_layout == "symbolic_development_freeze"
            else "absent"
        )
        symbolic = (
            symbolic_source_index(resolved)
            if layout == "readiness_freeze"
            else None
        )
        native = (
            d3_source_index(resolved)
            if method.source_layout == "d3_native_campaign"
            else None
        )
        for cell in plan.cells:
            for repetition in plan.repetitions:
                source = _resolve_one(
                    plan_method=method,
                    root=resolved,
                    symbolic=symbolic,
                    symbolic_layout=layout,
                    native=native,
                    cell=cell,
                    repetition=repetition,
                )
                sources.append(source)
                if source.adapter_requested:
                    requests.append(
                        SourceAdapterRequest(
                            request_id=source.request_id,
                            source_kind=method.source_kind,
                            source_path=Path(source.source_path),
                            expected_benchmark_id=cell.benchmark_id,
                            expected_tier=cell.tier,
                            expected_repetition=repetition,
                        )
                    )
    return tuple(requests), tuple(sources)


def load_reused_freeze(
    plan: ExternalBaselineFreezePlan,
    method: ExternalBaselineMethodPlan,
    manifest_path: Path,
) -> tuple[tuple[SourceAdapterRequest, ...], tuple[ExternalBaselineSource, ...]]:
    """Import an independently prepared method freeze and verify its roster."""
    resolved = manifest_path.expanduser().resolve()
    manifest = _read_object(resolved)
    if manifest.get("test_data_opened") is not False:
        raise ValueError(f"reused freeze already opened test data: {resolved}")
    if manifest.get("parameter_refit_applied") is not False:
        raise ValueError(f"reused freeze applied a parameter refit: {resolved}")
    requests_path = resolved.parent / "source_adapter_requests.jsonl"
    if not requests_path.is_file():
        raise ValueError(f"reused freeze has no adapter requests: {requests_path}")
    declared = manifest.get("source_adapter_requests_sha256")
    actual = _sha256(requests_path)
    if declared != actual:
        raise ValueError(
            f"reused freeze request digest differs: expected={declared}, "
            f"actual={actual}"
        )
    imported = tuple(
        SourceAdapterRequest.model_validate_json(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    by_request = {item.request_id: item for item in imported}
    if len(by_request) != len(imported):
        raise ValueError(f"reused freeze repeats a request identifier: {resolved}")
    rows = manifest.get("sources")
    if not isinstance(rows, list):
        raise ValueError(f"reused freeze records no source roster: {resolved}")
    sources = tuple(
        _reused_source(method, row, by_request, resolved) for row in rows
    )
    _require_complete_roster(plan, method, sources)
    # The producer emits a request for every planned row, including rows whose
    # artifacts are absent. Only available, evaluator-ready rows are executable.
    executable = {item.request_id for item in sources if item.adapter_requested}
    return tuple(item for item in imported if item.request_id in executable), sources


def verify_recorded_hashes(sources: tuple[ExternalBaselineSource, ...]) -> None:
    """Re-hash every available source and fail on drift before adaptation."""
    for source in sources:
        if source.artifact_status != "available":
            continue
        directory = Path(source.source_path)
        base = directory if directory.is_dir() else directory.parent
        for name, digest in source.artifact_sha256.items():
            path = base / name if (base / name).is_file() else Path(source.source_path)
            if not path.is_file():
                raise ValueError(f"{source.request_id}: recorded artifact vanished")
            if _sha256(path) != digest:
                raise ValueError(f"{source.request_id}: source hash drift at {path}")


def public_development_identity(
    public_data_root: Path,
    plan: ExternalBaselineFreezePlan,
) -> str:
    """Identify the public inputs by their existing development fingerprints.

    Reuses the train/validation split fingerprints and proposer-prompt hash that
    the loader already computes for each planned cell, so the numerical
    trajectories are covered rather than only JSON and text files. Test data is
    never loaded.
    """
    from autoformalism.rebuttal.baseline_validation import load_public

    root = public_data_root.expanduser().resolve()
    digest = hashlib.sha256()
    for cell in sorted(plan.cells, key=lambda item: (item.benchmark_id, item.tier)):
        _, _, identity = load_public(root, cell.benchmark_id, cell.tier)
        digest.update(f"{cell.benchmark_id}/{cell.tier}".encode())
        for key in sorted(identity):
            digest.update(f"{key}={identity[key]}".encode())
    return digest.hexdigest()


def unavailable_reused_rows(
    plan: ExternalBaselineFreezePlan,
    method: ExternalBaselineMethodPlan,
    reason: str,
) -> tuple[ExternalBaselineSource, ...]:
    """Keep an unprepared imported method in the roster as missing rows.

    An absent upstream freeze must not silently drop its method from the
    planned cross-product; every identity still owns exactly one outcome.
    """
    return tuple(
        ExternalBaselineSource(
            request_id=(
                f"{method.method_id}__{cell.benchmark_id}__{cell.tier}"
                f"__rep{repetition}"
            ),
            method_id=method.method_id,
            benchmark_id=cell.benchmark_id,
            tier=cell.tier,
            repetition=repetition,
            source_kind=method.source_kind,
            source_path=reason,
            artifact_status="missing",
            evaluator_status=method.evaluator_status,
            adapter_requested=False,
            reason=reason,
            missing_artifacts=(reason,),
        )
        for cell in plan.cells
        for repetition in plan.repetitions
    )


def verify_request_roster_association(
    sources: tuple[ExternalBaselineSource, ...],
    requests: tuple[SourceAdapterRequest, ...],
) -> None:
    """Require the executable requests to be exactly the eligible roster rows."""
    by_request = {item.request_id: item for item in sources}
    eligible = {item.request_id for item in sources if item.adapter_requested}
    emitted = {item.request_id for item in requests}
    if len(emitted) != len(requests):
        raise ValueError("frozen requests repeat an identifier")
    if emitted != eligible:
        raise ValueError(
            "frozen requests differ from the eligible roster rows; "
            f"unexpected={sorted(emitted - eligible)[:5]}, "
            f"absent={sorted(eligible - emitted)[:5]}"
        )
    for request in requests:
        row = by_request[request.request_id]
        actual = (
            request.source_kind,
            request.expected_benchmark_id,
            request.expected_tier,
            request.expected_repetition,
            str(request.source_path),
        )
        expected = (
            row.source_kind,
            row.benchmark_id,
            row.tier,
            row.repetition,
            row.source_path,
        )
        if actual != expected:
            raise ValueError(
                f"request {request.request_id} points away from its roster row; "
                f"request={actual}, roster={expected}"
            )


def reconcile(
    plan: ExternalBaselineFreezePlan,
    sources: tuple[ExternalBaselineSource, ...],
) -> dict[str, object]:
    """Confirm exactly one roster row per planned identity and count states.

    Reconciliation keys on the verified method/cell/tier/repetition identity, not
    on a constructed identifier string: imported freezes legitimately carry their
    producer's request identifiers, which this campaign preserves for provenance.
    """
    planned = {
        (method.method_id, cell.benchmark_id, cell.tier, repetition)
        for method in plan.methods
        for cell in plan.cells
        for repetition in plan.repetitions
    }
    seen = [
        (item.method_id, item.benchmark_id, item.tier, item.repetition)
        for item in sources
    ]
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate roster identities in the resolved sources")
    request_ids = [item.request_id for item in sources]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate request identifiers in the resolved sources")
    unexpected = sorted(set(seen) - planned)
    absent = sorted(planned - set(seen))
    if unexpected or absent:
        raise ValueError(
            f"roster differs from the plan; unexpected={unexpected[:5]}, "
            f"absent={absent[:5]}"
        )
    return {
        "expected_source_count": plan.expected_source_count,
        "roster_row_count": len(sources),
        "available_source_count": sum(
            item.artifact_status == "available" for item in sources
        ),
        "missing_source_count": sum(
            item.artifact_status == "missing" for item in sources
        ),
        "evaluator_unsupported_count": sum(
            item.evaluator_status == "evaluator_unsupported" for item in sources
        ),
        "adapter_request_count": sum(item.adapter_requested for item in sources),
        "counts_by_method": {
            method.method_id: {
                "available": sum(
                    item.artifact_status == "available"
                    for item in sources
                    if item.method_id == method.method_id
                ),
                "missing": sum(
                    item.artifact_status == "missing"
                    for item in sources
                    if item.method_id == method.method_id
                ),
                "requested": sum(
                    item.adapter_requested
                    for item in sources
                    if item.method_id == method.method_id
                ),
            }
            for method in plan.methods
        },
    }


def _resolve_one(
    *,
    plan_method: ExternalBaselineMethodPlan,
    root: Path,
    symbolic: dict[tuple[str, str, str, int], int] | None,
    symbolic_layout: SymbolicLayout,
    native: dict[tuple[str, str, int], int] | None,
    cell: FinalEvaluationPilotCell,
    repetition: int,
) -> ExternalBaselineSource:
    """Resolve one planned identity into exactly one roster row."""
    request_id = (
        f"{plan_method.method_id}-{cell.benchmark_id}-{cell.tier}-rep{repetition}"
    )
    evaluator_status = plan_method.evaluator_status
    common = {
        "request_id": request_id,
        "method_id": plan_method.method_id,
        "benchmark_id": cell.benchmark_id,
        "tier": cell.tier,
        "repetition": repetition,
        "source_kind": plan_method.source_kind,
        "evaluator_status": evaluator_status,
    }
    location = _locate(
        plan_method, root, symbolic, symbolic_layout, native, cell, repetition
    )
    if location is None:
        return ExternalBaselineSource(
            **common,
            source_path=str(root),
            artifact_status="missing",
            adapter_requested=False,
            reason=(
                f"no campaign record for {cell.benchmark_id}/{cell.tier}"
                f"/rep{repetition} under {root}"
            ),
            missing_artifacts=(str(root),),
        )
    source_path, required, optional = location
    absent = tuple(sorted(str(item) for item in required if not item.is_file()))
    if absent:
        terminal, detail = _terminal_outcome(source_path)
        return ExternalBaselineSource(
            **common,
            source_path=str(source_path),
            artifact_status="missing",
            adapter_requested=False,
            reason=detail or f"{len(absent)} required artifact(s) absent",
            missing_artifacts=absent,
            terminal_status=terminal,
        )
    present = tuple(item for item in (*required, *optional) if item.is_file())
    if evaluator_status == "ready":
        # A native campaign seals its selection beside the result wrapper it
        # names as the source, so the guards read that file instead.
        if plan_method.source_layout == "d3_native_campaign":
            guarded = source_path.with_name("native-selection.json")
            nested = "selection"
        else:
            guarded, nested = source_path, None
        reject_refit_derived_source(guarded, nested=nested)
        _validate_identity(guarded, cell, repetition, nested=nested)
        status_path = source_path.with_name("run_status.json")
        if status_path.is_file():
            require_complete_run_status(status_path)
    return ExternalBaselineSource(
        **common,
        source_path=str(source_path),
        artifact_status="available",
        adapter_requested=evaluator_status == "ready",
        reason=(None if evaluator_status == "ready" else plan_method.blocking_gap),
        artifact_sha256={item.name: _sha256(item) for item in present},
    )


def _terminal_outcome(
    source_path: Path,
) -> tuple[Literal["failed", "timed_out"] | None, str | None]:
    """Recover why a run produced no result, when the producer recorded it.

    A run that hit its wall clock is budget-limited; one that raised is a
    terminal outcome for the method on that data. Collapsing both into "absent"
    would license reading a compute limit as a method failure.
    """
    status_path = source_path.with_name("run_status.json")
    if not status_path.is_file():
        return None, None
    payload = _read_object(status_path)
    status = payload.get("status")
    if status not in {"failed", "timed_out"}:
        return None, None
    message = str(payload.get("message") or "").strip()
    elapsed = payload.get("elapsed_wall_seconds")
    limit = payload.get("wall_timeout_seconds")
    detail = f"development run terminated {status}"
    if status == "timed_out" and isinstance(elapsed, int | float):
        detail += f" after {float(elapsed):.0f}s of a {limit}s budget"
    if message:
        detail += f": {message[:400]}"
    return status, detail


def _locate(
    method: ExternalBaselineMethodPlan,
    root: Path,
    symbolic: dict[tuple[str, str, str, int], int] | None,
    symbolic_layout: SymbolicLayout,
    native: dict[tuple[str, str, int], int] | None,
    cell: FinalEvaluationPilotCell,
    repetition: int,
) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]] | None:
    """Return the source path plus required and optional companions."""
    if method.source_layout == "symbolic_development_freeze":
        if symbolic_layout == "readiness_freeze":
            if symbolic is None:
                return None
            identity = (method.source_kind, cell.benchmark_id, cell.tier, repetition)
            task_index = symbolic.get(identity)
            if task_index is None:
                return None
            path = root / "tasks" / f"task_{task_index:03d}.json"
            return path, (path,), ()
        if symbolic_layout == "absent":
            return None
        # The producer names runs "<benchmark_id>_<tier>_seed<repetition>".
        directory = (
            root
            / "runs"
            / method.source_kind
            / f"{cell.benchmark_id}_{cell.tier}_seed{repetition}"
        )
        path = directory / "result.json"
        return path, (path, directory / "run_status.json"), ()
    if native is None:
        return None
    position = native.get((cell.benchmark_id, cell.tier, repetition))
    if position is None:
        return None
    directory = root / "results" / str(position)
    required = tuple(directory / name for name in D3_REQUIRED_ARTIFACTS)
    optional = tuple(directory / name for name in D3_OPTIONAL_ARTIFACTS)
    return directory / "result.json", required, optional


def _reused_source(
    method: ExternalBaselineMethodPlan,
    row: object,
    by_request: dict[str, SourceAdapterRequest],
    manifest_path: Path,
) -> ExternalBaselineSource:
    """Convert one imported freeze row into a master-roster row."""
    if not isinstance(row, dict):
        raise ValueError(f"reused freeze roster row is not an object: {manifest_path}")
    request_id = str(row["request_id"])
    benchmark_id = str(row["benchmark_id"])
    tier = str(row["tier"])
    repetition = int(row["repetition"])
    source_path = str(row["source_path"])
    available = str(row["artifact_status"]) == "available"
    ready = method.evaluator_status == "ready"

    request = by_request.get(request_id)
    if available and request is None:
        raise ValueError(f"reused freeze omits a request for {request_id}")
    if request is not None:
        # Provenance is the producer's identifier; correctness is the verified
        # identity behind it. Both must agree before the row is executable.
        actual = (
            request.source_kind,
            request.expected_benchmark_id,
            request.expected_tier,
            request.expected_repetition,
            str(request.source_path),
        )
        expected = (
            method.source_kind,
            benchmark_id,
            tier,
            repetition,
            source_path,
        )
        if actual != expected:
            raise ValueError(
                f"reused request {request_id} disagrees with its roster row; "
                f"request={actual}, roster={expected}"
            )

    reason: str | None = None
    if not available:
        reason = "source artifact absent in the reused freeze"
    elif not ready:
        reason = method.blocking_gap
    return ExternalBaselineSource(
        request_id=request_id,
        method_id=method.method_id,
        benchmark_id=benchmark_id,
        tier=tier,
        repetition=repetition,
        source_kind=method.source_kind,
        source_path=source_path,
        artifact_status="available" if available else "missing",
        evaluator_status=method.evaluator_status,
        adapter_requested=available and ready,
        reason=reason,
        missing_artifacts=tuple(row.get("missing_artifacts", ())),
        artifact_sha256=dict(row.get("artifact_sha256", {})),
    )


def _require_complete_roster(
    plan: ExternalBaselineFreezePlan,
    method: ExternalBaselineMethodPlan,
    sources: tuple[ExternalBaselineSource, ...],
) -> None:
    """Require the reused freeze to cover the exact planned cross-product."""
    expected = {
        (cell.benchmark_id, cell.tier, repetition)
        for cell in plan.cells
        for repetition in plan.repetitions
    }
    actual = {(item.benchmark_id, item.tier, item.repetition) for item in sources}
    if actual != expected:
        raise ValueError(
            f"reused {method.method_id} roster differs from the plan; "
            f"missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )


def _validate_identity(
    path: Path,
    cell: FinalEvaluationPilotCell,
    repetition: int,
    *,
    nested: str | None = None,
) -> None:
    """Confirm the saved result names the exact planned cell and repetition."""
    payload = development_result(path, nested=nested)
    actual = (
        str(payload.get("benchmark_id")),
        str(payload.get("tier")),
        int(payload.get("seed", -1)),
    )
    if actual != (cell.benchmark_id, cell.tier, repetition):
        raise ValueError(
            f"source identity differs from the plan: expected="
            f"{(cell.benchmark_id, cell.tier, repetition)}, actual={actual}"
        )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
