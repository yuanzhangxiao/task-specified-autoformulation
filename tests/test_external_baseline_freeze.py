"""External-baseline roster resolution before sealed test evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median

import pytest

from autoformalism.rebuttal.external_baseline_freeze import (
    ExternalBaselineFreezePlan,
    ExternalBaselineSource,
    load_external_baseline_plan,
    load_reused_freeze,
    public_development_identity,
    reconcile,
    reject_refit_derived_source,
    require_executable_freeze,
    resolve_external_baseline_sources,
    symbolic_layout,
    unavailable_reused_rows,
    verify_recorded_hashes,
    verify_request_roster_association,
)
from autoformalism.rebuttal.final_evaluation import (
    ComplexityEndpoint,
    FinalEvaluationRecord,
    FrozenParameterization,
    PublicMechanismEndpoint,
    RuntimeValidityEndpoint,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
)
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome
from scripts.assemble_phase_b_final_evaluation import _validate_outcomes
from scripts.summarize_external_baseline_evaluation import (
    full_roster_median,
    join,
    summarize,
)

CONFIG = Path("configs/external_baseline_frozen_test_evaluation_v1.json")
CELL = {"benchmark_id": "phase_b_cell", "tier": "easy"}


def _development_result(method: str, benchmark_id: str, tier: str, seed: int) -> dict:
    return {
        "schema_version": "phase-b-baseline-development-result-1",
        "method": method,
        "benchmark_id": benchmark_id,
        "tier": tier,
        "seed": seed,
        "equations": {"y": "-0.5 * y"},
        "selected_hyperparameters": {"threshold": 0.05},
        "training_normalized_mse": 0.4,
        "validation_normalized_mse": 0.5,
        "status": "development_complete",
        "test_data_opened": False,
    }


def _method(method_id: str, **overrides: object) -> dict:
    base = {
        "method_id": method_id,
        "source_kind": method_id,
        "source_layout": "symbolic_development_freeze",
        "execution_semantics": "continuous_ode_free_rollout",
        "parameter_provenance": "equation_embedded_train_only",
        "fitting_data_access": "train_only",
        "prompt_sensitive": False,
        "implementation_status": "ready",
        "blocking_gap": None,
    }
    base.update(overrides)
    return base


def _d3_method() -> dict:
    return _method(
        "d3_native_no_tools",
        source_kind="d3",
        source_layout="d3_native_campaign",
        execution_semantics="discrete_increment_recursive_rollout",
        parameter_provenance="checkpoint_fitted_train_only",
        prompt_sensitive=True,
        implementation_status="blocked",
        blocking_gap="campaign reader and discrete test protocol are missing",
    )


def _plan(
    methods: list[dict], cells: list[dict] | None = None
) -> ExternalBaselineFreezePlan:
    return ExternalBaselineFreezePlan.model_validate(
        {
            "schema_version": (
                "phase-b-external-baseline-frozen-test-evaluation-plan-1"
            ),
            "status": "proposed_pending_review",
            "cells": cells or [CELL],
            "repetitions": [0],
            "methods": methods,
            "hidden_contract_audit": {
                "schema_version": "phase-b-hidden-subspace-contract-audit-2",
                "sha256": "0" * 64,
                "required_status": "pass",
            },
            "paired_numeric_data_across_semantic_variants": True,
            "reporting_roster": "all_40_conditions",
            "train_plus_validation_refit_permitted": False,
            "parameter_refit_applied": False,
            "model_selection_uses_test_results": False,
            "weighted_overall_score_defined": False,
            "test_data_opened": False,
            "private_reference_opened": False,
        }
    )


def _symbolic_freeze(
    tmp_path: Path, *, tasks: list[dict], write: dict[int, dict]
) -> Path:
    root = tmp_path / "common-readiness-freeze"
    (root / "inputs").mkdir(parents=True)
    (root / "tasks").mkdir()
    (root / "inputs" / "task_plan.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in tasks), encoding="utf-8"
    )
    for index, payload in write.items():
        (root / "tasks" / f"task_{index:03d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return root


def _d3_campaign(tmp_path: Path, *, rows: list[dict], results: dict[int, bool]) -> Path:
    root = tmp_path / "d3-native-full-v1"
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    for index, complete in results.items():
        directory = root / "results" / str(index)
        directory.mkdir(parents=True)
        # The production wrapper has no schema_version and is not a development result.
        (directory / "result.json").write_text(
            json.dumps(
                {
                    **rows[index],
                    "status": "complete",
                    "protocol": "phase-b-d3-native-validation-1",
                    "plan_sha256": "a" * 64,
                    "selection_metric": "native_one_step_validation",
                    "test_data_opened": False,
                }
            ),
            encoding="utf-8",
        )
        if complete:
            (directory / "native-selection.json").write_text(
                json.dumps(
                    {
                        "plan_sha256": "a" * 64,
                        "selection": _development_result(
                            "d3_native_no_tools",
                            rows[index]["benchmark_id"],
                            rows[index]["tier"],
                            rows[index]["repetition"],
                        ),
                    }
                ),
                encoding="utf-8",
            )
    return root


# --- committed plan -------------------------------------------------------


def test_committed_plan_forbids_refit_keeps_40_cells_and_blocks_d3() -> None:
    plan = load_external_baseline_plan(CONFIG)
    assert len(plan.cells) == 40
    assert plan.expected_source_count == 480
    assert plan.reporting_roster == "all_40_conditions"
    assert plan.train_plus_validation_refit_permitted is False
    assert plan.model_selection_uses_test_results is False
    blocked = {
        item.method_id: item.blocking_gap
        for item in plan.methods
        if item.implementation_status == "blocked"
    }
    assert set(blocked) == {"d3_native_no_tools"}
    assert "native-selection.json" in blocked["d3_native_no_tools"]
    assert "x_next = x + f" in blocked["d3_native_no_tools"]


def test_a_draft_plan_cannot_authorize_execution() -> None:
    """Only a finalized plan may open test data."""
    with pytest.raises(ValueError, match="cannot authorize execution"):
        require_executable_freeze(_plan([_method("sindy")]))


def test_the_committed_plan_is_finalized_for_execution() -> None:
    plan = load_external_baseline_plan(CONFIG)
    assert plan.status == "frozen_before_test_or_private_evaluation"
    require_executable_freeze(plan)


# --- refit guard ----------------------------------------------------------


def test_refit_derived_sources_are_rejected(tmp_path: Path) -> None:
    refit = tmp_path / "frozen_model.json"
    refit.write_text(
        json.dumps({"schema_version": "phase-b-frozen-baseline-model-1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refit-derived source"):
        reject_refit_derived_source(refit)

    declared = tmp_path / "declared.json"
    declared.write_text(
        json.dumps(
            {
                "schema_version": "phase-b-baseline-development-result-1",
                "finalization_protocol": (
                    "selected_threshold_refit_on_train_plus_validation"
                ),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train-plus-validation refit"):
        reject_refit_derived_source(declared)

    development = tmp_path / "result.json"
    development.write_text(
        json.dumps(_development_result("sindy", "phase_b_cell", "easy", 0)),
        encoding="utf-8",
    )
    reject_refit_derived_source(development)


def test_refit_model_in_the_source_slot_fails_the_freeze(tmp_path: Path) -> None:
    root = _symbolic_freeze(
        tmp_path,
        tasks=[{"task_index": 0, "method": "sindy", **CELL, "repetition": 0}],
        write={0: {"schema_version": "phase-b-frozen-baseline-model-1"}},
    )
    with pytest.raises(ValueError, match="refit-derived source"):
        resolve_external_baseline_sources(
            _plan([_method("sindy")]), roots={"sindy": root}
        )


def test_source_identity_must_match_the_frozen_plan(tmp_path: Path) -> None:
    root = _symbolic_freeze(
        tmp_path,
        tasks=[{"task_index": 0, "method": "sindy", **CELL, "repetition": 0}],
        write={0: _development_result("sindy", "other_cell", "easy", 0)},
    )
    with pytest.raises(ValueError, match="source identity differs"):
        resolve_external_baseline_sources(
            _plan([_method("sindy")]), roots={"sindy": root}
        )


# --- availability versus evaluator readiness ------------------------------


def test_absent_root_yields_missing_rows_without_blocking_other_methods(
    tmp_path: Path,
) -> None:
    root = _symbolic_freeze(
        tmp_path,
        tasks=[{"task_index": 0, "method": "sindy", **CELL, "repetition": 0}],
        write={0: _development_result("sindy", "phase_b_cell", "easy", 0)},
    )
    plan = _plan([_method("sindy"), _d3_method()])
    requests, sources = resolve_external_baseline_sources(
        plan, roots={"sindy": root, "d3_native_no_tools": tmp_path / "absent"}
    )

    assert len(requests) == 1
    assert requests[0].source_kind == "sindy"
    by_method = {item.method_id: item for item in sources}
    assert by_method["sindy"].artifact_status == "available"
    assert by_method["sindy"].adapter_requested is True
    d3 = by_method["d3_native_no_tools"]
    assert d3.artifact_status == "missing"
    assert d3.adapter_requested is False
    assert d3.missing_artifacts
    assert reconcile(plan, sources)["roster_row_count"] == 2


def test_plan_row_absent_from_the_campaign_is_a_missing_outcome(
    tmp_path: Path,
) -> None:
    root = _d3_campaign(
        tmp_path,
        rows=[{"benchmark_id": "other_cell", "tier": "easy", "repetition": 0}],
        results={0: True},
    )
    plan = _plan([_d3_method()])
    _, sources = resolve_external_baseline_sources(
        plan, roots={"d3_native_no_tools": root}
    )
    assert sources[0].artifact_status == "missing"
    assert "no campaign record" in sources[0].reason


def test_blocked_method_is_available_but_never_requested(tmp_path: Path) -> None:
    rows = [
        {"benchmark_id": "other_cell", "tier": "hard", "repetition": 0},
        {"benchmark_id": "phase_b_cell", "tier": "easy", "repetition": 0},
    ]
    root = _d3_campaign(tmp_path, rows=rows, results={0: True, 1: True})
    plan = _plan([_d3_method()])
    requests, sources = resolve_external_baseline_sources(
        plan, roots={"d3_native_no_tools": root}
    )

    assert requests == ()
    row = sources[0]
    assert row.artifact_status == "available"
    assert row.evaluator_status == "evaluator_unsupported"
    assert row.adapter_requested is False
    assert row.reason == "campaign reader and discrete test protocol are missing"
    # nontrivial row index: the planned cell is the second campaign row
    assert row.source_path.endswith("results/1/result.json")
    assert set(row.artifact_sha256) == {"result.json", "native-selection.json"}


def test_incomplete_d3_task_missing_its_sealed_selection_is_missing(
    tmp_path: Path,
) -> None:
    root = _d3_campaign(
        tmp_path,
        rows=[{"benchmark_id": "phase_b_cell", "tier": "easy", "repetition": 0}],
        results={0: False},
    )
    _, sources = resolve_external_baseline_sources(
        _plan([_d3_method()]), roots={"d3_native_no_tools": root}
    )
    assert sources[0].artifact_status == "missing"
    assert sources[0].missing_artifacts[0].endswith("native-selection.json")


# --- reused Sol freeze ----------------------------------------------------


def _reused_freeze(tmp_path: Path, *, available: bool = True) -> Path:
    """Build a freeze shaped exactly like the production Sol producer.

    The producer uses double-underscore request identifiers and emits a request
    for every planned row, including rows whose artifacts are absent.
    """
    root = tmp_path / "raw-agent-freeze"
    root.mkdir(parents=True)
    run = root / "runs" / "openai_gpt-5-6-sol_phase_b_cell_easy_rep0"
    run.mkdir(parents=True)
    request_id = "raw_data_agent_gpt-5.6-sol__phase_b_cell__easy__rep0"
    requests = [
        json.dumps(
            {
                "request_id": request_id,
                "source_kind": "raw_data_agent",
                "source_path": str(run),
                "expected_benchmark_id": "phase_b_cell",
                "expected_tier": "easy",
                "expected_repetition": 0,
            }
        )
    ]
    requests_path = root / "source_adapter_requests.jsonl"
    requests_path.write_text(
        "".join(item + "\n" for item in requests), encoding="utf-8"
    )
    manifest = {
        "method_id": "raw_data_agent:gpt-5.6-sol",
        "test_data_opened": False,
        "parameter_refit_applied": False,
        "source_adapter_requests_sha256": hashlib.sha256(
            requests_path.read_bytes()
        ).hexdigest(),
        "sources": [
            {
                "request_id": request_id,
                "benchmark_id": "phase_b_cell",
                "tier": "easy",
                "repetition": 0,
                "source_path": str(run),
                "artifact_status": "available" if available else "missing",
                "missing_artifacts": (
                    [] if available else [str(run / "candidate.json")]
                ),
                "artifact_sha256": {},
            }
        ],
    }
    manifest_path = root / "raw_agent_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _sol_method() -> dict:
    return _method(
        "raw_data_agent:gpt-5.6-sol",
        source_kind="raw_data_agent",
        source_layout="reused_external_freeze",
        parameter_provenance="provider_returned_no_refit",
        prompt_sensitive=True,
    )


def test_reused_sol_freeze_keeps_producer_identifiers(tmp_path: Path) -> None:
    plan = _plan([_sol_method()])
    requests, sources = load_reused_freeze(
        plan, plan.methods[0], _reused_freeze(tmp_path)
    )
    assert len(requests) == 1
    assert requests[0].request_id.startswith("raw_data_agent_gpt-5.6-sol__")
    assert sources[0].request_id == requests[0].request_id
    assert sources[0].adapter_requested is True
    # reconciliation keys on verified identity, not on the identifier string
    reconcile(plan, sources)
    verify_request_roster_association(sources, requests)


def test_reused_missing_row_keeps_its_producer_request_out_of_execution(
    tmp_path: Path,
) -> None:
    """The producer emits a request even when the artifact is absent."""
    plan = _plan([_sol_method()])
    requests, sources = load_reused_freeze(
        plan, plan.methods[0], _reused_freeze(tmp_path, available=False)
    )
    assert requests == ()
    assert sources[0].artifact_status == "missing"
    assert sources[0].adapter_requested is False
    reconcile(plan, sources)
    verify_request_roster_association(sources, requests)
    assert sources[0].outcome().status == "missing"


def test_reused_request_pointing_away_from_its_roster_row_is_rejected(
    tmp_path: Path,
) -> None:
    manifest_path = _reused_freeze(tmp_path)
    requests_path = manifest_path.parent / "source_adapter_requests.jsonl"
    row = json.loads(requests_path.read_text(encoding="utf-8"))
    row["source_path"] = str(manifest_path.parent / "somewhere-else")
    requests_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_adapter_requests_sha256"] = hashlib.sha256(
        requests_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = _plan([_sol_method()])
    with pytest.raises(ValueError, match="disagrees with its roster row"):
        load_reused_freeze(plan, plan.methods[0], manifest_path)


def test_reused_freeze_roster_must_cover_the_planned_cross_product(
    tmp_path: Path,
) -> None:
    plan = _plan(
        [_sol_method()], cells=[CELL, {"benchmark_id": "second_cell", "tier": "hard"}]
    )
    with pytest.raises(ValueError, match="roster differs from the plan"):
        load_reused_freeze(plan, plan.methods[0], _reused_freeze(tmp_path))


def test_reused_freeze_rejects_a_tampered_request_file(tmp_path: Path) -> None:
    manifest_path = _reused_freeze(tmp_path)
    (manifest_path.parent / "source_adapter_requests.jsonl").write_text(
        "", encoding="utf-8"
    )
    plan = _plan([_sol_method()])
    with pytest.raises(ValueError, match="request digest differs"):
        load_reused_freeze(plan, plan.methods[0], manifest_path)


# --- integrity and reporting ---------------------------------------------


def test_hash_drift_is_detected_before_adaptation(tmp_path: Path) -> None:
    root = _symbolic_freeze(
        tmp_path,
        tasks=[{"task_index": 0, "method": "sindy", **CELL, "repetition": 0}],
        write={0: _development_result("sindy", "phase_b_cell", "easy", 0)},
    )
    _, sources = resolve_external_baseline_sources(
        _plan([_method("sindy")]), roots={"sindy": root}
    )
    verify_recorded_hashes(sources)

    (root / "tasks" / "task_000.json").write_text(
        json.dumps(
            _development_result("sindy", "phase_b_cell", "easy", 0)
            | {"training_normalized_mse": 0.9}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hash drift"):
        verify_recorded_hashes(sources)


def test_reconcile_rejects_a_roster_that_differs_from_the_plan() -> None:
    plan = _plan([_method("sindy")])
    stray = ExternalBaselineSource(
        request_id="sindy-other_cell-easy-rep0",
        method_id="sindy",
        benchmark_id="other_cell",
        tier="easy",
        repetition=0,
        source_kind="sindy",
        source_path="/tmp/x.json",
        artifact_status="missing",
        evaluator_status="ready",
        adapter_requested=False,
        reason="absent",
        missing_artifacts=("/tmp/x.json",),
    )
    with pytest.raises(ValueError, match="roster differs from the plan"):
        reconcile(plan, (stray,))


def test_withheld_rows_become_assembly_outcomes_and_survive_validation(
    tmp_path: Path,
) -> None:
    root = _d3_campaign(
        tmp_path,
        rows=[{"benchmark_id": "phase_b_cell", "tier": "easy", "repetition": 0}],
        results={0: True},
    )
    _, sources = resolve_external_baseline_sources(
        _plan([_d3_method()]), roots={"d3_native_no_tools": root}
    )
    outcome = sources[0].outcome()
    assert isinstance(outcome, SourceAdapterOutcome)
    assert outcome.status == "evaluator_unsupported"
    assert outcome.error_type == "EvaluatorUnsupported"
    assert outcome.subject_id is None

    # assembly tolerates non-adapted rows alongside the adapted subject set
    _validate_outcomes((), (outcome,))


def test_missing_rows_become_missing_outcomes(tmp_path: Path) -> None:
    _, sources = resolve_external_baseline_sources(
        _plan([_d3_method()]), roots={"d3_native_no_tools": tmp_path / "absent"}
    )
    outcome = sources[0].outcome()
    assert outcome is not None
    assert outcome.status == "missing"
    assert outcome.error_type == "MissingSourceArtifact"
    _validate_outcomes((), (outcome,))


def test_adapted_rows_contribute_no_withheld_outcome(tmp_path: Path) -> None:
    root = _symbolic_freeze(
        tmp_path,
        tasks=[{"task_index": 0, "method": "sindy", **CELL, "repetition": 0}],
        write={0: _development_result("sindy", "phase_b_cell", "easy", 0)},
    )
    _, sources = resolve_external_baseline_sources(
        _plan([_method("sindy")]), roots={"sindy": root}
    )
    assert sources[0].outcome() is None


# --- submission chain -----------------------------------------------------

HPC = Path("scripts/hpc")


def test_submission_chain_is_valid_and_guards_test_access() -> None:
    stages = (
        "external_baseline_eval_prepare.slurm",
        "external_baseline_eval_postfreeze.slurm",
        "external_baseline_eval_merge.slurm",
        "external_baseline_eval_hidden.slurm",
        "external_baseline_eval_finalize.slurm",
        "submit_external_baseline_evaluation_delta.sh",
    )
    for name in stages:
        path = HPC / name
        assert path.is_file(), name
        subprocess.run(["bash", "-n", str(path)], check=True)

    prepare = (HPC / "external_baseline_eval_prepare.slurm").read_text(
        encoding="utf-8"
    )
    assert "--authorize-execution" in prepare
    assert "--verify" in prepare

    postfreeze = (HPC / "external_baseline_eval_postfreeze.slurm").read_text(
        encoding="utf-8"
    )
    assert ".execution_authorized == true" in postfreeze

    hidden = (HPC / "external_baseline_eval_hidden.slurm").read_text(encoding="utf-8")
    assert "--shard-index" in hidden and "--shard-count" in hidden

    finalize = (HPC / "external_baseline_eval_finalize.slurm").read_text(
        encoding="utf-8"
    )
    assert "withheld_source_outcomes.jsonl" in finalize
    assert "expected_source_count" in finalize

    submit = (HPC / "submit_external_baseline_evaluation_delta.sh").read_text(
        encoding="utf-8"
    )
    for job in ("afterok", "AF_EVALUATOR_CODE_COMMIT", "submission_manifest.json"):
        assert job in submit


# --- inventory without an upstream freeze ---------------------------------


def test_missing_upstream_freeze_keeps_sol_in_the_roster_as_missing() -> None:
    plan = _plan([_sol_method()])
    rows = unavailable_reused_rows(plan, plan.methods[0], "manifest not supplied")
    assert len(rows) == len(plan.cells) * len(plan.repetitions)
    assert all(item.artifact_status == "missing" for item in rows)
    assert all(item.adapter_requested is False for item in rows)
    reconcile(plan, rows)
    assert rows[0].outcome().status == "missing"


# --- roster-joined reporting ----------------------------------------------


def _row(state: str, request_id: str, method: str = "sindy") -> ExternalBaselineSource:
    missing = state == "missing"
    return ExternalBaselineSource(
        request_id=request_id,
        method_id=method,
        benchmark_id="phase_b_cell",
        tier="easy",
        repetition=0,
        source_kind="sindy",
        source_path="/tmp/task.json",
        artifact_status="missing" if missing else "available",
        evaluator_status=(
            "evaluator_unsupported" if state == "evaluator_unsupported" else "ready"
        ),
        adapter_requested=state == "evaluated",
        reason=None if state == "evaluated" else state,
        missing_artifacts=("/tmp/task.json",) if missing else (),
    )


def _outcome(status: str, request_id: str) -> SourceAdapterOutcome:
    if status == "adapted":
        return SourceAdapterOutcome(
            request_id=request_id,
            source_kind="sindy",
            source_path="/tmp/task.json",
            status="adapted",
            subject_id=f"subject-{request_id}",
        )
    return SourceAdapterOutcome(
        request_id=request_id,
        source_kind="sindy",
        source_path="/tmp/task.json",
        status=status,
        error_type="Reason",
        error="reason",
    )


def _record(
    request_id: str, *, nmse: float = 0.5, failed: str | None = None
) -> FinalEvaluationRecord:
    """Minimal evaluated record; the report reads only these fields."""
    return FinalEvaluationRecord(
        subject_id=f"subject-{request_id}",
        method="sindy",
        benchmark_id="phase_b_cell",
        tier="easy",
        repetition=0,
        source_provenance=SourceArtifactProvenance(
            adapter="sindy_result",
            request_id=request_id,
            source_path="/tmp/task.json",
            source_sha256="0" * 64,
            candidate_sha256="1" * 64,
        ),
        parameterization=FrozenParameterization(status="not_required"),
        runtime=RuntimeValidityEndpoint(valid=True),
        public_mechanism=PublicMechanismEndpoint(status="missing"),
        target_prediction=(
            TargetPredictionEndpoint(
                status="failed",
                evaluation_protocol="unseen_condition_free_rollout",
                trajectory_count=1,
                successful_trajectory_count=0,
                failed_trajectories=("t0",),
                message=failed,
            )
            if failed is not None
            else TargetPredictionEndpoint(
                status="available",
                evaluation_protocol="unseen_condition_free_rollout",
                normalized_mse=nmse,
                per_target_normalized_mse={"y": nmse},
                normalization_scales={"y": 1.0},
                trajectory_count=1,
                successful_trajectory_count=1,
            )
        ),
        hidden_mechanisms=(),
        interventions=(),
        complexity=ComplexityEndpoint(
            state_count=1,
            latent_state_count=0,
            process_count=0,
            parameter_count=0,
            additive_term_count=1,
        ),
        qualitative_llm=None,
    )


def test_report_keeps_missing_and_unsupported_out_of_failures() -> None:
    roster = (
        _row("evaluated", "a"),
        _row("missing", "b"),
        _row("evaluator_unsupported", "c"),
    )
    outcomes = (
        _outcome("adapted", "a"),
        _outcome("missing", "b"),
        _outcome("evaluator_unsupported", "c"),
    )
    rows = join(roster, outcomes, (_record("a"),))
    states = {row["request_id"]: row["state"] for row in rows}
    assert states == {
        "a": "evaluated",
        "b": "missing",
        "c": "evaluator_unsupported",
    }
    report = summarize(rows)
    counts = report["by_method"]["sindy"]
    assert counts["planned"] == 3
    assert counts["missing"] == 1
    assert counts["evaluator_unsupported"] == 1
    assert counts["adaptation_failed"] == 0
    assert report["pooled_cross_method_mean_reported"] is False
    assert report["headline_median"] == "target_nmse_median_full_roster"
    assert report["conditional_median_is_diagnostic_only"] is True


def test_report_rejects_same_count_wrong_identifiers() -> None:
    roster = (_row("evaluated", "a"), _row("missing", "b"))
    outcomes = (_outcome("adapted", "a"), _outcome("missing", "WRONG"))
    assert len(outcomes) == len(roster)
    with pytest.raises(ValueError, match="differ from the frozen roster"):
        join(roster, outcomes, ())


def test_report_requires_a_record_for_every_adapted_outcome() -> None:
    roster = (_row("evaluated", "a"),)
    outcomes = (_outcome("adapted", "a"),)
    with pytest.raises(ValueError, match="adapted outcome has no record"):
        join(roster, outcomes, ())


# --- execution-record preflight -------------------------------------------

PREPARE = Path("scripts/prepare_external_baseline_frozen_test_evaluation.py")


def test_public_identity_covers_every_planned_cell(monkeypatch) -> None:
    """Identity aggregates the loader's split fingerprints and prompt hash."""
    seen: list[tuple[str, str]] = []

    def fake_load_public(root, benchmark, tier):
        seen.append((benchmark, tier))
        return None, None, {
            "train": f"train-{benchmark}-{tier}",
            "validation": f"validation-{benchmark}-{tier}",
            "prompt": f"prompt-{benchmark}-{tier}",
        }

    import autoformalism.rebuttal.baseline_validation as bv

    monkeypatch.setattr(bv, "load_public", fake_load_public)
    plan = _plan(
        [_method("sindy")],
        cells=[CELL, {"benchmark_id": "phase_b_other", "tier": "hard"}],
    )
    first = public_development_identity(Path("/public"), plan)
    assert len(seen) == 2
    assert set(seen) == {("phase_b_cell", "easy"), ("phase_b_other", "hard")}
    assert first == public_development_identity(Path("/public"), plan)

    def changed_load_public(root, benchmark, tier):
        payload = fake_load_public(root, benchmark, tier)[2]
        if benchmark == "phase_b_other":
            payload["validation"] = "different"
        return None, None, payload

    monkeypatch.setattr(bv, "changed", changed_load_public, raising=False)
    monkeypatch.setattr(bv, "load_public", changed_load_public)
    assert public_development_identity(Path("/public"), plan) != first


def _sealed_tree(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Write a minimal frozen/adapted pair plus a consistent execution record."""
    frozen = tmp_path / "frozen"
    adapted = tmp_path / "adapted"
    frozen.mkdir(parents=True)
    adapted.mkdir(parents=True)
    (frozen / "external_baseline_roster.jsonl").write_text("{}\n", encoding="utf-8")
    (frozen / "source_adapter_requests.jsonl").write_text("{}\n", encoding="utf-8")
    (frozen / "withheld_source_outcomes.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = {"execution_authorized": True}
    (frozen / "external_baseline_freeze.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (adapted / "frozen_evaluation_subjects.jsonl").write_text("{}\n", encoding="utf-8")
    (adapted / "source_adapter_outcomes.jsonl").write_text("{}\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = {
        "schema_version": "phase-b-external-baseline-execution-record-2",
        "freeze_manifest_sha256": digest(frozen / "external_baseline_freeze.json"),
        "source_adapter_requests_sha256": digest(
            frozen / "source_adapter_requests.jsonl"
        ),
        "external_baseline_roster_sha256": digest(
            frozen / "external_baseline_roster.jsonl"
        ),
        "withheld_source_outcomes_sha256": digest(
            frozen / "withheld_source_outcomes.jsonl"
        ),
        "frozen_evaluation_subjects_sha256": digest(
            adapted / "frozen_evaluation_subjects.jsonl"
        ),
        "source_adapter_outcomes_sha256": digest(
            adapted / "source_adapter_outcomes.jsonl"
        ),
        "public_data_root": "/public",
        "public_data_identity_sha256": "public-identity",
        "evaluator_code_commit": "c" * 40,
        "chain_inputs_digest": "d" * 64,
        "maximum_trajectory_wall_time_seconds": 300.0,
        "test_data_opened": False,
    }
    record_path = frozen / "execution_record.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    record_path.with_name(f"{record_path.name}.sha256").write_text(
        f"{digest(record_path)}  {record_path.name}\n", encoding="utf-8"
    )
    return frozen, adapted, record


def _preflight(frozen: Path, adapted: Path, **overrides: str):
    """Run the real preflight from the repository working directory."""
    arguments = {
        "--evaluator-commit": "c" * 40,
        "--chain-inputs-digest": "d" * 64,
        "--maximum-trajectory-wall-time-seconds": "300",
        "--public-identity": "public-identity",
    }
    arguments.update(overrides)
    command = [
        ".venv/bin/python",
        str(PREPARE),
        "--config",
        str(CONFIG),
        "--output-root",
        str(frozen),
        "--verify-execution-record",
        "--adapted-root",
        str(adapted),
    ]
    for key, value in arguments.items():
        command += [key, value]
    return subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        check=False,
    )


def test_preflight_passes_an_unchanged_execution(tmp_path: Path) -> None:
    frozen, adapted, _ = _sealed_tree(tmp_path)
    result = _preflight(frozen, adapted)
    assert result.returncode == 0, result.stderr
    assert "execution record verified" in result.stdout


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"--maximum-trajectory-wall-time-seconds": "600"}, "trajectory wall-time"),
        ({"--evaluator-commit": "e" * 40}, "evaluator commit"),
        ({"--chain-inputs-digest": "f" * 64}, "chain inputs digest"),
        ({"--public-identity": "other"}, "public data identity"),
    ],
)
def test_preflight_rejects_changed_settings(
    tmp_path: Path, overrides: dict, expected: str
) -> None:
    frozen, adapted, _ = _sealed_tree(tmp_path)
    result = _preflight(frozen, adapted, **overrides)
    assert result.returncode != 0
    assert expected in result.stderr


def test_preflight_rejects_a_changed_authorization_manifest(tmp_path: Path) -> None:
    frozen, adapted, _ = _sealed_tree(tmp_path)
    (frozen / "external_baseline_freeze.json").write_text(
        json.dumps({"execution_authorized": True, "tampered": True}), encoding="utf-8"
    )
    result = _preflight(frozen, adapted)
    assert result.returncode != 0
    assert "freeze manifest" in result.stderr


def test_preflight_rejects_changed_adapted_subjects(tmp_path: Path) -> None:
    frozen, adapted, _ = _sealed_tree(tmp_path)
    (adapted / "frozen_evaluation_subjects.jsonl").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    result = _preflight(frozen, adapted)
    assert result.returncode != 0
    assert "adapted subjects" in result.stderr


def test_preflight_rejects_a_tampered_record(tmp_path: Path) -> None:
    frozen, adapted, record = _sealed_tree(tmp_path)
    record["maximum_trajectory_wall_time_seconds"] = 600.0
    (frozen / "execution_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = _preflight(frozen, adapted)
    assert result.returncode != 0
    assert "companion digest" in result.stderr


# --- resume guard ---------------------------------------------------------

SUBMIT = Path("scripts/hpc/submit_external_baseline_evaluation_delta.sh")


def _chain_digest() -> str:
    result = subprocess.run(
        ["bash", "scripts/hpc/external_baseline_inputs_digest.sh"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        env={"AF_REPO_ROOT": str(Path.cwd()), "PATH": "/usr/bin:/bin:/sbin"},
        check=True,
    )
    return result.stdout.strip()


def _resume(tmp_path: Path, sealed_digest: str) -> subprocess.CompletedProcess:
    """Run the wrapper's resume path with sbatch stubbed out."""
    output_root = tmp_path / "evaluation"
    (output_root / "frozen").mkdir(parents=True)
    (output_root / "frozen" / "execution_record.json").write_text(
        json.dumps({"chain_inputs_digest": sealed_digest}), encoding="utf-8"
    )
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "sbatch").write_text("#!/bin/bash\necho 12345\n", encoding="utf-8")
    (stub / "sbatch").chmod(0o755)
    return subprocess.run(
        ["bash", str(SUBMIT)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        env={
            "USER": "tester",
            "PATH": f"{stub}:/usr/bin:/bin:/sbin",
            "AF_REPO_ROOT": str(Path.cwd()),
            "AF_OUTPUT_ROOT": str(output_root),
            "AF_RESUME_FROM": "postfreeze",
            "AF_POSTFREEZE_ARRAY": "3",
        },
        check=False,
    )


def test_resume_rejects_code_identity_that_differs_from_the_sealed_record(
    tmp_path: Path,
) -> None:
    result = _resume(tmp_path, "0" * 64)
    assert result.returncode == 2
    assert "chain inputs differ from the sealed execution record" in result.stderr


def test_resume_proceeds_when_code_identity_matches_the_sealed_record(
    tmp_path: Path,
) -> None:
    result = _resume(tmp_path, _chain_digest())
    assert result.returncode == 0, result.stderr
    assert "postfreeze_JOB=12345" in result.stdout
    # resuming skips the already-completed preparation stage
    assert "prepare_JOB=" not in result.stdout


def test_offline_checks_never_submit_the_sealed_chain() -> None:
    """The copy-paste pre-check must stop before test data can be opened."""
    path = HPC / "external_baseline_offline_checks.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    executable, _, note = text.partition("cat <<'NOTE'")
    # The submit command may appear in the closing note, never as executed code.
    assert "submit_external_baseline_evaluation_delta.sh" in note
    assert "submit_external_baseline_evaluation_delta.sh" not in executable
    assert "sbatch" not in executable
    for forbidden in (
        "evaluate_phase_b_postfreeze",
        "--authorize-execution",
        "--verify-execution-record",
        "--seal-execution-record",
    ):
        assert forbidden not in executable


# --- unsealed development-run layout --------------------------------------


def _development_runs(
    tmp_path: Path, *, complete: bool = True, status: str = "complete"
) -> Path:
    """Mirror the producer's unsealed layout: runs/<method>/<id>_<tier>_seed<r>."""
    root = tmp_path / "public-baselines-full-v1"
    run = root / "runs" / "sindy" / "phase_b_cell_easy_seed0"
    run.mkdir(parents=True)
    if complete:
        (run / "result.json").write_text(
            json.dumps(_development_result("sindy", "phase_b_cell", "easy", 0)),
            encoding="utf-8",
        )
        (run / "run_status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "elapsed_wall_seconds": 1.0,
                    "wall_timeout_seconds": 10.0,
                }
            ),
            encoding="utf-8",
        )
    return root


def test_layout_detection_prefers_a_readiness_freeze(tmp_path: Path) -> None:
    assert symbolic_layout(tmp_path / "absent") == "absent"
    assert symbolic_layout(_development_runs(tmp_path)) == "development_runs"
    freeze = _symbolic_freeze(
        tmp_path / "sealed",
        tasks=[{"task_index": 0, "method": "sindy", **CELL, "repetition": 0}],
        write={0: _development_result("sindy", "phase_b_cell", "easy", 0)},
    )
    assert symbolic_layout(freeze) == "readiness_freeze"


def test_unsealed_development_runs_are_resolved(tmp_path: Path) -> None:
    """The readiness freeze cannot exist while any cell is unfinished."""
    root = _development_runs(tmp_path)
    plan = _plan([_method("sindy")])
    requests, sources = resolve_external_baseline_sources(plan, roots={"sindy": root})

    assert len(requests) == 1
    assert sources[0].artifact_status == "available"
    assert sources[0].adapter_requested is True
    assert sources[0].source_path.endswith(
        "runs/sindy/phase_b_cell_easy_seed0/result.json"
    )
    assert set(sources[0].artifact_sha256) == {"result.json", "run_status.json"}


def test_unfinished_development_run_fails_closed(tmp_path: Path) -> None:
    root = _development_runs(tmp_path, status="timed_out")
    plan = _plan([_method("sindy")])
    with pytest.raises(ValueError, match="did not terminate complete"):
        resolve_external_baseline_sources(plan, roots={"sindy": root})


def test_absent_development_run_is_a_missing_row(tmp_path: Path) -> None:
    root = _development_runs(tmp_path, complete=False)
    plan = _plan([_method("sindy")])
    requests, sources = resolve_external_baseline_sources(plan, roots={"sindy": root})
    assert requests == ()
    assert sources[0].artifact_status == "missing"
    assert sources[0].missing_artifacts[0].endswith("result.json")


def test_timeout_and_failure_are_recorded_distinctly(tmp_path: Path) -> None:
    """A wall-clock limit is a budget artifact, not a method failure."""
    plan = _plan([_method("sindy")])
    for status, message, expected in (
        ("timed_out", "baseline wall-clock limit reached", "of a 1800.0s budget"),
        (
            "failed",
            "ValueError: no SINDy support passed safe validation rollout",
            "no SINDy support",
        ),
    ):
        root = tmp_path / status
        run = root / "runs" / "sindy" / "phase_b_cell_easy_seed0"
        run.mkdir(parents=True)
        (run / "run_status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "elapsed_wall_seconds": 1800.0,
                    "wall_timeout_seconds": 1800.0,
                    "message": message,
                }
            ),
            encoding="utf-8",
        )
        _, sources = resolve_external_baseline_sources(plan, roots={"sindy": root})
        row = sources[0]
        assert row.artifact_status == "missing"
        assert row.terminal_status == status
        assert expected in row.reason
        assert row.outcome().error == row.reason


# --- full-roster median ---------------------------------------------------


def test_unscored_rows_rank_worst_rather_than_being_dropped() -> None:
    """Conditioning on success flatters whichever method fails most."""
    scored = [1.0, 2.0, 3.0]
    # Three of five scored. The two failures rank beyond every observed value,
    # so the middle of the ranking sits at the worst thing the method managed.
    assert full_roster_median(scored, 5) == 3.0
    # Conditioning on success discards them and reports the middle of what
    # happened to work, which is better than the method actually did.
    assert median(scored) == 2.0
    # With nothing unscored the two agree.
    assert full_roster_median([*scored, 4.0, 5.0], 5) == 3.0
    assert median([*scored, 4.0, 5.0]) == 3.0


def test_the_median_is_undefined_when_most_of_the_roster_is_unscored() -> None:
    """The middle of the ranking then falls among the failures."""
    # half scored: the lower median is still an observed value
    assert full_roster_median([1.0, 2.0], 4) == 2.0
    # fewer than half: the middle of the ranking is itself a failure
    assert full_roster_median([1.0], 4) is None
    assert full_roster_median([], 3) is None
    assert full_roster_median([1.0], 0) is None


def test_report_separates_a_compute_limit_from_a_diverged_model() -> None:
    roster = (_row("evaluated", "a"), _row("evaluated", "b"), _row("evaluated", "c"))
    outcomes = tuple(_outcome("adapted", item) for item in ("a", "b", "c"))
    records = (
        _record("a", nmse=0.5),
        _record("b", failed="one or more held-out free rollouts failed"),
        _record("c", failed="TimeoutError: fitting wall-clock limit reached"),
    )
    report = summarize(join(roster, outcomes, records))
    counts = report["by_method"]["sindy"]
    assert counts["scored_count"] == 1
    assert counts["evaluated_but_unscored"] == 2
    assert counts["unscored_diverged"] == 1
    assert counts["unscored_timeout"] == 1
    # one of three scored, so the middle of the ranking is a failure
    assert counts["target_nmse_median_full_roster"] is None
    assert counts["full_roster_median_defined"] is False
    # the conditional statistic still reports a number, which is why it is
    # labelled diagnostic rather than headline
    assert counts["target_nmse_median_conditional_on_success"] == 0.5
    assert report["headline_median"] == "target_nmse_median_full_roster"
