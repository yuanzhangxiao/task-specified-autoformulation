"""Portable public model snapshots from existing campaigns; no refitting or tests."""

from __future__ import annotations

from pathlib import Path

from autoformalism.baselines.d3_rollout import NativeMap
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import (
    CandidateModel,
    InitialConditionSpec,
    ObservationMapping,
)
from autoformalism.schemas.public_fitting import PublicFitRequest

BUNDLE_PROTOCOL = "public-mechanism-model-bundle-1"


def baseline_rows(root: Path) -> list[dict]:
    """Reuse exact embedded development models from the saved replay plan."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != "saved-baseline-validation-1":
        raise ValueError("expected saved baseline validation plan")
    rows = []
    for source in plan["rows"]:
        if source["cohort"] != "phase_b" or source["source_kind"] == "d3":
            continue  # D3 comes only from its native, exact-cell campaign below.
        row = {
            "method": source["source_kind"],
            "benchmark_id": source["benchmark_id"],
            "tier": source["tier"],
            "repetition": source["repetition"],
            "source_plan_sha256": plan["artifact_sha256"],
            "source_id": str(root / "plan.json") + f"#row={source['index']}",
            "semantics": "continuous_time",
            "status": "unavailable",
            "error": source.get("error"),
            "provenance": source.get("audit"),
            "validation_nmse": None,
        }
        if source["status"] == "ready":
            subject = FrozenEvaluationSubject.model_validate(source["subject"])
            if (subject.benchmark_id, subject.tier, subject.repetition) != (
                row["benchmark_id"],
                row["tier"],
                row["repetition"],
            ):
                raise ValueError("embedded subject identity differs from plan")
            row.update(
                status="ready",
                candidate=subject.candidate.model_dump(mode="json"),
                parameters=subject.parameterization.global_parameters,
                initials=subject.parameterization.global_initial_conditions,
                context=subject.validation_context.model_dump(mode="json"),
                public_prompt_sha256=source["data_identity"]["prompt"],
            )
        path = root / "results" / f"{source['index']:04d}.json"
        if path.exists():
            result = sealed_read(path)
            if (result["plan_sha256"], result["index"]) != (
                plan["artifact_sha256"],
                source["index"],
            ):
                raise ValueError("saved validation score has a different identity")
            row.update(
                validation_nmse=result.get("normalized_mse"),
                validation_status=result["status"],
                validation_result_sha256=result["artifact_sha256"],
            )
        rows.append(row)
    return rows


def d3_rows(root: Path) -> list[dict]:
    """Preserve D3 increments and native training checkpoint coefficients."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != "phase-b-d3-native-validation-1":
        raise ValueError("expected native Phase-B D3 plan")
    rows = []
    for source in plan["rows"]:
        row = {
            "method": "d3",
            "benchmark_id": source["benchmark_id"],
            "tier": source["tier"],
            "repetition": source["repetition"],
            "source_plan_sha256": plan["artifact_sha256"],
            "source_id": str(root / "results" / str(source["index"])),
            "semantics": "native_increment",
            "status": "unavailable",
            "validation_nmse": None,
            "error": "native selection unavailable",
        }
        path = root / "results" / str(source["index"]) / "native-selection.json"
        if path.exists():
            saved = sealed_read(path)
            if saved["plan_sha256"] != plan["artifact_sha256"]:
                raise ValueError("D3 selection differs from plan")
            selected = BaselineDevelopmentResult.model_validate(saved["selection"])
            if (selected.benchmark_id, selected.tier, selected.seed) != (
                row["benchmark_id"],
                row["tier"],
                row["repetition"],
            ):
                raise ValueError("D3 selection has different cell or repetition")
            payload = selected.selection_payload
            context = ValidationContext.model_validate(source["validation_context"])
            original = CandidateModel.model_validate(payload["candidate"])
            NativeMap.build(
                original,
                payload["parameters"],
                (*context.targets, *context.auxiliaries),
                context.external_inputs,
            )
            # Native rollout overwrites supplied auxiliaries at every sample and
            # returns target state values directly. Their fitted increment laws
            # and arbitrary observation/initial fields cannot supply fake paths.
            effective = original.model_copy(
                update={
                    "state_equations": tuple(
                        e
                        for e in original.state_equations
                        if e.state in context.targets
                    ),
                    "observation_mappings": tuple(
                        ObservationMapping(channel=t, expression=t)
                        for t in context.targets
                    ),
                    "initial_conditions": tuple(
                        InitialConditionSpec(state=t, scope="global", expression=t)
                        for t in context.targets
                    ),
                }
            )
            row.update(
                status="ready",
                error=None,
                candidate=effective.model_dump(mode="json"),
                parameters=payload["parameters"],
                initials={},
                context=source["validation_context"],
                public_prompt_sha256=source["public_identity"]["prompt"],
                source_selection_sha256=saved["artifact_sha256"],
                native_auxiliary_equations_ignored=[
                    e.model_dump()
                    for e in original.state_equations
                    if e.state in context.auxiliaries
                ],
                native_boundary="observed_initial_targets_supplied_auxiliary_paths",
            )
            result_path = path.with_name("result.json")
            if result_path.exists():
                result = sealed_read(result_path)
                if result["plan_sha256"] != plan["artifact_sha256"]:
                    raise ValueError("D3 score differs from plan")
                if result.get("source_files", {}).get("native-selection.json"):
                    from autoformalism.rebuttal.baseline_validation import file_hash

                    if result["source_files"]["native-selection.json"] != file_hash(
                        path
                    ):
                        raise ValueError("D3 scored selection has changed")
                evaluation = result.get("evaluation", {})
                row.update(
                    validation_nmse=evaluation.get("phase_b_rollout", {}).get(
                        "normalized_mse"
                    )
                    if evaluation.get("saved_one_step_matches")
                    else None,
                    validation_status=result["status"],
                    validation_result_sha256=result["artifact_sha256"],
                )
        rows.append(row)
    return rows


def review_rows(root: Path, round_index: int, arms: tuple[str, ...]) -> list[dict]:
    """Snapshot an explicit completed round without closing or mutating search."""
    plan = sealed_read(root / "plan.json")
    if not plan["protocol"].startswith("review-deadline-"):
        raise ValueError("expected review campaign")
    offset = plan.get("continuation", {}).get("source_round", 0)
    local = round_index - offset
    if not 0 <= local < plan["config"]["rounds"]:
        raise ValueError(f"global round {round_index} is outside this campaign")
    rows = []
    for task in plan["tasks"]:
        if task["arm"] not in arms:
            continue
        cell = plan["cells"][task["cell"]]
        path = root / "results" / task["task_id"] / f"round_{local:02d}" / "result.json"
        row = {
            "method": "autoformalism:" + task["arm"],
            "round": round_index,
            "benchmark_id": task["cell"],
            "tier": cell["target_contract"]["tier"],
            "repetition": task["seed"],
            "source_id": str(path),
            "source_plan_sha256": plan["artifact_sha256"],
            "semantics": "continuous_time",
            "status": "unavailable",
            "validation_nmse": None,
            "error": "round result or retained model unavailable",
            "evaluation_scope": "full_public_task_posthoc",
            "task_specification_visible_during_search": task["arm"] != "no_spec",
        }
        if path.exists():
            result = sealed_read(path)
            if result["task"] != task or result["round"] != local:
                raise ValueError("review result identity differs from requested round")
            selected = result.get("selected")
            row["source_result_sha256"] = result["artifact_sha256"]
            if selected:
                model, _, _ = public._lower(
                    PublicFitRequest.model_validate(selected["request"])
                )
                row.update(
                    status="ready",
                    error=None,
                    candidate=model.validated.candidate.model_dump(mode="json"),
                    parameters=selected["fit"]["parameters"],
                    initials={},
                    context=model.validated.context.model_dump(mode="json"),
                    public_prompt_sha256=cell["assets"]["proposer_prompt.txt"],
                    validation_nmse=selected["fit"]
                    .get("validation", {})
                    .get("normalized_mse"),
                )
        rows.append(row)
    return rows


def export_bundle(
    output: Path,
    *,
    baseline_root: Path | None = None,
    d3_root: Path | None = None,
    review_root: Path | None = None,
    round_index: int | None = None,
    arms: tuple[str, ...] = ("full",),
) -> dict:
    """Transport only public equations/scalars and existing validation scores."""
    rows = []
    if baseline_root:
        rows.extend(baseline_rows(baseline_root))
    if d3_root:
        rows.extend(d3_rows(d3_root))
    if review_root:
        if round_index is None:
            raise ValueError("review snapshots require explicit --round")
        rows.extend(review_rows(review_root, round_index, arms))
    if not rows:
        raise ValueError("no expected model rows; specify existing source campaigns")
    return sealed_write(
        output,
        {
            "protocol": BUNDLE_PROTOCOL,
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
            "live_llm_calls": 0,
            "parameter_refit_applied": False,
        },
    )
