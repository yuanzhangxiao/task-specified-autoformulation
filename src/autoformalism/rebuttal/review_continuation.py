"""A new development phase from immutable retained public-only checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicFitResult
from autoformalism.search import review_model_edits


def _check_result(
    result: dict,
    task: dict,
    index: int,
    *,
    profile: str = "collocation-single-target-v2",
) -> None:
    if result.get("task") != task or result.get("round") != index:
        raise ValueError("source checkpoint task/round mismatch")
    if result.get("test_data_opened") is not False:
        raise ValueError("source checkpoint lacks a public-only declaration")
    selected = result.get("selected")
    if selected is None:
        return
    request = PublicFitRequest.model_validate(selected["request"])
    fit = PublicFitResult.model_validate(selected["fit"])
    if request.profile != profile or fit.profile != profile:
        raise ValueError("source fitter profile differs")
    if (
        fit.parameters is None
        or not fit.training.available
        or not fit.validation.available
    ):
        raise ValueError("retained source model lacks finite development scores")
    if selected.get("packet") is not None:
        # Checks candidate/parameter identities without interpreting held-out scores.
        review_model_edits.payload(
            selected["bundle"], selected["packet"], fit.parameters
        )


def _draft_record(
    source: Path, task: dict, result: dict, index: int
) -> tuple[dict, str] | None:
    """Recover only recorded executable drafts, never infer a missing proposal."""
    if result.get("construction_draft") is not None:
        return result, "construction_draft"
    for at in range(index, -1, -1):
        path = io.round_path(source, task, at) / "proposal.json"
        if not path.exists():
            continue
        proposal = sealed_read(path)
        if (
            proposal.get("task") != task
            or proposal.get("round") != at
            or proposal.get("test_data_opened") is not False
        ):
            raise ValueError("draft proposal identity/public boundary differs")
        for field in ("construction_draft", "bundle"):
            if proposal.get(field) is not None:
                return proposal, field
    return None


def verify_imports(root: Path, plan: dict) -> None:
    """Copied source and imported checkpoint must agree, including learned initials."""
    ledger = plan["continuation"]
    if plan["protocol"] == io.INTEGRITY_PROTOCOL:
        from autoformalism.search.review_integrity import (
            PARAMETER_POLICY,
            selection_policy,
        )

        if (
            ledger.get("parameter_declaration_policy") != PARAMETER_POLICY
            or ledger.get("selection_policy") != selection_policy()
        ):
            raise ValueError("continuation integrity policies differ")
    if set(ledger["results"]) != {t["task_id"] for t in plan["tasks"]}:
        raise ValueError("import ledger omits a lineage")
    for task in plan["tasks"]:
        source = sealed_read(root / "imports" / task["task_id"] / "result.json")
        if source["artifact_sha256"] != ledger["results"][task["task_id"]]:
            raise ValueError("source checkpoint hash differs")
        _check_result(
            source,
            task,
            ledger.get("source_phase_round", ledger["source_round"]),
            profile=plan["config"]["fit_profile"]
            if plan["protocol"] in io.MULTI_PROTOCOLS
            else "collocation-single-target-v2",
        )
        draft = None
        if plan["protocol"] in io.MULTI_PROTOCOLS:
            entry = ledger["construction_drafts"].get(task["task_id"])
            if entry:
                record = sealed_read(root / "imports" / task["task_id"] / "draft.json")
                if (
                    record["artifact_sha256"] != entry["sha256"]
                    or record["task"] != task
                    or record["round"] != entry["round"]
                    or record.get("test_data_opened") is not False
                ):
                    raise ValueError("imported construction draft differs")
                draft = record[entry["field"]]
        anchor = io.read_round(root, task, 0)
        if (
            anchor is None
            or anchor.get("selected") != source.get("selected")
            or anchor.get("source_result_sha256") != source["artifact_sha256"]
            or anchor.get("round") != 0
            or anchor.get("task") != task
            or anchor.get("status") != "imported_checkpoint"
            or anchor.get("cost") != {"physical_requests": 0}
            or anchor.get("closed")
            != (
                False
                if plan["protocol"] in io.MULTI_PROTOCOLS
                else source.get("selected") is None
            )
            or (
                plan["protocol"] in io.MULTI_PROTOCOLS
                and anchor.get("construction_draft") != draft
            )
        ):
            raise ValueError("imported anchor differs from its source")


def prepare(
    source: Path,
    root: Path,
    source_round: int = 2,
    visits: int = 5,
    *,
    protocol: str = io.CONTINUATION_PROTOCOL,
) -> dict:
    """Copy development assets and retained models; source results are never edited."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and continuation must be disjoint directories")
    if protocol not in io.CONTINUATION_PROTOCOLS:
        raise ValueError("unsupported continuation protocol")
    limit = 12 if protocol in io.MULTI_PROTOCOLS else 5
    if source_round < 0 or not 1 <= visits <= limit:
        raise ValueError(
            f"choose a source round and between one and {limit} new visits"
        )
    with io.execution_lease(source, exclusive=True), public._lock(root):
        io.require_open(source)
        original = io.verify(source, execution=False)
        allowed = (
            {io.CONTENT_PROTOCOL}
            if protocol == io.CONTINUATION_PROTOCOL
            else {io.CONTENT_PROTOCOL, io.CONTINUATION_PROTOCOL, io.PARAMETER_PROTOCOL}
            if protocol == io.PARAMETER_PROTOCOL
            else {io.CONTENT_PROTOCOL, *io.CONTINUATION_PROTOCOLS}
        )
        if original["protocol"] not in allowed:
            raise ValueError(f"source protocol must be one of {sorted(allowed)}")
        phase_round = source_round - original.get("continuation", {}).get(
            "source_round", 0
        )
        if not 0 <= phase_round < original["config"]["rounds"]:
            raise ValueError("source round is outside the original campaign")
        results, drafts = {}, {}
        for task in original["tasks"]:
            result = io.read_round(source, task, phase_round)
            if result is None:
                raise ValueError(f"source round incomplete: {task['task_id']}")
            _check_result(
                result,
                task,
                phase_round,
                profile=original["config"]["fit_profile"]
                if protocol in io.MULTI_PROTOCOLS
                else "collocation-single-target-v2",
            )
            results[task["task_id"]] = result
            if protocol in io.RESPONSE_PROTOCOLS and result["selected"] is None:
                raise ValueError(
                    "response continuation requires a fitted incumbent for every task"
                )
            if protocol in io.MULTI_PROTOCOLS and result["selected"] is None:
                item = _draft_record(source, task, result, phase_round)
                if item:
                    from autoformalism.rebuttal.review_deadline_pipeline import (
                        certificates,
                    )

                    record, field = item
                    certificates(record[field], original["cells"][task["cell"]], task)
                    drafts[task["task_id"]] = item
        ledger = {
            "source_root": str(source),
            "source_plan_sha256": original["artifact_sha256"],
            "source_code_sha256": original["source_sha256"],
            "source_round": source_round,
            "additional_visits": visits,
            "results": {k: v["artifact_sha256"] for k, v in results.items()},
            "source_cost_in_incremental_counts": False,
            "fitter_changed": False,
            "closed_valid_incumbents_reopened": True,
            "on_revision_failure": "same_visit_incumbent_refit_then_next_visit",
            "automatic_expansion": False,
        }
        if protocol in io.PARAMETER_PROTOCOLS or protocol in io.MULTI_PROTOCOLS:
            ledger.update(
                source_phase_round=phase_round, source_protocol=original["protocol"]
            )
        if protocol == io.REVISION_PROTOCOL:
            ledger["revision_size_policy"] = "advisory_initial_construction_references"
            ledger["unused_new_parameter_policy"] = "discard_with_audit"
        if protocol in io.MULTI_PROTOCOLS:
            ledger.update(
                construction_drafts={
                    key: {
                        "sha256": rec["artifact_sha256"],
                        "field": field,
                        "round": rec["round"],
                    }
                    for key, (rec, field) in drafts.items()
                },
                failed_construction_policy="bounded_public_contract_repair_then_retry_next_visit",
                public_target_gate="unchanged_frozen_contract_exposed_before_construction",
                revision_policy="scientific-content-multi-revision-1",
            )
        if protocol in io.RESPONSE_PROTOCOLS:
            ledger.update(
                revision_policy="response-oriented-revision-1",
                evidence_policy="training-response-evidence-1",
                prompt_policy={
                    "input_limit": 24000,
                    "context_tokens": 32768,
                    "output_reserve": original["config"]["model_settings"][
                        "max_output_tokens"
                    ],
                    "safety_tokens": 512,
                },
                provider_failure_policy="record_and_preserve_without_refit_or_blind_retry",
            )
        if protocol == io.INTEGRITY_PROTOCOL:
            from autoformalism.search.review_integrity import (
                PARAMETER_POLICY,
                selection_policy,
            )

            ledger.update(
                parameter_declaration_policy=PARAMETER_POLICY,
                selection_policy=selection_policy(),
            )
        if (root / "plan.json").exists():
            plan = io.verify(root)
            if plan.get("continuation") != ledger or plan["protocol"] != protocol:
                raise ValueError("continuation source or additional budget differs")
            return plan
        config = io.DeadlineConfig.model_validate(
            {
                **original["config"],
                "protocol": protocol,
                "rounds": visits + 1,
            }
        )
        for cell, value in original["cells"].items():
            for name, digest in value["assets"].items():
                data = (source / "public/phase_b_v1" / cell / name).read_bytes()
                if name not in io.FILES or hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("source public content differs")
                dest = root / "public/phase_b_v1" / cell / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.read_bytes() != data:
                    raise ValueError("partial public import differs")
                dest.write_bytes(data)
        for task in original["tasks"]:
            result = results[task["task_id"]]
            item = drafts.get(task["task_id"])
            if item:
                record, field = item
                sealed_write(
                    root / "imports" / task["task_id"] / "draft.json",
                    {k: v for k, v in record.items() if k != "artifact_sha256"},
                )
            sealed_write(
                root / "imports" / task["task_id"] / "result.json",
                {k: v for k, v in result.items() if k != "artifact_sha256"},
            )
            sealed_write(
                io.round_path(root, task, 0) / "result.json",
                {
                    "task": task,
                    "round": 0,
                    "status": "imported_checkpoint",
                    "trial": None,
                    "selected": result["selected"],
                    "closed": False
                    if protocol in io.MULTI_PROTOCOLS
                    else result["selected"] is None,
                    **(
                        {"construction_draft": item[0][item[1]] if item else None}
                        if protocol in io.MULTI_PROTOCOLS
                        else {}
                    ),
                    "source_result_sha256": result["artifact_sha256"],
                    "cost": {"physical_requests": 0},
                    "test_data_opened": False,
                },
            )
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": protocol,
                "config": config.model_dump(mode="json"),
                "cells": original["cells"],
                "tasks": original["tasks"],
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "launcher_sha256": io.launcher_hash(protocol),
                "continuation": ledger,
                "test_data_opened": False,
                "private_reference_opened": False,
            },
        )
        verify_imports(root, plan)
        return plan
