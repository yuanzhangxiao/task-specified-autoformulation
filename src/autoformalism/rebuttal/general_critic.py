"""Advisory calibrated reviews around one general whole-model revision episode."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.llm.exceptions import LLMError
from autoformalism.llm.review_revision import RevisionClient
from autoformalism.llm.staged_topology import StagedModelSettings, visible_response
from autoformalism.rebuttal import general_critic_io as io
from autoformalism.rebuttal import process_pruning as pruning
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_evidence import model_hash
from autoformalism.rebuttal.review_deadline_pipeline import (
    certificates,
    replay_packet,
    request_for,
)
from autoformalism.rebuttal.revision_decision import parameter_aliases, translate_names
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)
from autoformalism.search import shared_model_revision as revision
from autoformalism.staged_topology import content_hash

ADVICE_NOTE = """
Scientific judge findings below are advisory, not certified defects or instructions.
Keep them separate from executable public requirements and training mismatch.
The calibrated judge saw symbolic models and the public task, not numerical results.
You decide what deserves investigation and may revise all coupled definitions in
one coherent patch. This is experimental proposer routing, not calibrated judging.
A mismatch at current fitted parameters does not establish structural impossibility.
Unavailable/indeterminate advice is neither a pass nor a model failure. You may
return no_change. The complete current equations and shared consumers are shown.
"""


def directory(root: Path, row: dict) -> Path:
    """Use the frozen task identity for all stage checkpoints."""
    return root / "results" / row["task"]["task_id"]


def read_stage(path: Path, plan: dict) -> dict:
    """Prevent a receipt from another campaign being used as feedback."""
    value = sealed_read(path)
    if value["identity"] != plan["artifact_sha256"]:
        raise ValueError("stage belongs to another campaign")
    return value


def prepare_evidence(root: Path, index: int) -> dict:
    """Reuse fixed-parameter training replay; no new optimizer calls."""
    plan = io.verify(root)
    row = plan["rows"][index]
    work = directory(root, row)
    with public._lock(work):
        path = work / "evidence.json"
        if path.exists():
            return read_stage(path, plan)
        request = PublicFitRequest.model_validate(row["request"])
        packet = replay_packet(
            root,
            work / "training",
            request,
            PublicFitResult.model_validate(row["fit"]),
            PublicSplit.model_validate(plan["cells"][row["task"]["cell"]]["training"]),
        )
        return sealed_write(
            path,
            {
                "identity": plan["artifact_sha256"],
                "packet": packet,
                "status": "available" if packet else "unavailable",
            },
        )


def critic_request(plan: dict, row: dict, child: dict | None = None) -> dict:
    """The original allowlisted judge contract; numerical artifacts never enter."""
    request = PublicFitRequest.model_validate(row["request"])
    parent_model, _, _ = public._lower(request)
    candidate = (
        CandidateModel.model_validate(child["candidate"])
        if child
        else parent_model.validated.candidate
    )
    return judge.review_request(
        parent_model.validated.candidate,
        candidate,
        ValidationContext.model_validate(child["initialization"]["context"])
        if child
        else parent_model.validated.context,
        plan["cells"][row["task"]["cell"]]["public_prompt"],
        row["task"]["seed"],
        plan["judge_revision"],
    )


def review_one(root: Path, index: int, stage: str, base_url: str) -> dict:
    """Self-pair parent, then paired changed equations; exact requests share cache."""
    if stage not in {"parent", "child"}:
        raise ValueError("expected parent or child review")
    plan = io.verify(root)
    row = plan["rows"][index]
    work = directory(root, row)
    with public._lock(work):
        path = work / f"{stage}_review.json"
        if path.exists():
            return read_stage(path, plan)
        child = None
        if stage == "child":
            proposal = read_stage(work / "proposal.json", plan)
            child = proposal.get("bundle")
            if child is None or (
                model_hash(CandidateModel.model_validate(child["candidate"]))
                == model_hash(CandidateModel.model_validate(row["bundle"]["candidate"]))
                and child["initialization"]["context"]
                == row["bundle"]["initialization"]["context"]
            ):
                parent = read_stage(work / "parent_review.json", plan)
                return sealed_write(
                    path,
                    {
                        **{k: v for k, v in parent.items() if k != "artifact_sha256"},
                        "reused_parent_review": True,
                        "parent_review_sha256": parent["artifact_sha256"],
                    },
                )
        request = critic_request(plan, row, child)
        key = content_hash(request)
        cache = root / "judge_cache" / key
        with public._lock(cache):
            request_path = cache / "request.json"
            if request_path.exists():
                if sealed_read(request_path)["request"] != request:
                    raise ValueError("cached scientific request differs")
            else:
                sealed_write(request_path, {"request": request})
            receipt = cache / "receipt.json"
            if receipt.exists():
                saved = sealed_read(receipt)
                if saved["request"] != request:
                    raise ValueError("cached scientific request differs")
            else:
                try:
                    review = judge.perform_review(request, cache, base_url)
                except (OSError, TimeoutError, LLMError) as error:
                    review = {
                        "request_sha256": key,
                        "status": "unavailable",
                        "findings": [],
                        "error": f"{type(error).__name__}: {error}",
                        "cost": judge.review_cost(cache),
                    }
                if review["request_sha256"] != key:
                    raise ValueError("judge response names another model pair")
                saved = sealed_write(receipt, {"request": request, "review": review})
        return sealed_write(
            path,
            {
                "identity": plan["artifact_sha256"],
                "review": saved["review"],
                "request_sha256": key,
                "receipt_sha256": saved["artifact_sha256"],
                "candidate_sha256": model_hash(
                    CandidateModel.model_validate(request["candidate"])
                ),
                "reused_parent_review": False,
                "placement": "after_pruning_before_revision"
                if stage == "parent"
                else "after_revision_before_fit",
                "legacy_prefit_unpruned_flag_not_applicable": True,
            },
        )


def advisory_payload(row: dict, packet: dict, review: dict, retry=None) -> dict:
    """Join separately labelled evidence only in the proposer's existing payload."""
    bundle = row["bundle"]
    if review["candidate_sha256"] != model_hash(
        CandidateModel.model_validate(bundle["candidate"])
    ):
        raise ValueError("scientific advice belongs to another candidate")
    value = revision.payload(bundle, packet, row["fit"]["parameters"], retry)
    evidence = review["review"]
    aliases = parameter_aliases(
        CandidateModel.model_validate(bundle["initialization"]["base_candidate"])
    )
    value["scientific_advice"] = translate_names(
        {
            "availability": judge.review_status(evidence),
            "findings": evidence["findings"],
            "request_sha256": review["request_sha256"],
            "can_override_public_checks": False,
            "used_for_selection": False,
        },
        aliases,
    )
    value["public_requirement_findings"] = row["certificate"]
    value["fitting_status"] = {
        k: row["fit"].get(k)
        for k in ("status", "budget_exhausted", "native_optimizer_converged", "message")
    }
    value["feedback_scope"] = ADVICE_NOTE
    return value


def propose_one(root: Path, index: int, base_url: str, *, transport=None) -> dict:
    """At most three cached replies, with ordinary compiler repair and no judge veto."""
    plan = io.verify(root)
    row = plan["rows"][index]
    work = directory(root, row)
    with public._lock(work):
        path = work / "proposal.json"
        if path.exists():
            return read_stage(path, plan)
        evidence = read_stage(work / "evidence.json", plan)
        review = read_stage(work / "parent_review.json", plan)
        packet = evidence["packet"]
        if packet is None:
            return sealed_write(
                path,
                {
                    "identity": plan["artifact_sha256"],
                    "status": "residual_evidence_unavailable",
                    "bundle": None,
                    "attempts": [],
                    "cost": _cost([]),
                },
            )
        kwargs = {} if transport is None else {"transport": transport}
        client = RevisionClient(
            settings=StagedModelSettings.model_validate(plan["proposer_settings"]),
            seed=row["task"]["seed"],
            namespace=content_hash([plan["artifact_sha256"], row["task"]]),
            directory=work / "calls",
            base_url=base_url,
            can_start=lambda: True,
            **kwargs,
        )
        attempts, retry, child, decision, certificate = [], None, None, None, None
        for attempt in range(3):
            user = advisory_payload(row, packet, review, retry)
            record = client.call(
                system=revision.SYSTEM_PROMPT + ADVICE_NOTE,
                user=json.dumps(user, sort_keys=True),
                response_model=revision.ScientificRevision,
                step="general_critic_revision",
                attempt=attempt,
            )
            raw = None
            try:
                raw = visible_response(record)
                decision = revision.apply_edits(row["bundle"], packet, raw)
                child = decision["bundle"]
                if child:
                    certificate = certificates(
                        child, plan["cells"][row["task"]["cell"]], row["task"]
                    )
                    if not certificate["eligible_for_development_selection"]:
                        raise ValueError(
                            "PUBLIC_MODEL_REQUIREMENTS: "
                            + json.dumps(certificate, sort_keys=True)
                        )
                    request = request_for(
                        child,
                        {"config": {"fit_profile": io.POLICY["fit_profile"]}},
                        row["task"],
                        1,
                    )
                    sibling_fit.compatible_seed(
                        PublicFitRequest.model_validate(row["request"]),
                        request,
                        row["fit"]["parameters"],
                        PublicSplit.model_validate(
                            plan["cells"][row["task"]["cell"]]["training"]
                        ),
                        allow_initialization_changes=True,
                    )
                attempts.append(
                    {
                        "accepted": True,
                        "raw": raw,
                        "request_hash": record["request_hash"],
                    }
                )
                break
            except (ValueError, KeyError, TypeError, ModelValidationError) as error:
                retry = revision.feedback(
                    row["bundle"], packet, row["fit"]["parameters"], raw, error
                )
                attempts.append(
                    {
                        "accepted": False,
                        "raw": raw,
                        "request_hash": record["request_hash"],
                        "feedback": retry,
                    }
                )
                child, decision, certificate = None, None, None
        return sealed_write(
            path,
            {
                "identity": plan["artifact_sha256"],
                "status": decision["outcome"] if decision else "revision_failed",
                "bundle": child,
                "certificate": certificate,
                "decision": decision,
                "attempts": attempts,
                "cost": _cost(client.records),
                "parent_review_sha256": review["artifact_sha256"],
            },
        )


def select(
    parent: dict,
    child: dict | None,
    certificate: dict | None,
    parent_request: PublicFitRequest,
    child_request: PublicFitRequest | None,
) -> str:
    """Critic verdict is deliberately absent from the selection interface."""
    if (
        child is None
        or child_request is None
        or not certificate
        or not certificate["eligible_for_development_selection"]
        or pruning.score(child) is None
    ):
        return "parent"

    def key(fit, request):
        return pruning.score(fit), pruning.pruning.complexity(request)["terms"]

    return (
        "child" if key(child, child_request) < key(parent, parent_request) else "parent"
    )


def checked_result(work: Path, plan: dict, row: dict) -> dict:
    """Bind a published outcome to the proposal, review and actual fit receipts."""
    result = read_stage(work / "result.json", plan)
    proposal = read_stage(work / "proposal.json", plan)
    review = read_stage(work / "child_review.json", plan)
    if (
        result["task"] != row["task"]
        or result["proposal_sha256"] != proposal["artifact_sha256"]
        or result["child_review_sha256"] != review["artifact_sha256"]
    ):
        raise ValueError("published outcome provenance differs")
    request = None
    if result["child_fit"] is not None:
        request = request_for(
            proposal["bundle"],
            {"config": {"fit_profile": io.POLICY["fit_profile"]}},
            row["task"],
            1,
        )
        if request.model_dump(mode="json") != result["child_request"]:
            raise ValueError("published child request differs")
        actual = sibling_fit.inspect_child_fit(work / "fit")
        if actual["result"] != result["child_fit"]:
            raise ValueError("published child fit differs")
    expected = select(
        row["fit"],
        result["child_fit"],
        result["child_certificate"],
        PublicFitRequest.model_validate(row["request"]),
        request,
    )
    retained = result["child_fit"] if expected == "child" else row["fit"]
    if expected != result["selected"] or retained != result["retained_fit"]:
        raise ValueError("published selection differs")
    return result


def fit_one(root: Path, index: int) -> dict:
    """Fit one legal child even if its advisory review was unavailable."""
    plan = io.verify(root)
    row = plan["rows"][index]
    work = directory(root, row)
    with public._lock(work):
        path = work / "result.json"
        if path.exists():
            return checked_result(work, plan, row)
        proposal = read_stage(work / "proposal.json", plan)
        review = read_stage(work / "child_review.json", plan)
        parent = PublicFitRequest.model_validate(row["request"])
        request, fit = None, None
        certificate = proposal.get("certificate")
        if proposal.get("bundle") is not None:
            bundle = proposal["bundle"]
            if review["candidate_sha256"] != model_hash(
                CandidateModel.model_validate(bundle["candidate"])
            ):
                raise ValueError("child was not reviewed under its current equations")
            certificate = certificates(
                bundle, plan["cells"][row["task"]["cell"]], row["task"]
            )
            if certificate["eligible_for_development_selection"]:
                request = request_for(
                    bundle,
                    {"config": {"fit_profile": io.POLICY["fit_profile"]}},
                    row["task"],
                    1,
                )
                cell = plan["cells"][row["task"]["cell"]]
                sibling_fit.prepare_child_fit(
                    parent,
                    request,
                    row["fit"]["parameters"],
                    PublicSplit.model_validate(cell["training"]),
                    PublicSplit.model_validate(cell["validation"]),
                    work / "fit",
                    lineage={
                        "campaign": plan["artifact_sha256"],
                        "proposal": proposal["artifact_sha256"],
                    },
                    allow_initialization_changes=True,
                )
                fit = sibling_fit.execute_child_fit(work / "fit").model_dump(
                    mode="json"
                )
        selected = select(row["fit"], fit, certificate, parent, request)
        return sealed_write(
            path,
            {
                "identity": plan["artifact_sha256"],
                "task": row["task"],
                "status": fit["status"]
                if fit
                else (
                    "requirement_failed"
                    if certificate
                    and not certificate["eligible_for_development_selection"]
                    else proposal["status"]
                ),
                "selected": selected,
                "child_fit": fit,
                "child_request": request.model_dump(mode="json") if request else None,
                "child_certificate": certificate,
                "proposal_sha256": proposal["artifact_sha256"],
                "child_review_sha256": review["artifact_sha256"],
                "retained_fit": fit if selected == "child" else row["fit"],
                "critic_used_for_selection": False,
                "test_data_opened": False,
                "independent_replay": "not_performed",
                "automatic_followup": False,
            },
        )


def proposer_usage(work: Path, plan: dict, row: dict) -> dict:
    """Count persisted calls even when the terminal proposal was not published."""
    namespace = content_hash([plan["artifact_sha256"], row["task"]])
    records = []
    for path in sorted((work / "calls").glob("*.json")):
        record = public._read(path)
        if (
            record["request"]["namespace"] != namespace
            or content_hash(record["request"]) != path.stem
            or record["request_hash"] != path.stem
        ):
            raise ValueError("revision cache provenance differs")
        records.append(record)
    return _cost(records)


def review_receipts(root: Path, requests: dict) -> list[dict]:
    """Preserve observed usage and unknown completion for interrupted reviews."""
    receipts = []
    for cache in sorted((root / "judge_cache").glob("*")):
        path = cache / "request.json"
        if not path.exists():
            continue
        request = sealed_read(path)["request"]
        if content_hash(request) != cache.name or requests.get(cache.name) != request:
            raise ValueError("scientific usage request provenance differs")
        receipt = cache / "receipt.json"
        if receipt.exists():
            saved = sealed_read(receipt)
            if (
                saved["request"] != request
                or saved["review"]["request_sha256"] != cache.name
            ):
                raise ValueError("scientific usage receipt differs")
        else:
            saved = {
                "request": request,
                "review": {
                    "status": "unpublished",
                    "cost": {**judge.review_cost(cache), "usage_complete": False},
                },
            }
        receipts.append(saved)
    return receipts


def report(root: Path) -> dict:
    """Report advisory availability, hard predicates, fit and costs independently."""
    plan = io.verify(root)
    rows, proposer_cost, requests = [], [], {}
    for row in plan["rows"]:
        work = directory(root, row)
        stages = {
            name: read_stage(work / f"{name}.json", plan)
            if (work / f"{name}.json").exists()
            else None
            for name in ("parent_review", "proposal", "child_review", "result")
        }
        cost = proposer_usage(work, plan, row)
        if stages["proposal"] and stages["proposal"]["cost"] != cost:
            raise ValueError("published proposer usage differs from call records")
        proposer_cost.append(cost)
        for child in (None, (stages["proposal"] or {}).get("bundle")):
            request = critic_request(plan, row, child)
            requests[content_hash(request)] = request
        result = stages["result"]
        if result is not None:
            result = checked_result(work, plan, row)
        rows.append(
            {
                "task": row["task"]["task_id"],
                "status": result["status"] if result else "missing",
                "parent": row["fit"],
                "parent_certificate": row["certificate"],
                "pruning_origin": row["pruning_origin"],
                "parent_complexity": pruning.pruning.complexity(
                    PublicFitRequest.model_validate(row["request"])
                ),
                "child_complexity": pruning.pruning.complexity(
                    PublicFitRequest.model_validate(result["child_request"])
                )
                if result and result["child_request"]
                else None,
                **stages,
            }
        )
    receipts = review_receipts(root, requests)
    totals = {
        "physical_requests": sum(
            x["review"]["cost"]["physical_requests"] for x in receipts
        ),
        "observed_total_tokens": sum(
            x["review"]["cost"]["observed_total_tokens"] for x in receipts
        ),
        "usage_missing_events": sum(
            x["review"]["cost"]["usage_missing_events"] for x in receipts
        ),
        "unique_review_requests": len(receipts),
        "usage_complete": bool(receipts)
        and all(x["review"]["cost"].get("usage_complete", False) for x in receipts),
        "scope": "Recorded calls/tokens; interrupted requests may have "
        "unobserved usage.",
    }
    value = {
        "protocol": io.PROTOCOL,
        "identity": plan["artifact_sha256"],
        "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "selection_counts": dict(
            Counter(r["result"]["selected"] for r in rows if r["result"])
        ),
        "review_status_counts": dict(Counter(x["review"]["status"] for x in receipts)),
        "judge_cost": totals,
        "proposer_cost": {
            k: sum(x[k] for x in proposer_cost)
            for k in (
                "physical_requests",
                "observed_tokens",
                "requests_with_unknown_usage",
            )
        },
        "test_data_opened": False,
        "automatic_followup": False,
        "critic_used_for_selection": False,
        "calibrated_for_routing": False,
    }
    public._write(root / "summary.json", value)
    lines = [
        "# General calibrated critic integration",
        "",
        "Existing judge, advisory only. One revision episode; frozen fitter. "
        "Validation selects the incumbent.",
        "Routing is experimental; calibration does not certify repair advice. "
        "No test access or automatic follow-up.",
        "",
        "| Task | Status | Parent review | Child review | Parent val NMSE | "
        "Child val NMSE | Selected |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in rows:
        outcome = r["result"] or {}
        lines.append(
            "| "
            + " | ".join(
                str(v)
                for v in (
                    r["task"],
                    r["status"],
                    (r["parent_review"] or {}).get("review", {}).get("status"),
                    (r["child_review"] or {}).get("review", {}).get("status"),
                    pruning.score(r["parent"]),
                    pruning.score(outcome.get("child_fit")),
                    outcome.get("selected"),
                )
            )
            + " |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return value
