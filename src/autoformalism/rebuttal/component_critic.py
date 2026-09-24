"""Portable corrected-judge service for fresh component campaigns; no fit access."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.judging.hybrid import FACTOR_SIGN_POLICY as SIGN_POLICY
from autoformalism.llm import jetstream
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.general_critic import ADVICE_NOTE
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_evidence import model_hash
from autoformalism.rebuttal.revision_decision import parameter_aliases, translate_names
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash

CALIBRATION_PROTOCOL = "jetstream-known-case-revalidation-1"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def authorize(root: Path, calibration_plan: Path, summary_path: Path) -> dict:
    """Admit the tested managed transport without claiming fresh calibration."""
    io.require_open(root)
    plan = io.verify(root)
    calibration = sealed_read(calibration_plan)
    summary = json.loads(summary_path.read_text())
    expected = judge.judge_protocol(sign_policy=SIGN_POLICY)
    expected.pop("distinct_seed_attempts")
    if (
        summary.get("protocol") != CALIBRATION_PROTOCOL
        or summary.get("identity") != calibration["artifact_sha256"]
        or summary.get("known_case_gate_passed") is not True
        or summary.get("status") != "complete"
        or summary.get("planned_orientations") != 140
        or summary.get("test_data_opened") is not False
        or calibration.get("scientific_protocol") != expected
        or calibration.get("transport_policy") != jetstream.POLICY
        or calibration.get("output_contract") != jetstream.OUTPUT_CONTRACT
        or calibration.get("endpoint") != jetstream.ENDPOINT
    ):
        raise ValueError(
            "complete compatible known-case revalidation with passed gates required"
        )
    return sealed_write(
        root / "critic_authorization.json",
        {
            "identity": plan["artifact_sha256"],
            "calibration_plan_sha256": calibration["artifact_sha256"],
            "summary_sha256": content_hash(summary),
            "known_case_gate_passed": True,
            "fresh_calibration_claim": False,
            "transport_policy": jetstream.POLICY,
            "endpoint": jetstream.ENDPOINT,
            "served_revision_verified": False,
        },
    )


def checked_authorization(root: Path, plan: dict) -> dict:
    """An API smoke response alone cannot enable the production critic."""
    value = sealed_read(root / "critic_authorization.json")
    if (
        value["identity"] != plan["artifact_sha256"]
        or value["known_case_gate_passed"] is not True
    ):
        raise ValueError("critic authorization differs from this campaign")
    return value


def needed_bundle(proposal: dict, parent: dict | None):
    """A failed/unchanged revision retains its existing review and fit allowance."""
    bundle = proposal.get("bundle")
    if bundle is not None:
        return bundle
    if (parent or {}).get("selected") and proposal["status"] in {
        "no_change",
        "revision_failed",
        "residual_evidence_unavailable",
        "unchanged_refit",
    }:
        return parent["selected"]["bundle"]
    return None


def incumbent_review(root: Path, task: dict, selected: dict) -> dict:
    """Attach advice only from the exact retained candidate's originating visit."""
    path = io.round_path(root, task, selected["origin_round"]) / "critic.json"
    value = sealed_read(path)
    if value.get("candidate_sha256") != model_hash(
        CandidateModel.model_validate(selected["bundle"]["candidate"])
    ):
        raise ValueError("critic receipt belongs to another candidate")
    return value


def advice(root: Path, task: dict, parent: dict | None) -> dict | None:
    """Only symbolic findings are joined to training feedback; no judge score."""
    if not task["critic"] or not (parent or {}).get("selected"):
        return None
    selected = parent["selected"]
    receipt = incumbent_review(root, task, selected)
    evidence = receipt["review"]
    aliases = parameter_aliases(
        CandidateModel.model_validate(
            selected["bundle"]["initialization"]["base_candidate"]
        )
    )
    return translate_names(
        {
            "availability": judge.review_status(evidence),
            "findings": evidence.get("findings", []),
            "request_sha256": receipt["request_sha256"],
            "can_override_public_checks": False,
            "used_for_selection": False,
            "scope": ADVICE_NOTE,
        },
        aliases,
    )


def review_one(
    root: Path,
    plan: dict,
    task: dict,
    index: int,
    key_supplier,
    *,
    progress=None,
    performer=None,
) -> dict:
    """Review before fit; cache exact symbolic pairs and never reset interruption."""
    checked_authorization(root, plan)
    if not task["critic"]:
        raise ValueError("critic disabled for this arm")
    directory = io.round_path(root, task, index)
    path = directory / "critic.json"
    if path.exists():
        value = sealed_read(path)
        if value["identity"] != plan["artifact_sha256"]:
            raise ValueError("critic receipt identity differs")
        return value
    proposal = sealed_read(directory / "proposal.json")
    parent = io.read_round(root, task, index - 1) if index else None
    bundle = needed_bundle(proposal, parent)
    payload = {
        "identity": plan["artifact_sha256"],
        "task": task,
        "round": index,
        "proposal_sha256": proposal["artifact_sha256"],
    }
    if bundle is None:
        return sealed_write(path, {**payload, "status": "no_executable_candidate"})
    candidate = CandidateModel.model_validate(bundle["candidate"])
    selected = (parent or {}).get("selected")
    if (
        selected
        and model_hash(candidate)
        == model_hash(CandidateModel.model_validate(selected["bundle"]["candidate"]))
        and bundle["initialization"]["context"]
        == selected["bundle"]["initialization"]["context"]
    ):
        old = incumbent_review(root, task, selected)
        return sealed_write(
            path,
            {
                **{k: v for k, v in old.items() if k != "artifact_sha256"},
                **payload,
                "reused": True,
            },
        )
    previous = (
        CandidateModel.model_validate(selected["bundle"]["candidate"])
        if selected
        else candidate
    )
    request = judge.review_request(
        previous,
        candidate,
        ValidationContext.model_validate(bundle["initialization"]["context"]),
        (root / "public/phase_b_v1" / task["cell"] / "proposer_prompt.txt").read_text(),
        task["seed"],
        REVISION,
        sign_policy=SIGN_POLICY,
    )
    cache = root / "judge_cache" / content_hash(request)
    with public._lock(cache):
        sealed_write(cache / "request.json", {"request": request})
        transport = jetstream.JetstreamTransport(cache / "wire", key_supplier, progress)
        review = (performer or judge.perform_review)(
            request,
            cache,
            jetstream.BASE_URL,
            client_factory=lambda config: jetstream.client(config, transport),
        )
        if review["request_sha256"] != content_hash(request):
            raise ValueError("judge request identity differs")
    return sealed_write(
        path,
        {
            **payload,
            "status": review["status"],
            "review": review,
            "request_sha256": content_hash(request),
            "reused": False,
            "candidate_sha256": model_hash(candidate),
            "numerical_scores_visible": False,
            "selection_uses_critic": False,
        },
    )


def cost(root: Path) -> dict:
    """Physical HTTP starts include interrupted attempts lacking usage."""
    requests = tokens = missing = 0
    for start in (root / "judge_cache").glob("*/wire/call-*/started.json"):
        requests += 1
        path = start.parent / "result.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        response = record.get("response")
        usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
        total = usage.get("total_tokens")
        if type(total) is int and total >= 0:
            tokens += total
        else:
            missing += 1
    return {
        "physical_requests_started": requests,
        "observed_total_tokens": tokens,
        "usage_missing_events": missing,
        "usage_complete": missing == 0,
    }


def review_final(root: Path, plan: dict, task: dict, key_supplier, *, performer=None):
    """Review an equation-changing final prune; reuse unchanged symbolic evidence."""
    from autoformalism.rebuttal import fresh_shared
    from autoformalism.schemas.public_fitting import PublicFitRequest
    from autoformalism.search import scientific_verification

    checked_authorization(root, plan)
    directory = root / "pruning" / task["task_id"]
    path = directory / "critic.json"
    if path.exists():
        return sealed_read(path)
    row = fresh_shared.final_row(root, plan, task)
    with scientific_verification.scope(task["scientific_verifier"]):
        pruned = fresh_shared._checked(directory, plan, row)
    selected = row["parent"]
    if not selected:
        raise ValueError("no final executable model to review")
    selection = pruned.get("selection") or {}
    request_data = (
        pruned["choice"]["request"]
        if selection.get("selected") == "pruned"
        else selected["request"]
    )
    model, _, _ = public._lower(PublicFitRequest.model_validate(request_data))
    candidate = model.validated.candidate
    old = incumbent_review(root, task, selected)
    payload = {
        "identity": plan["artifact_sha256"],
        "task": task,
        "pruning_sha256": pruned["artifact_sha256"],
        "used_for_selection": False,
        "candidate_sha256": model_hash(candidate),
    }
    if model_hash(candidate) == old["candidate_sha256"]:
        return sealed_write(
            path,
            {
                **{k: v for k, v in old.items() if k != "artifact_sha256"},
                **payload,
                "reused": True,
            },
        )
    request = judge.review_request(
        CandidateModel.model_validate(selected["bundle"]["candidate"]),
        candidate,
        model.validated.context,
        (root / "public/phase_b_v1" / task["cell"] / "proposer_prompt.txt").read_text(),
        task["seed"],
        REVISION,
        sign_policy=SIGN_POLICY,
    )
    cache = root / "judge_cache" / content_hash(request)
    with public._lock(cache):
        sealed_write(cache / "request.json", {"request": request})
        transport = jetstream.JetstreamTransport(cache / "wire", key_supplier)
        review = (performer or judge.perform_review)(
            request,
            cache,
            jetstream.BASE_URL,
            client_factory=lambda config: jetstream.client(config, transport),
        )
        if review["request_sha256"] != content_hash(request):
            raise ValueError("judge request identity differs")
    return sealed_write(
        path,
        {
            **payload,
            "status": review["status"],
            "review": review,
            "request_sha256": content_hash(request),
            "reused": False,
        },
    )
