"""Numerically blind adapter around the established paired scientific judge.

The calibrated prompts, atomic evidence stage, orientation consensus and scoring
are reused unchanged. Only the scheduling and extraction of advisory concerns is
new. A self-pair supplies the initial absolute review; later pairs compare the
previous model with its proposed replacement. This is a search experiment, not a
new validation of the judge on unpruned candidates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from autoformalism.execution import _prediction_protocol_prompt, _symbol_contract
from autoformalism.expressions import ValidationContext
from autoformalism.judging import HybridScoringConfig, extract_public_requirements
from autoformalism.judging.prompts import (
    ATOMIC_EVIDENCE_PROMPT,
    ATOMIC_STAGE_TWO_NOTE,
    HYBRID_JUDGE_PROMPT,
)
from autoformalism.llm import LLMConfig, create_llm_client
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.repair_evidence import Finding, model_hash
from autoformalism.schemas import CandidateModel
from autoformalism.search.hybrid_pair import PairedHybridJudge
from autoformalism.staged_topology import content_hash


def judge_protocol() -> dict:
    """Freeze existing scientific protocol independently from repair prompts."""
    return {
        "model": "openai/gpt-oss-120b",
        "reasoning_effort": "low",
        "temperature": 0.2,
        "max_output_tokens": 6144,
        "max_attempts": 10,
        "distinct_seed_attempts": 2,
        "scoring": asdict(
            HybridScoringConfig(
                comparative_indeterminate_policy="neutral_fixed_denominator"
            )
        ),
        "prompts": {
            "hybrid": HYBRID_JUDGE_PROMPT,
            "atomic": ATOMIC_EVIDENCE_PROMPT,
            "second_stage_note": ATOMIC_STAGE_TWO_NOTE,
        },
    }


def review_request(
    parent: CandidateModel,
    candidate: CandidateModel,
    context: ValidationContext,
    public_prompt: str,
    seed: int,
    revision: str,
) -> dict:
    """Closed allowlist: no fits, trajectories, certificates or repair history."""
    return {
        "parent": parent.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "public_prompt": public_prompt,
        "seed": seed,
        "model_revision": revision,
        "protocol": judge_protocol(),
    }


def perform_review(request: dict, directory: Path, base_url: str) -> dict:
    """Cache both orientations and report advisory scientific concerns only."""
    identity = content_hash(request)
    path = directory / "review.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["request_sha256"] != identity:
            raise ValueError("scientific review identity differs")
        return saved
    marker = directory / "review_started.json"
    if marker.exists():
        saved = {
            "request_sha256": identity,
            "status": "interrupted",
            "findings": [],
            "error": "review interrupted; partial calls retained without budget reset",
            "cost": review_cost(directory),
        }
        atomic_json(path, saved)
        return saved
    atomic_json(marker, {"request_sha256": identity})
    context = ValidationContext.model_validate(request["context"])
    parent, candidate = (
        CandidateModel.model_validate(request[k]) for k in ("parent", "candidate")
    )
    protocol = request["protocol"]
    if protocol != judge_protocol():
        raise ValueError("frozen scientific judge protocol differs")
    model = protocol["model"]
    seeds = [12000 + 2 * request["seed"] + i for i in range(2)]
    clients = tuple(
        (
            seed,
            create_llm_client(
                LLMConfig(
                    provider="vllm",
                    model=model,
                    cache_directory=directory / f"seed_{seed}",
                    log_path=directory / "events.jsonl",
                    max_attempts=10,
                    initial_backoff_seconds=1,
                    max_backoff_seconds=30,
                    jitter_fraction=0,
                    vllm_base_url=base_url,
                    vllm_reasoning_effort="low",
                    vllm_temperature=0.2,
                    vllm_seed=seed,
                    timeout_seconds=900,
                    max_output_tokens=6144,
                )
            ),
        )
        for seed in seeds
    )
    common = (
        f"Configured judge model: vllm:{model}\n\nPublic scientific task:\n"
        f"{request['public_prompt']}\n\n"
    )
    judge = PairedHybridJudge(
        seeded_clients=clients,
        requirements=extract_public_requirements(request["public_prompt"]),
        task_inputs=tuple(context.external_inputs),
        system_prompt=common
        + _prediction_protocol_prompt(context)
        + "\n\n"
        + _symbol_contract(context)
        + "\n\nHybrid judge protocol:\n"
        + HYBRID_JUDGE_PROMPT
        + ATOMIC_STAGE_TWO_NOTE,
        atomic_system_prompt=common
        + _symbol_contract(context)
        + "\n\nAtomic evidence protocol:\n"
        + ATOMIC_EVIDENCE_PROMPT,
        scoring=HybridScoringConfig(**protocol["scoring"]),
        identity=identity,
    )
    pair = judge.compare(parent, candidate)
    findings = []
    if pair.consensus_result:
        for item in pair.consensus_result.absolute_assessments:
            if item.candidate_b.verdict.value == "fail":
                findings.append(
                    Finding(
                        source="scientific_judge",
                        stage="prefit",
                        candidate_sha256=model_hash(candidate),
                        category="scientific_requirement",
                        code=item.criterion.value,
                        component=item.subject_id,
                        certainty="advisory",
                        observation=item.candidate_b.evidence,
                        recheck="same frozen paired scientific judge",
                        evidence={
                            "request_sha256": identity,
                            "subject_id": item.subject_id,
                        },
                    ).model_dump(mode="json")
                )
        for item in pair.consensus_result.comparative_assessments:
            if item.verdict.value == "candidate_a":
                findings.append(
                    Finding(
                        source="scientific_judge",
                        stage="prefit",
                        candidate_sha256=model_hash(candidate),
                        category="scientific_comparison",
                        code=item.criterion.value,
                        certainty="advisory",
                        observation=item.evidence,
                        recheck="same frozen paired scientific judge",
                        evidence={
                            "request_sha256": identity,
                            "interpretation": (
                                "parent preferred on this criterion; "
                                "not an absolute defect"
                            ),
                        },
                    ).model_dump(mode="json")
                )
    result = {
        "request_sha256": identity,
        "status": "reviewed" if pair.consensus_result else "indeterminate",
        "comparison": pair.model_dump(mode="json"),
        "findings": findings,
        "numerical_results_visible": False,
        "external_runtime_findings_visible": False,
        "prefit_unpruned": True,
        "protocol_validated_for_this_use": False,
        "cost": review_cost(directory),
    }
    atomic_json(path, result)
    return result


def review_cost(directory: Path) -> dict:
    """Account retries without double-counting success records' cumulative totals."""
    path = directory / "events.jsonl"
    events = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    groups = defaultdict(list)
    for event in events:
        if event.get("event") in {"llm_response", "llm_failure"} and not event.get(
            "cache_hit", False
        ):
            groups[event["request_hash"]].append(event)
    requests = tokens = missing = 0
    for group in groups.values():
        # A final success total includes all preceding failed attempts in that call.
        failed = 0
        for event in group:
            count = event.get("provider_attempts", 0)
            if event["event"] == "llm_failure":
                if event.get("attempt") == 1:
                    failed = 0
                failed += count
                requests += count
            else:
                requests += max(0, count - failed)
                failed = 0
            usage = event.get("usage") or {}
            total = usage.get("total_tokens")
            if total is None:
                missing += 1
            else:
                tokens += total
    return {
        "physical_requests": requests,
        "observed_total_tokens": tokens,
        "usage_missing_events": missing,
        "usage_complete": bool(groups) and missing == 0,
        "cache_hit_events": sum(e.get("cache_hit", False) for e in events),
    }
