"""Frozen graph audit plus bounded, blinded equation reviews of saved models."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import signal
import time
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.llm.review_revision import RevisionClient
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
    visible_response,
)
from autoformalism.rebuttal.mechanism_audit import (
    PROTOCOL,
    REVIEW_PROMPT,
    Review,
    check_review,
    consensus,
    equation_packet,
    structural_result,
)
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash

REPO = Path(__file__).resolve().parents[3]


def source_identity() -> dict:
    """Pin code and rubric; tolerate moving a byte-identical experiment checkout."""
    return {
        "package": runtime_source_hash(),
        "cli": hashlib.sha256(
            (REPO / "scripts/mechanism_audit.py").read_bytes()
        ).hexdigest(),
        "review_prompt": content_hash(REVIEW_PROMPT),
        "review_schema": content_hash(Review.model_json_schema()),
    }


def _public_prompt(root: Path, benchmark: str) -> str:
    paths = [
        root / "phase_b_v1" / benchmark / "proposer_prompt.txt",
        root / benchmark / "proposer_prompt.txt",
    ]
    available = [p for p in paths if p.is_file()]
    if len(available) != 1:
        raise ValueError(f"require exactly one public prompt for {benchmark}: {paths}")
    return available[0].read_text()


def prepare(
    bundles: list[Path],
    public_root: Path,
    root: Path,
    *,
    model_revision: str,
    model: str = "openai/gpt-oss-120b",
    cells: tuple[str, ...] | None = None,
) -> dict:
    """Create an immutable snapshot and CPU graph results; never simulate or fit."""
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("supply the exact cached judge model revision (40 hex)")
    if model not in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
        raise ValueError("unsupported prespecified local judge model")
    config = json.loads((REPO / "configs/review_deadline_v1.json").read_text())
    cells = cells or tuple(config["public_cells"])
    selected, hashes, seen = [], [], set()
    for bundle_path in bundles:
        bundle = sealed_read(bundle_path)
        if bundle["protocol"] != BUNDLE_PROTOCOL:
            raise ValueError("unknown model bundle protocol")
        hashes.append(bundle["artifact_sha256"])
        for source in bundle["rows"]:
            if source["benchmark_id"] not in cells:
                continue
            key = (
                source["method"],
                source["benchmark_id"],
                source["tier"],
                source["repetition"],
                source.get("round"),
            )
            if key in seen:
                raise ValueError(
                    f"duplicate source identity (no score selection): {key}"
                )
            seen.add(key)
            row = {**source, "index": len(selected)}
            row["row_sha256"] = content_hash(source)
            if source["status"] == "ready":
                try:
                    prompt = _public_prompt(public_root, source["benchmark_id"])
                    digest = hashlib.sha256(prompt.encode()).hexdigest()
                    if digest != source["public_prompt_sha256"]:
                        raise ValueError("public prompt differs from model source")
                    candidate = CandidateModel.model_validate(source["candidate"])
                    packet = equation_packet(
                        candidate,
                        ValidationContext.model_validate(source["context"]),
                        source["parameters"],
                        source["initials"],
                        prompt,
                        source["semantics"],
                    )
                    spec = MechanismEvaluationSpec.model_validate_json(
                        (
                            REPO
                            / "configs/mechanism_eval/phase_b_v1/specs"
                            / f"{source['benchmark_id']}.json"
                        ).read_text()
                    )
                    graph = (
                        structural_result(candidate, spec)
                        if spec.public_prompt_sha256 == digest
                        else {
                            "status": "spec_prompt_mismatch",
                            "graph_mechanism_compliance": None,
                            "message": "graph spec belongs to another public prompt",
                        }
                    )
                    row.update(packet=packet, graph=graph)
                    row["packet_sha256"] = content_hash(packet)
                except (
                    ValueError,
                    OSError,
                    KeyError,
                    TypeError,
                    ModelValidationError,
                ) as exc:
                    row.update(status="audit_input_failed", error=str(exc))
            selected.append(row)
    if not selected:
        raise ValueError("no source rows match the six-cell roster")
    settings = StagedModelSettings(
        model=model,
        model_revision=model_revision,
        temperature=0.2,
        reasoning_effort="low",
        max_output_tokens=8192,
        timeout_seconds=300,
        maximum_requests=3,
        attempts_per_step=3,
    )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "source_identity": source_identity(),
                "bundle_hashes": hashes,
                "cells": list(cells),
                "rows": selected,
                "settings": settings.model_dump(mode="json"),
                "review_seeds": [173, 941],
                "review_prompt": REVIEW_PROMPT,
                "maximum_provider_calls": 6
                * sum(r["status"] == "ready" for r in selected),
                "serving_image_sha256": config["serving_image_sha256"],
                "tensor_parallel_size": 2 if model.endswith("120b") else 1,
                "test_data_opened": False,
                "private_reference_opened": False,
                "parameter_refit_applied": False,
                "behavioral_tests_performed": False,
                "rubric_calibrated": False,
            },
        )
        for row in selected:
            if "packet" in row:
                sealed_write(
                    root / "packets" / f"{row['index']:04d}.json",
                    {
                        "packet_sha256": row["packet_sha256"],
                        "packet": row["packet"],
                    },
                )
        report(root)
        return plan


def _verify(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["source_identity"] != source_identity():
        raise ValueError("audit code/rubric changed; use the pinned checkout")
    return plan


def _review_one(root, plan, row, slot, base_url, can_start, transport=None):
    directory = root / "reviews" / f"{row['index']:04d}" / str(slot)
    directory.mkdir(parents=True, exist_ok=True)
    namespace = content_hash([plan["artifact_sha256"], row["row_sha256"], slot])
    with (directory / "task.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "result.json"
        if path.exists():
            saved = sealed_read(path)
            if saved["namespace"] != namespace:
                raise ValueError("review checkpoint identity differs")
            return
        client = RevisionClient(
            directory=directory / "calls",
            namespace=namespace,
            settings=StagedModelSettings.model_validate(plan["settings"]),
            seed=plan["review_seeds"][slot],
            base_url=base_url,
            can_start=can_start,
            **({"transport": transport} if transport else {}),
        )
        failures, review = [], None
        for attempt in range(3):
            user = json.dumps(
                {
                    "packet": row["packet"],
                    "format_feedback": failures[-1:] if failures else [],
                },
                sort_keys=True,
            )
            record = client.call(
                system=plan["review_prompt"],
                user=user,
                response_model=Review,
                step="public_equation_review",
                attempt=attempt,
            )
            try:
                review = check_review(
                    Review.model_validate(visible_response(record)), row["packet"]
                )
                break
            except ValueError as exc:
                failures.append(str(exc))
        sealed_write(
            path,
            {
                "namespace": namespace,
                "packet_sha256": row["packet_sha256"],
                "status": "reviewed" if review else "review_failed",
                "review": review.model_dump() if review else None,
                "failures": failures,
                "accounting": {
                    "physical_requests": len(client.records),
                    "observed_tokens": sum(
                        r.get("observed_total_tokens") or 0 for r in client.records
                    ),
                    "unknown_usage_requests": sum(
                        r.get("observed_total_tokens") is None for r in client.records
                    ),
                },
            },
        )


def run(root: Path, base_url: str, *, wall_seconds: float = 18000, transport=None):
    """Two seed reviews per model, at most three physical calls per seed."""
    plan = _verify(root)
    deadline = time.monotonic() + wall_seconds
    draining = False

    def drain(*_):
        nonlocal draining
        draining = True

    old = signal.signal(signal.SIGTERM, drain)
    try:
        can_start = lambda: (  # noqa: E731
            not draining
            and time.monotonic() + plan["settings"]["timeout_seconds"] + 15 < deadline
        )
        for row in plan["rows"]:
            if row["status"] != "ready":
                continue
            for slot in range(2):
                try:
                    _review_one(root, plan, row, slot, base_url, can_start, transport)
                except DeferredCall:
                    return report(root)
            report(root)
        return report(root)
    finally:
        signal.signal(signal.SIGTERM, old)


def report(root: Path) -> dict:
    """Show all requested rows, review uncertainty and independent graph results."""
    plan = _verify(root)
    rows = []
    totals = Counter()
    for source in plan["rows"]:
        row = {
            k: source.get(k)
            for k in (
                "index",
                "method",
                "benchmark_id",
                "tier",
                "repetition",
                "round",
                "status",
                "error",
                "semantics",
                "validation_nmse",
                "provenance",
            )
        }
        row["graph"] = source.get("graph")
        if source["status"] == "ready":
            reviews = []
            for slot in range(2):
                path = (
                    root
                    / "reviews"
                    / f"{source['index']:04d}"
                    / str(slot)
                    / "result.json"
                )
                saved = sealed_read(path) if path.exists() else None
                if saved and (
                    saved["packet_sha256"] != source["packet_sha256"]
                    or saved["namespace"]
                    != content_hash(
                        [plan["artifact_sha256"], source["row_sha256"], slot]
                    )
                ):
                    raise ValueError("review belongs to a different model or plan")
                # Charge calls even if the worker could not publish a terminal
                # review. Unknown usage is not zero usage.
                records = []
                for call in sorted((path.parent / "calls").glob("*.json")):
                    record = json.loads(call.read_text())
                    expected_namespace = content_hash(
                        [plan["artifact_sha256"], source["row_sha256"], slot]
                    )
                    if (
                        record["request"]["namespace"] != expected_namespace
                        or content_hash(record["request"]) != call.stem
                        or record["request_hash"] != call.stem
                    ):
                        raise ValueError("review request cache identity differs")
                    records.append(record)
                if len(records) > 3:
                    raise ValueError("review slot exceeded its physical call budget")
                totals.update(
                    {
                        "physical_requests": len(records),
                        "observed_tokens": sum(
                            r.get("observed_total_tokens") or 0 for r in records
                        ),
                        "unknown_usage_requests": sum(
                            r.get("observed_total_tokens") is None for r in records
                        ),
                    }
                )
                reviews.append(
                    Review.model_validate(saved["review"])
                    if saved and saved["review"]
                    else None
                )
            row["equation_review"] = consensus(source["packet"], reviews)
        rows.append(row)
    groups = []
    for method, benchmark in sorted({(r["method"], r["benchmark_id"]) for r in rows}):
        group = [
            r for r in rows if (r["method"], r["benchmark_id"]) == (method, benchmark)
        ]
        counts = Counter()
        for row in group:
            counts.update(
                row.get("equation_review", {})
                .get("counts", {})
                .get("task_mechanism", {})
            )
        groups.append(
            {
                "method": method,
                "benchmark_id": benchmark,
                "requested_models": len(group),
                "available_models": sum(r["status"] == "ready" for r in group),
                "task_requirements": {
                    k: counts[k] for k in ("total", "pass", "fail", "unresolved")
                },
                "supported_fraction_available_models": counts["pass"] / counts["total"]
                if counts["total"]
                else None,
            }
        )
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "judge": {
            "model": plan["settings"]["model"],
            "revision": plan["settings"]["model_revision"],
            "seeds": plan["review_seeds"],
        },
        "requested_models": len(rows),
        "input_status_counts": dict(Counter(r["status"] for r in rows)),
        "review_status_counts": dict(
            Counter(
                r.get("equation_review", {}).get("status", "source_unavailable")
                for r in rows
            )
        ),
        "accounting": dict(totals),
        "groups": groups,
        "rows": rows,
        "behavioral_evidence": "not_assessed",
        "scientific_correctness_certified": False,
        "test_data_opened": False,
        "rubric_calibrated": False,
    }
    atomic_json(root / "summary.json", result)
    lines = [
        "# Public mechanism audit",
        "",
        "Graph compliance is syntactic. Equation reviews are advisory, "
        "not scientific certification.",
        "Two seeds of the same judge measure agreement, "
        "not independent expert validation.",
        "No new behavioral tests, fits, or test/private data. "
        "NMSE is the saved validation output rollout.",
        "Unavailable models remain in model counts; "
        "unresolved requirements remain in rubric denominators.",
        "",
        "| Method | Benchmark | Seed | Source | Graph | "
        "Equation pass/fail/unresolved | Validation NMSE |",
        "| --- | --- | ---: | --- | ---: | --- | ---: |",
    ]
    for row in rows:
        c = row.get("equation_review", {}).get("counts", {}).get("task_mechanism", {})
        graph = (row["graph"] or {}).get("graph_mechanism_compliance")
        count = "/".join(str(c.get(k, "—")) for k in ("pass", "fail", "unresolved"))
        lines.append(
            f"| {row['method']} | {row['benchmark_id']} | {row['repetition']} | "
            f"{row['status']} | {graph if graph is not None else '—'} | {count} | "
            f"{row['validation_nmse'] if row['validation_nmse'] is not None else '—'} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return result
