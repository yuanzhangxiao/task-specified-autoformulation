"""One portable paired-review compatibility pilot on managed Jetstream inference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from autoformalism.llm import jetstream
from autoformalism.llm.exceptions import LLMError
from autoformalism.rebuttal import judge_sign_delta as portable
from autoformalism.rebuttal import judge_sign_recheck as original
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

PROTOCOL = "jetstream-paired-judge-compatibility-1"


def launcher_hash() -> str:
    """Bind executable entry points in addition to the package source identity."""
    names = ("scripts/judge_jetstream.py", "scripts/smoke_judge_jetstream.py")
    return original.content_hash(
        {n: hashlib.sha256((original.REPO / n).read_bytes()).hexdigest() for n in names}
    )


def _selection(origin: dict, index: int | None) -> dict:
    """Choose without scores: largest symbolic request, or an explicit source index."""
    eligible = [i for i, row in enumerate(origin["reviews"]) if row["affected"]]
    if not eligible or (index is not None and index not in eligible):
        raise ValueError("select an affected review from the source plan")
    sizes = {
        i: len(json.dumps(origin["reviews"][i]["request"], sort_keys=True).encode())
        for i in eligible
    }
    selected = max(eligible, key=lambda i: (sizes[i], -i)) if index is None else index
    return {
        "source_index": selected,
        "rule": "largest_symbolic_request_bytes" if index is None else "explicit_index",
        "symbolic_request_bytes": sizes[selected],
        "note": "Byte size is a selection heuristic, not a model-token count.",
    }


def _plan(origin: dict, index: int | None) -> dict:
    """Derive the complete closed plan, including explicit managed-runtime limits."""
    selection = _selection(origin, index)
    row = origin["reviews"][selection["source_index"]]
    return {
        "protocol": PROTOCOL,
        "origin_plan_sha256": origin["artifact_sha256"],
        "selection": selection,
        "request": original.corrected_request(row["request"]),
        "historical_request_sha256": row["historical_request_sha256"],
        "placements": [
            p
            for p in origin["placements"]
            if p["request_sha256"] == row["historical_request_sha256"]
        ],
        "execution": {
            "transport_policy": jetstream.POLICY,
            "endpoint": jetstream.ENDPOINT,
            "model_alias": jetstream.MODEL,
            "served_model_revision": None,
            "serving_software_version": None,
            "origin_model_revision": origin["judge_revision"],
            "origin_revision_match": "unverified",
            "calibration_status": "not_established",
            "settings_forwarding": "requested_not_independently_verified",
        },
        "maximum_paired_reviews": 1,
        "maximum_physical_requests": jetstream.MAX_CALLS,
        "source_sha256": original.public._source_identity(),
        "runtime": original.public._runtime(),
        "launcher_sha256": launcher_hash(),
        **dict.fromkeys(portable.ZERO_FIELDS, 0),
        **dict.fromkeys(portable.FLAG_FIELDS, False),
    }


def freeze(source: Path, root: Path, index: int | None = None) -> dict:
    """Import verified symbolic evidence into a separate, immutable one-pair plan."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root):
        raise ValueError("source plan and pilot output must be separate")
    origin = portable.imported_plan(source)
    plan = _plan(origin, index)
    with original.public._lock(root):
        sealed_write(
            root / "origin_plan.json",
            {k: v for k, v in origin.items() if k != "artifact_sha256"},
        )
        return sealed_write(root / "plan.json", plan)


def verify(root: Path) -> dict:
    """Reject source/request/transport drift without reopening cluster paths."""
    original.history.require_open(root)
    plan = sealed_read(root / "plan.json")
    origin = portable.imported_plan(root / "origin_plan.json")
    selection = plan["selection"]
    index = selection["source_index"] if selection["rule"] == "explicit_index" else None
    if {k: v for k, v in plan.items() if k != "artifact_sha256"} != _plan(
        origin, index
    ):
        raise ValueError("frozen Jetstream pilot identity differs")
    return plan


def run(root: Path, key_supplier: Callable[[], str]) -> dict:
    """Run once; reuse completed results and retain interrupted-review accounting."""
    plan = verify(root)
    with original.public._lock(root):
        path = root / "result.json"
        if path.exists():
            result = sealed_read(path)
            if result["identity"] != plan["artifact_sha256"]:
                raise ValueError("result belongs to another Jetstream pilot")
            return result
        cache = root / "judge"
        sealed_write(cache / "request.json", {"request": plan["request"]})
        if not (cache / "review_started.json").exists():
            key = key_supplier().strip()
            if not key or any(c in key for c in "\r\n"):
                raise ValueError("a nonempty, single-line Jetstream key is required")

            def key_supplier() -> str:
                return key

        transport = jetstream.JetstreamTransport(root / "calls", key_supplier)
        try:
            review = judge.perform_review(
                plan["request"],
                cache,
                jetstream.BASE_URL,
                client_factory=lambda config: jetstream.client(config, transport),
            )
        except (OSError, TimeoutError, LLMError) as error:
            review = {
                "request_sha256": original.content_hash(plan["request"]),
                "status": "unavailable",
                "findings": [],
                "error": f"{type(error).__name__}: {error}",
                "cost": judge.review_cost(cache),
            }
        return sealed_write(
            path,
            {
                "identity": plan["artifact_sha256"],
                "status": review["status"],
                "review": review,
            },
        )


def report(root: Path) -> dict:
    """Separate API/schema compatibility from model equivalence and calibration."""
    plan = verify(root)
    result = (
        sealed_read(root / "result.json") if (root / "result.json").exists() else None
    )
    if result and result["identity"] != plan["artifact_sha256"]:
        raise ValueError("result belongs to another Jetstream pilot")
    calls = []
    for work in sorted((root / "calls").glob("call-*")):
        if not (work / "started.json").exists():
            continue
        record = (
            json.loads((work / "result.json").read_text())
            if (work / "result.json").exists()
            else {"status": "started_without_result"}
        )
        request = json.loads((work / "request.json").read_text())["body"]
        response = record.get("response")
        response = response if isinstance(response, dict) else {}
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        choice = choice if isinstance(choice, dict) else {}
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        total = usage.get("total_tokens")
        observed = isinstance(total, int) and not isinstance(total, bool) and total >= 0
        calls.append(
            {
                "call": work.name,
                "status": record["status"],
                "http_status": record.get("http_status"),
                "seconds": record.get("seconds"),
                "schema": request["response_format"]["json_schema"]["name"],
                "seed": request.get("seed"),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage or None,
                "total_tokens": total if observed else None,
                "model": response.get("model"),
                "system_fingerprint": response.get("system_fingerprint"),
            }
        )
    status = result["status"] if result else "pending"
    value = {
        "protocol": PROTOCOL,
        "identity": plan["artifact_sha256"],
        "status": status,
        "selection": plan["selection"],
        "placements": plan["placements"],
        "execution": plan["execution"],
        "paired_review_completed": status == "reviewed",
        "calibration_established": False,
        "calls": calls,
        "schema_attempt_counts": dict(Counter(c["schema"] for c in calls)),
        "cost": {
            "physical_requests_started": len(calls),
            "observed_total_tokens": sum(c["total_tokens"] or 0 for c in calls),
            "usage_missing_events": sum(c["total_tokens"] is None for c in calls),
            "usage_complete": bool(calls)
            and all(c["total_tokens"] is not None for c in calls),
            "scope": (
                "Started HTTP attempts, including failures; "
                "interrupted usage may be unknown."
            ),
        },
        "review_availability": judge.review_status(
            result["review"] if result else None
        ),
        **dict.fromkeys(portable.ZERO_FIELDS, 0),
        **dict.fromkeys(portable.FLAG_FIELDS, False),
    }
    original.public._write(root / "summary.json", value)
    lines = [
        "# Jetstream paired-judge compatibility",
        "",
        "One symbolic pair; existing stages, orientations, seeds and settings.",
        "Managed revision and setting forwarding are unverified. No calibration claim.",
        "No fitting, model promotion, automatic follow-up or test-data access.",
        "",
        f"Status: {status}. HTTP attempts started: {len(calls)}. "
        f"Observed tokens: {value['cost']['observed_total_tokens']}.",
        "",
        "| Call | Schema | HTTP | Finish | Seconds | Tokens |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for c in calls:
        lines.append(
            f"| {c['call']} | {c['schema']} | {c['http_status']} | "
            f"{c['finish_reason']} | {c['seconds']} | {c['total_tokens']} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return value
