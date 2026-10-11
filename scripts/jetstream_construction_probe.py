#!/usr/bin/env python3
"""Four independent, cached construction requests against a local GPU VM server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.response_revision import diagnostic_transport
from autoformalism.llm.staged_topology import (
    atomic_json,
    strict_provider_schema,
    visible_response,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search.construction_prompts import response_model
from autoformalism.staged_topology import content_hash

PROTOCOL = "jetstream-construction-compatibility-1"
STAGES = ("variables", "shared_laws", "equations", "repair")
MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
IMAGE = (
    "vllm/vllm-openai@sha256:"
    "0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
)
BASE_URL = "http://127.0.0.1:8000"


def source_identity() -> str:
    """Bind the shipped runtime and probe tools without depending on Git metadata."""
    repo = Path(__file__).resolve().parents[1]
    files = sorted((repo / "src/autoformalism").rglob("*.py"))
    files += [Path(__file__), repo / "scripts/jetstream_server_record.py"]
    return content_hash(
        {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        }
    )


def validate_request(stage: str, body: dict) -> None:
    """Refuse a changed model, sampling policy, or stage response schema."""
    expected = {
        "model": MODEL,
        "reasoning_effort": "low",
        "temperature": 0.2,
        "max_tokens": 8192,
        "stream": False,
    }
    if any(body.get(k) != v for k, v in expected.items()):
        raise ValueError("request settings differ from ACES construction")
    if set(body) != {*expected, "seed", "messages", "response_format"}:
        raise ValueError("unexpected request fields")
    if type(body["seed"]) is not int or not 0 <= body["seed"] < 2**31:
        raise ValueError("invalid seed")
    model = response_model(stage)
    if body["response_format"] != {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": strict_provider_schema(model.model_json_schema()),
        },
    }:
        raise ValueError("stage schema differs")
    messages = body["messages"]
    if len(messages) != 2 or [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("expected complete system and user messages")
    packet = json.loads(messages[1]["content"])
    if packet.get("stage") != stage or "model_view" not in packet:
        raise ValueError("expected focused construction request")


def prepare(review: Path, output: Path) -> dict:
    """Select the largest projected request per stage, without response selection."""
    rows = json.loads(review.read_text())
    selected = []
    for stage in STAGES:
        choices = [r for r in rows if r["stage"] == stage]
        row = max(choices, key=lambda r: len(json.dumps(r["after"]).encode()))
        validate_request(stage, row["after"])
        selected.append(
            {
                "stage": stage,
                "task": row["task"],
                "event": row["event"],
                "historical_request_sha256": row["source_request_sha256"],
                "body": row["after"],
            }
        )
    return sealed_write(
        output,
        {
            "protocol": PROTOCOL,
            "model_revision": REVISION,
            "requests": selected,
            "selection": "largest_projected_request_bytes_per_stage",
            "source_review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
            "interpretation": (
                "Independent displayed predecessors; not a new model chain."
            ),
            "maximum_generation_requests": 4,
            "test_data_opened": False,
        },
    )


def validate_server(server: dict) -> None:
    """Bind the probe to the pinned container and observed CUDA kernel check."""
    expected = {
        "image": IMAGE,
        "model": MODEL,
        "revision": REVISION,
        "context_tokens": 32768,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
    }
    if any(server.get(k) != v for k, v in expected.items()):
        raise ValueError("server settings differ")
    cuda = server["cuda"]
    if (
        cuda.get("vllm") != "0.27.1"
        or cuda.get("kernel") != "passed"
        or cuda.get("device_count") != 1
        or "H100" not in cuda.get("gpu", "")
        or cuda.get("memory_bytes", 0) < 75 * 1024**3
    ):
        raise ValueError("requires the verified single H100 CUDA environment")
    if len(server.get("weights_manifest_sha256", "")) != 64:
        raise ValueError("missing verified model weights manifest")


def cached_post(path: Path, url: str, body: dict, transport) -> dict:
    """Checkpoint before HTTP; never silently resend an interrupted request."""
    identity = {"url": url, "body": body}
    if path.exists():
        saved = sealed_read(path)
        if saved["request"] != identity:
            raise ValueError("cached request differs")
        return saved
    sealed_write(path, {"request": identity, "status": "inflight"})
    started = time.monotonic()
    record = {"request": identity}
    try:
        record.update(status="responded", raw_response=transport(url, body, 180))
    except (OSError, ValueError, TimeoutError) as exc:
        record.update(status="provider_failure", error=str(exc)[:8192])
    record["seconds"] = time.monotonic() - started
    # A response replaces only our own inflight checkpoint, under the run lock.
    record["artifact_sha256"] = content_hash(record)
    atomic_json(path, record)
    return record


def assess(record: dict, stage: str) -> dict:
    """Check final content/schema, not scientific quality or topology acceptance."""
    result = {"status": record["status"], "seconds": record.get("seconds")}
    try:
        value = visible_response(record)
        model = response_model(stage)
        if not isinstance(value, dict) or set(value) != set(model.model_fields):
            raise ValueError("final JSON must include every requested field")
        model.model_validate_json(json.dumps(value), strict=True)
        result["status"] = "schema_passed"
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        result.update(status="response_unusable", error=str(exc)[:4096])
    raw = record.get("raw_response", {})
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    result["observed_total_tokens"] = (
        total if type(total) is int and total >= 0 else None
    )
    return result


def run(requests: Path, server_path: Path, root: Path, *, transport=None) -> dict:
    """Run at most four fixed physical generations, preserving all outcomes."""
    transport = transport or diagnostic_transport
    inputs = sealed_read(requests)
    if inputs["protocol"] != PROTOCOL or inputs["model_revision"] != REVISION:
        raise ValueError("probe input protocol/revision differs")
    if [r["stage"] for r in inputs["requests"]] != list(STAGES):
        raise ValueError("expected exactly four staged requests")
    for row in inputs["requests"]:
        validate_request(row["stage"], row["body"])
    server = json.loads(server_path.read_text())
    validate_server(server)
    with _lock(root):
        sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "input_sha256": inputs["artifact_sha256"],
                "server": server,
                "source_identity": source_identity(),
                "maximum_generation_requests": 4,
            },
        )
        rows = []
        for index, row in enumerate(inputs["requests"]):
            stage, body = row["stage"], row["body"]
            token_body = {
                "model": MODEL,
                "messages": body["messages"],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            }
            token = cached_post(
                root / f"tokenize-{index}.json",
                BASE_URL + "/tokenize",
                token_body,
                transport,
            )
            reply = token.get("raw_response", {})
            if not isinstance(reply, dict):
                reply = {}
            count = reply.get("count")
            if (
                token["status"] != "responded"
                or type(count) is not int
                or not 0 < count <= 24000
                or reply.get("max_model_len") != 32768
            ):
                rows.append(
                    {
                        "stage": stage,
                        "status": "token_preflight_failed",
                        "input_tokens": count,
                        "generation_started": False,
                    }
                )
                continue
            record = cached_post(
                root / f"call-{index}.json",
                BASE_URL + "/v1/chat/completions",
                body,
                transport,
            )
            rows.append(
                {
                    "stage": stage,
                    "task": row["task"],
                    "input_tokens": count,
                    "generation_started": True,
                    **assess(record, stage),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        started = [r for r in rows if r["generation_started"]]
        missing = sum(r["observed_total_tokens"] is None for r in started)
        passed = all(r["status"] == "schema_passed" for r in rows) and not missing
        summary = {
            "protocol": PROTOCOL,
            "compatibility_passed": passed,
            "status": "passed" if passed else "needs_inspection",
            "rows": rows,
            "physical_generation_requests_started": len(started),
            "observed_total_tokens": sum(
                r["observed_total_tokens"] or 0 for r in started
            ),
            "usage_missing_events": missing,
            "usage_complete": not missing,
            "optimizer_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
            "scientific_correctness_assessed": False,
            "interpretation": (
                "Serving/schema compatibility, not accuracy or ACES equivalence."
            ),
        }
        atomic_json(root / "summary.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--review", type=Path)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.review is None:
            parser.error("prepare requires --review")
        value = prepare(args.review, args.requests)
        print(json.dumps({"identity": value["artifact_sha256"], "requests": 4}))
    else:
        if args.server is None or args.root is None:
            parser.error("run requires --server and --root")
        result = run(args.requests, args.server, args.root)
        print(json.dumps(result, indent=2))
        if not result["compatibility_passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
