#!/usr/bin/env python3
"""Fresh construction seed rosters on the verified local H100 server."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
import urllib.request
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.staged_topology import content_hash
from scripts import jetstream_construction_probe as probe

PROTOCOL = "jetstream-focused-construction-pilot-1"
PLATFORM = "jetstream2-h100x1-docker"
REPO = Path(__file__).resolve().parents[1]


def operation_identity() -> str:
    """Bind the launcher and provenance checks in addition to the core runtime."""
    names = (
        "scripts/jetstream_construction_campaign.py",
        "scripts/jetstream_construction_probe.py",
        "scripts/jetstream_server_record.py",
        "scripts/jetstream/run_construction.sh",
        "scripts/jetstream/run_readable_construction.sh",
        "scripts/jetstream/run_completion_construction.sh",
        "scripts/jetstream/run_coverage_construction.sh",
        "scripts/jetstream/run_decision_construction.sh",
        "scripts/jetstream/run_edit_feedback_construction.sh",
        "scripts/jetstream/run_ownership_feedback_construction.sh",
        "scripts/jetstream/run_protocol_clarity_construction.sh",
        "scripts/jetstream/run_evidence_feedback_construction.sh",
    )
    return content_hash(
        {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in names}
    )


def compatibility(work: Path, requests: Path) -> dict:
    """Reassess all four saved replies against the exact delivered requests."""
    inputs = sealed_read(requests)
    plan = sealed_read(work / "probe/plan.json")
    server = json.loads((work / "server.json").read_text())
    probe.validate_server(server)
    if (
        inputs["protocol"] != probe.PROTOCOL
        or plan["protocol"] != probe.PROTOCOL
        or inputs["model_revision"] != probe.REVISION
        or plan["input_sha256"] != inputs["artifact_sha256"]
        or plan["server"] != server
        or plan["maximum_generation_requests"] != 4
        or [r["stage"] for r in inputs["requests"]] != list(probe.STAGES)
    ):
        raise ValueError("compatibility inputs or server differ")
    if (
        hashlib.sha256((work / "weights.json").read_bytes()).hexdigest()
        != server["weights_manifest_sha256"]
    ):
        raise ValueError("weights manifest differs")
    tokens = 0
    for i, row in enumerate(inputs["requests"]):
        body, stage = row["body"], row["stage"]
        probe.validate_request(stage, body)
        call = sealed_read(work / f"probe/call-{i}.json")
        tokenizer = sealed_read(work / f"probe/tokenize-{i}.json")
        if call["request"] != {
            "url": probe.BASE_URL + "/v1/chat/completions",
            "body": body,
        } or tokenizer["request"] != {
            "url": probe.BASE_URL + "/tokenize",
            "body": {
                "model": probe.MODEL,
                "messages": body["messages"],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
        }:
            raise ValueError("compatibility request differs")
        count = tokenizer.get("raw_response", {}).get("count")
        if (
            tokenizer["status"] != "responded"
            or type(count) is not int
            or not 0 < count <= 24000
            or tokenizer["raw_response"].get("max_model_len") != 32768
        ):
            raise ValueError("compatibility tokenizer did not pass")
        assessment = probe.assess(call, stage)
        if (
            assessment["status"] != "schema_passed"
            or assessment["observed_total_tokens"] is None
        ):
            raise ValueError("compatibility response did not pass")
        tokens += assessment["observed_total_tokens"]
    return {
        "plan_sha256": plan["artifact_sha256"],
        "input_sha256": inputs["artifact_sha256"],
        "physical_requests": 4,
        "observed_total_tokens": tokens,
        "server": server,
    }


def validate_settings(config: dict) -> None:
    """Keep the ACES prompt, sampling and per-construction budget settings."""
    settings = config["model_settings"]
    expected = {
        "model": probe.MODEL,
        "model_revision": probe.REVISION,
        "reasoning_effort": "low",
        "temperature": 0.2,
        "max_output_tokens": 8192,
        "timeout_seconds": 180,
        "maximum_requests": 128,
        "maximum_total_tokens": 524288,
        "attempts_per_step": 3,
    }
    if config.get("bookkeeping_policy") in {
        "minimal-protocol-clarity-1",
        "minimal-mechanism-realizations-1",
        "minimal-evidence-feedback-1",
    }:
        expected.update(maximum_requests=133, maximum_total_tokens=655360)
        if config.get("repair_requests") != 10 or config.get("repair_tokens") != 262144:
            raise ValueError("protocol clarity repair allowance differs")
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("construction settings differ from the reviewed ACES pilot")
    if config["served_context_tokens"] != 32768:
        raise ValueError("construction context differs")


def prepare(
    source: Path,
    work: Path,
    requests: Path,
    root: Path,
    *,
    study: str = "focused_prompt_confirmation",
    seeds: list[int] | None = None,
    training_evidence: Path | None = None,
) -> dict:
    """Freeze a distinct campaign; preserve the source and compatibility run."""
    if study not in {
        "focused_prompt_confirmation",
        "readable_prompt_confirmation",
        "completion_feedback_confirmation",
        "public_coverage_confirmation",
        "coverage_decision_confirmation",
        "edit_feedback_confirmation",
        "ownership_feedback_confirmation",
        "protocol_clarity_confirmation",
        "mechanism_realization_confirmation",
        "evidence_feedback_confirmation",
    }:
        raise ValueError("unsupported Jetstream construction study")
    selected = list(campaign.sampling_seeds(study, seeds))
    evidence = compatibility(work, requests)
    old = sealed_read(source / "plan.json")
    validate_settings(old["config"])
    plan = campaign.freeze(
        source,
        root,
        study=study,
        seeds=selected,
        training_evidence=training_evidence,
        serving_environment={
            "platform": PLATFORM,
            "serving_image_sha256": probe.IMAGE.split("sha256:")[1],
        },
    )
    plan = campaign.verify(root)
    if len(plan["tasks"]) != 8 * len(selected) or plan["sampling_seeds"] != selected:
        raise ValueError("construction seed roster differs")
    sealed_write(
        root / "execution.json",
        {
            "protocol": PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "operation_identity": operation_identity(),
            "compatibility": evidence,
            "base_url": probe.BASE_URL,
            "functions_or_fitting": False,
            "test_data_opened": False,
            "automatic_followup": False,
        },
    )
    return plan


def verify(root: Path) -> dict:
    """Require the same source, provenance and seed roster on every resume."""
    plan = campaign.verify(root)
    execution = sealed_read(root / "execution.json")
    validate_settings(plan["config"])
    seeds = campaign.sampling_seeds(plan["study"], plan.get("sampling_seeds"))
    if (
        execution["protocol"] != PROTOCOL
        or execution["plan_sha256"] != plan["artifact_sha256"]
        or execution["operation_identity"] != operation_identity()
        or plan["study"]
        not in {
            "focused_prompt_confirmation",
            "readable_prompt_confirmation",
            "completion_feedback_confirmation",
            "public_coverage_confirmation",
            "coverage_decision_confirmation",
            "edit_feedback_confirmation",
            "ownership_feedback_confirmation",
            "protocol_clarity_confirmation",
            "mechanism_realization_confirmation",
            "evidence_feedback_confirmation",
        }
        or len(plan["tasks"]) != 8 * len(seeds)
        or plan["config"]["platform"] != PLATFORM
        or plan["config"]["serving_image_sha256"] != probe.IMAGE.split("sha256:")[1]
    ):
        raise ValueError("Jetstream execution identity differs")
    return plan


def live_server() -> None:
    """Read serving metadata on loopback only; never issue a model generation."""

    def read(route):
        with urllib.request.urlopen(probe.BASE_URL + route, timeout=10) as response:
            return json.load(response)

    if read("/version").get("version") != "0.27.1":
        raise ValueError("live vLLM version differs")
    models = read("/v1/models").get("data", [])
    if (
        len(models) != 1
        or models[0].get("id") != probe.MODEL
        or models[0].get("max_model_len") != 32768
    ):
        raise ValueError("live model or context differs")


def run(root: Path) -> dict:
    """Reuse the existing constructor and reports within one three-hour window."""
    plan = verify(root)
    live_server()
    print(json.dumps({"storage_check": campaign.check_storage(root)}), flush=True)
    deadline = time.monotonic() + plan["config"]["wall_seconds"] - 300
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for task in plan["tasks"]:
            if stopping or time.monotonic() >= deadline:
                break
            try:
                result = campaign.propose(
                    root,
                    plan,
                    task,
                    probe.BASE_URL,
                    can_start=lambda: not stopping and time.monotonic() < deadline,
                )
                print(
                    json.dumps({"task": task["task_id"], "status": result["status"]}),
                    flush=True,
                )
            except DeferredCall:
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # Even a failed worker retains the reports for its completed predecessors.
        summary = campaign.report(root, plan)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--training-evidence", type=Path)
    parser.add_argument(
        "--study",
        choices=(
            "focused_prompt_confirmation",
            "readable_prompt_confirmation",
            "completion_feedback_confirmation",
            "public_coverage_confirmation",
            "coverage_decision_confirmation",
            "edit_feedback_confirmation",
            "ownership_feedback_confirmation",
            "protocol_clarity_confirmation",
            "mechanism_realization_confirmation",
            "evidence_feedback_confirmation",
        ),
        default="focused_prompt_confirmation",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    with _lock(args.root / "campaign-lock"):
        if args.command == "run":
            if any(x is None for x in (args.work, args.source, args.requests)):
                parser.error("run requires --work, --source and --requests")
            prepare(
                args.source,
                args.work,
                args.requests,
                args.root,
                study=args.study,
                seeds=args.seeds,
                training_evidence=args.training_evidence,
            )
            result = run(args.root)
        else:
            result = campaign.report(args.root, verify(args.root))
        keys = (
            "status_counts",
            "all_tasks_terminal",
            "physical_requests",
            "observed_total_tokens",
            "unmeasured_requests",
            "test_data_opened",
        )
        print(json.dumps({k: result[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
