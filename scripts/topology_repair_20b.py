"""Five independent, single-edit GPT-OSS-20B draws on one frozen topology witness."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import StagedModelSettings, atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import topology_repair_trial as trial
from autoformalism.staged_topology import content_hash
from scripts import t2_breadth as pilot
from scripts.construction_functions import file_hash

PROTOCOL = pilot.TOPOLOGY_20B
SEEDS = tuple(range(100, 105))
ENTRYPOINTS = (
    *pilot.ENTRYPOINTS,
    "scripts/topology_repair_20b.py",
    "scripts/hpc/start_topology_repair_20b_aces.sh",
)
SETTINGS = StagedModelSettings(
    model_revision=pilot.MODELS["20b"]["revision"],
    max_output_tokens=4096,
    maximum_requests=1,
    attempts_per_step=1,
    maximum_total_tokens=32768,
)


def _plan(case: dict) -> dict:
    trial.verify_case(case)
    system, user = trial.messages(case)
    return {
        "protocol": PROTOCOL,
        "case_sha256": case["case_sha256"],
        "public_packet_sha256": content_hash(json.loads(user)),
        "messages_sha256": content_hash([system, user]),
        "response_schema_sha256": content_hash(trial.RepairReply.model_json_schema()),
        "seeds": list(SEEDS),
        "maximum_generation_requests": len(SEEDS),
        "settings": SETTINGS.model_dump(mode="json"),
        "runtime_source": runtime_source_hash(),
        "entrypoints": {p: file_hash(pilot.REPO / p) for p in ENTRYPOINTS},
        "test_data_opened": False,
        "optimizer_calls": 0,
        "automatic_followup": False,
    }


def prepare(root: Path, source: Path) -> dict:
    """Freeze one case, the exact shared evidence and independent sampling plan."""
    case = json.loads(source.read_text())
    plan = _plan(case)
    with _lock(root / "prepare-lock"):
        sealed_write(root / "case.json", {"case": case})
        return sealed_write(root / "plan.json", plan)


def load(root: Path) -> tuple[dict, dict]:
    case = sealed_read(root / "case.json")["case"]
    plan = sealed_read(root / "plan.json")
    expected = _plan(case)
    if plan["artifact_sha256"] != content_hash(expected):
        raise ValueError("frozen case, prompt, settings or runtime differs")
    return plan, case


def _row(root: Path, plan: dict, seed: int) -> dict:
    """Check saved scoring and its underlying generation receipt before reporting."""
    row = sealed_read(root / "results" / f"seed-{seed}.json")
    slot = root / "steps" / f"seed-{seed}"
    reply = sealed_read(slot / "reply.json")
    if (
        row["plan_sha256"] != plan["artifact_sha256"]
        or row["reply_sha256"] != reply["artifact_sha256"]
        or row["seed"] != seed
    ):
        raise ValueError(f"changed checkpoint for seed {seed}")
    if (
        "request_hash" in reply
        and file_hash(slot / "calls" / (reply["request_hash"] + ".json"))
        != reply["physical_receipt_sha256"]
    ):
        raise ValueError(f"changed provider record for seed {seed}")
    return row


def status(root: Path) -> dict:
    """Report scientific and operational outcomes, including partial runs."""
    plan, _ = load(root)
    rows = [
        _row(root, plan, seed)
        for seed in SEEDS
        if (root / "results" / f"seed-{seed}.json").exists()
    ]
    records = [
        json.loads(f.read_text()) for f in sorted(root.glob("steps/*/calls/*.json"))
    ]
    return {
        "status": "complete" if len(rows) == len(SEEDS) else "incomplete",
        "expected": len(SEEDS),
        "recorded": len(rows),
        "plan_sha256": plan["artifact_sha256"],
        "case_sha256": plan["case_sha256"],
        "public_packet_sha256": plan["public_packet_sha256"],
        "status_counts": dict(Counter(r["assessment"]["status"] for r in rows)),
        "distinct_witness_breaking_edits": len(
            {
                r["assessment"]["edit_sha256"]
                for r in rows
                if r["assessment"]["status"] == "distinguishes"
            }
        ),
        "physical_generation_requests": len(records),
        "provider_status_counts": dict(Counter(r["status"] for r in records)),
        "observed_tokens": sum(r.get("observed_total_tokens") or 0 for r in records),
        "unknown_usage": sum(r.get("observed_total_tokens") is None for r in records),
        "optimizer_calls": 0,
        "rows": rows,
    }


def run(root: Path, endpoint: str, *, transport=None, token_transport=None) -> dict:
    """Each seed sees the same original case, with no findings from earlier draws."""
    with _lock(root / "worker-lock"):
        plan, case = load(root)
        system, user = trial.messages(case)
        for seed in SEEDS:
            slot = f"seed-{seed}"
            if (root / "results" / f"{slot}.json").exists():
                _row(root, plan, seed)
                continue
            response = pilot._ask(
                root,
                plan,
                slot,
                seed,
                system,
                user,
                trial.RepairReply,
                endpoint,
                transport,
                token_transport,
                {"physical_requests": 0, "slots": {}},
                len(SEEDS),
            )
            assessment = {"status": response["status"]}
            if response["status"] == "visible":
                try:
                    edit = trial.RepairReply.model_validate(response["value"])
                except ValidationError as exc:
                    assessment = {"status": "invalid", "error": str(exc)}
                else:
                    assessment = trial.assess_reply(case, edit)
            sealed_write(
                root / "results" / f"{slot}.json",
                {
                    "seed": seed,
                    "plan_sha256": plan["artifact_sha256"],
                    "reply_sha256": response["artifact_sha256"],
                    "assessment": assessment,
                },
            )
    return status(root)


def inspect(root: Path) -> dict:
    """Write the summary and portable archive of prompts, responses and assessments."""
    with _lock(root / "worker-lock"), _lock(root / "inspection-lock"):
        summary = status(root)
        atomic_json(root / "summary.json", summary)
        archive = root / "inspection.tar.gz"
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            for name in (
                "plan.json",
                "case.json",
                "summary.json",
                "steps",
                "results",
                "execution.json",
                "submissions",
                "runtime",
                "logs",
            ):
                path = root / name
                if path.exists() and not path.is_symlink():
                    tar.add(path, arcname=name)
        temporary.replace(archive)
    return {
        "archive": str(archive),
        **{k: v for k, v in summary.items() if k != "rows"},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "preview", "run", "status", "inspect"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--endpoint")
    p.add_argument("--runtime-record", type=Path)
    args = p.parse_args()
    if args.action == "prepare":
        if not args.source:
            p.error("prepare needs --source")
        result = prepare(args.root, args.source)
    elif args.action == "run":
        if not args.endpoint or not args.runtime_record:
            p.error("run needs a bound ACES runtime and endpoint")
        from scripts.submit_t2_breadth import bind

        bind(args.root, args.runtime_record, args.endpoint)
        result = run(args.root, args.endpoint)
    elif args.action == "preview":
        _, case = load(args.root)
        system, user = trial.messages(case)
        result = {
            "system": system,
            "user": user,
            "schema": trial.RepairReply.model_json_schema(),
        }
    elif args.action == "inspect":
        result = inspect(args.root)
    else:
        result = status(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
