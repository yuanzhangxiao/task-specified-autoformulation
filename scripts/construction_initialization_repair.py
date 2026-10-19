#!/usr/bin/env python3
"""Repair deterministic initializer conflicts over fixed saved models and seeds."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import signal
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
    strict_provider_schema,
    visible_response,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.search.construction_graph_repair import (
    GraphDependencyReply,
    construct_graph_dependencies,
    dependency_construction_identity,
    dependency_request,
    system_prompt,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    validate_handoff,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import construction_initialization_graph as legacy  # noqa: E402

PROTOCOL = "saved-construction-initialization-repair-1"
FEEDBACK_PROTOCOLS = {
    "legacy-1": PROTOCOL,
    "scope-aware-2": "saved-construction-initialization-repair-2",
}
PROPOSAL_SEEDS = (0, 1, 2, 3, 4)
BUDGETS = {
    "maximum_calls_per_episode": 3,
    "maximum_charged_tokens_per_episode": 393216,
    "maximum_output_tokens": 8192,
}
ENTRYPOINTS = (
    "scripts/construction_initialization_repair.py",
    "scripts/construction_initialization_graph.py",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
    "scripts/submit_initialization_repair_aces.py",
    "scripts/hpc/start_initialization_repair_aces.sh",
    "scripts/hpc/run_initializer_choice_aces.sh",
)


def _terminal_value(value: dict) -> dict:
    """Validate a saved runner seal without importing historical execution code."""
    if value.get("result_sha256") != content_hash(
        {key: item for key, item in value.items() if key != "result_sha256"}
    ):
        raise ValueError("source initializer result digest differs")
    return value


def _source(source: Path) -> tuple[dict, dict, dict[str, dict]]:
    """Verify complete V1 choices and cached replies before making fresh episodes."""
    with tarfile.open(source, "r:gz") as archive:
        plan = legacy._sealed(legacy._archive_json(archive, "plan.json"))
        summary = legacy._archive_json(archive, "summary.json")
        entries = plan.get("cases")
        if (
            plan.get("protocol") != legacy.PROTOCOL
            or not isinstance(entries, list)
            or not 1 <= len(entries) <= 2
            or summary.get("plan_sha256") != plan["artifact_sha256"]
            or summary.get("status") != "complete"
            or plan.get("test_data_opened") is not False
            or summary.get("test_data_opened") is not False
            or plan.get("optimizer_calls") != 0
            or summary.get("optimizer_calls") != 0
            or plan.get("automatic_followup") is not False
        ):
            raise ValueError("requires a completed V1 initialization graph inspection")
        models, source_calls = {}, []
        for entry in entries:
            name = entry["case_id"]
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", name)
                or name in models
            ):
                raise ValueError("duplicate or unsafe source model name")
            case = legacy._sealed(legacy._archive_json(archive, f"inputs/{name}.json"))
            if case["artifact_sha256"] != entry["input_sha256"]:
                raise ValueError("source case does not match plan")
            evidence = TrainingCompatibilityEvidence.model_validate(
                case["training_evidence"]
            )
            validate_evidence(evidence)
            if evidence.context.model_dump(mode="json") != case["context"]:
                raise ValueError("source evidence context differs")
            draft = Draft.model_validate(case["draft"])
            contract = GraphConstructionContract.model_validate(case["contract"])
            result = _terminal_value(
                legacy._archive_json(archive, f"results/{name}/result.json")
            )
            if (
                result.get("status") != "complete"
                or result.get("handoff") is None
                or result.get("contract") != case["contract"]
                or result.get("test_data_opened") is not False
                or result.get("optimizer_calls") != 0
            ):
                raise ValueError("source model lacks a completed initializer choice")
            handoff = GraphDependencyHandoff.model_validate(result["handoff"])
            validate_handoff(
                handoff,
                contract,
                draft=draft,
                context=evidence.context,
                scientific_context=case["scientific_context"],
            )
            attempts = result["attempts"]
            if not attempts or len(attempts) > plan["settings"]["maximum_requests"]:
                raise ValueError("source attempt count differs")
            calls = []
            for attempt in attempts:
                key = attempt["request_hash"]
                record = legacy._archive_json(
                    archive, f"results/{name}/calls/{key}.json"
                )
                if (
                    content_hash(record["request"]) != key
                    or record.get("request_hash") != key
                    or record["request"].get("namespace") != entry["namespace"]
                    or record["request"].get("settings") != plan["settings"]
                ):
                    raise ValueError("source cached request provenance differs")
                calls.append(record)
            accepted = [i for i, attempt in enumerate(attempts) if attempt["accepted"]]
            if accepted != [len(attempts) - 1] or (
                GraphDependencyHandoff.model_validate(
                    visible_response(calls[-1])
                ).model_dump(mode="json")
                != result["handoff"]
            ):
                raise ValueError("source accepted choice differs from cached reply")
            source_calls.extend(calls)
            models[name] = {
                "source_model_id": name,
                "source_input_sha256": case["artifact_sha256"],
                "source_result_sha256": result["result_sha256"],
                "source_model_seed": entry["seed"],
                "input": case,
                "initial_handoff": handoff.model_dump(mode="json"),
                "source_graph_cost": _cost(calls),
            }
        if _cost(source_calls) != summary.get("cost"):
            raise ValueError("source diagnostic cost differs from cached calls")
    return plan, summary, models


def _prepared(
    source: Path, *, feedback_policy: str = "legacy-1"
) -> tuple[dict, dict[str, dict]]:
    """Replicate independent proposal seeds, never historical work or model search."""
    if feedback_policy not in FEEDBACK_PROTOCOLS:
        raise ValueError("unknown initializer repair feedback policy")
    protocol = FEEDBACK_PROTOCOLS[feedback_policy]
    entrypoints = ENTRYPOINTS + (
        ("scripts/hpc/start_initialization_scope_repair_aces.sh",)
        if feedback_policy == "scope-aware-2"
        else ()
    )
    prior, summary, models = _source(source)
    settings = StagedModelSettings.model_validate(prior["settings"]).model_copy(
        update={
            "maximum_requests": BUDGETS["maximum_calls_per_episode"],
            "attempts_per_step": BUDGETS["maximum_calls_per_episode"],
            "maximum_total_tokens": BUDGETS["maximum_charged_tokens_per_episode"],
            "max_output_tokens": BUDGETS["maximum_output_tokens"],
        }
    )
    episodes = []
    for name in sorted(models):
        for seed in PROPOSAL_SEEDS:
            episode = f"{name}__proposal_seed{seed}"
            episodes.append(
                {
                    "case_id": episode,
                    "source_model_id": name,
                    "seed": seed,
                    "namespace": f"{protocol}/{prior['artifact_sha256']}/{episode}",
                    "input_sha256": content_hash(models[name]),
                }
            )
    return {
        "protocol": protocol,
        **(
            {"feedback_policy": feedback_policy}
            if feedback_policy != "legacy-1"
            else {}
        ),
        "source_archive_sha256": legacy._file_hash(source),
        "source_plan_sha256": prior["artifact_sha256"],
        "source_diagnostic_summary": summary,
        "runtime_source_hash": runtime_source_hash(),
        "entrypoint_sha256": {
            name: legacy._file_hash(REPO / name) for name in entrypoints
        },
        "settings": settings.model_dump(mode="json"),
        "budgets": BUDGETS,
        "budget_reservation_policy": MEASURED_RESERVATION,
        "proposal_seeds": list(PROPOSAL_SEEDS),
        "source_models": sorted(models),
        "cases": episodes,
        "acceptance_policy": "deterministic_conflicts",
        "selection_rule": "all_source_models_all_proposal_seeds_no_score_selection",
        "test_data_opened": False,
        "optimizer_calls": 0,
        "automatic_followup": False,
        "model_promotion": False,
        "topology_edits_allowed": False,
        "historical_budget_refunded": False,
    }, models


def prepare(source: Path, root: Path, *, feedback_policy: str = "legacy-1") -> dict:
    """Seal a new child plan and keep the entire predecessor archive unchanged."""
    source = source.resolve()
    if (root / "results").exists() and not (root / "plan.json").exists():
        raise ValueError("existing results lack a plan; use a fresh root")
    plan, models = _prepared(source, feedback_policy=feedback_policy)
    if (root / "plan.json").exists() and sealed_read(root / "plan.json") != {
        **plan,
        "artifact_sha256": content_hash(plan),
    }:
        raise ValueError("repair identity differs; use a fresh root")
    historical = root / "historical/source.tar.gz"
    historical.parent.mkdir(parents=True, exist_ok=True)
    if historical.exists():
        if legacy._file_hash(historical) != plan["source_archive_sha256"]:
            raise ValueError("frozen source differs")
    else:
        temporary = historical.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(historical)
    for name, model in models.items():
        sealed_write(root / "inputs" / f"{name}.json", model)
    result = sealed_write(root / "plan.json", plan)
    preview(root)
    report(root)
    return result


def verify(root: Path) -> dict:
    """Reject changed source, code, settings, seeds or inputs before any request."""
    plan = sealed_read(root / "plan.json")
    expected, models = _prepared(
        root / "historical/source.tar.gz",
        feedback_policy=plan.get("feedback_policy", "legacy-1"),
    )
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("repair plan, source or runtime identity differs")
    for name, model in models.items():
        if sealed_read(root / "inputs" / f"{name}.json") != {
            **model,
            "artifact_sha256": content_hash(model),
        }:
            raise ValueError("frozen repair input differs")
    if (root / "execution.json").exists():
        execution = sealed_read(root / "execution.json")
        if (
            execution.get("protocol") != "initialization-repair-aces-execution-1"
            or execution.get("plan_sha256") != plan["artifact_sha256"]
            or execution.get("launcher_sha256")
            != legacy._file_hash(REPO / "scripts/hpc/run_initializer_choice_aces.sh")
            or execution.get("automatic_followup") is not False
        ):
            raise ValueError("repair execution plan or launcher identity differs")
    return plan


def _calls(root: Path, entry: dict, plan: dict) -> list[dict]:
    """Verify cached new charges separately for every proposal-seed namespace."""
    records = []
    for path in sorted((root / "results" / entry["case_id"] / "calls").glob("*.json")):
        record = json.loads(path.read_text())
        key = content_hash(record["request"])
        if (
            path.stem != key
            or record.get("request_hash") != key
            or record["request"].get("namespace") != entry["namespace"]
            or record["request"].get("settings") != plan["settings"]
            or record["request"]["body"].get("seed")
            != int(
                content_hash([entry["seed"], record["step"], record["attempt"]])[:8], 16
            )
            % (2**31)
            or type(record.get("budget_charge")) is not int
            or record["budget_charge"] < 0
        ):
            raise ValueError("new repair request provenance differs")
        records.append(record)
    if len(records) > plan["budgets"]["maximum_calls_per_episode"]:
        raise ValueError("new repair request ceiling exceeded")
    return records


def _parts(root: Path, entry: dict):
    model = sealed_read(root / "inputs" / f"{entry['source_model_id']}.json")
    case = model["input"]
    evidence = TrainingCompatibilityEvidence.model_validate(case["training_evidence"])
    return (
        case,
        Draft.model_validate(case["draft"]),
        evidence,
        GraphConstructionContract.model_validate(case["contract"]),
        GraphDependencyHandoff.model_validate(model["initial_handoff"]),
    )


def run(
    root: Path,
    base_url: str,
    runtime_record: Path | None = None,
    *,
    seconds: int = 3600,
    transport=None,
    token_transport=None,
) -> dict:
    """Execute each bounded seed episode once with cached deterministic resume."""
    plan = verify(root)
    base_url = legacy._loopback(base_url)
    endpoint = {"plan_sha256": plan["artifact_sha256"], "base_url": base_url}
    sealed_write(root / "endpoint_history" / f"{content_hash(endpoint)}.json", endpoint)
    atomic_json(root / "endpoint.json", endpoint)
    if runtime_record is not None:
        legacy.bind_server(root, runtime_record, base_url)
    stopped = False
    deadline = time.monotonic() + seconds

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for entry in plan["cases"]:
            directory = root / "results" / entry["case_id"]
            _calls(root, entry, plan)
            if (directory / "execution_error.json").exists():
                sealed_read(directory / "execution_error.json")
                continue
            case, draft, evidence, contract, initial = _parts(root, entry)
            client = ConstructionClient(
                settings=StagedModelSettings.model_validate(plan["settings"]),
                base_url=base_url,
                directory=directory / "calls",
                namespace=entry["namespace"],
                seed=entry["seed"],
                reservation_policy=MEASURED_RESERVATION,
                token_transport=token_transport,
                can_start=lambda: not stopped
                and time.monotonic() + plan["settings"]["timeout_seconds"] < deadline,
                **({"transport": transport} if transport is not None else {}),
            )
            client.budget_diagnostics_enabled = True
            try:
                construct_graph_dependencies(
                    draft,
                    evidence.context,
                    case["scientific_context"],
                    contract,
                    client,
                    directory,
                    evidence=evidence,
                    previous_questions=tuple(case["previous_questions"]),
                    initial_handoff=initial,
                    feedback_policy=plan.get("feedback_policy", "legacy-1"),
                )
            except DeferredCall:
                break
            except (PromptPreflightError, RepairBudgetExceeded) as exc:
                sealed_write(
                    directory / "execution_error.json",
                    {
                        "plan_sha256": plan["artifact_sha256"],
                        "status": "unavailable",
                        "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
                        "automatic_retry": False,
                    },
                )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        summary = report(root)
    return summary


def preview(root: Path) -> dict:
    """Render exact first requests for all seeds without any provider access."""
    plan = verify(root)
    feedback_policy = plan.get("feedback_policy", "legacy-1")
    system = system_prompt(feedback_policy)
    sections = []
    for entry in plan["cases"]:
        case, draft, evidence, contract, initial = _parts(root, entry)
        client = ConstructionClient(
            settings=StagedModelSettings.model_validate(plan["settings"]),
            base_url="http://127.0.0.1",
            directory=root / "preview/unused-calls" / entry["case_id"],
            namespace=entry["namespace"],
            seed=entry["seed"],
            reservation_policy=MEASURED_RESERVATION,
        )
        identity = dependency_construction_identity(
            draft,
            evidence.context,
            case["scientific_context"],
            contract,
            client,
            evidence=evidence,
            previous_questions=tuple(case["previous_questions"]),
            initial_handoff=initial,
            feedback_policy=feedback_policy,
        )
        user = dependency_request(
            draft,
            evidence.context,
            case["scientific_context"],
            contract,
            evidence=evidence,
            previous_questions=tuple(case["previous_questions"]),
            construction_identity=identity,
            initial_handoff=initial,
            feedback_policy=feedback_policy,
        )
        packet = {
            "system": system,
            "user": user,
            "response_schema": strict_provider_schema(
                GraphDependencyReply.model_json_schema()
            ),
        }
        atomic_json(root / "preview" / f"{entry['case_id']}.json", packet)
        sections.append(
            f"<h2>{html.escape(entry['case_id'])}</h2>"
            f"<h3>System</h3><pre>{html.escape(system)}</pre>"
            f"<h3>User</h3>{legacy._html_user_message(user)}"
            "<h3>Schema</h3>"
            f"<pre>{html.escape(json.dumps(packet['response_schema'], indent=2))}</pre>"
        )
    path = root / "PROMPTS.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Initializer repair prompts</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}dt{font-weight:600}"
        "dd{margin:0 0 1rem 1rem}h2{border-top:1px solid #bbb;padding-top:1rem}</style>"
        "<h1>Deterministic initializer conflict repair</h1>"
        "<p>Offline exact first-request preview. Two saved models, five proposal "
        "seeds each; at most three calls per episode. No score selection, fitting "
        "or test data. Deferred evidence is not a demonstrated repair.</p>"
        + "".join(sections)
    )
    return {"preview": str(path), "episodes": len(plan["cases"]), "llm_calls": 0}


def report(root: Path) -> dict:
    """Keep acceptance, active conflicts, deferrals and source costs distinct."""
    plan = verify(root)
    rows, records = [], []
    for entry in plan["cases"]:
        directory = root / "results" / entry["case_id"]
        current = _calls(root, entry, plan)
        records.extend(current)
        result = legacy._terminal(directory / "result.json")
        error = (
            sealed_read(directory / "execution_error.json")
            if (directory / "execution_error.json").exists()
            else None
        )
        result = result or {}
        rows.append(
            {
                "case_id": entry["case_id"],
                "source_model_id": entry["source_model_id"],
                "proposal_seed": entry["seed"],
                "status": result.get("status", (error or {}).get("status", "pending")),
                "structural_valid": result.get("structural_valid"),
                "accepted": result.get("accepted"),
                "compatibility_status": result.get(
                    "compatibility_status", "unassessed"
                ),
                "conflict_count": result.get("conflict_count"),
                "deferred_count": result.get("deferred_count"),
                "cost": _cost(current),
                "scientific_correctness_established": False,
            }
        )
    counts = dict(Counter(row["status"] for row in rows))
    history = {}
    for name in plan["source_models"]:
        model = sealed_read(root / "inputs" / f"{name}.json")
        case = model["input"]
        history[name] = {
            "source_model_seed": model["source_model_seed"],
            "source_graph_cost": model["source_graph_cost"],
            "historical_source_model_cost": case["historical_cost"],
            "predecessor_historical_cost": case["predecessor_historical_cost"],
        }
    summary = {
        "protocol": plan["protocol"],
        **(
            {"feedback_policy": plan["feedback_policy"]}
            if "feedback_policy" in plan
            else {}
        ),
        "plan_sha256": plan["artifact_sha256"],
        "status": "pending" if counts.get("pending") else "complete",
        "availability_status": "unavailable"
        if counts.get("unavailable")
        else "available",
        "status_counts": counts,
        "structural_valid_interpretation": "Graph/schema validity only; acceptance "
        "also requires absence of a qualified active deterministic conflict.",
        "compatibility_status_counts": dict(
            Counter(row["compatibility_status"] for row in rows)
        ),
        "structurally_valid_episodes": sum(
            row["structural_valid"] is True for row in rows
        ),
        "accepted_episodes": sum(row["accepted"] is True for row in rows),
        "rows": rows,
        "cost": _cost(records),
        "historical_by_source_model": history,
        "historical_source_diagnostic_cost": plan["source_diagnostic_summary"]["cost"],
        "cost_scope": "cost covers new calls only. Historical source/model costs "
        "are recorded once per source, never multiplied by proposal seeds. "
        "The source graph costs are a breakdown of the historical diagnostic "
        "total, not additional charges. No historical budget is refunded.",
        "source_models": len(plan["source_models"]),
        "proposal_seeds": plan["proposal_seeds"],
        "maximum_calls": BUDGETS["maximum_calls_per_episode"] * len(rows),
        "maximum_charged_tokens": BUDGETS["maximum_charged_tokens_per_episode"]
        * len(rows),
        "acceptance_policy": plan["acceptance_policy"],
        "prefix_window_policy": "unresolved_fitting_hyperparameter",
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "scientific_correctness_established": False,
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Saved initializer conflict repair",
        "",
        f"Status: {summary['status']}; availability: {summary['availability_status']}.",
        "",
        "Qualified deterministic conflicts block acceptance. Deferred evidence "
        "remains unresolved; acceptance is not scientific correctness.",
        "",
        "| Source model | Proposal seed | Status | Graph valid | Accepted | "
        "Compatibility | New calls | New tokens |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['source_model_id']} | {row['proposal_seed']} | {row['status']} | "
        f"{row['structural_valid']} | {row['accepted']} | "
        f"{row['compatibility_status']} | "
        f"{row['cost']['physical_requests']} | {row['cost']['observed_tokens']} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            summary["cost_scope"],
            "",
            "No model selection, promotion, "
            "fitting, test-data access or automatic follow-up.",
        ]
    )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


def inspect(root: Path) -> dict:
    """Archive saved evidence and provenance, without rerunning model requests."""
    summary = report(root)
    preview(root)
    path = root / "inspection.tar.gz"
    count = 0
    with tarfile.open(path, "w:gz") as archive:
        for name in (
            "plan.json",
            "endpoint.json",
            "endpoint_history",
            "execution.json",
            "serving_runtime.json",
            "summary.json",
            "SUMMARY.md",
            "PROMPTS.html",
            "historical",
            "inputs",
            "preview",
            "results",
        ):
            item = root / name
            paths = sorted(item.rglob("*")) if item.is_dir() else [item]
            for entry in paths:
                if entry.is_symlink():
                    raise ValueError("inspection does not follow symlinks")
                if entry.is_file():
                    archive.add(
                        entry, arcname=str(entry.relative_to(root)), recursive=False
                    )
                    count += 1
    return {"archive": str(path), "files": count, "status": summary["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "preview", "run", "report", "inspect")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--runtime-record", type=Path)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--feedback-policy", choices=tuple(FEEDBACK_PROTOCOLS))
    args = parser.parse_args()
    if args.feedback_policy is not None and args.command != "prepare":
        parser.error(
            "--feedback-policy applies only to prepare; "
            "other commands use the sealed plan"
        )
    with _lock(args.root / "campaign-lock"):
        if args.command == "prepare":
            if args.source is None:
                parser.error("prepare requires --source")
            result = prepare(
                args.source,
                args.root,
                feedback_policy=args.feedback_policy or "legacy-1",
            )
        elif args.command == "run":
            if args.base_url is None or args.seconds <= 0:
                parser.error("run requires --base-url and positive --seconds")
            result = run(
                args.root, args.base_url, args.runtime_record, seconds=args.seconds
            )
        else:
            result = {
                "verify": verify,
                "preview": preview,
                "report": report,
                "inspect": inspect,
            }[args.command](args.root)
    print(json.dumps(result, indent=2))
    if args.command == "run":
        raise SystemExit(
            int(
                result["status"] != "complete"
                or result["availability_status"] != "available"
            )
        )


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    main()
