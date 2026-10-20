#!/usr/bin/env python3
"""Construct functions for every accepted saved initializer; export, never fit."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import signal
import sys
import tarfile
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
    strict_provider_schema,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research.interaction_fit_handoff import build_handoff, verify_handoff
from autoformalism.research.interaction_source import load_sources
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.construction_graph_formulas import SYSTEM_PROMPT
from autoformalism.search.staged_function_prompts import (
    render_equation_function_batch_system_prompt,
    render_interaction_function_system_prompt,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.training_evidence import TrainingEvidence
from autoformalism.staged_topology import content_hash

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "saved-construction-functions-1"
ENTRYPOINTS = (
    "scripts/construction_functions.py",
    "scripts/submit_construction_functions_aces.py",
    "scripts/hpc/start_construction_functions_aces.sh",
    "scripts/hpc/run_construction_functions_aces.sh",
    "scripts/submit_review_continuation.py",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
)
BUDGETS = {
    "maximum_calls_per_case": 96,
    "maximum_charged_tokens_per_case": 2097152,
    "maximum_output_tokens": 8192,
    "attempts_per_step": 2,
}
POLICIES = {
    "generation_granularity": "equation_batch_atomic_repair",
    "function_repair_policy": "certified_outer_gain",
    "initialization_policy": "initialization_graph",
    "shared_process_guidance": True,
    "dependency_policy": "strict",
    "assembly_policy": "legacy",
    "function_delivery_policy": "legacy",
}


def file_hash(path: Path) -> str:
    """Hash source bytes without interpreting or extracting them."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _prepared(source: Path) -> tuple[dict, dict]:
    """Freeze all accepted identities; neither ranking nor deduplication is used."""
    loaded = load_sources(source)
    settings = StagedModelSettings.model_validate(
        loaded["source_plan"]["settings"]
    ).model_copy(
        update={
            "maximum_requests": BUDGETS["maximum_calls_per_case"],
            "maximum_total_tokens": BUDGETS["maximum_charged_tokens_per_case"],
            "attempts_per_step": BUDGETS["attempts_per_step"],
            "max_output_tokens": BUDGETS["maximum_output_tokens"],
        }
    )
    cases = []
    for name, value in loaded["cases"].items():
        identity = {
            "protocol": PROTOCOL,
            "case_id": name,
            "input_sha256": content_hash(value),
            "source_result_sha256": value["result"]["result_sha256"],
        }
        cases.append(
            {
                **identity,
                "source_model_id": value["entry"]["source_model_id"],
                "seed": value["entry"]["seed"],
                "namespace": f"{PROTOCOL}:{content_hash(identity)}",
                "initializer_rules": value["result"]
                .get("handoff", {})
                .get("records", []),
            }
        )
    plan = {
        "protocol": PROTOCOL,
        "source_archive_sha256": file_hash(source),
        "source_plan_sha256": loaded["source_plan"]["artifact_sha256"],
        "source_summary": loaded["source_summary"],
        "roster": loaded["roster"],
        "cases": cases,
        "settings": settings.model_dump(mode="json"),
        "budgets": BUDGETS,
        "policies": POLICIES,
        "runtime_source_sha256": runtime_source_hash(),
        "entrypoints": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "selection": "all accepted saved declarations; no family or score selection",
        "prefix_window_policy": "unresolved_fitting_hyperparameter",
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
    }
    return plan, loaded["cases"]


def prepare(source: Path, root: Path) -> dict:
    """Copy and seal one source archive before constructing any new functions."""
    if source.is_symlink() or not source.is_file():
        raise ValueError("source must be a regular inspection archive")
    plan, cases = _prepared(source)
    frozen = root / "historical/source.tar.gz"
    if frozen.exists() and file_hash(frozen) != plan["source_archive_sha256"]:
        raise ValueError("historical source differs; use a fresh root")
    if (root / "results").exists() and not (root / "plan.json").exists():
        raise ValueError("unbound results exist; use a fresh root")
    result = sealed_write(root / "plan.json", plan)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        if file_hash(temporary) != plan["source_archive_sha256"]:
            raise ValueError("source changed while copying")
        temporary.replace(frozen)
    for name, value in cases.items():
        sealed_write(root / "inputs" / f"{name}.json", value)
    return result


def verify(root: Path) -> dict:
    """Rebuild identities from original replies, full drafts and current code."""
    plan = sealed_read(root / "plan.json")
    expected, cases = _prepared(root / "historical/source.tar.gz")
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("function campaign source, runtime or plan differs")
    for name, value in cases.items():
        if sealed_read(root / "inputs" / f"{name}.json") != {
            **value,
            "artifact_sha256": content_hash(value),
        }:
            raise ValueError("frozen function input differs")
    return plan


def _calls(root: Path, entry: dict, plan: dict) -> list[dict]:
    """Verify the new namespace, physical charges and deterministic call seeds."""
    records = []
    for path in sorted((root / "results" / entry["case_id"] / "calls").glob("*.json")):
        record = json.loads(path.read_text())
        request = record["request"]
        key = content_hash(request)
        if (
            path.is_symlink()
            or path.stem != key
            or record.get("request_hash") != key
            or request.get("namespace") != entry["namespace"]
            or request.get("settings") != plan["settings"]
            or request["body"].get("seed")
            != int(
                content_hash([entry["seed"], record["step"], record["attempt"]])[:8], 16
            )
            % (2**31)
            or type(record.get("budget_charge")) is not int
            or record["budget_charge"] < 0
        ):
            raise ValueError("function request provenance differs")
        records.append(record)
    if len(records) > plan["settings"]["maximum_requests"]:
        raise ValueError("function physical-request ceiling exceeded")
    return records


def _tokenizations(root: Path, entry: dict, plan: dict) -> list[dict]:
    """Verify tokenizer receipts separately from charged generation requests."""
    directory = root / "results" / entry["case_id"] / "calls" / "tokenization"
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        request = record["request"]
        if (
            path.is_symlink()
            or path.stem != content_hash(request)
            or request.get("namespace") != entry["namespace"]
            or request.get("policy") != "response-token-preflight-1"
            or request.get("model_revision") != plan["settings"]["model_revision"]
        ):
            raise ValueError("function tokenizer provenance differs")
        records.append(record)
    return records


def _construct(value: dict, client, directory: Path) -> dict:
    """Use the existing restricted function runner with immutable dependencies."""
    return run_staged_functions(
        PublicScientificBrief.model_validate(value["brief"]),
        ValidationContext.model_validate(value["case"]["context"]),
        value["function_source"],
        client,
        directory,
        training_evidence=(
            TrainingEvidence.model_validate(value["training_evidence"])
            if value["training_evidence"] is not None
            else None
        ),
        **POLICIES,
    )


def _provenance(plan: dict, entry: dict, value: dict) -> dict:
    """Bind exports to the exact accepted initializer, topology and public packet."""
    return {
        "plan_sha256": plan["artifact_sha256"],
        "case_id": entry["case_id"],
        "source_model_id": entry["source_model_id"],
        "proposal_seed": entry["seed"],
        "source_result_sha256": entry["source_result_sha256"],
        "function_source_sha256": content_hash(value["function_source"]),
        "public_brief_sha256": content_hash(value["brief"]),
        "training_evidence_sha256": content_hash(value["training_evidence"]),
        "retained_compatibility_status": value["result"]["compatibility_status"],
        "retained_compatibility_evidence": value["case"]["training_evidence"],
        "prefix_window_policy": plan["prefix_window_policy"],
    }


def _receipt(root: Path, entry: dict, plan: dict, calls: list[dict]) -> dict | None:
    """Check a terminal receipt before skipping cached work or reporting exports."""
    directory = root / "results" / entry["case_id"]
    if not (directory / "completion.json").exists():
        return None
    value = sealed_read(directory / "completion.json")
    if (
        value["plan_sha256"] != plan["artifact_sha256"]
        or value["input_sha256"] != entry["input_sha256"]
        or value["calls_sha256"] != content_hash(calls)
        or value["tokenizations_sha256"]
        != content_hash(_tokenizations(root, entry, plan))
    ):
        raise ValueError("function completion provenance differs")
    for name, digest in value["files"].items():
        if name not in {"result.json", "fit_handoff.json", "execution_error.json"}:
            raise ValueError("unexpected function completion file")
        if file_hash(directory / name) != digest:
            raise ValueError("completed function artifact differs")
    return value


def _availability(
    status: str, error: str | None, calls: list[dict], tokenizations: list[dict]
) -> str:
    """Keep transport, truncation and admission failures apart from formula errors."""
    if status == "pending":
        return "pending"
    if status == "complete":
        return "available"
    if status == "unavailable":
        return "unavailable"
    if any("error" in record for record in tokenizations):
        return "unavailable"
    if any(
        marker in (error or "")
        for marker in (
            "complete construction context",
            "total provider request budget",
            "total token budget",
        )
    ):
        return "unavailable"
    last_by_step = {}
    for record in calls:
        old = last_by_step.get(record["step"])
        if old is None or record["attempt"] > old["attempt"]:
            last_by_step[record["step"]] = record
    for record in last_by_step.values():
        choices = record.get("raw_response", {}).get("choices", [])
        if (
            record["status"] != "responded"
            or not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
            or not isinstance(choices[0].get("message"), dict)
            or not choices[0].get("message", {}).get("content")
        ):
            return "unavailable"
    return "available"


def run(
    root: Path,
    base_url: str,
    runtime_record: Path | None = None,
    *,
    seconds: int = 5400,
    transport=None,
    token_transport=None,
) -> dict:
    """Resume bounded episodes through the same cache; never invoke the fitter."""
    plan = verify(root)
    url = urlparse(base_url)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.username is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("function server must be a loopback HTTP endpoint")
    if runtime_record is not None:
        from scripts import submit_construction_functions_aces as submitter
        from scripts.saved_initialization_aces import bind_server

        _, execution = submitter.verify(root)
        runtime = json.loads(runtime_record.read_text())
        identity_keys = (
            "artifact_sha256",
            "model",
            "model_revision",
            "image_sha256",
            "served_context_tokens",
            "tensor_parallel_size",
            "maximum_sequences",
        )
        if runtime.get("serving_identity") != {
            key: execution[key] for key in identity_keys
        }:
            raise ValueError("serving runtime does not match frozen execution")
        bind_server(root, runtime_record, base_url)
    elif transport is None:
        raise ValueError("live generation requires a verified runtime record")
    endpoint = {"plan_sha256": plan["artifact_sha256"], "base_url": base_url}
    sealed_write(root / "endpoints" / f"{content_hash(endpoint)}.json", endpoint)
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
            if _receipt(root, entry, plan, _calls(root, entry, plan)) is not None:
                continue
            value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
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
            files = []
            try:
                result = _construct(value, client, directory)
                files.append("result.json")
                status = result["status"]
                if result["complete_model"]:
                    handoff = build_handoff(
                        result, provenance=_provenance(plan, entry, value)
                    )
                    sealed_write(
                        directory / "fit_handoff.json",
                        {k: v for k, v in handoff.items() if k != "artifact_sha256"},
                    )
                    files.append("fit_handoff.json")
            except DeferredCall:
                break
            except (ValueError, RuntimeError, OSError) as exc:
                status = "unavailable"
                sealed_write(
                    directory / "execution_error.json",
                    {
                        "plan_sha256": plan["artifact_sha256"],
                        "status": status,
                        "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
                        "automatic_retry": False,
                    },
                )
                files.append("execution_error.json")
            sealed_write(
                directory / "completion.json",
                {
                    "plan_sha256": plan["artifact_sha256"],
                    "input_sha256": entry["input_sha256"],
                    "calls_sha256": content_hash(_calls(root, entry, plan)),
                    "tokenizations_sha256": content_hash(
                        _tokenizations(root, entry, plan)
                    ),
                    "status": status,
                    "files": {name: file_hash(directory / name) for name in files},
                },
            )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        summary = report(root)
    return summary


class _PreviewClient:
    """Capture the actual first runner request without transport or fitting."""

    def __init__(self, settings: dict):
        self.settings = StagedModelSettings.model_validate(settings)
        self.records = []
        self.packet = None

    def call(self, *, system, user, response_model, step, attempt):
        self.packet = {
            "step": step,
            "attempt": attempt,
            "system": system,
            "user": user,
            "response_schema": strict_provider_schema(
                response_model.model_json_schema()
            ),
        }
        raise DeferredCall("offline request preview")


def preview(root: Path) -> dict:
    """Render exact first calls plus labeled later-stage instruction templates."""
    plan = verify(root)
    sections = []
    for entry in plan["cases"]:
        value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
        client = _PreviewClient(plan["settings"])
        with suppress(DeferredCall):
            _construct(value, client, root / "preview" / entry["case_id"])
        if client.packet is None:
            raise ValueError("function preflight did not reach an initial request")
        atomic_json(root / "preview" / f"{entry['case_id']}.json", client.packet)
        sections.append(
            f"<h2>{html.escape(entry['case_id'])}</h2>"
            + "".join(
                f"<h3>{html.escape(key)}</h3><pre>{html.escape(text)}</pre>"
                for key, text in (
                    ("System", client.packet["system"]),
                    ("User", client.packet["user"]),
                    (
                        "Response schema",
                        json.dumps(client.packet["response_schema"], indent=2),
                    ),
                )
            )
        )
    templates = [
        ("Equation batch", render_equation_function_batch_system_prompt()),
        ("Targeted function repair", render_interaction_function_system_prompt()),
        ("Initializer formulas/roles", SYSTEM_PROMPT),
    ]
    path = root / "PROMPTS.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Function construction</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "h2{border-top:1px solid #bbb;padding-top:1rem}</style>"
        "<h1>Function construction and fitting handoff</h1>"
        "<p>All eight accepted declarations are retained. Topology, initializer "
        "dependencies and scopes are fixed. Construct equation batches, repair "
        "individual failed terms, then request initializer formulas or root roles. "
        "Two attempts per slot, at most 96 calls and 2,097,152 charged tokens per "
        "proposal across all stages. No fitting, selection or test access.</p>"
        "<p>Below: exact first requests, produced offline by the real runner. "
        "Later repair messages depend on replies; their instruction templates "
        "follow. Actual requests are saved verbatim under results/*/calls.</p>"
        + "".join(sections)
        + "<h2>Later-stage templates (not fabricated requests)</h2>"
        + "".join(
            f"<h3>{html.escape(name)}</h3><pre>{html.escape(text)}</pre>"
            for name, text in templates
        )
    )
    return {"preview": str(path), "cases": len(plan["cases"]), "llm_calls": 0}


def report(root: Path) -> dict:
    """Distinguish functions, fitting export and forecast capability for each seed."""
    plan = verify(root)
    rows, records = [], []
    for entry in plan["cases"]:
        directory = root / "results" / entry["case_id"]
        calls = _calls(root, entry, plan)
        records.extend(calls)
        receipt = _receipt(root, entry, plan, calls)
        result = (
            json.loads((directory / "result.json").read_text())
            if receipt and "result.json" in receipt["files"]
            else {}
        )
        execution_error = (
            sealed_read(directory / "execution_error.json")
            if receipt and "execution_error.json" in receipt["files"]
            else {}
        )
        export = (
            sealed_read(directory / "fit_handoff.json")
            if receipt and "fit_handoff.json" in receipt["files"]
            else {}
        )
        if export:
            value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
            verify_handoff(result, export, provenance=_provenance(plan, entry, value))
        rows.append(
            {
                "case_id": entry["case_id"],
                "source_model_id": entry["source_model_id"],
                "proposal_seed": entry["seed"],
                "status": receipt["status"] if receipt else "pending",
                "complete_model": result.get("complete_model", False),
                "accepted_functions": len(result.get("accepted_functions", [])),
                "initializer_formula_status": (result.get("initialization") or {}).get(
                    "status"
                ),
                "fitting_exported": bool(export),
                "readiness": export.get("readiness"),
                "error": result.get("error") or execution_error.get("error"),
                "availability": _availability(
                    receipt["status"] if receipt else "pending",
                    result.get("error") or execution_error.get("error"),
                    calls,
                    _tokenizations(root, entry, plan),
                ),
                "cost": _cost(calls),
            }
        )
    counts = dict(Counter(row["status"] for row in rows))
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status": "pending" if counts.get("pending") else "complete",
        "status_counts": counts,
        "availability_status": "unavailable"
        if any(row["availability"] == "unavailable" for row in rows)
        else "pending"
        if counts.get("pending")
        else "available",
        "rows": rows,
        "source_roster": plan["roster"],
        "cost": _cost(records),
        "historical_initializer_cost": plan["source_summary"]["cost"],
        "cost_scope": "New calls only; historical costs are retained once, "
        "not repeated per seed.",
        "maximum_calls": len(rows) * plan["settings"]["maximum_requests"],
        "maximum_charged_tokens": len(rows) * plan["settings"]["maximum_total_tokens"],
        "prefix_window_policy": plan["prefix_window_policy"],
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "scientific_correctness_established": False,
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Saved function construction",
        "",
        "All accepted initializer families retained. No fit, selection or test access.",
        "Formula completion is not fitted accuracy or mechanism compliance.",
        "Forecast capability and an unresolved prefix window are reported separately.",
        "",
        "| Proposal | Status | Availability | Functions | Fit packet | "
        "New calls | Tokens |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    lines += [
        f"| {r['case_id']} | {r['status']} | {r['availability']} | "
        f"{r['accepted_functions']} | "
        f"{r['fitting_exported']} | {r['cost']['physical_requests']} | "
        f"{r['cost']['observed_tokens']} |"
        for r in rows
    ]
    lines += ["", summary["cost_scope"], ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    _render_results(root, rows)
    return summary


def _render_results(root: Path, rows: list[dict]) -> None:
    """Show complete assembled equations and remaining gates without fit scores."""
    sections = []
    for row in rows:
        directory = root / "results" / row["case_id"]
        equations = []
        if row["fitting_exported"]:
            export = sealed_read(directory / "fit_handoff.json")
            model = export["prefix_fit_problem"]["base_candidate"]
            equations += [
                f"{p['name']} = {p['expression']}" for p in model["processes"]
            ]
            equations += [
                f"d({eq['state']})/dt = {eq['rhs']}" for eq in model["state_equations"]
            ]
            equations += [
                f"observed {m['channel']} = {m['expression']}"
                for m in model["observation_mappings"]
            ]
            for state, node in export["prefix_fit_problem"]["graph"]["rules"].items():
                rule = node["initial"]
                equations.append(f"{state}(0): {json.dumps(rule, sort_keys=True)}")
            equations.append(
                "Compiler aliases: "
                + json.dumps(
                    export["initialization_dependency_contract"]["state_aliases"]
                )
            )
        displayed = "\n".join(equations) or "No complete executable export."
        sections.append(
            f"<h2>{html.escape(row['case_id'])}</h2>"
            f"<p>Status: {html.escape(row['status'])}; "
            f"accepted functions: {row['accepted_functions']}.</p>"
            f"<pre>{html.escape(displayed)}</pre>"
            "<h3>Readiness</h3>"
            f"<pre>{html.escape(json.dumps(row['readiness'], indent=2))}</pre>"
            "<h3>Construction error</h3>"
            f"<pre>{html.escape(row['error'] or 'None')}</pre>"
        )
    (root / "REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Function results</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Function construction results</h1>"
        "<p>Every accepted source retained; no fitting, ranking or scientific "
        "certification. These equations contain unfitted parameters. Dependency "
        "evidence may remain deferred. Initializer aliases are local to each map.</p>"
        + "".join(sections)
    )


def inspect(root: Path) -> dict:
    """Package exact requests, formulas and exports, without new model requests."""
    summary = report(root)
    preview(root)
    path = root / "inspection.tar.gz"
    count = 0
    with tarfile.open(path, "w:gz") as archive:
        for name in (
            "plan.json",
            "summary.json",
            "SUMMARY.md",
            "PROMPTS.html",
            "REPORT.html",
            "execution.json",
            "serving_runtime.json",
            "historical",
            "inputs",
            "results",
            "preview",
            "endpoints",
            "submissions",
            "logs",
            "runtime",
        ):
            item = root / name
            for entry in sorted(item.rglob("*")) if item.is_dir() else [item]:
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
    parser.add_argument("--seconds", type=int, default=5400)
    args = parser.parse_args()
    with _lock(args.root / "campaign-lock"):
        if args.command == "prepare":
            if args.source is None:
                parser.error("prepare requires --source")
            result = prepare(args.source, args.root)
        elif args.command == "run":
            if (
                args.base_url is None
                or args.runtime_record is None
                or args.seconds <= 0
            ):
                parser.error(
                    "run requires --base-url, --runtime-record and positive --seconds"
                )
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
        if args.command in {"prepare", "verify"}:
            result = {
                "identity": result["artifact_sha256"],
                "cases": len(result["cases"]),
            }
        print(json.dumps(result, indent=2))
        if args.command == "run":
            if result["status"] != "complete":
                sys.exit(2)
            if result["availability_status"] != "available":
                sys.exit(1)


if __name__ == "__main__":
    main()
