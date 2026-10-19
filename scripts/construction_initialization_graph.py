#!/usr/bin/env python3
"""Prepare and run a bounded, saved-topology initializer dependency diagnostic."""

from __future__ import annotations

import argparse
import hashlib
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
from urllib.parse import urlsplit

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
    strict_provider_schema,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.search.construction_graph_formulas import (
    SYSTEM_PROMPT as FORMULA_SYSTEM_PROMPT,
)
from autoformalism.search.construction_graph_formulas import (
    GraphMapReply,
    GraphRootReply,
)
from autoformalism.search.construction_graph_runner import (
    SYSTEM,
    construct_graph_dependencies,
    dependency_construction_identity,
    dependency_request,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    build_contract,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "saved-construction-initialization-graph-1"
SOURCE_PROTOCOL = "saved-initializer-choice-diagnostic-2"
ENTRYPOINTS = (
    "scripts/construction_initialization_graph.py",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
)
BUDGETS = {
    "maximum_calls_per_case": 2,
    "maximum_charged_tokens_per_case": 262144,
    "maximum_output_tokens": 8192,
}


def bind_server(root: Path, runtime_record: Path, base_url: str) -> None:
    """Reuse the verified H100/Harmony preflight only when a receipt is supplied."""
    from scripts.saved_initialization_aces import bind_server as bind

    bind(root, runtime_record, base_url)


def _file_hash(path: Path) -> str:
    """Hash an immutable source without interpreting its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed(value: dict) -> dict:
    """Validate a historical seal without requiring its historical source code."""
    if value.get("artifact_sha256") != content_hash(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    ):
        raise ValueError("historical artifact digest differs")
    return value


def _archive_json(archive: tarfile.TarFile, name: str) -> dict:
    """Read one bounded regular member; never extract archive paths."""
    matches = [member for member in archive.getmembers() if member.name == name]
    if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 64 * 1024**2:
        raise ValueError(f"missing, duplicate or unsafe archive member: {name}")
    stream = archive.extractfile(matches[0])
    if stream is None:
        raise ValueError(f"unreadable archive member: {name}")
    value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"archive member is not a JSON object: {name}")
    return value


def _source(source: Path) -> tuple[dict, dict]:
    """Require the selected completed public diagnostic and retain its own costs."""
    with tarfile.open(source, "r:gz") as archive:
        plan = _sealed(_archive_json(archive, "plan.json"))
        summary = _archive_json(archive, "summary.json")
    if (
        plan.get("protocol") != SOURCE_PROTOCOL
        or not isinstance(plan.get("cases"), dict)
        or not 1 <= len(plan["cases"]) <= 2
        or plan.get("test_data_opened") is not False
        or plan.get("optimizer_calls") != 0
        or plan.get("automatic_followup") is not False
        or summary.get("plan_sha256") != plan["artifact_sha256"]
    ):
        raise ValueError("requires the bounded public saved initializer-choice source")
    return plan, summary


def _case(name: str, case: dict) -> dict:
    """Project only initializer-relevant ledger fields while retaining provenance."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", name):
        raise ValueError("unsafe diagnostic case name")
    source = _sealed(case["source_result"])
    if source["draft"] != case["start"]["draft"]:
        raise ValueError("historical starting draft differs from source result")
    fields = set(Draft.model_fields)
    projected = {key: value for key, value in source["draft"].items() if key in fields}
    omitted = {
        key: value for key, value in source["draft"].items() if key not in fields
    }
    draft = Draft.model_validate(projected)
    evidence = TrainingCompatibilityEvidence.model_validate(
        case["cell"]["training_evidence"]
    )
    validate_evidence(evidence)
    if evidence.context.model_dump(mode="json") != case["cell"]["context"]:
        raise ValueError(
            "historical public context differs from sealed training evidence"
        )
    scientific_context = case["cell"]["brief"]["scientific_context"]
    if not isinstance(scientific_context, str) or not scientific_context:
        raise ValueError("missing public scientific context")
    initial_information = case["start"]["initial_information_contract"]
    if initial_information.get("measured_initial_bindings", []) not in ([], ()):
        raise ValueError(
            "historical measured initial bindings require a reviewed graph projection"
        )
    states = {v.name for v in draft.variables if v.definition == "differential"}
    observed = {
        state: state
        for state, rule in evidence.contract.initial_rules.items()
        if state in states and rule.mode == "observed_initial"
    }
    contract = build_contract(
        draft,
        evidence.context,
        scientific_context=scientific_context,
        permitted_public_initials=initial_information["permitted_symbols"],
        observed_initial_channels=observed,
        trajectory_initialization="observation_prefix",
        provenance="Opt-in saved-topology initialization diagnostic. Observed "
        "identities come only from sealed runtime observed_initial rules. Public "
        "t0 access retains the historical allowlist. Prefix duration is a fitting "
        "hyperparameter, unresolved here; no scientific correctness is asserted.",
    )
    state = case["start"]["scientific_state"]
    questions, locations = [], []
    for collection in ("tracked_conflicts", "advisory_questions"):
        for index, question in enumerate(state.get(collection, [])):
            if not isinstance(question, dict):
                raise ValueError("historical question is not a record")
            questions.append(question)
            locations.append({"collection": collection, "index": index})
    seed = case["task"]["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("case seed must be a nonnegative integer")
    return {
        "case_id": name,
        "seed": seed,
        "source_task": case["task"],
        "draft": draft.model_dump(mode="json"),
        "context": evidence.context.model_dump(mode="json"),
        "scientific_context": scientific_context,
        "contract": contract.model_dump(mode="json"),
        "training_evidence": evidence.model_dump(mode="json"),
        "previous_questions": questions,
        "previous_question_locations": locations,
        "source_result_sha256": source["artifact_sha256"],
        "source_draft_sha256": content_hash(source["draft"]),
        "projection": {
            "retained_fields": sorted(projected),
            "omitted_fields": {
                key: content_hash(value) for key, value in sorted(omitted.items())
            },
            "interpretation": "The complete original draft remains in the copied "
            "historical archive. This projection supports initializer analysis "
            "only; omitted declarations are not removed from the historical model. "
            "Only historical observed_initial identities are supported; sources "
            "with measured_initial_bindings require a separate reviewed projection.",
        },
        "historical_cost": source["cost"],
        "predecessor_historical_cost": case["start"].get("predecessor_historical_cost"),
    }


def _prepared(source: Path) -> tuple[dict, dict[str, dict]]:
    """Derive deterministic, isolated inputs and fixed per-case request ceilings."""
    prior, summary = _source(source)
    settings = StagedModelSettings.model_validate(prior["parent_plan"]["settings"])
    settings = settings.model_copy(
        update={
            "maximum_requests": 2,
            "attempts_per_step": 2,
            "maximum_total_tokens": 262144,
            "max_output_tokens": 8192,
        }
    )
    inputs = {name: _case(name, case) for name, case in sorted(prior["cases"].items())}
    return {
        "protocol": PROTOCOL,
        "source_archive_sha256": _file_hash(source),
        "source_plan_sha256": prior["artifact_sha256"],
        "source_diagnostic_summary": summary,
        "runtime_source_hash": runtime_source_hash(),
        "entrypoint_sha256": {name: _file_hash(REPO / name) for name in ENTRYPOINTS},
        "settings": settings.model_dump(mode="json"),
        "budgets": BUDGETS,
        "budget_reservation_policy": MEASURED_RESERVATION,
        "cases": [
            {
                "case_id": name,
                "seed": case["seed"],
                "namespace": f"{PROTOCOL}/{prior['artifact_sha256']}/{name}",
                "input_sha256": content_hash(case),
            }
            for name, case in inputs.items()
        ],
        "test_data_opened": False,
        "optimizer_calls": 0,
        "automatic_followup": False,
        "model_promotion": False,
        "topology_edits_allowed": False,
        "historical_budget_refunded": False,
        "qualification": "structural_only",
    }, inputs


def prepare(source: Path, root: Path) -> dict:
    """Freeze an independent diagnostic; never modify or import historical caches."""
    source = source.resolve()
    if (root / "results").exists() and not (root / "plan.json").exists():
        raise ValueError("existing results lack a diagnostic plan; use a fresh root")
    plan, inputs = _prepared(source)
    historical = root / "historical" / "source.tar.gz"
    historical.parent.mkdir(parents=True, exist_ok=True)
    if historical.exists():
        if _file_hash(historical) != plan["source_archive_sha256"]:
            raise ValueError("frozen historical source differs")
    else:
        temporary = historical.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(historical)
    for name, case in inputs.items():
        sealed_write(root / "inputs" / f"{name}.json", case)
    result = sealed_write(root / "plan.json", plan)
    preview(root)
    report(root)
    return result


def verify(root: Path) -> dict:
    """Rebuild the frozen projection and verify source, code, settings and inputs."""
    plan = sealed_read(root / "plan.json")
    expected, inputs = _prepared(root / "historical" / "source.tar.gz")
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("diagnostic plan, source or runtime identity differs")
    for name, case in inputs.items():
        if sealed_read(root / "inputs" / f"{name}.json") != {
            **case,
            "artifact_sha256": content_hash(case),
        }:
            raise ValueError("frozen initializer case differs")
    return plan


def _loopback(base_url: str) -> str:
    """Permit only the explicit unauthenticated IPv4 loopback serving endpoint."""
    url = urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("server must be an unauthenticated loopback HTTP endpoint")
    return base_url.rstrip("/")


def _calls(root: Path, entry: dict, plan: dict) -> list[dict]:
    """Validate every new physical request before reporting retained charges."""
    records = []
    for path in sorted((root / "results" / entry["case_id"] / "calls").glob("*.json")):
        record = json.loads(path.read_text())
        key = content_hash(record["request"])
        if (
            path.stem != key
            or record.get("request_hash") != key
            or record["request"].get("namespace") != entry["namespace"]
            or record["request"].get("settings") != plan["settings"]
            or type(record.get("budget_charge")) is not int
            or record["budget_charge"] < 0
        ):
            raise ValueError("new diagnostic request provenance differs")
        records.append(record)
    if len(records) > 2:
        raise ValueError("new diagnostic request ceiling exceeded")
    return records


def _terminal(path: Path) -> dict | None:
    """Check a runner terminal seal before using its scientific status."""
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("result_sha256") != content_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ):
        raise ValueError("initializer result digest differs")
    return result


def run(
    root: Path,
    base_url: str,
    runtime_record: Path | None = None,
    *,
    seconds: int = 3600,
    transport=None,
    token_transport=None,
) -> dict:
    """Run each frozen unit once; cache failures and stop without automatic retries."""
    plan = verify(root)
    base_url = _loopback(base_url)
    endpoint = {"plan_sha256": plan["artifact_sha256"], "base_url": base_url}
    sealed_write(root / "endpoint_history" / f"{content_hash(endpoint)}.json", endpoint)
    atomic_json(root / "endpoint.json", endpoint)
    if runtime_record is not None:
        bind_server(root, runtime_record, base_url)
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
            case = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
            directory = root / "results" / entry["case_id"]
            _calls(root, entry, plan)
            if (directory / "execution_error.json").exists():
                sealed_read(directory / "execution_error.json")
                continue
            options = {"transport": transport} if transport is not None else {}
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
                **options,
            )
            client.budget_diagnostics_enabled = True
            evidence = TrainingCompatibilityEvidence.model_validate(
                case["training_evidence"]
            )
            try:
                construct_graph_dependencies(
                    Draft.model_validate(case["draft"]),
                    evidence.context,
                    case["scientific_context"],
                    GraphConstructionContract.model_validate(case["contract"]),
                    client,
                    directory,
                    evidence=evidence,
                    previous_questions=tuple(case["previous_questions"]),
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


def _formula_stage_templates() -> dict:
    """Show actual later-stage contracts with labeled, schema-valid illustrations."""
    return {
        "scope": "illustrative_later_formula_stage",
        "called_in_this_diagnostic": False,
        "system": FORMULA_SYSTEM_PROMPT,
        "response_schemas": {
            "independent_root": strict_provider_schema(
                GraphRootReply.model_json_schema()
            ),
            "dependent_map": strict_provider_schema(GraphMapReply.model_json_schema()),
        },
        "illustrative_examples": [
            {
                "upstream_dependency": {
                    "state": "z1",
                    "rule": {"mode": "value", "scope": "shared_training"},
                    "justification": "Illustrative shared initial preparation.",
                },
                "reply": GraphRootReply(role="coefficient").model_dump(mode="json"),
                "interpretation": "The earlier graph fixes shared or trajectory "
                "scope. The formula stage selects only the root's qualitative role; "
                "it does not fit or choose a numerical initial value.",
            },
            {
                "upstream_dependency": {
                    "state": "z2",
                    "rule": {
                        "mode": "map",
                        "state_initials": ["z1"],
                        "public_initials": ["y"],
                    },
                    "justification": "Illustrative dependent initial preparation.",
                },
                "aliases": {"state_1": "z1(0)", "public_1": "y(0)"},
                "reply": GraphMapReply.model_validate(
                    {
                        "expression": "a*state_1 + b*public_1",
                        "parameters": [
                            {"name": "a", "role": "coefficient"},
                            {"name": "b", "role": "coefficient"},
                        ],
                    }
                ).model_dump(mode="json"),
                "interpretation": "Shared coefficients a and b define a calculated "
                "map output. These invented coordinates and this example law are "
                "not selected for either saved model.",
            },
        ],
    }


def _html_json(value: object) -> str:
    """Render JSON values with real string line breaks and escaped HTML content."""
    if isinstance(value, dict):
        return (
            "<dl>"
            + "".join(
                f"<dt>{html.escape(str(key))}</dt><dd>{_html_json(item)}</dd>"
                for key, item in value.items()
            )
            + "</dl>"
        )
    if isinstance(value, list):
        return (
            "<ol>" + "".join(f"<li>{_html_json(item)}</li>" for item in value) + "</ol>"
        )
    text = value if isinstance(value, str) else json.dumps(value)
    return f"<pre>{html.escape(text)}</pre>"


def _html_user_message(message: str) -> str:
    """Pretty-render a JSON user payload without modifying the exact saved message."""
    try:
        return _html_json(json.loads(message))
    except json.JSONDecodeError:
        prefix, separator, body = message.partition("\n")
        if separator:
            try:
                return f"<p>{html.escape(prefix)}</p>" + _html_json(json.loads(body))
            except json.JSONDecodeError:
                pass
    return f"<pre>{html.escape(message)}</pre>"


def preview(root: Path) -> dict:
    """Write complete initial requests and a local readable HTML preview offline."""
    plan = verify(root)
    sections = []
    for entry in plan["cases"]:
        case = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
        evidence = TrainingCompatibilityEvidence.model_validate(
            case["training_evidence"]
        )
        draft = Draft.model_validate(case["draft"])
        contract = GraphConstructionContract.model_validate(case["contract"])
        client = ConstructionClient(
            settings=StagedModelSettings.model_validate(plan["settings"]),
            base_url="http://127.0.0.1",
            directory=root / "preview" / "unused-calls" / entry["case_id"],
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
        )
        user = dependency_request(
            draft,
            evidence.context,
            case["scientific_context"],
            contract,
            previous_questions=tuple(case["previous_questions"]),
            construction_identity=identity,
        )
        packet = {
            "system": SYSTEM,
            "user": user,
            "response_schema": strict_provider_schema(
                GraphDependencyHandoff.model_json_schema()
            ),
        }
        atomic_json(root / "preview" / f"{entry['case_id']}.json", packet)
        sections.append(
            f"<h2>{html.escape(entry['case_id'])}</h2>"
            f"<h3>System</h3><pre>{html.escape(SYSTEM)}</pre>"
            f"<h3>User</h3>{_html_user_message(user)}"
            "<h3>Schema</h3>"
            f"<pre>{html.escape(json.dumps(packet['response_schema'], indent=2))}</pre>"
        )
    templates = _formula_stage_templates()
    atomic_json(root / "preview" / "formula-stage-templates.json", templates)
    sections.append(
        "<section><h2>Later interaction/formula stage: illustrative only</h2>"
        "<p>This stage is not called in the saved dependency diagnostic. "
        "The system prompt and response schemas below come from the implemented "
        "formula constructor. The root and dependent-map examples are invented "
        "illustrations, not requests or scientific choices for the saved models.</p>"
        f"<h3>System</h3><pre>{html.escape(templates['system'])}</pre>"
        "<h3>Illustrative root and dependent-map examples</h3>"
        f"{_html_json(templates['illustrative_examples'])}"
        "<h3>Schemas</h3>"
        f"<pre>{html.escape(json.dumps(templates['response_schemas'], indent=2))}</pre>"
        "</section>"
    )
    path = root / "PROMPTS.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Initialization prompts</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}dt{font-weight:600}"
        "dd{margin:0 0 1rem 1rem}li{margin-bottom:.5rem}"
        "h2{border-top:1px solid #bbb;padding-top:1.5rem}</style>"
        "<h1>Initialization dependency prompts</h1><p>Offline preview; no LLM calls "
        "or fitting. Prefix duration remains unresolved.</p>"
        "<h2>Exact saved dependency requests</h2>"
        "<p>The case sections contain the exact initial messages and provider "
        "response schemas used by this diagnostic, formatted for reading.</p>"
        + "".join(sections)
    )
    return {"preview": str(path), "cases": len(plan["cases"]), "llm_calls": 0}


def report(root: Path) -> dict:
    """Separate structural status, conditional evidence and all historical charges."""
    plan = verify(root)
    rows, records = [], []
    for entry in plan["cases"]:
        case = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
        directory = root / "results" / entry["case_id"]
        current = _calls(root, entry, plan)
        records.extend(current)
        result = _terminal(directory / "result.json")
        error = (
            sealed_read(directory / "execution_error.json")
            if (directory / "execution_error.json").exists()
            else None
        )
        assessment = (result or {}).get("assessment") or {}
        diagnostics = [
            row
            for name in (
                "declared_report",
                "initial_time_report",
                "retained_witness_report",
            )
            for row in assessment.get(name, {}).get("diagnostics", [])
        ]
        checks = assessment.get("retained_witness_report", {}).get("checks", [])
        rows.append(
            {
                "case_id": entry["case_id"],
                "seed": entry["seed"],
                "status": result["status"]
                if result
                else error["status"]
                if error
                else "pending",
                "cost": _cost(current),
                "historical_source_model_cost": case["historical_cost"],
                "predecessor_historical_cost": case["predecessor_historical_cost"],
                "conditional_conflicts": sum(
                    row["status"] == "conditional_conflict" for row in diagnostics
                ),
                "retained_unresolved_witness_checks": sum(
                    check.get("retained_unresolved", False) for check in checks
                ),
                "trajectory_roots": sorted(
                    {
                        root
                        for deps in assessment.get(
                            "initialization_dependency_summary", {}
                        ).values()
                        for root in deps["trajectory_roots"]
                    }
                ),
                "scientific_correctness_established": False,
            }
        )
    counts = dict(Counter(row["status"] for row in rows))
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status": "pending" if counts.get("pending") else "complete",
        "availability_status": "unavailable"
        if counts.get("unavailable")
        else "available",
        "status_counts": counts,
        "rows": rows,
        "cost": _cost(records),
        "historical_source_diagnostic_cost": plan["source_diagnostic_summary"].get(
            "cost"
        ),
        "cost_scope": "New diagnostic costs only in cost; historical source model, "
        "predecessor and source diagnostic costs remain separate "
        "and are never refunded.",
        "maximum_calls": 2 * len(rows),
        "maximum_charged_tokens": 262144 * len(rows),
        "qualification": "structural_only",
        "prefix_window_policy": "unresolved_fitting_hyperparameter",
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "scientific_correctness_established": False,
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Saved initialization dependency diagnostic",
        "",
        f"Status: {summary['status']}; availability: {summary['availability_status']}.",
        "",
        "Structural acceptance does not establish a scientific repair. "
        "Prefix duration remains a fitting hyperparameter.",
        "",
        "| Case | Status | New requests | New observed tokens | "
        "Unresolved saved checks |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['case_id']} | {row['status']} | {row['cost']['physical_requests']} | "
        f"{row['cost']['observed_tokens']} | "
        f"{row['retained_unresolved_witness_checks']} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            summary["cost_scope"],
            "",
            "Full historical source: `historical/source.tar.gz`. "
            "Exact new requests/results: `results/`. Prompt preview: `PROMPTS.html`.",
        ]
    )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


def inspect(root: Path) -> dict:
    """Package review artifacts and the unchanged compressed historical source."""
    summary = report(root)
    preview(root)
    archive = root / "inspection.tar.gz"
    included = 0
    with tarfile.open(archive, "w:gz") as output:
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
            path = root / name
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                if item.is_symlink():
                    raise ValueError("inspection does not follow symlinks")
                if item.is_file():
                    output.add(
                        item, arcname=str(item.relative_to(root)), recursive=False
                    )
                    included += 1
    return {"archive": str(archive), "files": included, "status": summary["status"]}


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
    args = parser.parse_args()
    with _lock(args.root / "campaign-lock"):
        if args.command == "prepare":
            if args.source is None:
                parser.error("prepare requires --source")
            result = prepare(args.source, args.root)
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
