#!/usr/bin/env python3
"""Restore promised process gains and retry only saved failed initializer maps."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
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
    visible_response,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research.interaction_fit_handoff import verify_handoff
from autoformalism.research.interaction_source import _Reader, _sealed, load_sources
from autoformalism.schemas import CandidateModel
from autoformalism.search.construction_graph_formulas import (
    RHS_ONLY_POLICY,
    GraphMapReply,
    _formula_repair_feedback,
    construct_graph_formulas,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    GraphDependencyRecord,
)
from autoformalism.search.scoped_process_gains import (
    POLICY,
    build_scoped_gain_handoff,
    lower_saved_function_gains,
    rebind_saved_initializer,
    verify_scoped_gain_handoff,
)
from autoformalism.search.training_evidence import TrainingEvidence, evidence_brief
from autoformalism.staged_topology import content_hash
from scripts import construction_functions as original

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "saved-function-handoff-repair-1"
BUDGETS = {
    "maximum_calls_per_case": 2,
    "maximum_charged_tokens_per_case": 65536,
    "maximum_output_tokens": 8192,
    "attempts_per_step": 2,
}
ENTRYPOINTS = (
    "scripts/repair_function_handoff.py",
    "scripts/submit_function_handoff_repair_aces.py",
    "scripts/hpc/start_function_handoff_repair_aces.sh",
    "scripts/hpc/run_function_handoff_repair_aces.sh",
    "scripts/hpc/run_construction_functions_aces.sh",
    "scripts/construction_functions.py",
    "scripts/submit_review_continuation.py",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
)
file_hash = original.file_hash


class _MemoryReader(_Reader):
    """Validate the archive first, then read its regular members in one pass."""

    def __init__(self, source: Path):
        super().__init__(source)
        self.contents = {}
        with tarfile.open(fileobj=io.BytesIO(self.blob), mode="r:gz") as archive:
            for member in archive:
                if member.isfile():
                    self.contents[member.name] = archive.extractfile(member).read()

    def read(self, name: str) -> bytes:
        if name not in self.contents:
            raise ValueError(f"missing regular source member: {name}")
        return self.contents[name]


def _records(reader: _MemoryReader, entry: dict, plan: dict, *, tokens=False) -> list:
    prefix = f"results/{entry['case_id']}/calls" + ("/tokenization" if tokens else "")
    records = []
    for name in sorted(reader.files(prefix)):
        record = reader.json(name)
        request = record["request"]
        if (
            Path(name).stem != content_hash(request)
            or request.get("namespace") != entry["namespace"]
        ):
            raise ValueError("historical request identity differs")
        if tokens:
            if (
                request.get("policy") != "response-token-preflight-1"
                or request.get("model_revision") != plan["settings"]["model_revision"]
            ):
                raise ValueError("historical tokenizer identity differs")
        elif (
            record.get("request_hash") != content_hash(request)
            or request.get("settings") != plan["settings"]
            or request["body"].get("seed")
            != int(
                content_hash([entry["seed"], record["step"], record["attempt"]])[:8], 16
            )
            % (2**31)
            or type(record.get("budget_charge")) is not int
            or record["budget_charge"] < 0
        ):
            raise ValueError("historical request charges or settings differ")
        records.append(record)
    if not tokens and len(records) > plan["settings"]["maximum_requests"]:
        raise ValueError("historical request ceiling exceeded")
    return records


def load_history(source: Path) -> dict:
    """Validate sealed historical receipts and rebuild the original public inputs."""
    from autoformalism.expressions import ModelValidationError
    from autoformalism.expressions.parser import RestrictedParser
    from autoformalism.llm.staged_topology import visible_response
    from autoformalism.search.construction_graph_formulas import GraphMapReply

    reader = _MemoryReader(source)
    plan = reader.json("plan.json", "artifact_sha256")
    if plan["protocol"] != original.PROTOCOL or plan["policies"] != original.POLICIES:
        raise ValueError("unsupported historical function campaign")
    blob = reader.read("historical/source.tar.gz")
    if hashlib.sha256(blob).hexdigest() != plan["source_archive_sha256"]:
        raise ValueError("historical nested source differs")
    loaded = load_sources(blob)
    if (
        loaded["source_plan"]["artifact_sha256"] != plan["source_plan_sha256"]
        or loaded["source_summary"] != plan["source_summary"]
        or loaded["roster"] != plan["roster"]
    ):
        raise ValueError("historical initializer identity differs")
    cases, records = {}, []
    for entry in plan["cases"]:
        name = entry["case_id"]
        if name in cases:
            raise ValueError("duplicate historical case")
        value = loaded["cases"][name]
        identity = {
            "protocol": original.PROTOCOL,
            "case_id": name,
            "input_sha256": content_hash(value),
            "source_result_sha256": value["result"]["result_sha256"],
        }
        expected = {
            **identity,
            "source_model_id": value["entry"]["source_model_id"],
            "seed": value["entry"]["seed"],
            "namespace": f"{original.PROTOCOL}:{content_hash(identity)}",
            "initializer_rules": value["result"].get("handoff", {}).get("records", []),
        }
        if entry != expected or reader.json(
            f"inputs/{name}.json", "artifact_sha256"
        ) != {**value, "artifact_sha256": content_hash(value)}:
            raise ValueError("historical public input differs")
        calls = _records(reader, entry, plan)
        tokens = _records(reader, entry, plan, tokens=True)
        receipt = reader.json(f"results/{name}/completion.json", "artifact_sha256")
        if (
            receipt["plan_sha256"] != plan["artifact_sha256"]
            or receipt["input_sha256"] != entry["input_sha256"]
            or receipt["calls_sha256"] != content_hash(calls)
            or receipt["tokenizations_sha256"] != content_hash(tokens)
            or "result.json" not in receipt["files"]
            or any(
                key not in {"result.json", "fit_handoff.json"}
                for key in receipt["files"]
            )
        ):
            raise ValueError("historical completion differs or is unavailable")
        for filename, digest in receipt["files"].items():
            if (
                hashlib.sha256(reader.read(f"results/{name}/{filename}")).hexdigest()
                != digest
            ):
                raise ValueError("historical completed artifact differs")
        result = reader.json(f"results/{name}/result.json")
        if result["status"] != receipt["status"]:
            raise ValueError("historical result status differs")
        if "fit_handoff.json" in receipt["files"]:
            verify_handoff(
                result,
                reader.json(f"results/{name}/fit_handoff.json", "artifact_sha256"),
                provenance=original._provenance(plan, entry, value),
            )
        cost = _cost(calls)
        if cost["budget_charge"] > plan["settings"]["maximum_total_tokens"] or any(
            result.get(result_key) != cost[cost_key]
            for result_key, cost_key in (
                ("physical_requests", "physical_requests"),
                ("budget_charge", "budget_charge"),
                ("observed_total_tokens", "observed_tokens"),
                ("unmeasured_requests", "requests_with_unknown_usage"),
            )
        ):
            raise ValueError("historical function cost or token ceiling differs")
        initializer = result.get("initialization")
        rejected_initializer_choices = []
        checkpoint = reader.json(
            f"results/{name}/initialization_graph/state.json", "state_sha256"
        )
        # Legacy successful formula results have no status field. Exhausted
        # legacy maps leave a running checkpoint and initialization=None.
        if result["status"] == "complete" and isinstance(initializer, dict):
            _sealed(initializer, "result_sha256")
            if (
                not result.get("complete_model")
                or "fit_handoff.json" not in receipt["files"]
                or checkpoint.get("status") != "complete"
                or checkpoint.get("result") != initializer
                or checkpoint.get("identity") != initializer.get("identity")
                or checkpoint.get("choices") != initializer.get("choices")
                or checkpoint.get("attempts") != initializer.get("attempts")
            ):
                raise ValueError("historical complete initializer checkpoint differs")
            action = "migrate_saved_initializer"
        elif result["status"] == "failed" and initializer is None:
            graph = value["function_source"]["initialization_graph_handoff"]["records"]
            if (
                len(graph) != 1
                or graph[0]["rule"]["mode"] != "map"
                or result.get("complete_model") is not False
                or "fit_handoff.json" in receipt["files"]
                or checkpoint.get("status") != "running"
                or checkpoint.get("choices") != {}
                or checkpoint.get("result") is not None
            ):
                raise ValueError("historical failed initializer graph differs")
            state_name = graph[0]["state"]
            attempts = checkpoint.get("attempts", [])
            if (
                result.get("error")
                != f"bounded graph formula repair exhausted for {state_name}"
                or len(attempts) != plan["settings"]["attempts_per_step"]
            ):
                raise ValueError("historical initializer was not exhausted")
            graph_calls = {
                record["request_hash"]: record
                for record in calls
                if record["step"].startswith("graph_initial_")
            }
            if len(graph_calls) != len(attempts):
                raise ValueError("historical initializer attempts differ from calls")
            used = set()
            public_brief = value["brief"]
            if value["training_evidence"] is not None:
                public_brief = {
                    **public_brief,
                    "training_observations": value["training_evidence"],
                }
            for index, attempt in enumerate(attempts):
                key = attempt.get("request_hash")
                record = graph_calls.get(key)
                if (
                    key in used
                    or record is None
                    or record["status"] != "responded"
                    or record["step"] != f"graph_initial_{state_name}"
                    or record["attempt"] != index
                    or attempt.get("attempt") != index
                    or attempt.get("state") != state_name
                    or attempt.get("accepted") is not False
                ):
                    raise ValueError(
                        "historical initializer attempt provenance differs"
                    )
                used.add(key)
                message = record["request"]["body"]["messages"][1]["content"]
                packet = json.loads(message.split("\n", 1)[1])
                if (
                    packet.get("construction_identity") != checkpoint.get("identity")
                    or packet.get("selected_state") != state_name
                    or packet.get("selected_initializer_dependency") != graph[0]
                    or packet.get("initializer_graph") != graph
                    or packet.get("public_brief") != public_brief
                    or packet.get("accepted_formulas") != {}
                    or packet.get("diagnostics") != attempts[max(0, index - 1) : index]
                ):
                    raise ValueError("historical initializer request graph differs")
                choice = GraphMapReply.model_validate(visible_response(record))
                try:
                    RestrictedParser().parse(
                        choice.expression, location=f"initial:{state_name}"
                    )
                except ModelValidationError as exc:
                    if (
                        not str(exc).startswith("SYNTAX_ERROR at ")
                        or attempt.get("error") != str(exc)[:2000]
                    ):
                        raise ValueError(
                            "historical initializer rejection differs"
                        ) from exc
                else:
                    raise ValueError("historical initializer is not a syntax failure")
                rejected_initializer_choices.append(choice.model_dump(mode="json"))
            action = "retry_failed_initializer"
        else:
            raise ValueError(
                "source does not contain one exhausted initializer or a complete graph"
            )
        cases[name] = {
            "source": value,
            "result": result,
            "action": action,
            "initializer_checkpoint": checkpoint,
            "rejected_initializer_choices": rejected_initializer_choices,
        }
        records.extend(calls)
    if set(cases) != set(loaded["cases"]) or not 1 <= len(cases) <= 8:
        raise ValueError("historical source roster differs")
    if sum(v["action"] == "retry_failed_initializer" for v in cases.values()) > 3:
        raise ValueError("at most three failed initializer maps may be retried")
    return {"plan": plan, "cases": cases, "cost": _cost(records)}


def _prepared(source: Path) -> tuple[dict, dict]:
    historical = load_history(source)
    settings = StagedModelSettings.model_validate(
        historical["plan"]["settings"]
    ).model_copy(
        update={
            "maximum_requests": 2,
            "maximum_total_tokens": 65536,
            "attempts_per_step": 2,
            "max_output_tokens": 8192,
        }
    )
    cases = []
    for name, value in historical["cases"].items():
        identity = {
            "protocol": PROTOCOL,
            "case_id": name,
            "input_sha256": content_hash(value),
        }
        cases.append(
            {
                **identity,
                "seed": value["source"]["entry"]["seed"],
                "source_model_id": value["source"]["entry"]["source_model_id"],
                "action": value["action"],
                "namespace": f"{PROTOCOL}:{content_hash(identity)}",
            }
        )
    plan = {
        "protocol": PROTOCOL,
        "source_archive_sha256": file_hash(source),
        "source_plan_sha256": historical["plan"]["artifact_sha256"],
        "historical_function_cost": historical["cost"],
        "historical_initializer_cost": historical["plan"]["source_summary"]["cost"],
        "source_runtime_sha256": historical["plan"]["runtime_source_sha256"],
        "runtime_source_sha256": runtime_source_hash(),
        "entrypoints": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "cases": cases,
        "retry_case_ids": [
            e["case_id"] for e in cases if e["action"] == "retry_failed_initializer"
        ],
        "settings": settings.model_dump(mode="json"),
        "budgets": BUDGETS,
        "gain_policy": POLICY,
        "initializer_formula_policy": RHS_ONLY_POLICY,
        "prefix_window_policy": "unresolved_fitting_hyperparameter",
        "selection": (
            "all source cases retained; only failed initializer formulas retried"
        ),
        "dynamics_generation_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "automatic_followup": False,
    }
    return plan, historical["cases"]


def prepare(source: Path, root: Path) -> dict:
    """Seal a fresh derivative campaign; historical requests are never overwritten."""
    if source.is_symlink() or not source.is_file():
        raise ValueError("source must be a regular inspection archive")
    plan, cases = _prepared(source)
    if (root / "results").exists() and not (root / "plan.json").exists():
        raise ValueError("unbound results exist")
    frozen = root / "historical/source.tar.gz"
    if frozen.exists() and file_hash(frozen) != plan["source_archive_sha256"]:
        raise ValueError("source differs; use a fresh root")
    sealed = sealed_write(root / "plan.json", plan)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        if file_hash(temporary) != plan["source_archive_sha256"]:
            raise ValueError("source changed while copying")
        temporary.replace(frozen)
    for name, value in cases.items():
        sealed_write(root / "inputs" / f"{name}.json", value)
    _migrate(root, sealed)
    return sealed


def verify(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    expected, cases = _prepared(root / "historical/source.tar.gz")
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("repair source, runtime or plan differs")
    for name, value in cases.items():
        if sealed_read(root / "inputs" / f"{name}.json") != {
            **value,
            "artifact_sha256": content_hash(value),
        }:
            raise ValueError("repair input differs")
    return plan


def _lower(value: dict) -> dict:
    return lower_saved_function_gains(value["result"], value["source"], policy=POLICY)


def _provenance(plan: dict, entry: dict) -> dict:
    return {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "source_plan_sha256": plan["source_plan_sha256"],
        "case_id": entry["case_id"],
        "input_sha256": entry["input_sha256"],
        "action": entry["action"],
        "prefix_window_policy": plan["prefix_window_policy"],
        "dynamics_reused": True,
    }


def _save(root: Path, plan: dict, entry: dict, lowered: dict, result: dict) -> None:
    directory = root / "results" / entry["case_id"]
    atomic_json(directory / "result.json", result)
    names = ["result.json"]
    if result.get("result_sha256"):
        handoff = build_scoped_gain_handoff(
            lowered, result, provenance=_provenance(plan, entry)
        )
        sealed_write(
            directory / "fit_handoff.json",
            {k: v for k, v in handoff.items() if k != "artifact_sha256"},
        )
        names.append("fit_handoff.json")
    _complete(root, plan, entry, "complete", names)


def _complete(
    root: Path, plan: dict, entry: dict, status: str, names: list[str]
) -> None:
    directory = root / "results" / entry["case_id"]
    sealed_write(
        directory / "completion.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "input_sha256": entry["input_sha256"],
            "calls_sha256": content_hash(original._calls(root, entry, plan)),
            "tokenizations_sha256": content_hash(
                original._tokenizations(root, entry, plan)
            ),
            "status": status,
            "files": {name: file_hash(directory / name) for name in names},
        },
    )


def _migrate(root: Path, plan: dict) -> None:
    for entry in plan["cases"]:
        if entry["action"] != "migrate_saved_initializer":
            continue
        calls = original._calls(root, entry, plan)
        if calls:
            raise ValueError("offline migration cannot have new provider calls")
        if original._receipt(root, entry, plan, calls):
            continue
        value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
        lowered = _lower(value)
        _save(
            root,
            plan,
            entry,
            lowered,
            rebind_saved_initializer(lowered, value["result"]["initialization"]),
        )


def _construct(value: dict, lowered: dict, client, directory: Path) -> dict:
    source = value["source"]
    brief = evidence_brief(
        source["brief"],
        ValidationContext.model_validate(source["case"]["context"]),
        TrainingEvidence.model_validate(source["training_evidence"])
        if source["training_evidence"] is not None
        else None,
    )
    return construct_graph_formulas(
        CandidateModel.model_validate(lowered["base_candidate"]),
        ValidationContext.model_validate(lowered["context"]),
        brief,
        GraphConstructionContract.model_validate(
            lowered["initialization_graph_contract"]
        ),
        GraphDependencyHandoff.model_validate(lowered["initialization_graph_handoff"]),
        client,
        directory / "initializer",
        state_aliases=lowered["state_aliases"],
        formula_policy=RHS_ONLY_POLICY,
    )


def run(
    root: Path,
    base_url: str,
    runtime_record: Path | None = None,
    *,
    seconds=1800,
    transport=None,
    token_transport=None,
) -> dict:
    """Retry only three initializer maps under new, independently charged ceilings."""
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
        raise ValueError("repair server must be a loopback HTTP endpoint")
    if runtime_record is not None:
        from scripts import submit_function_handoff_repair_aces as submitter
        from scripts.saved_initialization_aces import bind_server

        _, execution = submitter.verify(root)
        runtime = json.loads(runtime_record.read_text())
        keys = (
            "artifact_sha256",
            "model",
            "model_revision",
            "image_sha256",
            "served_context_tokens",
            "tensor_parallel_size",
            "maximum_sequences",
        )
        if runtime.get("serving_identity") != {key: execution[key] for key in keys}:
            raise ValueError("serving runtime differs from frozen execution")
        bind_server(root, runtime_record, base_url)
    elif transport is None:
        raise ValueError("live generation requires a verified runtime record")
    _migrate(root, plan)
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
            if original._receipt(root, entry, plan, original._calls(root, entry, plan)):
                continue
            if entry["action"] != "retry_failed_initializer":
                raise ValueError("only failed initializer maps may generate")
            directory = root / "results" / entry["case_id"]
            value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
            lowered = _lower(value)
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
                result = _construct(value, lowered, client, directory)
                _save(root, plan, entry, lowered, result)
            except DeferredCall:
                break
            except (ValueError, RuntimeError, OSError) as exc:
                checkpoint = directory / "initializer/state.json"
                status = "unavailable"
                if checkpoint.exists():
                    saved = _sealed(json.loads(checkpoint.read_text()), "state_sha256")
                    if saved.get("status") == "failed":
                        status = "failed"
                sealed_write(
                    directory / "execution_error.json",
                    {
                        "status": status,
                        "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
                        "automatic_retry": False,
                    },
                )
                _complete(root, plan, entry, status, ["execution_error.json"])
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return report(root)


class _PreviewClient(original._PreviewClient):
    """Use the live initializer identity while stopping before any HTTP request."""

    def __init__(self, settings: dict, entry: dict):
        super().__init__(settings)
        self.namespace = entry["namespace"]
        self.seed = entry["seed"]

    def request_policy(self) -> dict:
        return {"budget_reservation_policy": MEASURED_RESERVATION}


def preview(root: Path) -> dict:
    plan = verify(root)
    sections = []
    for entry in plan["cases"]:
        if entry["action"] != "retry_failed_initializer":
            continue
        value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
        client = _PreviewClient(plan["settings"], entry)
        with suppress(DeferredCall):
            _construct(
                value, _lower(value), client, root / "preview" / entry["case_id"]
            )
        if client.packet is None:
            raise ValueError("initializer preview did not reach a request")
        atomic_json(root / "preview" / f"{entry['case_id']}.json", client.packet)
        examples = []
        for choice in value.get("rejected_initializer_choices", []):
            record = GraphDependencyRecord.model_validate(
                value["source"]["function_source"]["initialization_graph_handoff"][
                    "records"
                ][0]
            )
            examples.append(_formula_repair_feedback(record, choice))
        if examples:
            sections.append(
                "<h2>Updated feedback for saved rejected replies</h2>"
                "<p>Offline illustration using actual historical replies; "
                "not a new response or an automatically adopted correction.</p><pre>"
                + html.escape(json.dumps(examples, indent=2))
                + "</pre>"
            )
        sections.append(
            f"<h2>{html.escape(entry['case_id'])}</h2>"
            + "".join(
                f"<h3>{html.escape(key)}</h3><pre>{html.escape(text)}</pre>"
                for key, text in (
                    ("System", client.packet["system"]),
                    ("User", client.packet["user"]),
                    ("Schema", json.dumps(client.packet["response_schema"], indent=2)),
                )
            )
        )
    (root / "PROMPTS.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Initializer formula repair</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Saved initializer repair and gain handoff</h1>"
        "<p>Changed: JSON expression is the right-hand side only. "
        "Assignment errors name the rejected expression, state and required aliases. "
        "No automatic expression stripping. "
        "Existing proposed laws, topology and initializer dependencies/scopes "
        "are retained; runtime consumer gains are added. "
        "Five successful graphs migrate offline; "
        "only three failed maps may generate, at most two calls each. "
        "No fitting or test access.</p>"
        "<p>Exact first retry requests follow. Later error feedback depends "
        "on the returned expression.</p>" + "".join(sections)
    )
    return {
        "preview": str(root / "PROMPTS.html"),
        "retry_cases": len(plan["retry_case_ids"]),
        "llm_calls": 0,
    }


def _verify_result(
    directory: Path,
    entry: dict,
    value: dict,
    lowered: dict,
    result: dict,
    calls: list[dict],
) -> None:
    """Bind exported initializer choices to migrated or actually accepted replies."""
    if entry["action"] == "migrate_saved_initializer":
        if calls or result != rebind_saved_initializer(
            lowered, value["result"]["initialization"]
        ):
            raise ValueError("migrated initializer differs from saved graph")
        return
    checkpoint = _sealed(
        json.loads((directory / "initializer/state.json").read_text()), "state_sha256"
    )
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("result") != result
        or checkpoint.get("choices") != result["choices"]
        or checkpoint.get("attempts") != result["attempts"]
        or checkpoint.get("identity") != result["identity"]
    ):
        raise ValueError("initializer result differs from checkpoint")
    by_hash = {record["request_hash"]: record for record in calls}
    accepted = {}
    for event in result["attempts"]:
        record = by_hash.get(event["request_hash"])
        if (
            record is None
            or record["attempt"] != event["attempt"]
            or record["step"] != f"graph_initial_{event['state']}"
        ):
            raise ValueError("initializer attempt differs from cached request")
        packet = json.loads(
            record["request"]["body"]["messages"][1]["content"].split("\n", 1)[1]
        )
        if packet.get("construction_identity") != result["identity"]:
            raise ValueError("initializer request construction identity differs")
        if event["accepted"]:
            accepted[event["state"]] = GraphMapReply.model_validate(
                visible_response(record)
            ).model_dump(mode="json")
    if accepted != result["choices"]:
        raise ValueError("initializer choices differ from accepted replies")


def report(root: Path) -> dict:
    plan = verify(root)
    rows, records = [], []
    sections = []
    for entry in plan["cases"]:
        directory = root / "results" / entry["case_id"]
        calls = original._calls(root, entry, plan)
        records.extend(calls)
        receipt = original._receipt(root, entry, plan, calls)
        value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
        lowered = _lower(value)
        result, handoff = {}, {}
        if receipt and "result.json" in receipt["files"]:
            result = _sealed(
                json.loads((directory / "result.json").read_text()), "result_sha256"
            )
            _verify_result(directory, entry, value, lowered, result, calls)
        if receipt and "fit_handoff.json" in receipt["files"]:
            handoff = sealed_read(directory / "fit_handoff.json")
            verify_scoped_gain_handoff(
                lowered, result, handoff, provenance=_provenance(plan, entry)
            )
        status = receipt["status"] if receipt else "pending"
        error = result.get("error")
        if receipt and "execution_error.json" in receipt["files"]:
            error = sealed_read(directory / "execution_error.json")["error"]
        rows.append(
            {
                "case_id": entry["case_id"],
                "source_model_id": entry["source_model_id"],
                "proposal_seed": entry["seed"],
                "action": entry["action"],
                "status": status,
                "fitting_exported": bool(handoff),
                "readiness": handoff.get("readiness"),
                "gain_compilation": lowered["gain_compilation"],
                "error": error,
                "availability": original._availability(
                    status, error, calls, original._tokenizations(root, entry, plan)
                ),
                "cost": _cost(calls),
            }
        )
        model = lowered["base_candidate"]
        equations = [f"{p['name']} = {p['expression']}" for p in model["processes"]]
        equations += [
            f"d({e['state']})/dt = {e['rhs']}" for e in model["state_equations"]
        ]
        sections.append(
            f"<h2>{html.escape(entry['case_id'])}</h2>"
            f"<p>{status}; {entry['action']}</p><pre>"
            + html.escape("\n".join(equations))
            + "</pre><h3>Readiness and gain audit</h3><pre>"
            + html.escape(
                json.dumps(
                    {
                        "readiness": handoff.get("readiness"),
                        "gain_compilation": lowered["gain_compilation"],
                    },
                    indent=2,
                )
            )
            + "</pre>"
        )
    counts = dict(Counter(row["status"] for row in rows))
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "rows": rows,
        "status": "pending" if counts.get("pending") else "complete",
        "status_counts": counts,
        "availability_status": "unavailable"
        if any(r["availability"] == "unavailable" for r in rows)
        else "pending"
        if counts.get("pending")
        else "available",
        "cost": _cost(records),
        "historical_function_cost": plan["historical_function_cost"],
        "historical_initializer_cost": plan["historical_initializer_cost"],
        "maximum_calls": 2 * len(plan["retry_case_ids"]),
        "maximum_charged_tokens": 65536 * len(plan["retry_case_ids"]),
        "dynamics_generation_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "interpretation": (
            "Unfitted functions; corrected gains do not establish "
            "scientific accuracy. Prefix policy remains unresolved. "
            "Historical usage is retained once separately."
        ),
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Saved initializer repair and gain handoff",
        "",
        summary["interpretation"],
        "",
        "| Proposal | Action | Status | Fit packet | Calls | Tokens |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    lines += [
        f"| {r['case_id']} | {r['action']} | {r['status']} | "
        f"{r['fitting_exported']} | {r['cost']['physical_requests']} | "
        f"{r['cost']['observed_tokens']} |"
        for r in rows
    ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (root / "REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Gain handoff</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Saved gain handoff</h1><p>"
        + summary["interpretation"]
        + "</p>"
        + "".join(sections)
    )
    return summary


def inspect(root: Path) -> dict:
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
    parser.add_argument("--seconds", type=int, default=1800)
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
                "retry_case_ids": result["retry_case_ids"],
            }
        print(json.dumps(result, indent=2))
        if args.command == "run":
            if result["status"] != "complete":
                sys.exit(2)
            if result["availability_status"] != "available":
                sys.exit(1)


if __name__ == "__main__":
    main()
