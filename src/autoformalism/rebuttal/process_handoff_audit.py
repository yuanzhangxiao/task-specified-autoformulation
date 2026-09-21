"""Read-only saved-response audit of signed topology additions.

Each attempt is tested against its original request context. This does not
continue construction, invent unasked responses, refit, or promote a candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
    equation_reply_model,
)
from autoformalism.search import shared_process_contract as shared
from autoformalism.search import signed_processes as signed
from autoformalism.search.staged_topology_runner import (
    _validate_memory_equation_obligations,
)
from autoformalism.staged_topology import (
    content_hash,
    validate_equation,
    validate_inventory_revision,
)

PROTOCOL = "signed-process-handoff-audit-1"


def request_payload(record: dict) -> dict:
    """Extract the original JSON request; never execute provider text."""
    if content_hash(record["request"]) != record["request_hash"]:
        raise ValueError("saved request hash differs")
    text = record["request"]["body"]["messages"][1]["content"]
    return json.loads(text if text.startswith("{") else text.split("\n", 1)[1])


def replay_attempt(record: dict, event: dict, bindings: list[dict]) -> dict:
    """Check old/new schemas and the actual assembled equation in saved context."""
    row = {
        "step": event["step"],
        "attempt": event["attempt"],
        "request_hash": event["request_hash"],
        "historically_accepted": event["accepted"],
        "historical_error": event.get("error"),
        "old_schema_valid": False,
        "new_local_valid": False,
        "newly_valid": False,
        "whole_model_recovered": False,
    }
    try:
        payload = request_payload(record)
        raw = visible_response(record)
        brief = PublicScientificBrief.model_validate(
            {
                k: v
                for k, v in payload["public_brief"].items()
                if k != "training_observations"
            }
        )
        inventory = tuple(
            ScientificVariable.model_validate(v) for v in payload["frozen_inventory"]
        )
        selected = next(
            v for v in inventory if v.name == payload["selected_lhs"]["name"]
        )
        required = shared.requirements(bindings, selected.name)
        if required != payload.get("required_process_contributions", []):
            raise ValueError("saved process requirements differ from the bindings")
        row["runtime_term_count"] = len(required)
        row["empty_additions"] = (
            isinstance(raw, dict)
            and raw.get("terms") == []
            and raw.get("inventory_revision") is None
        )
        old = equation_reply_model(
            tuple(payload["allowed_sources"]),
            maximum_terms=brief.limits.terms_per_equation,
        )
        try:
            old.model_validate(raw)
            row["old_schema_valid"] = True
        except ValueError as exc:
            row["old_schema_error"] = str(exc)[:4000]
        new = equation_reply_model(
            tuple(payload["allowed_sources"]),
            maximum_terms=brief.limits.terms_per_equation,
            runtime_supplies_terms=bool(required),
        )
        reply = new.model_validate(raw)
        if reply.inventory_revision is not None:
            validate_inventory_revision(brief, inventory, reply.inventory_revision)
            row["new_outcome"] = "inventory_revision_requested"
        else:
            definition = EquationDefinition(
                name=selected.name,
                definition=selected.definition,
                terms=signed.assemble(bindings, selected.name, reply.terms),
            )
            previous = tuple(
                EquationDefinition.model_validate(e)
                for e in payload["current_equation_sketch"]
            )
            shared.validate_equation_uses(bindings, definition)
            validate_equation(inventory, previous, definition, brief.limits)
            memory = {}
            for obligation in payload["agenda"].get("public_obligations", []):
                if obligation["kind"] == "required_memory_input":
                    memory[obligation["requirement_id"]] = {selected.name}
                elif obligation.get("candidate_memory_mediators"):
                    memory[obligation["requirement_id"]] = set(
                        obligation["candidate_memory_mediators"]
                    )
            _validate_memory_equation_obligations(
                brief, selected, previous, definition, memory
            )
            row.update(
                new_outcome="equation",
                assembled_equation=definition.model_dump(mode="json"),
            )
        row["new_local_valid"] = True
        row["newly_valid"] = not event["accepted"]
    except (ValueError, TypeError, KeyError, StopIteration) as exc:
        row["new_error"] = str(exc)[:4000]
    return row


def _read(path: Path, source: Path) -> dict:
    """Restrict reads to frozen source paths, including symlink resolution."""
    if not path.resolve().is_relative_to(source):
        raise ValueError("source artifact escapes campaign")
    return json.loads(path.read_text())


def source_files(source: Path, plan: dict) -> list[Path]:
    """Enumerate only public campaign metadata, constructions, calls and results."""
    paths = {source / "plan.json"}
    for task in plan["tasks"]:
        for key in (task["task_id"], task["construction_task"]["task_id"]):
            if not re.fullmatch(r"[A-Za-z0-9_]+", key):
                raise ValueError("invalid task path")
        directory = source / "results" / task["task_id"]
        paths.update(
            directory / name
            for name in ("proposal.json", "result.json")
            if (directory / name).exists()
        )
        common = source / "construction/results" / task["construction_task"]["task_id"]
        if (common / "proposal.json").exists():
            paths.add(common / "proposal.json")
        for route in ("review", "fallback"):
            stage = common / route / "topology_stage.json"
            if stage.exists():
                paths.add(stage)
                value = _read(stage, source)
                for event in value["result"]["events"]:
                    if "request_hash" not in event:
                        continue
                    if not re.fullmatch(r"[0-9a-f]{64}", event["request_hash"]):
                        raise ValueError("invalid request hash")
                    record = common / "calls" / (event["request_hash"] + ".json")
                    if record.exists():
                        paths.add(record)
    return sorted(paths)


def audit(source: Path, output: Path) -> dict:
    """Freeze read identities, checkpoint per construction, and publish diagnostics."""
    from autoformalism.rebuttal.process_equation_diagnostics import diagnose

    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("audit output must be separate from source")
    _read(source / "plan.json", source)
    plan = sealed_read(source / "plan.json")
    if plan["protocol"] != "detention-process-pilot-3":
        raise ValueError("expected saved gain-policy pilot")
    paths = source_files(source, plan)
    digests = {}
    for path in paths:
        _read(path, source)
        digests[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    identity = {
        "protocol": PROTOCOL,
        "source": str(source),
        "source_plan_sha256": plan["artifact_sha256"],
        "runtime_sha256": runtime_source_hash(),
        "files": digests,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
        "trajectory_values_used": False,
    }
    with public._lock(output):
        freeze = output / "freeze.json"
        if freeze.exists():
            saved = sealed_read(freeze)
            if {k: v for k, v in saved.items() if k != "artifact_sha256"} != identity:
                raise ValueError("audit source or runtime changed; use a new output")
        else:
            sealed_write(freeze, identity)
        constructions = []
        common_tasks = {t["construction_task"]["task_id"] for t in plan["tasks"]}
        for name in sorted(common_tasks):
            checkpoint = output / "checkpoints" / (name + ".json")
            if checkpoint.exists():
                constructions.append(sealed_read(checkpoint))
                continue
            base = source / "construction/results" / name
            common = (
                sealed_read(base / "proposal.json")
                if (base / "proposal.json").exists()
                else {}
            )
            attempts = []
            for route in ("review", "fallback"):
                stage = base / route / "topology_stage.json"
                if not stage.exists():
                    continue
                topology = sealed_read(stage)["result"]
                bindings = (topology.get("shared_process_contract") or {}).get(
                    "bindings", []
                )
                for event in topology["events"]:
                    if not event["step"].startswith("equation_"):
                        continue
                    if "request_hash" not in event:
                        attempts.append(
                            {
                                "route": route,
                                "step": event["step"],
                                "attempt": event.get("attempt"),
                                "status": "missing_request_hash",
                                "newly_valid": False,
                            }
                        )
                        continue
                    call = base / "calls" / (event["request_hash"] + ".json")
                    if call.exists():
                        row = replay_attempt(_read(call, source), event, bindings)
                    else:
                        row = {
                            "step": event["step"],
                            "attempt": event["attempt"],
                            "status": "missing_record",
                            "newly_valid": False,
                        }
                    attempts.append({"route": route, **row})
            row = {
                "task": name,
                "historical_status": common.get("status", "missing"),
                "historical_fallback": common.get("fallback_used"),
                "attempts": attempts,
                "newly_valid_attempts": sum(a["newly_valid"] for a in attempts),
                "whole_model_recovered": False,
            }
            constructions.append(sealed_write(checkpoint, row))
        models = []
        for task in plan["tasks"]:
            base = source / "results" / task["task_id"]
            proposal = (
                sealed_read(base / "proposal.json")
                if (base / "proposal.json").exists()
                else {}
            )
            result = (
                sealed_read(base / "result.json")
                if (base / "result.json").exists()
                else {}
            )
            if result and result["proposal_sha256"] != proposal.get("artifact_sha256"):
                raise ValueError("fit and proposal identity differ")
            bundle = proposal.get("bundle")
            models.append(
                {
                    "task": task["task_id"],
                    "status": result.get("status", "missing"),
                    "fit": {
                        k: (result.get("fit") or {}).get(k)
                        for k in (
                            "message",
                            "actual_residual_calls",
                            "budget_exhausted",
                        )
                    },
                    "diagnostics": diagnose(
                        bundle,
                        (result.get("fit") or {}).get("parameters"),
                        plan["cells"][task["case"]],
                    )
                    if bundle
                    else None,
                }
            )
        # Detect concurrent source changes before publishing a supposedly fixed audit.
        if source_files(source, plan) != paths or any(
            hashlib.sha256(p.read_bytes()).hexdigest()
            != digests[str(p.relative_to(source))]
            for p in paths
        ):
            raise ValueError("source changed during audit")
        summary = {
            "protocol": PROTOCOL,
            "identity": content_hash(identity),
            "historical_status_counts": dict(Counter(m["status"] for m in models)),
            "constructions": constructions,
            "models": models,
            "newly_valid_attempts": sum(
                c["newly_valid_attempts"] for c in constructions
            ),
            "unavailable_saved_attempts": sum(
                a.get("status") in {"missing_record", "missing_request_hash"}
                for c in constructions for a in c["attempts"]
            ),
            "previously_accepted_now_blocked": [
                {"task": c["task"], **a}
                for c in constructions
                for a in c["attempts"]
                if a.get("historically_accepted") and not a.get("new_local_valid")
            ],
            "llm_calls": 0,
            "optimizer_calls": 0,
            "whole_models_recovered": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
        sealed_write(output / "summary.json", summary)
        lines = [
            "# Saved process-handoff audit",
            "",
            "Original requests and replies; local equation checks only. "
            "No LLM calls, fitting or model promotion.",
            "",
            "| Construction | Historical status | Newly valid attempts |",
            "| --- | --- | ---: |",
        ]
        lines += [
            f"| {c['task']} | {c['historical_status']} | {c['newly_valid_attempts']} |"
            for c in constructions
        ]
        lines += [
            "",
            "Previously accepted attempts now blocked: "
            f"{len(summary['previously_accepted_now_blocked'])}",
            f"Unavailable saved attempts: {summary['unavailable_saved_attempts']}",
            "",
            "Equation diagnostics are advisory facts, not complete mechanism "
            "certification. See summary.json for exact expressions, duplicate laws, "
            "threshold structure, boundary-covariate use, and transfer cancellation.",
        ]
        (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        return summary
