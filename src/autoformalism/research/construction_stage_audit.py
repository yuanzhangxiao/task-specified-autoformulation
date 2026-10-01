"""Read saved construction decisions without replaying or changing the models.

This is an evidence index, not a scientific grader. In particular, prose is
displayed verbatim and never used to infer mechanism assignments.
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas.staged_topology import VariableReply
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-construction-stage-audit-1"
PROCESS_ERROR = "process uses are fixed; return only other equation terms"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _events(directory: Path) -> dict[str, list[dict]]:
    """Deduplicate decisions copied into progress, stage and result artifacts."""
    found: dict[str, dict[str, dict]] = defaultdict(dict)

    def visit(value: object, source: str) -> None:
        if isinstance(value, dict):
            key = value.get("request_hash")
            if isinstance(key, str) and "accepted" in value:
                record = found[key].setdefault(
                    content_hash(value), {"event": value, "files": []}
                )
                record["files"].append(source)
            for child in value.values():
                visit(child, source)
        elif isinstance(value, list):
            for child in value:
                visit(child, source)

    for path in sorted((directory / "construction").rglob("*.json")):
        visit(_read(path), str(path.relative_to(directory)))
    return {key: list(records.values()) for key, records in found.items()}


def _calls(directory: Path, identity: str) -> list[dict]:
    events = _events(directory)
    calls = []
    for path in sorted((directory / "calls").glob("*.json")):
        raw = _read(path)
        if (
            raw["request_hash"] != path.stem
            or content_hash(raw["request"]) != path.stem
            or raw["request"]["namespace"] != identity
        ):
            raise ValueError(f"call provenance differs: {path}")
        decisions = events.get(path.stem, [])
        accepted = {r["event"]["accepted"] for r in decisions}
        if len(accepted) > 1:
            raise ValueError(f"contradictory saved acceptance: {path.stem}")
        calls.append(
            {
                "request_hash": path.stem,
                "step": raw["step"],
                "attempt": raw["attempt"],
                "status": raw["status"],
                "request": raw["request"]["body"],
                "response": raw.get("raw_response"),
                "accepted": next(iter(accepted)) if accepted else None,
                "decisions": decisions,
            }
        )
    if set(events) - {c["request_hash"] for c in calls}:
        raise ValueError("saved decisions reference missing call records")
    return calls


def _payload(call: dict) -> dict:
    """Decode the runtime JSON displayed in the saved user message, not prose."""
    message = next(m for m in call["request"]["messages"] if m["role"] == "user")
    text = message["content"]
    payload, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    return payload


def _reply(call: dict) -> object:
    return visible_response(
        {"status": call["status"], "raw_response": call["response"]}
    )


def variable_call(call: dict) -> dict:
    """Separate schema validity, runtime completion, and existing-state evidence."""
    payload = _payload(call)
    reply, schema_error = None, None
    try:
        reply = VariableReply.model_validate(_reply(call)).model_dump(mode="json")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        schema_error = str(exc)
    agenda = payload["agenda"]
    targets = {
        v["name"]
        for v in payload["public_brief"]["public_variables"]
        if v["data_role"] == "target"
    }
    excluded = targets | set(agenda["targets"]) | set(agenda["drivers"])
    existing = [
        v
        for v in payload["current_inventory"]
        if v["definition"] == "differential" and v["name"] not in excluded
    ]
    gaps = sorted(
        {
            gap
            for r in call["decisions"]
            for gap in r["event"].get("unresolved_obligations", [])
        }
    )
    return {
        **call,
        "agenda": agenda,
        "displayed_inventory": payload["current_inventory"],
        "parsed_reply": reply,
        "schema_error": schema_error,
        "partial_acceptance": any(
            r["event"].get("partial_acceptance", False) for r in call["decisions"]
        ),
        "unresolved_obligations": gaps,
        "empty_reply_with_existing_dynamic_candidates": bool(
            reply == {"variables": []}
            and existing
            and any(g.startswith("dynamic_memory_mediator:") for g in gaps)
        ),
        "existing_dynamic_candidates": existing,
        "candidate_interpretation": (
            "Syntactic eligibility only; not a scientific assignment."
        ),
        "target_contract_marker_displayed": (
            "Runtime public target contract (enforced before fitting):"
            in payload["public_brief"]["scientific_context"]
        ),
    }


def process_conflicts(call: dict, bindings: list[dict]) -> list[dict]:
    """Compare structured uses only; tolerate exact repeats like the runtime."""
    payload = _payload(call)
    required = payload["required_process_contributions"]
    processes = {b["proposal"]["name"]: b for b in bindings}
    conflicts = []
    for term in _reply(call)["terms"]:
        mentioned = set(term["sources"]) & processes.keys()
        if not mentioned or any(
            set(term["sources"]) == set(r["sources"])
            and term["outer_weight_sign"] == r["outer_weight_sign"]
            for r in required
        ):
            continue
        declared_here = {s for r in required for s in r["sources"]}
        conflicts.append(
            {
                "term": term,
                "classification": (
                    "undeclared_consumer"
                    if mentioned - declared_here
                    else "declared_use_sign_or_group_conflict"
                ),
                "earlier_declarations": [processes[n] for n in sorted(mentioned)],
            }
        )
    return conflicts


def audit(source: Path) -> dict:
    """Audit saved calls and variable results; never invoke construction or fitting."""
    plan = sealed_read(source / "plan.json")
    if plan["protocol"] != "phase-c-construction-baseline-1":
        raise ValueError("expected a Phase C construction baseline")
    rows = []
    for task in plan["tasks"]:
        name = task["task_id"]
        if Path(name).name != name:
            raise ValueError("invalid task directory")
        directory = source / "results" / name
        proposal = sealed_read(directory / "proposal.json")
        identity = content_hash([plan["artifact_sha256"], task])
        if proposal["identity"] != identity:
            raise ValueError("proposal belongs to another plan/task")
        calls = _calls(directory, identity)
        if len(calls) != proposal["cost"]["physical_requests"]:
            raise ValueError("proposal call accounting differs")
        variables = [
            variable_call(c) for c in calls if c["step"].startswith("variables_")
        ]
        variables.sort(key=lambda c: (int(c["step"].split("_")[1]), c["attempt"]))
        result = _read(directory / "construction/variables/result.json")
        inventory = {v["name"]: v for v in result["inventory"]}
        contract = plan["cells"][task["benchmark_id"]]["target_contract"]
        mismatches = []
        roles = {"dynamic_state": "differential", "instantaneous_process": "algebraic"}
        for rule in contract["targets"]:
            expected = roles.get(rule["expected_representation"])
            actual = inventory.get(rule["target_channel"], {}).get("definition")
            if expected and expected != actual:
                mismatches.append({"rule": rule, "inventory_definition": actual})
        bindings_path = directory / "construction/process/topology/process_review.json"
        bindings = (
            _read(bindings_path).get("bindings", []) if bindings_path.exists() else []
        )
        conflicts = []
        for call in calls:
            if any(r["event"].get("error") == PROCESS_ERROR for r in call["decisions"]):
                conflicts.append(
                    {**call, "conflicts": process_conflicts(call, bindings)}
                )
        rows.append(
            {
                "task": task,
                "construction_status": proposal["status"],
                "variable_status": result["status"],
                "inventory": result["inventory"],
                "memory_candidates": result["memory_candidates"],
                "machine_contract_mismatches": mismatches,
                "mismatch_interpretation": (
                    "Policy conflict, not proof of a public/scientific violation."
                ),
                "variable_calls": variables,
                "process_conflict_calls": conflicts,
                "physical_calls": len(calls),
            }
        )
    variables = [c for r in rows for c in r["variable_calls"]]
    agendas = [list({c["step"] for c in r["variable_calls"]}) for r in rows]
    conflicts = [c for r in rows for c in r["process_conflict_calls"]]
    return {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "source": str(source.resolve()),
        "counts": {
            "constructions": len(rows),
            "variable_statuses": dict(Counter(r["variable_status"] for r in rows)),
            "variable_agendas_called": sum(map(len, agendas)),
            "variable_physical_calls": len(variables),
            "schema_valid_replies": sum(c["schema_error"] is None for c in variables),
            "accepted_variable_calls": sum(c["accepted"] is True for c in variables),
            "rejected_variable_calls": sum(c["accepted"] is False for c in variables),
            "unknown_variable_decisions": sum(c["accepted"] is None for c in variables),
            "first_attempt_completed_agendas": sum(
                c["accepted"] is True and c["attempt"] == 0 for c in variables
            ),
            "partial_acceptance_calls": sum(c["partial_acceptance"] for c in variables),
            "empty_replies_with_existing_candidates": sum(
                c["empty_reply_with_existing_dynamic_candidates"] for c in variables
            ),
            "constructions_with_policy_mismatch": sum(
                bool(r["machine_contract_mismatches"]) for r in rows
            ),
            "process_conflict_calls": len(conflicts),
            "process_conflict_terms": dict(
                Counter(
                    term["classification"] for c in conflicts for term in c["conflicts"]
                )
            ),
        },
        "scientific_adequacy": (
            "Not automatically scored; review prompts, meanings and equations."
        ),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "model_changes": 0,
        "test_data_opened": False,
        "rows": rows,
    }


def write_report(result: dict, output: Path) -> None:
    """Write an escaped, browsable evidence report outside the source snapshot."""
    source = Path(result["source"]).resolve()
    if output.resolve() == source or source in output.resolve().parents:
        raise ValueError("audit output must be outside the source snapshot")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")

    def block(value: object) -> str:
        return (
            "<pre>"
            + html.escape(json.dumps(value, indent=2, ensure_ascii=False))
            + "</pre>"
        )

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Construction stage audit</title>",
        "<style>body{max-width:1100px;margin:2em auto;font-family:system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}"
        "summary{cursor:pointer;padding:.6em}td,th{padding:.5em;text-align:left}"
        "table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>",
        "<h1>Saved construction stage audit</h1><p>Mechanical acceptance is not "
        "scientific adequacy. Candidates for memory assignment are syntactic, not "
        "scientifically selected. No LLM calls, fitting, rollouts or model "
        "changes.</p>",
        block(result["counts"]),
        "<nav>"
        + " · ".join(
            f'<a href="#{html.escape(r["task"]["task_id"])}">'
            + html.escape(r["task"]["task_id"])
            + "</a>"
            for r in result["rows"]
        )
        + "</nav>",
    ]
    for row in result["rows"]:
        name = html.escape(row["task"]["task_id"])
        parts.extend(
            [
                f'<h2 id="{name}">{name}</h2>',
                block(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"variable_calls", "process_conflict_calls"}
                    }
                ),
            ]
        )
        for call in [*row["variable_calls"], *row["process_conflict_calls"]]:
            label = html.escape(
                f"{call['step']} / attempt {call['attempt']} / "
                f"accepted={call['accepted']}"
            )
            parts.append(
                f'<details id="{call["request_hash"]}"><summary>{label}</summary>'
            )
            for heading, value in (
                (
                    "Displayed agenda and inventory",
                    {
                        "agenda": call.get("agenda"),
                        "inventory": call.get("displayed_inventory"),
                    },
                ),
                (
                    "Parsed reply / structured conflicts",
                    call.get("parsed_reply", call.get("conflicts")),
                ),
                ("Runtime decisions", call["decisions"]),
                ("Exact request (including schema)", call["request"]),
                ("Exact saved provider response", call["response"]),
            ):
                parts.append(f"<h3>{heading}</h3>" + block(value))
            parts.append("</details>")
    (output / "VARIABLES.html").write_text("\n".join(parts))
