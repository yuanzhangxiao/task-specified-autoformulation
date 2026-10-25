#!/usr/bin/env python3
"""Replay the two pinned structural-repair inspections without generation or fitting.

Each saved reply is checked independently against its original unchanged parent.
This is an offline gate audit, not a new trajectory of proposer decisions. Source
archives are read in memory; no archived code is extracted or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from contextlib import suppress
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research.construction_revision_edits import RevisionValidationError
from autoformalism.research.construction_structural_revision import (
    normalize_structural_reply,
)
from autoformalism.research.interaction_source import _Reader
from autoformalism.staged_topology import content_hash
from scripts import structural_revision_campaign as campaign

PROTOCOL = "saved-structural-revision-replay-1"
MAX_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
SOURCE_PINS = {
    "20b": {
        "archive_sha256": (
            "02c3dc8e67491c22157092604d10c35bed546b75f647ec0b5fa55e76923aa29c"
        ),
        "plan_sha256": (
            "7e139752205cd59c6073646f07b5928eb588b3edf143914f7363d52f4c984a08"
        ),
    },
    "120b": {
        "archive_sha256": (
            "d3821dc55fa7a3da0cc78c4c7b2b73f2cf1624bdc5a9142ca7669dd5301a7cb4"
        ),
        "plan_sha256": (
            "92bbfab97cdd577a109c86896547200cf354e8a2d3a19097823ec8a7571d0029"
        ),
    },
}


def load_source(path: Path, arm: str, *, archive_sha256: str | None = None) -> dict:
    """Authenticate bytes, parents and receipts against an explicit trusted pin.

    The default remains the original upload. A caller restoring verified saved
    members may supply its separately pinned repack; all inner checks still run.
    """
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("a bounded regular inspection archive is required")
    blob = path.read_bytes()
    expected_hash = archive_sha256 or SOURCE_PINS[arm]["archive_sha256"]
    if hashlib.sha256(blob).hexdigest() != expected_hash:
        raise ValueError("inspection archive identity differs")
    reader = _Reader(blob)
    if (
        len(reader.members) > 4096
        or sum(member.size for member in reader.members.values()) > MAX_EXPANDED_BYTES
    ):
        raise ValueError("inspection archive exceeds resource bounds")
    plan = reader.json("plan.json", "artifact_sha256")
    selected = reader.json("selected_sources.json", "artifact_sha256")
    if (
        plan["artifact_sha256"] != SOURCE_PINS[arm]["plan_sha256"]
        or plan["model_size"] != arm
        or plan["selected_sources_sha256"] != selected["artifact_sha256"]
        or hashlib.sha256(reader.read("saved-source.tar.gz")).hexdigest()
        != plan["source_archive_sha256"]
        or plan["test_data_opened"] is not False
    ):
        raise ValueError("inspection plan or source identity differs")
    campaign.old._validate_inputs(selected["inputs"])
    calls = []
    for entry in plan["cases"]:
        name = entry["parent_id"]
        parent = selected["parents"][name]
        if entry["input_sha256"] != content_hash(parent):
            raise ValueError("original parent identity differs")
        control = reader.json(f"controls/{name}.json", "artifact_sha256")
        if control["problem"] != parent["problem"] or not control["control_unchanged"]:
            raise ValueError("unchanged parent control differs")
        prefix = f"results/{entry['case_id']}"
        result = reader.json(f"{prefix}/result.json")
        receipt = reader.json(f"{prefix}/completion.json", "artifact_sha256")
        records = campaign.sources._records(reader, entry, plan)
        tokens = campaign.sources._records(reader, entry, plan, tokens=True)
        if (
            receipt["plan_sha256"] != plan["artifact_sha256"]
            or receipt["input_sha256"] != entry["input_sha256"]
            or receipt["calls_sha256"] != content_hash(records)
            or receipt["tokenizations_sha256"] != content_hash(tokens)
            or receipt["status"] != result["status"]
        ):
            raise ValueError("completed episode receipt differs")
        expected_files = {"result.json"} | (
            {"fit_handoff.json"} if result.get("problem") else set()
        )
        if set(receipt["files"]) != expected_files:
            raise ValueError("completion file roster differs")
        for filename, digest in receipt["files"].items():
            if (
                hashlib.sha256(reader.read(f"{prefix}/{filename}")).hexdigest()
                != digest
            ):
                raise ValueError("completed episode file differs")
        attempts = result["attempts"]
        if (
            [a["attempt"] for a in attempts] != list(range(len(attempts)))
            or len(attempts) != len(records)
            or {a["request_hash"] for a in attempts}
            != {r["request_hash"] for r in records}
        ):
            raise ValueError("completed call roster differs")
        for attempt in attempts:
            record = next(
                r for r in records if r["request_hash"] == attempt["request_hash"]
            )
            payload = json.loads(record["request"]["body"]["messages"][1]["content"])
            if (
                record["attempt"] != attempt["attempt"]
                or record["step"] != "structural_revision"
                or payload["current_model"]["problem_sha256"]
                != content_hash(parent["problem"])
                or payload["public_task"] != parent["public_brief"]
            ):
                raise ValueError("saved request parent or call position differs")
            calls.append(
                {
                    "case_id": entry["case_id"],
                    "parent_id": name,
                    "seed": entry["seed"],
                    "attempt": attempt["attempt"],
                    "historical_outcome": attempt.get("diagnostic", {}).get(
                        "code", attempt["status"]
                    ),
                    "historical_episode_status": result["status"],
                    "record": record,
                }
            )
    return {"arm": arm, "plan": plan, "selected": selected, "calls": calls}


def assess_reply(parent: dict, call: dict) -> dict:
    """Revalidate one raw reply; preserve rejection and normalization evidence."""
    normalized, changes = None, []
    try:
        raw = visible_response(call["record"])
        cited, citation_changes = campaign._normalize_displayed_citations(parent, raw)
        # The production boundary owns schema-error classification. An audit-only
        # normalization probe must not replace its REVISION_SCHEMA diagnosis.
        with suppress(ValueError):
            normalized, changes = normalize_structural_reply(cited)
        applied = campaign._apply(parent, raw)
        problem = applied.get("problem")
        return {
            "outcome": "scope_unresolved"
            if problem is None
            else "passes_current_gates",
            "applied_status": applied["status"],
            "assessment_status": applied.get("audit", {})
            .get("training_consistency", {})
            .get("status"),
            "audit": applied.get("audit"),
            "problem": problem.model_dump(mode="json") if problem else None,
            "normalizations": changes,
            "citation_normalizations": citation_changes,
            "normalized_reply": normalized,
        }
    except ValueError as exc:
        diagnostic = (
            exc.as_dict()
            if isinstance(exc, RevisionValidationError)
            else {"code": "RESPONSE_SCHEMA_OR_AVAILABILITY", "message": str(exc)[:4000]}
        )
        return {
            "outcome": diagnostic["code"],
            "diagnostic": diagnostic,
            "normalizations": changes,
            "normalized_reply": normalized,
            "problem": None,
        }


def _identity(loaded: list[dict]) -> dict:
    """Freeze source receipts and implementation for deterministic resume."""
    return {
        "protocol": PROTOCOL,
        "sources": {s["arm"]: SOURCE_PINS[s["arm"]] for s in loaded},
        "runtime_source_sha256": runtime_source_hash(),
        "campaign_entrypoints": campaign._entrypoints(),
        "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gate_policy": campaign.SYMBOLIC_INITIAL_POLICY,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "model_promotion": False,
        "fitting_exports": 0,
    }


def replay(archives: dict[str, Path], root: Path) -> dict:
    """Checkpoint each independent recheck; never alter the source experiment."""
    root = root.resolve()
    if any(path.resolve().is_relative_to(root) for path in archives.values()):
        raise ValueError("replay output must be separate from source archive storage")
    loaded = [load_source(archives[arm], arm) for arm in sorted(archives)]
    comparisons = {s["plan"]["comparison_sha256"] for s in loaded}
    if len(comparisons) != 1:
        raise ValueError("the original comparison identities differ")
    with _lock(root / "replay-lock"):
        plan = sealed_write(root / "plan.json", _identity(loaded))
        rows = []
        for source in loaded:
            for call in source["calls"]:
                record = call["record"]
                identity = {
                    "plan_sha256": plan["artifact_sha256"],
                    "arm": source["arm"],
                    **{k: v for k, v in call.items() if k != "record"},
                    "request_sha256": record["request_hash"],
                    "record_sha256": content_hash(record),
                    "historical_tokens": record["observed_total_tokens"],
                    "adopted": False,
                    "fitting_exported": False,
                }
                path = root / "rows" / f"{source['arm']}-{record['request_hash']}.json"
                if path.exists():
                    row = sealed_read(path)
                    if any(row.get(k) != v for k, v in identity.items()):
                        raise ValueError("cached replay input identity differs")
                else:
                    parent = campaign._parent(source["selected"], call["parent_id"])
                    row = sealed_write(path, {**identity, **assess_reply(parent, call)})
                rows.append(row)
        arms = {}
        for arm in archives:
            arm_rows = [r for r in rows if r["arm"] == arm]
            eligible = [r for r in arm_rows if r["outcome"] == "passes_current_gates"]
            arms[arm] = {
                "saved_calls_replayed": len(arm_rows),
                "outcome_counts": dict(Counter(r["outcome"] for r in arm_rows)),
                "gate_eligible_replies": len(eligible),
                "gate_eligible_episodes": len({r["case_id"] for r in eligible}),
                "assessment_counts": dict(
                    Counter(r["assessment_status"] for r in eligible)
                ),
            }
        summary = {
            **_identity(loaded),
            "plan_sha256": plan["artifact_sha256"],
            "status": "complete",
            "arms": arms,
            "interpretation": "Independent saved-reply gate replay, not new live "
            "success rates. Deferred checks do not establish scientific correctness.",
            "rows": [
                {
                    k: v
                    for k, v in r.items()
                    if k not in {"problem", "audit", "diagnostic", "normalized_reply"}
                }
                for r in rows
            ],
        }
        atomic_json(root / "summary.json", summary)
        _report(root, summary)
        return summary


def _report(root: Path, summary: dict) -> None:
    """Render a compact safe index of the complete saved replay records."""
    lines = [
        "<!doctype html><meta charset='utf-8'>"
        "<title>Saved structural reply replay</title>",
        "<style>body{max-width:1150px;margin:2rem auto;font:16px/1.5 sans-serif}"
        "table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border-bottom:"
        "1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>",
        "<h1>Saved structural reply replay</h1>",
        "<p>Original results unchanged. No new model calls, fits, rollouts "
        "or test access."
        " No candidate adoption. Each reply uses its original parent.</p>",
        "<pre>" + html.escape(json.dumps(summary["arms"], indent=2)) + "</pre>",
        "<p>Full row files preserve equations, normalizations and qualified evidence."
        " Passing the current finite gates is not proof of scientific correctness.</p>",
        "<table><tr><th>Arm</th><th>Episode / attempt</th>"
        "<th>Historical first outcome</th>"
        "<th>Current replay outcome</th><th>Evidence</th></tr>",
    ]
    for row in summary["rows"]:
        name = f"{row['arm']}-{row['request_sha256']}.json"
        lines.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    row["arm"],
                    f"{row['case_id']} / {row['attempt']}",
                    row["historical_outcome"],
                    row["outcome"],
                )
            )
            + f'<td><a href="rows/{name}">record</a></td></tr>'
        )
    lines.append("</table>")
    (root / "REPORT.html").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-20b", type=Path, required=True)
    parser.add_argument("--archive-120b", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = replay({"20b": args.archive_20b, "120b": args.archive_120b}, args.root)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
