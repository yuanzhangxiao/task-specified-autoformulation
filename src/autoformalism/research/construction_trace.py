"""Human-readable prompt/response evidence without executing provider content."""

from __future__ import annotations

import html
import json
from pathlib import Path

from autoformalism.staged_topology import content_hash


def render(directory: Path, namespace: str) -> dict:
    """Join raw requests with runtime events; unknown acceptance stays unknown."""
    events: dict[str, list] = {}

    def visit(value, source):
        if isinstance(value, dict):
            if isinstance(value.get("request_hash"), str):
                events.setdefault(value["request_hash"], []).append(
                    {"file": source, "event": value}
                )
            for v in value.values():
                visit(v, source)
        elif isinstance(value, list):
            for v in value:
                visit(v, source)

    for path in sorted((directory / "construction").rglob("*.json")):
        visit(json.loads(path.read_text()), str(path.relative_to(directory)))
    rows = []
    for path in sorted((directory / "calls").glob("*.json")):
        r = json.loads(path.read_text())
        if (
            r["request_hash"] != path.stem
            or content_hash(r["request"]) != path.stem
            or r["request"]["namespace"] != namespace
        ):
            raise ValueError("trace request provenance differs")
        rows.append(
            {
                "request_hash": path.stem,
                "step": r["step"],
                "attempt": r["attempt"],
                "status": r["status"],
                "request": r["request"]["body"],
                "raw_response": r.get("raw_response"),
                "error": r.get("error"),
                "observed_total_tokens": r.get("observed_total_tokens"),
                "runtime_events": events.get(path.stem, []),
                "scientific_adequacy": "not_assessed",
            }
        )
    if set(events) - {row["request_hash"] for row in rows}:
        raise ValueError("runtime evidence references missing call records")
    proposal = directory / "proposal.json"
    if proposal.exists():
        cost = json.loads(proposal.read_text()).get("cost")
        if cost and cost["physical_requests"] != len(rows):
            raise ValueError("proposal accounting differs from retained call records")
    # Step order is shown explicitly; file hashes are not a chronological order.
    result = {
        "calls": rows,
        "ordering": "request_hash; step and attempt shown explicitly",
        "scientific_adequacy": "requires independent review",
    }

    def block(v):
        return (
            "<pre>"
            + html.escape(json.dumps(v, indent=2, ensure_ascii=False))
            + "</pre>"
        )

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Construction trace</title>",
        "<style>body{max-width:1100px;margin:2em auto;font-family:system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}"
        "summary{cursor:pointer;padding:.6em}</style>",
        "<h1>Construction trace</h1><p>Raw provider content is evidence. "
        "Runtime acceptance is not scientific approval. Calls are indexed by hash; "
        "use step/attempt labels and stage artifacts to reconstruct order.</p>",
    ]
    if proposal.exists():
        parts.append(
            "<details><summary>Accepted model / construction outcome</summary>"
            + block(json.loads(proposal.read_text()))
            + "</details>"
        )
    for r in rows:
        label = html.escape(f"{r['step']} / attempt {r['attempt']} / {r['status']}")
        parts.append(
            f"<details><summary>{label}</summary><h3>Exact request</h3>"
            + block(r["request"])
            + "<h3>Provider response</h3>"
            + block(r["raw_response"])
            + "<h3>Runtime events</h3>"
            + block(r["runtime_events"])
            + "</details>"
        )
    (directory / "TRACE.html").write_text("\n".join(parts))
    (directory / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
