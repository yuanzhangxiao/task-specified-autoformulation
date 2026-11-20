"""Replay both topology checks offline and export the exact 20B prompt for review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json, strict_provider_schema
from autoformalism.research.topology_repair_case import load_case
from autoformalism.research.topology_repair_trial import (
    RepairReply,
    messages,
    offline_portfolio,
)
from autoformalism.staged_topology import content_hash


def _json_html(obj: object) -> str:
    return "<pre>" + escape(json.dumps(obj, indent=2, ensure_ascii=False)) + "</pre>"


def report(case: dict, outputs: Path) -> dict:
    """Write human-readable edits, per-pair findings and machine-readable topologies."""
    system, user = messages(case)
    rows = offline_portfolio(case)
    counts = dict(sorted(Counter(row["status"] for row in rows).items()))
    packet = json.loads(user)
    target = case["target"]
    pair_values = packet["paired_values"][target]["witness_values"]
    midpoint_error = abs(pair_values[0] - pair_values[1]) / 2
    summary = {
        "case_sha256": case["case_sha256"],
        "public_packet_sha256": content_hash(json.loads(user)),
        "target": case["target"],
        "pair": case["pair"],
        "witness_time": case["witness_time"],
        "counts": counts,
        "parent_minimax_error_against_recorded_pair": midpoint_error,
        "edits": rows,
    }
    outputs.mkdir(parents=True, exist_ok=True)
    result = outputs / "edits.json"
    if (
        result.exists()
        and json.loads(result.read_text())["case_sha256"] != case["case_sha256"]
    ):
        raise ValueError("Choose a new output directory for a different case")
    atomic_json(result, summary)
    style = (
        "<meta charset=utf-8><style>body{font:16px/1.5 system-ui;max-width:1080px;"
        "margin:3em auto;padding:0 1em;color:#172b3a}table{border-collapse:collapse;"
        "width:100%}th,td{border-bottom:1px solid #ccd4d9;"
        "text-align:left;padding:.45em}"
        "pre{background:#f4f6f7;padding:1em;overflow:auto;white-space:pre-wrap}"
        "summary{cursor:pointer}code{font-size:.96em}</style>"
    )
    prompt = (
        f"<!doctype html><title>20B topology repair prompt</title>{style}"
        "<h1>20B topology repair prompt</h1><h2>System</h2>"
        f"<p>{escape(system)}</p><h2>User evidence</h2>{_json_html(json.loads(user))}"
        "<h2>Reply schema</h2>"
        + _json_html(strict_provider_schema(RepairReply.model_json_schema()))
    )
    (outputs / "PROMPTS.html").write_text(prompt)
    table = []
    for row in rows:
        edit = row["edit"]
        route = [edit["source"]]
        if edit["kind"] == "via_new_memory":
            route.append(row.get("memory_state") or "(new memory)")
        route.append(edit["target"])
        remaining = (
            len(row["remaining_conflicts"]) if row["status"] != "invalid" else "—"
        )
        undecided = len(row["unresolved_pairs"]) if row["status"] != "invalid" else "—"
        table.append(
            "<tr><td><code>" + escape(" → ".join(route)) + "</code></td>"
            f"<td>{escape(row['status'])}</td><td>{escape(row['public_graph_check'])}</td>"
            f"<td>{remaining}</td><td>{undecided}</td>"
            "<td><details><summary>Evidence and edit</summary>"
            + _json_html(row)
            + "</details></td></tr>"
        )
    names = " and ".join(case["pair"])
    srcs = case["parent"]["targets"][case["target"]]["sources"]
    page = (
        f"<!doctype html><title>Topology repair audit</title>{style}"
        f"<h1>Topology repair audit</h1><p>The parent models the derivative of "
        f"<b>{escape(target)}</b> using {escape(', '.join(srcs))}.</p>"
        f"<p>At t = {case['witness_time']:g}, {escape(names)} have recorded "
        f"values {pair_values[0]:.4f} and {pair_values[1]:.4f}. The parent "
        "must give the same prediction for both. For at least one trajectory "
        f"its absolute error must be at least <b>{midpoint_error:.4f}</b>.</p>"
        "<p>Each row adds a source directly or through one differential state "
        "with a shared initial value. A distinguishing path permits different "
        "predictions at the selected time. The remaining-pair count includes "
        "earlier times on the same schedules. Expand a row to see unresolved "
        "pairs, full declarations and evidence. No parameter fits were run.</p><p>"
        '<a href="PROMPTS.html">Review the 20B request and schema</a> · '
        '<a href="edits.json">Download complete edits and assessments</a></p>'
        "<table><thead><tr><th>Proposed dependency</th><th>Selected pair</th>"
        "<th>Public graph check</th><th>Remaining conflicting pairs</th>"
        "<th>Unresolved pairs</th>"
        "<th>Details</th></tr></thead><tbody>" + "".join(table) + "</tbody></table>"
    )
    (outputs / "REPORT.html").write_text(page)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(load_case(args.case), args.out), indent=2))


if __name__ == "__main__":
    main()
