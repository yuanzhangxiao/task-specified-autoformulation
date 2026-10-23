"""Offline, self-contained equation and training-conflict inspection reports.

The renderer consumes saved audit dictionaries only. It cannot fit, simulate,
request a model response, access a dataset or make a model-selection decision.
All displayed evidence is escaped text, including expressions and explanations.
"""

from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

_CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{max-width:1240px;margin:32px auto;
padding:0 24px 50px;font:16px/1.55 system-ui,sans-serif;
color:#243345;background:#f5f7fa}
h1,h2,h3,h4{line-height:1.25;overflow-wrap:anywhere}a{color:#165f99}table{border-collapse:collapse;width:100%;
background:white;margin:14px 0}th,td{border:1px solid #ccd5df;
padding:9px;text-align:left;
vertical-align:top}th{background:#edf2f7}code,pre{font-family:ui-monospace,monospace;
font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f8;
border-radius:6px;padding:14px}.callout{padding:16px;border-left:5px solid #216d9b;
background:#e7f2fa}.warning{border-left-color:#ad4919;background:#fff0e4}
article,details{background:white;border:1px solid #cbd5df;border-radius:8px;
margin:16px 0;padding:16px}details details{background:#f7f9fc}summary{cursor:pointer;
font-weight:650}.muted{color:#56687a;font-size:14px}.equation{font-size:15px;
border-left:4px solid #639bb9}.scroll{overflow-x:auto}.plot{display:block;width:100%;
max-width:780px;margin:18px auto}.badge{font-weight:700}
.counts{font-variant-numeric:tabular-nums}
@media(max-width:650px){body{padding:0 12px}th,td{padding:6px;font-size:13px}}
"""


def _text(value: Any) -> str:
    """Escape every untrusted value before inserting it into markup."""
    return escape(str(value), quote=True)


def _json(value: Any) -> str:
    """Render a receipt as text, never executable JSON or HTML."""
    return "<pre>" + _text(json.dumps(value, indent=2, ensure_ascii=False)) + "</pre>"


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[dict]:
    return (
        [row for row in value if isinstance(row, dict)]
        if isinstance(value, list)
        else []
    )


def _reports(model: dict) -> list[tuple[str, dict]]:
    consistency = _dict(model.get("training_consistency"))
    result = [("Matched-condition equality", consistency)] if consistency else []
    function = _dict(consistency.get("function_consistency"))
    if function:
        result.append(("Function and parameter domains", function))
    return result


def _conflicts(model: dict) -> list[tuple[str, dict]]:
    """Deduplicate checks copied into both a combined and a nested receipt."""
    result, seen = [], set()
    for label, report in _reports(model):
        for check in _rows(report.get("conflicts")):
            identity = json.dumps(check, sort_keys=True, ensure_ascii=False)
            if identity not in seen:
                result.append((label, check))
                seen.add(identity)
    return result


def _classification(model: dict) -> tuple[str, str]:
    if _conflicts(model):
        return "blocked", "Blocked: qualified training conflict"
    reports = _reports(model)
    if any(report.get("status") == "conflict" for _, report in reports):
        return "blocked", "Blocked: conflict reported; inspect evidence availability"
    if not reports or any(
        report.get("status") not in {"deferred", "no_conflict_found"}
        for _, report in reports
    ):
        return "unavailable", "Assessment unavailable or incomplete"
    if any(report.get("status") == "deferred" for _, report in reports):
        return "deferred", "No qualified conflict; some checks deferred"
    return "not_detected", "No qualified conflict in the implemented checks"


def _chart(counts: Counter) -> str:
    """Use only computed counts and fixed markup in the inline SVG."""
    rows = [
        ("blocked", "Qualified conflict — blocked", "#b44c31"),
        ("deferred", "No conflict found; checks deferred", "#aa7624"),
        ("not_detected", "No conflict found by implemented checks", "#327898"),
        ("unavailable", "Assessment unavailable / incomplete", "#718092"),
    ]
    maximum = max([1, *counts.values()])
    parts = [
        '<svg class="plot" viewBox="0 0 780 190" role="img" '
        'aria-labelledby="coverage-chart-title coverage-chart-desc">',
        '<title id="coverage-chart-title">Construction check coverage</title>',
        '<desc id="coverage-chart-desc">Counts of saved models by qualified conflict '
        "and assessment coverage. No category certifies scientific correctness.</desc>",
    ]
    for index, (key, label, color) in enumerate(rows):
        y, count = 28 + 42 * index, counts[key]
        width = 340 * count / maximum
        parts.extend(
            [
                f'<text x="0" y="{y}" font-size="14">{label}</text>',
                f'<rect x="340" y="{y - 17}" width="{width:.2f}" height="24" '
                f'fill="{color}" rx="3"/>',
                f'<text x="{350 + width:.2f}" y="{y}" font-size="15">{count}</text>',
            ]
        )
    return "".join(parts) + "</svg>"


def _equations(audit: dict, *, label: str = "Assembled equations") -> str:
    rows = _rows(audit.get("equations"))
    compact = audit.get("expanded_equations", [])
    if not rows and isinstance(compact, list) and compact:
        return (
            "<h4>"
            + _text(label)
            + "</h4>"
            + '<p class="muted">Compact equation view retained in repair feedback.</p>'
            + "".join(
                '<pre class="equation">' + _text(row) + "</pre>" for row in compact
            )
        )
    if not rows:
        return "<h4>" + _text(label) + "</h4><p>No equation view was available.</p>"
    parts = ["<h4>" + _text(label) + "</h4>"]
    for row in rows:
        lhs = row.get("lhs", row.get("location", "unknown component"))
        expanded = row.get("expanded_expression", row.get("declared_expression", "?"))
        parts.append('<pre class="equation">' + _text(f"{lhs} = {expanded}") + "</pre>")
        if row.get("reason"):
            parts.append('<p class="muted">' + _text(row["reason"]) + "</p>")
        if row.get("declared_expression") != expanded or row.get("coefficients"):
            parts.append(
                "<details><summary>Declared expression and collected coefficients"
                "</summary>" + _json(row) + "</details>"
            )
    return "".join(parts)


def _coverage(model: dict) -> str:
    parts = [
        "<h4>Check coverage</h4><table><tr><th>Check</th><th>Target</th>"
        "<th>Status</th><th>Count</th></tr>"
    ]
    counts: Counter = Counter()
    for label, report in _reports(model):
        if not _rows(report.get("checks")):
            for status, count in _dict(report.get("status_counts")).items():
                if isinstance(count, int) and count >= 0:
                    counts[
                        (label, "compact coverage summary", "all targets", str(status))
                    ] += count
        for check in _rows(report.get("checks")):
            counts[
                (
                    label,
                    str(check.get("property", "unspecified")),
                    str(check.get("target", "—")),
                    str(check.get("status", "unknown")),
                )
            ] += 1
    for (label, prop, target, status), count in sorted(counts.items()):
        parts.append(
            "<tr><td>"
            + _text(label + ": " + prop)
            + "</td><td>"
            + _text(target)
            + "</td><td>"
            + _text(status)
            + f"</td><td>{count}</td></tr>"
        )
    if not counts:
        parts.append(
            '<tr><td colspan="4">No individual check records available.</td></tr>'
        )
    parts.append("</table>")
    for label, report in _reports(model):
        if report.get("coverage"):
            parts.append('<p class="muted">' + _text(report["coverage"]) + "</p>")
        deferred = _rows(report.get("deferred")) or [
            check
            for check in _rows(report.get("checks"))
            if check.get("status") == "deferred"
        ]
        if deferred:
            parts.append(
                "<details><summary>"
                + _text(label)
                + ": deferred reasons</summary>"
                + _json(deferred)
                + "</details>"
            )
    return "".join(parts)


def _proof(label: str, check: dict) -> str:
    parts = [
        "<article><h4>"
        + _text(check.get("target", "Model"))
        + " — "
        + _text(check.get("property", label))
        + "</h4>"
    ]
    if check.get("model_relation"):
        parts.append(
            '<pre class="equation">' + _text(check["model_relation"]) + "</pre>"
        )
    if "derivative_interval" in check:
        interval = check["derivative_interval"]
        parts.append(
            "<p>Proved derivative interval: <code>" + _text(interval) + "</code>.</p>"
        )
    if check.get("proven_nonnegative_states"):
        parts.append(
            "<p>Proved nonnegative states: <code>"
            + _text(", ".join(map(str, check["proven_nonnegative_states"])))
            + "</code>.</p>"
        )
    witnesses = _rows(check.get("witnesses"))
    if not witnesses and isinstance(check.get("witness"), dict):
        witnesses = [check["witness"]]
    for witness in witnesses:
        trajectory = witness.get(
            "trajectory", witness.get("trajectories", "unspecified")
        )
        parts.append(
            "<p><strong>Training witness:</strong> "
            + _text(trajectory)
            + "; times "
            + _text(witness.get("times", "unavailable"))
            + "; observed values "
            + _text(witness.get("observations", "unavailable"))
            + ".</p>"
        )
        required = _dict(witness.get("required_change"))
        if required:
            relation = {"at_least": "at least", "at_most": "at most"}.get(
                required.get("comparison"), str(required.get("comparison", "unknown"))
            )
            parts.append(
                "<p>The model requires a change <strong>"
                + _text(relation)
                + " "
                + _text(required.get("value"))
                + "</strong>; the observed change interval is "
                + _text(witness.get("observed_change_interval", "unavailable"))
                + ".</p>"
            )
    if check.get("repair_feedback"):
        parts.append(
            '<p class="callout warning">' + _text(check["repair_feedback"]) + "</p>"
        )
    parts.append(
        "<details><summary>Complete proof and assumptions for this check</summary>"
        + _json(check)
        + "</details></article>"
    )
    return "".join(parts)


def _parameter_view(audit: dict) -> str:
    parts = [
        "<h4>Effective coefficients and parameter sharing</h4>",
        "<p>These are bookkeeping findings, not automatic rewrites. A replacement must "
        "preserve the exact attainable domain, units and all parameter uses.</p>",
    ]
    candidates = _rows(audit.get("simplification_candidates")) or _rows(
        audit.get("private_coefficient_examples")
    )
    if candidates:
        parts.append(
            '<div class="scroll"><table><tr><th>Equation / argument</th>'
            "<th>Effective coefficient</th><th>Parameters</th>"
            "<th>Interpretation</th></tr>"
        )
        for row in candidates:
            parts.append(
                "<tr><td>"
                + _text(row.get("location", row.get("equation")))
                + " / "
                + _text(row.get("argument"))
                + "</td><td><code>"
                + _text(row.get("effective_coefficient"))
                + "</code></td><td>"
                + _text(", ".join(map(str, row.get("parameters", []))))
                + "</td><td>"
                + _text(row.get("kind", row.get("reason", "review candidate")))
                + "</td></tr>"
            )
        parts.append(
            "</table></div><details><summary>Original domains and review instructions"
            "</summary>" + _json(candidates) + "</details>"
        )
    else:
        parts.append(
            "<p>No private composite-coefficient candidates were reported. "
            "This does not certify parameter identifiability.</p>"
        )
    retained = _rows(audit.get("retained_composite_coefficients"))
    if retained:
        parts.append(
            "<details><summary>Composite coefficients with external uses: "
            "preserve sharing</summary>" + _json(retained) + "</details>"
        )
    certificate = _dict(audit.get("inactive_parameter_certificate"))
    if certificate:
        names = certificate.get("parameters", [])
        parts.append(
            "<p><strong>Syntactically inactive parameters:</strong> "
            + _text(", ".join(map(str, names)) if names else "none reported")
            + ". No optimizer coordinates were changed by this audit.</p>"
        )
        parts.append(
            "<details><summary>Inactivity scope, unused definitions and limitations"
            "</summary>"
            + _json(
                {
                    "certificate": certificate,
                    "unused_process_definitions": audit.get(
                        "unused_process_definitions", []
                    ),
                }
            )
            + "</details>"
        )
    fixed = _rows(audit.get("fixed_scaling_review")) or _rows(
        audit.get("fixed_scaling_examples")
    )
    if fixed:
        parts.append(
            "<details><summary>Fixed scales requiring scientific interpretation"
            "</summary>"
            "<p>A literal coefficient is not by itself a dimensional error. "
            "Review public units and any fitted scale before claiming a conflict.</p>"
            + _json(fixed)
            + "</details>"
        )
    for key, label in (
        ("parameters", "All parameter domains and consumers"),
        ("initializer_audit", "Scoped initialization"),
        ("historical_base_initials", "Recorded base initial conditions"),
    ):
        if audit.get(key):
            parts.append(
                "<details><summary>"
                + label
                + "</summary>"
                + _json(audit[key])
                + "</details>"
            )
    return "".join(parts)


def _model(record: dict, index: int) -> str:
    _, classification = _classification(record)
    model_id = record.get("model_id", f"model-{index}")
    parts = [
        f'<article id="model-{index}"><h2>' + _text(model_id) + "</h2>",
        '<p class="badge">' + _text(classification) + "</p>",
    ]
    scope = record.get("audit_scope", "supplied_model")
    scope_explanations = {
        "accepted_reply": "The equations and checks below describe the accepted reply.",
        "last_rejected_candidate": (
            "The checks below describe the last rejected candidate, which was not "
            "installed. Equations are unavailable unless that candidate's audit was "
            "retained. Any parent equations are shown separately."
        ),
        "unchanged_parent": (
            "No candidate assessment is available for this episode. The equations "
            "and checks below describe its unchanged parent, not a new child. "
            "They do not establish completion or acceptance of a pending attempt."
        ),
        "supplied_model": "The audit describes the supplied saved model.",
    }
    parts.append(
        '<p class="callout"><strong>Audit scope:</strong> '
        + _text(scope)
        + ". "
        + _text(scope_explanations.get(scope, "Use the caller's provenance."))
        + "</p>"
    )
    if record.get("status"):
        parts.append(
            "<p>Saved source outcome: <code>" + _text(record["status"]) + "</code>. "
            "This historical outcome is separate from the current audit.</p>"
        )
    if record.get("parent_id"):
        parts.append("<p>Parent: " + _text(record["parent_id"]) + "</p>")
    audit = _dict(record.get("parameter_audit"))
    if audit.get("reason"):
        parts.append('<p class="callout warning">' + _text(audit["reason"]) + "</p>")
    parts.append(_equations(audit))
    parent = _dict(record.get("parent_parameter_audit")) or _dict(
        _dict(record.get("parent_model")).get("parameter_audit")
    )
    if parent:
        parts.append(
            "<details><summary>Parent equations for comparison</summary>"
            + _equations(parent, label="Unchanged parent equations")
            + "</details>"
        )
    parts.append(_coverage(record))
    conflicts = _conflicts(record)
    if conflicts:
        parts.append(f"<h3>Qualified conflicts ({len(conflicts)} checks)</h3>")
        parts.extend(_proof(label, check) for label, check in conflicts[:3])
        if len(conflicts) > 3:
            parts.append(
                f"<details><summary>Remaining {len(conflicts) - 3} conflict checks"
                "</summary>"
                + "".join(_proof(label, check) for label, check in conflicts[3:])
                + "</details>"
            )
    parts.append(_parameter_view(audit))
    if record.get("attempts"):
        parts.append(
            "<details><summary>Saved proposal and repair attempts</summary>"
            + _json(record["attempts"])
            + "</details>"
        )
    for label, report in _reports(record):
        assumptions = report.get("assumptions")
        if assumptions:
            parts.append(
                "<details><summary>"
                + _text(label)
                + ": assessment assumptions</summary>"
                + _json(assumptions)
                + "</details>"
            )
    return "".join(parts) + "</article>"


def render(path: Path, summary: dict, models: list[dict], *, title: str) -> None:
    """Write an escaped, offline HTML audit without performance claims or I/O inputs."""
    records = _rows(models)
    counts = Counter(_classification(model)[0] for model in records)
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>" + _text(title) + "</title><style>" + _CSS + "</style></head><body>",
        "<h1>" + _text(title) + "</h1>",
        '<p class="callout">Construction-only review of saved equations and public '
        "training evidence. Qualified contradictions block acceptance. "
        "No conflict found means only "
        "that these implemented checks found none; deferred checks remain unresolved. "
        "No category establishes full scientific validity or optimization adequacy."
        "</p>",
        f"<p>{len(records)} model records. Chart counts refer to these audit records; "
        "they may include unchanged parent views for pending or rejected episodes, "
        "rather than distinct completed proposals. Inspect each audit scope. "
        "This report contains no NMSE ranking or parent-child performance conclusion."
        "</p>",
        _chart(counts),
    ]
    metadata = {
        key: summary[key]
        for key in (
            "protocol",
            "identity",
            "plan_sha256",
            "source_archive_sha256",
            "source_commit",
            "llm_calls",
            "optimizer_calls",
            "solver_rollouts",
            "test_data_opened",
        )
        if key in summary
    }
    if metadata:
        parts.append(
            "<details><summary>Provenance and activity recorded by the caller</summary>"
            + _json(metadata)
            + "</details>"
        )
    parts.append(
        '<div class="scroll"><table><tr><th>Model</th>'
        "<th>Source role / audit scope</th>"
        "<th>Saved outcome</th><th>Current audit</th><th>Private coefficients</th></tr>"
    )
    for index, model in enumerate(records):
        _, classification = _classification(model)
        audit = _dict(model.get("parameter_audit"))
        count = len(_rows(audit.get("simplification_candidates")))
        if not count:
            count = audit.get(
                "private_coefficient_count",
                len(_rows(audit.get("private_coefficient_examples"))),
            )
        parts.append(
            f'<tr><td><a href="#model-{index}">'
            + _text(model.get("model_id", f"model-{index}"))
            + "</a></td><td>"
            + _text(model.get("source_role", "unspecified"))
            + "<br><code>"
            + _text(model.get("audit_scope", "supplied_model"))
            + "</code>"
            + "</td><td>"
            + _text(model.get("status", "unavailable"))
            + "</td><td>"
            + _text(classification)
            + "</td><td>"
            + _text(count)
            + "</td></tr>"
        )
    if not records:
        parts.append('<tr><td colspan="5">No model records were supplied.</td></tr>')
    parts.append("</table></div>")
    parts.extend(_model(model, index) for index, model in enumerate(records))
    parts.append("</body></html>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
