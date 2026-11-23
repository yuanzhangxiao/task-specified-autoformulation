"""Inspect prompts, paired outcomes, equations and free validation rollouts."""

import json
import os
import tarfile
import tempfile
from collections import Counter
from html import escape
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.preproposal_observed_gate import check_bound_memories
from autoformalism.schemas.staged_topology import PublicScientificBrief


def _code(value):
    return "<pre>" + escape(json.dumps(value, indent=2, ensure_ascii=False)) + "</pre>"


def _page(title, body):
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><style>'
        "body{max-width:1080px;margin:36px auto;padding:0 18px;color:#213544;"
        "font:16px/1.5 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f3f6f6;padding:12px}summary{cursor:pointer}table{border-collapse:"
        "collapse;width:100%;font-size:.9em}th,td{text-align:left;border-bottom:1px "
        "solid #ccd6d9;padding:.35em;vertical-align:top}figure{margin:1em 0}img{"
        "width:100%}h1,h2{line-height:1.2}</style><title>" + escape(title) + "</title>"
        "<main><h1>" + escape(title) + "</h1>" + body + "</main></html>"
    )


def _graph_label(row):
    return escape(f"{row['case']} · {row['origin']} · {row['structure_sha256'][:10]}")


def prompt_review(root, p, request):
    body = [
        "<p>Each graph has two proposer seeds in each proposer arm. The hybrid can "
        "select a suggested model by ID or write its own equations.</p>"
    ]
    if p.get("profile") in ("source-review", "basin-review", "basin-replay"):
        body.append(
            "<p>The new request adds a choice to change equation sources. "
            "The runtime infers the patch, checks the graph on training data "
            "and tests the internal transfer when the equations permit it.</p>"
        )
    if p.get("profile") == "basin-replay":
        body.append(
            "<p>This is a preview for future calls. The original proposer "
            "received the fixed-graph prompt.</p>"
        )
    for g, row in enumerate(p["graphs"]):
        body.append(f"<h2>{_graph_label(row)}</h2>")
        for arm in ("proposer", "hybrid"):
            system, user, schema = request(p, g, arm)
            body.append(
                f"<details><summary>{arm} request</summary><p>{escape(system)}</p>"
                + _code(user)
                + "<details><summary>Reply schema</summary>"
                + _code(schema.model_json_schema())
                + "</details></details>"
            )
    (root / "PROMPTS.html").write_text(
        _page("Function-construction prompts", "".join(body))
    )


def _reaches(graph, source, target):
    stack = [target]
    seen = set()
    while stack:
        current = stack.pop()
        if current == source:
            return True
        if current not in seen:
            stack.extend(graph.get(current, ()))
            seen.add(current)
    return False


def path_checks(graph, case, fit):
    """Inspect fitted direct edges for public pathways and assigned memory."""
    edges = fit.get("effective_sources")
    if edges is None:
        return {}
    brief = PublicScientificBrief.model_validate(case["public_brief"])
    states = {k for k, v in graph["variable_types"].items() if v == "differential"}
    checks = {}
    for req in brief.requirements:
        values = [_reaches(edges, s, t) for s in req.drivers for t in req.targets]
        record = {"driver_target_paths": all(values) if values else None}
        bindings = [
            b
            for b in case.get("mechanism_bindings", [])
            if b["requirement_id"] == req.id
        ]
        if bindings:
            try:
                check_bound_memories(edges, states, brief, bindings)
                record["assigned_memory_path"] = True
            except ValueError as exc:
                record.update(assigned_memory_path=False, detail=str(exc))
        checks[req.id] = record
    return checks


def _plot(root, rows):
    """Plot the first validation trajectory for each retained model on the same axes."""
    available = [r for r in rows if r.get("curves")]
    if not available:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    images = []
    for g in sorted({r["graph_index"] for r in available}):
        group = [r for r in available if r["graph_index"] == g]
        first = group[0]["curves"][0]
        names = list(first["observed"])
        fig, axes = plt.subplots(
            1,
            len(names),
            figsize=(4.4 * len(names), 3.2),
            squeeze=False,
            layout="constrained",
        )
        for ax, name in zip(axes[0], names, strict=True):
            ax.plot(
                first["time"],
                first["observed"][name],
                c="#111111",
                lw=2.2,
                label="observed",
            )
            for row in group:
                curve = next(
                    c
                    for c in row["curves"]
                    if c["trajectory_id"] == first["trajectory_id"]
                )
                ax.plot(
                    curve["time"], curve["predicted"][name], lw=1.4, label=row["arm"]
                )
            ax.set(xlabel="Time (min)", title=name)
            ax.legend(fontsize=8)
        path = root / f"validation-graph-{g}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        images.append((g, path))
    return images


def _comparison(graphs, arms, jobs, fits, rows):
    """Summarize all planned slots, including incomplete and rejected fits."""
    validation = {(r["graph_index"], r["arm"]): r["validation"] for r in rows}
    comparison = []
    for g, graph in enumerate(graphs):
        outcomes = {}
        for arm in arms:
            slots = [i for i, (j, a, _) in enumerate(jobs) if (j, a) == (g, arm)]
            receipts = [fits[i] for i in slots if i in fits]
            score = validation.get((g, arm), {})
            outcomes[arm] = {
                "training_slots": len(slots),
                "recorded_fits": len(receipts),
                "verified_fits": sum(f["status"] == "fitted" for f in receipts),
                "validation_status": score.get("status", "pending"),
                "validation_nmse": score.get("pooled_nmse"),
            }
        comparison.append(
            {
                "graph_index": g,
                "case": graph["case"],
                "origin": graph["origin"],
                "structure_sha256": graph["structure_sha256"],
                "arms": outcomes,
            }
        )
    return comparison


def _comparison_table(comparison, arms):
    """Show the paired scores and their denominators before equation details."""
    table = [
        "<h2>Validation by graph</h2><p>Each cell shows the finalist's "
        "validation NMSE and how many training candidates passed the numerical "
        "grid check. A verified rollout has a maximum prediction change of "
        "at most 0.01 training standard deviations between the two integration "
        "grids. Verification measures numerical agreement over the recorded "
        "trajectories.</p><table><thead><tr><th>Graph</th>"
    ]
    table.extend(f"<th>{escape(arm.title())}</th>" for arm in arms)
    table.append("</tr></thead><tbody>")
    for graph in comparison:
        g = graph["graph_index"]
        table.append(f'<tr><th><a href="#graph-{g}">{_graph_label(graph)}</a></th>')
        for arm in arms:
            row = graph["arms"][arm]
            score = row["validation_nmse"]
            text = (
                f"{score:.5g}"
                if score is not None
                else escape(row["validation_status"].replace("_", " "))
            )
            pending = row["training_slots"] - row["recorded_fits"]
            table.append(
                f"<td><strong>{text}</strong><br>"
                f"{row['verified_fits']}/{row['training_slots']} verified training fits"
                + (f" · {pending} pending" if pending else "")
                + "</td>"
            )
        table.append("</tr>")
    table.append("<tr><th>Verified validation finalists / planned graphs</th>")
    for arm in arms:
        statuses = Counter(g["arms"][arm]["validation_status"] for g in comparison)
        count = f"{statuses['scored']}/{len(comparison)}"
        pending = statuses["pending"]
        table.append(
            f"<td>{count}" + (f" · {pending} pending" if pending else "") + "</td>"
        )
    return "".join(table) + "</tr></tbody></table>"


def inspect(root: Path, progress: dict):
    from scripts.trajectory_functions import fitting_graph, load, slots

    plan, p = load(root)
    rows = []
    for path in sorted((root / "validation").glob("*.json")):
        r = sealed_read(path)
        item = r["selection"]
        g = item["graph_index"]
        row = p["graphs"][g]
        score = dict(r["score"])
        curves = score.pop("curves", [])
        detail = {
            "case": row["case"],
            "structure_sha256": row["structure_sha256"],
            **item,
            "validation": score,
            "curves": curves,
        }
        if item["winner"] is not None:
            i = item["winner"]
            fit = sealed_read(root / "fits" / f"{i:03}" / "result.json")["result"]
            cand = sealed_read(root / "candidates" / f"{i:03}.json")
            detail.update(
                origin=cand["origin"],
                route=cand.get("route", "fixed_graph"),
                admitted_graph=fitting_graph(cand, row),
                source_changes=cand.get("patch", {}),
                scientific_reason=cand.get("scientific_reason"),
                balance_check=cand.get("balance_check"),
                formula=cand["formula"],
                fit=fit,
                public_path_checks=path_checks(
                    fitting_graph(cand, row), p["cases"][row["case"]], fit
                ),
            )
        rows.append(detail)
    receipts = [
        json.loads(x.read_text())
        for x in sorted((root / "steps").glob("*/calls/*.json"))
    ]
    costs = {
        "observed_tokens": sum(r.get("observed_total_tokens") or 0 for r in receipts),
        "unknown_usage": sum(r.get("observed_total_tokens") is None for r in receipts),
    }
    fits = {}
    for path in (root / "fits").glob("*/result.json"):
        receipt = sealed_read(path)
        fits[receipt["index"]] = receipt["result"]
    costs.update(
        fit_evaluations=sum(r.get("evaluations", 0) for r in fits.values()),
        fit_seconds=sum(r.get("seconds", 0) for r in fits.values()),
    )
    admissions = {arm: Counter() for arm in plan["arms"]}
    reviews = []
    for path in sorted((root / "candidates").glob("*.json")):
        r = sealed_read(path)
        admissions[r["arm"]][r["status"]] += 1
        if r.get("route") == "request_topology_change":
            reviews.append(
                {
                    k: r[k]
                    for k in (
                        "index",
                        "archived_index",
                        "archived_status",
                        "status",
                        "route",
                        "reason_source",
                        "scientific_reason",
                        "patch",
                        "balance_check",
                        "reason",
                        "assessment",
                        "child_graph_sha256",
                    )
                    if k in r
                }
            )
    comparison = _comparison(p["graphs"], plan["arms"], slots(p), fits, rows)
    body = [
        "<p>Two expression slots per graph per arm. The lowest verified training "
        "NMSE chooses the finalist. Validation NMSE uses training target variance.</p>",
        _comparison_table(comparison, plan["arms"]),
        "<details><summary>Campaign progress and cost</summary>"
        + _code(progress | costs)
        + "</details>",
        "<h2>Formula admission</h2>",
        _code(admissions),
    ]
    if p.get("profile") == "basin-replay":
        body.insert(
            0,
            "<p>This retrospective audit reused the original formula "
            "text and coefficient starts. The audit supplied the route and "
            "review reason for each original rejection. There were zero new "
            "proposer calls.</p>",
        )
    if reviews:
        body.append("<h2>Source changes reviewed</h2>")
        for r in reviews:
            slot = r.get("archived_index", r["index"])
            body.append(
                f"<details><summary>Slot {slot:03} · {escape(r['status'])}"
                "</summary>" + _code(r) + "</details>"
            )
    images = dict(_plot(root, rows))
    for g, row in enumerate(p["graphs"]):
        body.append(f'<h2 id="graph-{g}">{_graph_label(row)}</h2>')
        if g in images:
            body.append(
                f'<figure><img src="{images[g].name}" '
                'alt="Validation predictions and observations"></figure>'
            )
        if row["case"].startswith("basin"):
            body.append(
                "<p>The original basin graphs omit downstream depth and inflow from "
                "the downstream rate. The prompt review includes an outlet-law check "
                "using training data.</p>"
            )
        for r in rows:
            if r["graph_index"] == g:
                score = r["validation"]
                body.append(
                    "<details><summary>"
                    + escape(
                        f"{r['arm']} · {score['status']} · "
                        f"validation NMSE {score.get('pooled_nmse', '—')}"
                    )
                    + "</summary>"
                    + _code({k: v for k, v in r.items() if k != "curves"})
                    + "</details>"
                )
        body.append(
            "<details><summary>Topology and initial rules</summary>"
            + _code(row["graph"])
            + "</details>"
        )
    summary = progress | {
        "plan": plan["artifact_sha256"],
        "costs": costs,
        "admission_by_arm": admissions,
        "source_reviews": reviews,
        "comparison_by_graph": comparison,
        "comparisons": [{k: v for k, v in r.items() if k != "curves"} for r in rows],
    }
    atomic_json(root / "summary.json", summary)
    (root / "REPORT.html").write_text(
        _page("Function construction on the selected graphs", "".join(body))
    )
    include = (
        "plan.json",
        "intake.json",
        "execution.json",
        "generation.json",
        "finalists.json",
        "evaluation.json",
        "summary.json",
        "REPORT.html",
        "PROMPTS.html",
        "candidates",
        "fits",
        "steps",
        "validation",
        "logs",
        "submissions",
        "cpu-submissions",
        "runtime",
    )
    archive = root / "inspection.tar.gz"
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
        name = tmp.name
    try:
        with tarfile.open(name, "w:gz") as tar:
            for path in [root / i for i in include] + list(images.values()):
                if path.exists():
                    tar.add(path, arcname=path.name)
        os.replace(name, archive)
    finally:
        Path(name).unlink(missing_ok=True)
    return progress | {"archive": str(archive)}
