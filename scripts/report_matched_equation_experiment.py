"""Summarize matched graph/fold forecast receipts without further fitting."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from html import escape
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.matched_equation_experiment import DEFAULT
from scripts.matched_equation_hybrid import result_for


def _provider(root: Path, p: dict) -> dict:
    packet = sealed_read(root / "packets" / f"{p['panel_id']}.json")
    if packet["artifact_sha256"] != p["packet"]:
        raise ValueError("Provider packet differs from graph/fold plan")
    outcome = result_for(packet, root)
    info = {
        "status": outcome["status"],
        "action": outcome.get("action"),
        "reason": outcome.get("reason") or outcome.get("detail"),
        "observed_tokens": None,
    }
    path = root / "hybrid/generate-responses" / f"{p['panel_id']}.json"
    if path.is_file():
        receipt = json.loads(path.read_text())
        usage = receipt.get("response", {}).get("usage") or {}
        info.update(
            provider_status=receipt["status"],
            observed_tokens=usage.get("total_tokens"),
            seconds=receipt.get("seconds"),
        )
    return info


def _panel(root: Path, p: dict, arm: str, bounds: list) -> dict:
    name = p["panel_id"]
    record = {k: p[k] for k in ("panel_id", "inventory", "origin", "graph_id", "fold")}
    record.update(
        arm=arm,
        fit_statuses=["pending"] * 2,
        scored=False,
        status="pending",
        training_nmse=None,
        held_nmse=None,
        held_nonnegative=None,
        selected_slot=None,
        source=None,
        targets={},
    )
    if arm == "hybrid":
        record["provider"] = _provider(root, p)
    selection_path = root / "selections" / name / f"{arm}.json"
    if not selection_path.exists():
        return record
    selected = sealed_read(selection_path)
    fit_plan = sealed_read(root / "fit_plans" / name / f"{arm}.json")
    if fit_plan["artifact_sha256"] != selected["plan"] or (
        selected["arm"],
        selected["panel"],
    ) != (arm, name):
        raise ValueError("Selected model refers to a different fit plan")
    for slot, digest in enumerate(selected["receipts"]):
        receipt = sealed_read(root / "fits" / name / arm / f"{slot}.json")
        if receipt["artifact_sha256"] != digest or receipt["plan"] != selected["plan"]:
            raise ValueError("Fit differs from the frozen selection")
    slot = selected["selected_slot"]
    if slot is not None:
        record["source"] = (
            ("algorithm_A" if slot == 0 else record["provider"]["action"])
            if arm == "hybrid"
            else f"{arm}_{'AB'[slot]}"
        )
    record.update(
        selected_slot=slot,
        fit_statuses=selected["fit_statuses"],
        selection_sha256=selected["artifact_sha256"],
        training_nmse=selected["training_nmse"],
    )
    path = root / "scores" / name / f"{arm}.json"
    if not path.exists():
        return record
    score = sealed_read(path)
    if (
        score["selection"] != selected["artifact_sha256"]
        or score["held_ids"] != p["held_ids"]
    ):
        raise ValueError("Held score differs from the plan or training selection")
    record.update(status=score["status"], score_sha256=score["artifact_sha256"])
    if score["status"] == "scored":
        record.update(
            scored=True,
            held_nmse=score["pooled_nmse"],
            targets=score["targets"],
            held_nonnegative=all(
                score["targets"][k]["negative_samples"] == 0 for k in bounds
            ),
        )
    return record


def _avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def _med(values):
    values = [v for v in values if v is not None]
    return median(values) if values else None


def _graphs(rows: list[dict], arms: list[str]) -> list[dict]:
    by_graph = {}
    for r in rows:
        key = (r["inventory"], r["origin"], r["graph_id"])
        by_graph.setdefault(key, []).append(r)
    graphs = []
    for (inventory, origin, graph), members in by_graph.items():
        result = {
            "inventory": inventory,
            "origin": origin,
            "graph_id": graph,
            "arms": {},
        }
        for arm in arms:
            records = [r for r in members if r["arm"] == arm]
            if sorted(r["fold"] for r in records) != [0, 1]:
                raise ValueError("Each graph/arm must have two complementary folds")
            complete = all(r["scored"] for r in records)
            result["arms"][arm] = {
                "scored_folds": sum(r["scored"] for r in records),
                "held_nmse": mean(r["held_nmse"] for r in records)
                if complete
                else None,
                "training_nmse": mean(r["training_nmse"] for r in records)
                if all(r["training_nmse"] is not None for r in records)
                else None,
                "both_held_nonnegative": all(r["held_nonnegative"] for r in records)
                if complete
                else None,
            }
        graphs.append(result)
    return graphs


def _compare(reference: list, alternative: list) -> dict:
    """Credit a pair only if both arms have scores on the same unit."""
    wins = ties = losses = 0
    diffs = []
    for a, b in zip(reference, alternative, strict=True):
        if a is None or b is None:
            continue
        diffs.append(b - a)
        if math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12):
            ties += 1
        elif b < a:
            wins += 1
        else:
            losses += 1
    return {
        "expected": len(reference),
        "paired": len(diffs),
        "alternative_wins": wins,
        "ties": ties,
        "reference_wins": losses,
        "median_alternative_minus_reference": _med(diffs),
    }


def aggregate(rows: list[dict], arms: list[str]) -> dict:
    """Keep the full denominator and compare forecasts on the same graphs and folds."""
    graphs = _graphs(rows, arms)
    summaries = {}
    pairs = []
    for arm in arms:
        members = [r for r in rows if r["arm"] == arm]
        valid = [r for r in members if r["scored"]]
        full = [g["arms"][arm] for g in graphs]
        targets = {}
        for name in sorted({name for r in valid for name in r["targets"]}):
            detail = [r["targets"][name] for r in valid]
            n = sum(r["comparable_peaks"] for r in detail)
            observed = sum(r["observed_interior_peaks"] for r in detail)
            targets[name] = {
                "mean_panel_nmse": _avg(r["nmse"] for r in detail),
                "observed_interior_peaks": observed,
                "comparable_peaks": n,
                "missed_interior_peaks": observed - n,
                "mean_peak_time_absolute_error": (
                    sum(
                        r["mean_peak_absolute_time_error"] * r["comparable_peaks"]
                        for r in detail
                        if r["comparable_peaks"]
                    )
                    / n
                    if n
                    else None
                ),
                "mean_panel_late_recovery_mae_scaled": _avg(
                    r["mean_event_late_recovery_mae_scaled"] for r in detail
                ),
            }
        result = {
            "slots_expected": 2 * len(members),
            "fit_statuses": dict(
                Counter(status for r in members for status in r["fit_statuses"])
            ),
            "panels_expected": len(members),
            "panels_scored": len(valid),
            "panels_held_nonnegative": sum(r["held_nonnegative"] for r in valid),
            "graphs_expected": len(full),
            "graphs_both_folds_scored": sum(x["scored_folds"] == 2 for x in full),
            "graphs_both_folds_nonnegative": sum(
                x["both_held_nonnegative"] is True for x in full
            ),
            "median_graph_held_nmse": _med(x["held_nmse"] for x in full),
            "median_graph_training_nmse": _med(x["training_nmse"] for x in full),
            "targets": targets,
            "winner_sources": dict(
                Counter(r["source"] for r in members if r["source"])
            ),
        }
        if arm == "hybrid":
            model = [r["provider"] for r in members]
            result["provider"] = {
                "expected": len(model),
                "statuses": dict(Counter(r["status"] for r in model)),
                "actions": dict(Counter(r["action"] for r in model if r["action"])),
                "responded": sum(
                    r.get("provider_status") == "responded" for r in model
                ),
                "observed_tokens": sum(r["observed_tokens"] or 0 for r in model),
                "responses_missing_usage": sum(
                    r.get("provider_status") == "responded"
                    and r["observed_tokens"] is None
                    for r in model
                ),
            }
        summaries[arm] = result
    for reference, alternative in combinations(arms, 2):
        panels = {r["panel_id"]: r["held_nmse"] for r in rows if r["arm"] == reference}
        other = {r["panel_id"]: r["held_nmse"] for r in rows if r["arm"] == alternative}
        pairs.append(
            {
                "reference": reference,
                "alternative": alternative,
                "panels": _compare(list(panels.values()), [other[k] for k in panels]),
                "graphs": _compare(
                    [g["arms"][reference]["held_nmse"] for g in graphs],
                    [g["arms"][alternative]["held_nmse"] for g in graphs],
                ),
            }
        )
    return {"arms": summaries, "graphs": graphs, "paired": pairs}


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _table(columns: list[str], rows) -> str:
    heads = "<tr>" + "".join(f"<th>{escape(s)}</th>" for s in columns) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_fmt(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f"<div><table><thead>{heads}</thead><tbody>{body}</tbody></table></div>"


def _plot(root: Path, panels: list[dict], path: Path) -> str:
    """Illustrate the first jointly scorable panel, choosing its first nonflat event."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = None
    available = {r["arm"] for r in panels if r["scored"]}
    for panel in dict.fromkeys(r["panel_id"] for r in panels):
        group = [r for r in panels if r["panel_id"] == panel and r["scored"]]
        if group and {r["arm"] for r in group} == available:
            chosen = group
            break
    if not chosen:
        return ""
    curves = {}
    for r in chosen:
        receipt = sealed_read(root / "scores" / r["panel_id"] / f"{r['arm']}.json")
        curves[r["arm"]] = receipt["grid"]["curves"]
    rows = next(iter(curves.values()))
    index = next(
        (
            i
            for i, c in enumerate(rows)
            if any(max(y) > min(y) for y in c["observed"].values())
        ),
        0,
    )
    ref = rows[index]
    fig, axes = plt.subplots(
        len(ref["observed"]),
        1,
        squeeze=False,
        figsize=(8, 2 * len(ref["observed"])),
        sharex=True,
        layout="constrained",
    )
    for (target, values), (axis,) in zip(ref["observed"].items(), axes, strict=True):
        axis.plot(ref["time"], values, color="black", lw=2.5, label="Observed")
        for arm, candidates in curves.items():
            curve = candidates[index]
            if (
                curve["trajectory_id"] != ref["trajectory_id"]
                or curve["observed"] != ref["observed"]
            ):
                raise ValueError("Paired curves must score the same held observations")
            axis.plot(ref["time"], curve["predicted"][target], lw=1.35, label=arm)
        axis.set_ylabel(target)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8, ncol=len(curves) + 1)
    axes[-1, 0].set_xlabel("Time (min)")
    label = f"{chosen[0]['panel_id']}, {ref['trajectory_id']}"
    fig.suptitle(f"Forecasts for {label}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return (
        f"<h2>Example forecast</h2><p>{escape(label)} is the first event with any "
        "observed variation in the first panel scored by every available arm.</p>"
        f'<img src="{escape(path.name)}" alt="Held forecasts by target">'
    )


def _html(root: Path, report: dict, partial: bool) -> str:
    arms, panels = report["arms"], report["panels"]
    pieces = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Equation forecasts</title>'
        "<style>body{max-width:1150px;margin:2rem auto;padding:0 1rem;color:#182d30;"
        "font:16px/1.5 system-ui}table{border-collapse:collapse;"
        "font-size:14px;width:100%}td,th{padding:.45rem;text-align:left;"
        "border-bottom:1px solid "
        "#cbd5d6}div{overflow-x:auto}img{max-width:100%}a{color:#125b66}</style><main>",
        "<h1>Forecasting with three equation constructors</h1>",
        "<p>This is a partial snapshot of recorded outcomes.</p>" if partial else "",
        "<p>Each graph is paired. Each arm gets two formula slots and "
        "the same fitting budget. Training NMSE selects the winner before "
        "scoring. Each event appears once in fitting and once in "
        "forecast evaluation across the folds. Graphs were already screened "
        "on all 16 training events. This study measures "
        "equation construction conditional on them.</p>",
        "<h2>Coverage and error</h2><p>Graph NMSE is the arithmetic mean of its two "
        "folds. Medians span graphs with both corresponding scores. Held NMSE covers "
        "post-initial samples, using the fitting-fold target standard deviation. "
        "Training NMSE uses the fitter's original whole-trajectory objective.</p>",
        _table(
            [
                "Arm",
                "Fitted / slots",
                "Scored / panels",
                "Held bounds pass / panels",
                "Both folds / graphs",
                "Median graph train NMSE",
                "Median graph held NMSE",
            ],
            (
                (
                    a,
                    f"{s['fit_statuses'].get('fitted', 0)} / {s['slots_expected']}",
                    f"{s['panels_scored']} / {s['panels_expected']}",
                    f"{s['panels_held_nonnegative']} / {s['panels_expected']}",
                    f"{s['graphs_both_folds_scored']} / {s['graphs_expected']}",
                    s["median_graph_training_nmse"],
                    s["median_graph_held_nmse"],
                )
                for a, s in arms.items()
            ),
        ),
        "<h2>Paired graph comparison</h2>",
        _table(
            [
                "Reference",
                "Alternative",
                "Comparable / graphs",
                "Alternative wins",
                "Ties",
                "Reference wins",
                "Median paired NMSE difference (alt - ref)",
            ],
            (
                (
                    p["reference"],
                    p["alternative"],
                    f"{g['paired']} / {g['expected']}",
                    g["alternative_wins"],
                    g["ties"],
                    g["reference_wins"],
                    g["median_alternative_minus_reference"],
                )
                for p in report["paired"]
                for g in [p["graphs"]]
            ),
        ),
        "<h2>Target-specific behavior</h2><p>Timing error uses interior peaks "
        "above the initial value. Ties use the first sample. Peaks at the last "
        "sample are censored. Recovery MAE compares changes "
        "from the initial values over the final time quarter, "
        "in fitting-fold standard deviation units.</p>",
        _table(
            [
                "Arm",
                "Target",
                "Mean held NMSE",
                "Comparable / observed peaks",
                "Mean peak error (min)",
                "Mean late recovery MAE",
            ],
            (
                (
                    a,
                    k,
                    s["mean_panel_nmse"],
                    f"{s['comparable_peaks']} / {s['observed_interior_peaks']}",
                    s["mean_peak_time_absolute_error"],
                    s["mean_panel_late_recovery_mae_scaled"],
                )
                for a, arm in arms.items()
                for k, s in arm["targets"].items()
            ),
        ),
        "<h2>Scores for both folds</h2>",
        _table(
            ["Inventory", "Graph origin", "Graph ID", *arms],
            (
                (
                    g["inventory"],
                    g["origin"],
                    g["graph_id"][:9],
                    *(g["arms"][a]["held_nmse"] for a in arms),
                )
                for g in report["graphs"]
            ),
        ),
    ]
    if "hybrid" in arms:
        p = arms["hybrid"]["provider"]
        pieces.extend(
            (
                "<h2>20B contributions</h2><p>Slot A retains the first "
                "nomination. For B, 20B can select the other option or author "
                "equations. The table shows choices and winners of "
                "training selection.</p>",
                _table(
                    [
                        "Expected calls",
                        "Responses",
                        "Parsed outcomes",
                        "Eligible actions",
                        "Training-selected source",
                        "Observed tokens",
                    ],
                    [
                        (
                            p["expected"],
                            p["responded"],
                            json.dumps(p["statuses"]),
                            json.dumps(p["actions"]),
                            json.dumps(arms["hybrid"]["winner_sources"]),
                            p["observed_tokens"],
                        )
                    ],
                ),
            )
        )
    pieces.extend(
        (
            "<h2>All graph and fold outcomes</h2>",
            _table(
                [
                    "Panel",
                    "Arm",
                    "Fit slots",
                    "Winner",
                    "Train NMSE",
                    "Held NMSE",
                    "Held bounds",
                    "Status",
                ],
                (
                    (
                        r["panel_id"],
                        r["arm"],
                        ", ".join(r["fit_statuses"]),
                        r["source"],
                        r["training_nmse"],
                        r["held_nmse"],
                        {True: "pass", False: "violation", None: "—"}[
                            r["held_nonnegative"]
                        ],
                        r["status"],
                    )
                    for r in panels
                ),
            ),
            _plot(
                root,
                panels,
                root
                / ("matched-curves.partial.png" if partial else "matched-curves.png"),
            ),
            f'<p><a href="{"summary.partial.json" if partial else "summary.json"}"> '
            "Machine-readable results and fit receipt digests</a></p></main></html>",
        )
    )
    return "\n".join(pieces)


def report(root: Path, *, partial: bool = False) -> dict:
    plan = sealed_read(root / "plan.json")
    with _lock(root / "report-lock"):
        rows = [
            _panel(root, p, arm, plan["nonnegative_targets"])
            for p in plan["panels"]
            for arm in plan["arms"]
        ]
        if not partial and any(r["status"] == "pending" for r in rows):
            raise ValueError(
                "The final report requires a score receipt for every arm/panel"
            )
        result = {
            "plan": plan["artifact_sha256"],
            "partial": partial,
            "panels": rows,
            **aggregate(rows, plan["arms"]),
        }
        if partial:
            atomic_json(root / "summary.partial.json", result)
        else:
            sealed_write(root / "summary.json", result)
        path = root / ("REPORT.partial.html" if partial else "REPORT.html")
        path.write_text(_html(root, result, partial))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = report(args.root, partial=args.partial)
    print(json.dumps({"arms": result["arms"], "paired": result["paired"]}, indent=2))


if __name__ == "__main__":
    main()
