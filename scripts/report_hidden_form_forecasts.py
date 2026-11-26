"""Freeze choices using selection events and compare them on omitted T2 events."""

# ruff: noqa: E501 -- HTML paragraphs and table labels are written as prose.

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean, median

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import forecast_disagreement as rule
from scripts.construction_functions import file_hash

ROOT = Path(__file__).resolve().parents[1] / "output/hidden-form-forecast-20261126"
LABELS = {
    "fixed_affine": "Always affine",
    "all_events": "All four selection events",
    "high_disagreement": "Two largest forecast differences",
    "random_same_size": "Random two (exact expectation)",
}


def _read_plan(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    if not plan["panels"] or len({p["panel_id"] for p in plan["panels"]}) != len(
        plan["panels"]
    ):
        raise ValueError("Missing or duplicated panels")
    for p in plan["panels"]:
        fit = set(p["fit_ids"])
        rots = p["choice_rotations"]
        if len(fit) != 8 or len(rots) != 2:
            raise ValueError("Expected eight fitting events and two rotations")
        for r in rots:
            select, held = set(r["selection_ids"]), set(r["held_ids"])
            together = fit | select | held
            if (
                len(select) != len(r["selection_ids"])
                or len(held) != len(r["held_ids"])
                or len(select) != 4
                or len(held) != 4
                or len(together) != 16
                or not all(n.startswith("train_") for n in together)
            ):
                raise ValueError(
                    "Expected disjoint training, selection and held events"
                )
        if set(rots[0]["selection_ids"]) != set(rots[1]["held_ids"]) or set(
            rots[0]["held_ids"]
        ) != set(rots[1]["selection_ids"]):
            raise ValueError("Expected complementary rotations")
    return plan


def _selection_plan(root: Path, plan: dict) -> dict:
    return sealed_write(
        root / "selection_plan.json",
        {
            "protocol": "matched-forecast-disagreement-policy-1",
            "fit_plan": plan["artifact_sha256"],
            "selector_sha256": file_hash(Path(rule.__file__)),
            "families": list(rule.FAMILIES),
            "policies": list(rule.POLICIES),
            "ranking": "Mean squared post-initial forecast gap in fitting-scale units.",
            "count": "Highest half of selection events. Exact gap ties by event ID.",
            "selection": "Lower mean selection event NMSE. Numerical ties affine.",
            "tie_atol": rule.TIE_ATOL,
            "tie_rtol": rule.TIE_RTOL,
            "random": "Exact expectation over all subsets of the same size.",
            "qualification": "Require eligible forecast on every selection event.",
            "held_metric": "Equal weight per output, then whole event, then rotation.",
        },
    )


def _curve(root, plan, panel, family, event):
    path = root / "curves" / panel / family / f"{event}.json"
    data = sealed_read(path)
    fit = sealed_read(root / "fits" / panel / family / "best.json")
    if (
        data["fit"] != fit["artifact_sha256"]
        or fit["plan"] != plan["artifact_sha256"]
        or (fit["panel_id"], fit["family"]) != (panel, family)
    ):
        raise ValueError("Forecast uses a different fit")
    if (data["plan"], data["panel_id"], data["family"], data["event_id"]) != (
        plan["artifact_sha256"],
        panel,
        family,
        event,
    ):
        raise ValueError(f"Forecast identity differs in {path}")
    return data


def _load_event(root: Path, plan: dict, panel: str, event: str) -> dict:
    curves = {f: _curve(root, plan, panel, f, event) for f in rule.FAMILIES}
    good = [c for c in curves.values() if c["status"] == "scored"]
    for c in good:
        n = len(c["time"])
        keys = set(c["scales"])
        if not keys or set(c["predicted"]) != keys or set(c["observed"]) != keys:
            raise ValueError("Missing observed or forecast channels")
        if n < 2 or not np.all(np.diff(c["time"]) > 0):
            raise ValueError("Invalid forecast times")
        errors = []
        for k in sorted(keys):
            y, pred = (
                np.asarray(c[s][k], dtype=float) for s in ("observed", "predicted")
            )
            scale = c["scales"][k]
            if (
                y.shape != (n,)
                or pred.shape != (n,)
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError("Invalid forecast grid or normalizer")
            errors.append(np.mean(np.square((y[1:] - pred[1:]) / scale)))
        if not np.isclose(c["nmse"], mean(errors), atol=1e-10, rtol=1e-8):
            raise ValueError("Event NMSE differs from the saved forecast")
    gap = None
    if len(good) == 2:
        if any(good[0][k] != good[1][k] for k in ("time", "observed", "scales")):
            raise ValueError("The candidates use different event data or scales")
        gap = rule.prediction_gap(
            {f: c["predicted"] for f, c in curves.items()}, good[0]["scales"]
        )
    return {
        "losses": {
            f: c["nmse"] if c["status"] == "scored" else None for f, c in curves.items()
        },
        "statuses": {f: c["status"] for f, c in curves.items()},
        "digests": {f: c["artifact_sha256"] for f, c in curves.items()},
        "gap": gap,
    }


def _choice_path(root, panel, rotation):
    return root / "choices" / panel / f"rotation{rotation}.json"


def select(root: Path) -> dict:
    """Freeze all selection decisions. Only selection-event curves are opened per rotation."""
    plan = _read_plan(root)
    with _lock(root / "selection_lock"):
        policy = _selection_plan(root, plan)
        receipts = []
        for p in plan["panels"]:
            for i, r in enumerate(p["choice_rotations"]):
                panel = p["panel_id"]
                rows = {
                    e: _load_event(root, plan, panel, e)
                    for e in sorted(r["selection_ids"])
                }
                decision = rule.choose(
                    {e: v["losses"] for e, v in rows.items()},
                    {e: v["gap"] for e, v in rows.items() if v["gap"] is not None},
                )
                rec = sealed_write(
                    _choice_path(root, panel, i),
                    {
                        "policy": policy["artifact_sha256"],
                        "panel": panel,
                        "rotation": i,
                        "events": {e: v["digests"] for e, v in rows.items()},
                        "statuses": {e: v["statuses"] for e, v in rows.items()},
                        **decision,
                    },
                )
                receipts.append(rec["artifact_sha256"])
        return sealed_write(
            root / "choices.json",
            {"policy": policy["artifact_sha256"], "receipts": receipts},
        )


def compare(rows: list[dict]) -> dict:
    """Summarize on one common denominator and report each policy's full coverage."""
    complete = [
        r
        for r in rows
        if all(s["mean_event_nmse"] is not None for s in r["scores"].values())
    ]
    summary = {}
    for policy in rule.POLICIES:
        paired = [r["scores"][policy]["mean_event_nmse"] for r in complete]
        own = [
            r["scores"][policy]["mean_event_nmse"]
            for r in rows
            if r["scores"][policy]["mean_event_nmse"] is not None
        ]
        summary[policy] = {
            "fully_scored": len(own),
            "available_mean": mean(own) if own else None,
            "expected_complete_rotations": sum(
                r["scores"][policy]["full_forecast_probability"] for r in rows
            ),
            "common_mean": mean(paired) if paired else None,
            "common_median": median(paired) if paired else None,
        }
    comparisons = {}
    for policy in ("fixed_affine", "all_events", "random_same_size"):
        pairs = [
            (
                r["scores"][policy]["mean_event_nmse"],
                r["scores"]["high_disagreement"]["mean_event_nmse"],
            )
            for r in complete
        ]
        delta = [a - b for a, b in pairs]
        tied = sum(
            math.isclose(a, b, rel_tol=rule.TIE_RTOL, abs_tol=rule.TIE_ATOL)
            for a, b in pairs
        )
        wins = sum(
            b < a
            and not math.isclose(a, b, rel_tol=rule.TIE_RTOL, abs_tol=rule.TIE_ATOL)
            for a, b in pairs
        )
        comparisons[policy] = {
            "reference_minus_high": mean(delta) if delta else None,
            "wins": wins,
            "ties": tied,
            "losses": len(pairs) - wins - tied,
        }
    return {
        "rotations": len(rows),
        "common": len(complete),
        "policies": summary,
        "comparisons": comparisons,
    }


def score(root: Path) -> dict:
    """Score choices only after every rotation's selection receipt has been sealed."""
    plan = _read_plan(root)
    policy = sealed_read(root / "selection_plan.json")
    manifest = sealed_read(root / "choices.json")
    if (
        manifest["policy"] != policy["artifact_sha256"]
        or policy["fit_plan"] != plan["artifact_sha256"]
        or policy["selector_sha256"] != file_hash(Path(rule.__file__))
    ):
        raise ValueError("Selection protocol differs")
    rows = []
    with _lock(root / "selection_report_lock"):
        for p in plan["panels"]:
            for i, r in enumerate(p["choice_rotations"]):
                panel = p["panel_id"]
                dec = sealed_read(_choice_path(root, panel, i))
                if (
                    dec["panel"],
                    dec["rotation"],
                    dec["policy"],
                    set(dec["events"]),
                    dec["artifact_sha256"],
                ) != (
                    panel,
                    i,
                    manifest["policy"],
                    set(r["selection_ids"]),
                    manifest["receipts"][len(rows)],
                ):
                    raise ValueError("Decision differs from the registered rotation")
                held = {e: _load_event(root, plan, panel, e) for e in r["held_ids"]}
                held_losses = {e: v["losses"] for e, v in held.items()}
                scores = {
                    k: rule.score_votes(votes, held_losses)
                    for k, votes in dec["votes"].items()
                }
                rows.append(
                    {
                        "panel": panel,
                        "graph": p["graph_name"],
                        "source": p["source"],
                        "target": p["target"],
                        "rotation": i,
                        "decision": dec["artifact_sha256"],
                        "votes": dec["votes"],
                        "eligible": dec["eligible"],
                        "selected_events": dec["high_disagreement_events"],
                        "selection_statuses": dec["statuses"],
                        "held_statuses": {e: v["statuses"] for e, v in held.items()},
                        "held_curves": {e: v["digests"] for e, v in held.items()},
                        "scores": scores,
                    }
                )
        if len(rows) != len(manifest["receipts"]):
            raise ValueError("Unexpected selection receipts")
        result = sealed_write(
            root / "policy_scores.json",
            {
                "choice_manifest": manifest["artifact_sha256"],
                "rows": rows,
                "summary": compare(rows),
                "competitive_only": compare(
                    [r for r in rows if len(r["eligible"]) == 2]
                ),
            },
        )
    return result


def _format(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _table(headers: list[str], rows: list[list]) -> str:
    headings = "".join(f"<th scope='col'>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in r) + "</tr>"
        for r in rows
    )
    return f"<div class='scroll'><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>"


def _example(root: Path, plan: dict, rows: list[dict]) -> str:
    """Plot curves from the first competitive rotation whose chosen family changes."""
    import base64
    from io import BytesIO

    import matplotlib as mpl
    from matplotlib.figure import Figure

    example = next(
        (
            r
            for r in rows
            if r["votes"]["high_disagreement"] == {"square": 1}
            and len(r["eligible"]) == 2
            and all(s["mean_event_nmse"] is not None for s in r["scores"].values())
        ),
        None,
    )
    if not example:
        return ""
    panel, rotation, target = example["panel"], example["rotation"], example["target"]
    name = escape(panel)
    decision = sealed_read(_choice_path(root, panel, rotation))
    gap_event = decision["high_disagreement_events"][0]
    held_event = sorted(example["held_curves"])[0]
    with mpl.rc_context(
        {"svg.hashsalt": "hidden-form-policy-20261126", "font.size": 10}
    ):
        fig = Figure(figsize=(8.5, 3.2), layout="constrained")
        axes = fig.subplots(1, 2, sharey=True)
        for ax, event, title in zip(
            axes, [gap_event, held_event], ["Selection", "Held out"], strict=True
        ):
            for family, color in (("affine", "#d26d13"), ("square", "#1b7990")):
                row = _curve(root, plan, panel, family, event)
                t = row["time"]
                if family == "affine":
                    ax.plot(
                        t,
                        row["observed"][target],
                        color="#2b313e",
                        linestyle="--",
                        label="Observed",
                    )
                ax.plot(
                    t, row["predicted"][target], color=color, label=family.capitalize()
                )
            ax.set(title=f"{title} · {event}", xlabel="Time (min)")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(f"{target} ({'mg/kg/min' if target == 'U' else 'pmol/L'})")
        axes[0].legend(frameon=False, fontsize=9)
        buf = BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
    image = base64.b64encode(buf.getvalue()).decode("ascii")
    changed = _format(example["scores"]["high_disagreement"]["mean_event_nmse"])
    base = _format(example["scores"]["fixed_affine"]["mean_event_nmse"])
    return f"""<h2>One whole-event forecast</h2><p>The first fully scored rotation choosing the square term was {name}, rotation {rotation}. The left panel shows its largest forecast disagreement on the selection events. The right panel shows the first held event by identifier. The mean all-target NMSE across the four held events was {base} for affine and {changed} for square.</p>
    <figure><img src="data:image/svg+xml;base64,{image}" alt="Observed and forecast {escape(target)} trajectories for the selection event {escape(gap_event)} and held event {escape(held_event)}."></figure>"""


def _statuses(rows: list[dict], field: str) -> str:
    counts = Counter(
        s for r in rows for event in r[field].values() for s in event.values()
    )
    return ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in sorted(counts.items()))


def _policy_table(summary: dict) -> str:
    """Show coverage on planned rotations and NMSE on the paired denominator."""
    table_rows = []
    for name in rule.POLICIES:
        r = summary["policies"][name]
        table_rows.append(
            [
                LABELS[name],
                f"{r['fully_scored']}/{summary['rotations']}",
                _format(r["common_mean"]),
                _format(r["common_median"]),
            ]
        )
    return _table(
        [
            "Selection policy",
            "Fully scored rotations",
            "Common mean NMSE",
            "Common median NMSE",
        ],
        table_rows,
    )


def _validation_section(root: Path, frozen_choices: str) -> str:
    path = root / "official-validation/summary.json"
    if not path.exists():
        return ""
    receipt = sealed_read(path)
    if receipt["choice_manifest"] != frozen_choices:
        raise ValueError("Official validation uses different training choices")
    summary = receipt["summary"]
    n = len(receipt["events"])
    if (
        n == 0
        or len(set(receipt["events"])) != n
        or not all(e.startswith("val_") for e in receipt["events"])
    ):
        raise ValueError("Unexpected official validation event identifiers")
    failures = Counter(f["status"].replace("_", " ") for f in receipt["failures"])
    failure_text = ", ".join(
        f"{count} {name}" for name, count in sorted(failures.items())
    )
    coverage = (
        f"Across the validation forecasts, {escape(failure_text)}."
        if failures
        else "Every individual validation forecast passed the checks."
    )
    return f"""<h2>Official validation events</h2>
    <p>The {n} official validation events were scored after all training selections and parameter vectors were frozen. Each of the {summary["rotations"]} existing selection decisions was carried forward, and every decision was evaluated on the same {n} events. The common NMSE comparison covers {summary["common"]} rotations where every policy produced qualified forecasts for every validation event.</p>
    {_policy_table(summary)}
    <p>{coverage} Validation events repeat across rotations. Each pair of rotations shares a fitted parameter vector for a given candidate. The benchmark test split remains unopened. <a href="official-validation/summary.json">Validation replay and qualification results</a></p>"""


def render(root: Path, result: dict) -> Path:
    """Render the paired errors, available denominators and selected whole events."""
    plan = _read_plan(root)
    summary = result["summary"]
    rows = result["rows"]
    comp = []
    for key, r in summary["comparisons"].items():
        comp.append(
            [
                LABELS[key],
                _format(r["reference_minus_high"]),
                r["wins"],
                r["ties"],
                r["losses"],
            ]
        )
    both = result["competitive_only"]
    per_contrast = []
    for graph, source, target in sorted(
        {(r["graph"], r["source"], r["target"]) for r in rows}
    ):
        subset = [
            r
            for r in rows
            if (r["graph"], r["source"], r["target"]) == (graph, source, target)
        ]
        metrics = compare(subset)
        per_contrast.append(
            [
                graph,
                f"{source} → {target}",
                metrics["common"],
                *[
                    _format(metrics["policies"][p]["common_mean"])
                    for p in rule.POLICIES
                ],
            ]
        )
    without_00 = compare([r for r in rows if r["graph"] != "graph00"])
    panel_rows = []
    for r in rows:
        votes = r["votes"]
        panel_rows.append(
            [
                r["panel"],
                r["rotation"],
                ", ".join(r["eligible"]) or "Neither",
                ", ".join(r["selected_events"]) or "—",
                next(iter(votes["high_disagreement"]), "—"),
                next(iter(votes["all_events"]), "—"),
                *[_format(r["scores"][p]["mean_event_nmse"]) for p in rule.POLICIES],
            ]
        )
    family_counts = Counter(len(r["eligible"]) for r in rows)
    budget = plan["per_start_budget"]
    filename = root / "REPORT.html"
    title = "Do events with different forecasts help choose an equation?"
    body = f"""<h1>{title}</h1><p>For three frozen T2-easy graphs, we changed one term at a time between an affine and a square dependence on its source. The four contrasts were EGP → U on graph00, EGP → I on graph04 and graph05, and I → U on graph05. A fitted coefficient and intercept can rescale either shape. All other equations, argument lists and free parameter counts were held fixed within a pair.</p>
    {_validation_section(root, result["choice_manifest"])}
    <h2>Held-out events within the training split</h2>
    <p>Means and medians below use the same {summary["common"]} complete rotations. Each rotation contributes its mean across four whole omitted events. Coverage uses all {summary["rotations"]} planned rotations.</p>
    {_policy_table(summary)}
    {_table(["Compared with largest differences", "Reference NMSE minus largest-difference NMSE", "Largest differences wins", "Ties", "Losses"], comp)}
    <p>Both forms qualified on every selection event in {family_counts[2]} rotations, one in {family_counts[1]}, neither in {family_counts[0]}. Restricting to actual choices with complete held forecasts leaves {both["common"]} rotations. There, selecting the largest differences averaged {_format(both["policies"]["high_disagreement"]["common_mean"])} compared with {_format(both["policies"]["all_events"]["common_mean"])} using all four events.</p>
    {_example(root, plan, rows)}
    <h2>Which equation benefited?</h2>
    {_table(["Graph", "Changed dependence", "Common rotations", "Fixed affine", "All events", "Largest differences", "Random two"], per_contrast)}
    <p>As a descriptive sensitivity check, omitting graph00 leaves {without_00["common"]} common rotations. Mean held NMSE there was {_format(without_00["policies"]["fixed_affine"]["common_mean"])} for fixed affine and {_format(without_00["policies"]["high_disagreement"]["common_mean"])} for the largest forecast differences.</p>
    <h2>Study design</h2><p>Each family received the same {len(plan["starts"])} starts per fitting fold with up to {budget["max_evaluations"]} evaluations and {budget["wall_seconds"]:g} seconds per start. Each fit used eight events, with fitted parameters fixed afterward. Four distinct training events selected a family, and the remaining four scored the choice. The selection and scoring groups then swapped roles within that fitting fold. Each contrast used both complementary fitting folds.</p>
    <p>To rank selection events, average the squared difference between the two predicted trajectories across post-initial times and targets, using common target scales from the fitting events. Reveal the observed errors for the two largest differences and choose the smaller mean event NMSE. A numerical tie favors affine. The all-event policy uses all four observed errors. The random comparator averages the results of all six two-event subsets.</p>
    <p>Every adaptive policy requires each eligible family to give a qualified forecast on all four selection events. If only one qualifies, the three adaptive policies choose it. If neither qualifies they abstain. A failed held event makes the rotation incomplete for the policy that chose its family. The fixed-affine policy always attempts affine. Forecast statuses across all selection roles were {escape(_statuses(rows, "selection_statuses"))}.</p>
    <p>All events used for fitting or model choice come from the original training split. The previous pooled shape audit motivated the four contrasts using this same 16-event collection. The internal held scores describe these fixed contrasts retrospectively. The rotations reuse training events and related graphs, and the fitted parameters have a finite budget.</p>
    <h2>Each paired choice</h2>
    {_table(["Contrast and fit fold", "Rotation", "Qualified at selection", "Two largest differences", "Chosen on two", "Chosen on all", "Fixed affine NMSE", "All-event NMSE", "Largest-difference NMSE", "Random-two expected NMSE"], panel_rows)}
    <p>Machine-readable evidence includes the <a href="plan.json">fit plan</a>, <a href="selection_plan.json">selection rules</a>, <a href="choices.json">sealed choice receipts</a> and <a href="policy_scores.json">paired scores</a>.</p>"""
    filename.write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
    <style>body{{max-width:66rem;margin:2.2rem auto;padding:0 1rem 4rem;color:#18343d;font:1rem/1.6 system-ui,sans-serif}}h1,h2{{line-height:1.18}}p{{max-width:85ch}}.scroll{{overflow-x:auto;margin:1rem 0}}table{{border-collapse:collapse;font-size:.92rem;font-variant-numeric:tabular-nums}}td,th{{padding:.44rem .6rem;text-align:left;border-bottom:1px solid #c1ccce}}th{{background:#eaf1f1}}img{{max-width:100%}}figure{{margin:1.5rem 0}}a{{color:#135d84}}</style><main>{body}</main></html>""",
        encoding="utf-8",
    )
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["select", "report", "all"], nargs="?", default="all"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage in {"select", "all"}:
        print(select(args.root)["artifact_sha256"])
    if args.stage in {"report", "all"}:
        result = score(args.root)
        print(json.dumps(result["summary"], indent=2))
        print(render(args.root, result))
