"""Render sealed T2 and basin workflow decisions with validation coverage and costs."""

# ruff: noqa: E501 -- HTML paragraphs and scientific headings are written as prose.

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
from os.path import relpath
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.compare_evidence_decisions import ROOT, summarize
from scripts.report_hidden_form_forecasts import _table

NAMES = {
    "adaptive": "Exact finite-horizon",
    "one_step": "One-step lookahead",
    "simple_first": "Fit simple first",
    "extended_first": "Fit extended first",
    "screens_first": "Screen both first",
    "check_simple": "Screen simple",
    "check_extended": "Screen extended",
    "fit_simple": "Fit simple",
    "fit_extended": "Fit extended",
    "stop": "Stop",
    "basin": "Coupled basin",
    "t2": "T2 disposal",
}
FRAMEWORK = (
    Path(__file__).resolve().parents[1] / "docs/PHASE_C_EVIDENCE_DECISION_PROBLEM.md"
)


def _setting(rec: dict) -> tuple:
    return (rec["mode"], rec["transfer"], rec["budget"])


def _description(setting: tuple) -> str:
    mode, transfer, budget = setting
    objective = {
        "forecast": "Forecast loss",
        "precision": "Precision required, basin tolerance 0.1 mm",
        "precision_1mm": "Precision required, basin tolerance 1 mm",
        "precision_10mm": "Precision required, basin tolerance 10 mm",
    }[mode]
    return f"{objective} · {transfer:.0%} cross-domain prior weight · {budget} units"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _comparison(rows: list[dict], design: dict) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["policy"])].append(r)
    body = []
    for benchmark in sorted({r["benchmark"] for r in rows}):
        ids = {r["panel"] for r in rows if r["benchmark"] == benchmark}
        for policy in design["policy"]:
            group = groups[(benchmark, policy)]
            if len(group) != len(ids) or {r["panel"] for r in group} != ids:
                raise ValueError("Unmatched policy episodes in evaluation")
            s = summarize(group)
            n = s["episodes"]
            body.append(
                [
                    NAMES[benchmark],
                    NAMES[policy],
                    f"{s['qualified']}/{n}",
                    _number(s["mean_nmse"]),
                    _number(s["all_episode_loss"]),
                    _number(s["loss_with_cost"]),
                    f"{s['cost_units'] / n:.2f}",
                    s["checks"],
                    s["fits"],
                    s["optimizer_evaluations_replayed"],
                ]
            )
    return _table(
        [
            "Benchmark",
            "Workflow",
            "Qualified",
            "NMSE on qualified",
            "Loss, all episodes",
            "Loss + cost",
            "Units / episode",
            "Screens",
            "Fits",
            "Replayed optimizer evaluations",
        ],
        body,
    )


def _example(rec: dict, key: str) -> str:
    dec = rec["decisions"]["adaptive"]
    body = []
    for entry in dec["trace"]:
        values = entry["action_values"]
        action = entry["action"]
        body.append(
            [
                entry["spent"],
                _number(entry["stop_risk"]),
                *[
                    _number(values.get(a))
                    for a in (
                        "check_simple",
                        "check_extended",
                        "fit_simple",
                        "fit_extended",
                    )
                ],
                NAMES[action],
                entry.get("observation", dec["model"] or "Abstained"),
            ]
        )
    heading = f"{rec['panel']} · {_description(_setting(rec))}"
    table = _table(
        [
            "Units spent",
            "Stop",
            "Screen simple",
            "Screen extended",
            "Fit simple",
            "Fit extended",
            "Decision",
            "Observed result or returned model",
        ],
        body,
    )
    return f'<h3>{escape(heading)}</h3>{table}<p><a href="decisions/{escape(key)}.json">Decision receipt</a></p>'


def render(root: Path) -> Path:
    """Read registered results and decisions. Never run a fit or open event data."""
    plan = sealed_read(root / "plan.json")
    choices = sealed_read(root / "choices.json")
    results = sealed_read(root / "results.json")
    if plan["artifact_sha256"] != choices["plan"] or (
        results["choices"] != choices["artifact_sha256"]
    ):
        raise ValueError("Unmatched plan, decisions and results")
    design = plan["design"]
    primary = tuple(design["primary"])
    grouped = defaultdict(list)
    for row in results["rows"]:
        grouped[_setting(row)].append(row)
    if not grouped.get(primary):
        raise ValueError("Missing primary results")
    sensitivity = []
    for setting in sorted(grouped):
        if setting != primary:
            sensitivity.append(
                f"<details><summary>{escape(_description(setting))}</summary>"
                f"{_comparison(grouped[setting], design)}</details>"
            )
    decisions = []
    for key, digest in choices["receipts"]:
        rec = sealed_read(root / "decisions" / f"{key}.json")
        if (rec["artifact_sha256"], rec["plan"]) != (digest, choices["plan"]):
            raise ValueError("Unmatched decision receipt")
        decisions.append((key, rec))
    registered = sorted(decisions, key=lambda pair: pair[0])
    examples = []
    for domain in sorted({row["benchmark"] for row in grouped[primary]}):
        examples.append(
            next(
                r
                for r in registered
                if r[1]["benchmark"] == domain and _setting(r[1]) == primary
            )
        )
    if not any(r[1]["decisions"]["adaptive"]["checks"] for r in examples):
        screened = [r for r in registered if r[1]["decisions"]["adaptive"]["checks"]]
        if screened:
            examples.append(
                next((r for r in screened if _setting(r[1]) == primary), screened[0])
            )
    # Show both responses to the first basin fit when a second fit is affordable.
    larger = [
        item
        for item in registered
        if item[1]["benchmark"] == "basin"
        and _setting(item[1]) == (*primary[:2], max(design["budgets"]))
        and item[1]["budget"] > primary[2]
    ]
    for fit_count in (1, 2):
        example = next(
            (
                item
                for item in larger
                if item[1]["decisions"]["adaptive"]["fits"] == fit_count
            ),
            None,
        )
        if example:
            examples.append(example)
    tolerance_rows = [
        [
            f"{float(tol) * 1000:g}",
            values["simple"]["family_rejections"],
            values["simple"]["eligible_fits"],
            values["extended"]["family_rejections"],
            values["extended"]["eligible_fits"],
        ]
        for tol, values in sorted(
            plan["basin_screens"].items(), key=lambda t: float(t[0])
        )
    ]
    tolerance_table = _table(
        [
            "Tolerance (mm)",
            "Linear families rejected",
            "Linear fits eligible",
            "Three-halves families rejected",
            "Three-halves fits eligible",
        ],
        tolerance_rows,
    )
    headlines = []
    for domain in sorted({r["benchmark"] for r in grouped[primary]}):
        scores = {
            policy: summarize(
                [
                    r
                    for r in grouped[primary]
                    if r["benchmark"] == domain and r["policy"] == policy
                ]
            )["loss_with_cost"]
            for policy in (
                "adaptive",
                "simple_first",
                "extended_first",
                "screens_first",
            )
        }
        best = min(v for k, v in scores.items() if k != "adaptive")
        fixed = " and ".join(
            NAMES[k].lower() for k, v in scores.items() if k != "adaptive" and v == best
        )
        headlines.append(
            f"<p><strong>{NAMES[domain]}.</strong> The exact policy's mean loss "
            f"including cost was {_number(scores['adaptive'])}. The lowest fixed-workflow "
            f"loss was {_number(best)} ({fixed}).</p>"
        )
    report = root / "REPORT.html"
    framework = escape(relpath(FRAMEWORK, root))
    prices = design["prices"]
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Choosing when to screen, fit or stop</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:84rem;margin:2rem auto;padding:0 1rem;color:#1a313d}}h1,h2,h3{{line-height:1.2}}p{{max-width:87ch}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font:0.9rem/1.35 system-ui;font-variant-numeric:tabular-nums}}td,th{{padding:.48rem;text-align:left;border-bottom:1px solid #c6d3d9}}th{{background:#eaf0f3}}summary{{padding:.6rem;cursor:pointer}}a{{color:#235779}}</style>
<main><h1>Choosing when to screen, fit or stop</h1>
<p>Each workflow received the same two candidate equations on each episode. Only paid actions could reveal a diagnostic or fitting result. Model choices were frozen before validation. Candidate fits came from earlier runs, so the counts show how much computation each policy would request in replay.</p>
<h2>Primary comparison</h2>{"".join(headlines)}<p>{escape(_description(primary))}.</p>
{_comparison(grouped[primary], design)}
<p>Lower loss is better. Qualified NMSE averages whole-event validation error only for delivered models that passed qualification on all validation events. All-episode loss caps each available NMSE at {design["validation_loss_cap"]:g} and charges {design["abstention"]:g} for an abstention or failed forecast. Loss + cost adds {prices["loss_per_unit"]:g} per compute unit. A screen costs {prices["check"]} unit and a fit costs {prices["fit"]}. Totals reuse some archived fits and events across episodes.</p>
<p>T2 compares latent insulin action alone with a tissue-glucose interaction in the disposal equation. Basin compares discharge proportional to head with discharge proportional to head to the three-halves power. A training discrepancy informs the forecast policy. The precision condition makes a certified contradiction exclude that family and can disqualify a particular fitted endpoint when its coefficient lies outside the admissible interval.</p>
<h2>How the exact policy made its choices</h2>
<p>The first registered primary episode per benchmark is shown. When available, an additional example shows the first use of a screen if neither primary example purchased one. For the larger forecast budget the first basin episode using one fit and the first using two are also shown when available. Equal action values use a stable ordering by action name. In particular, equal values for the two fits provide no evidence that the selected equation was scientifically preferable. Each cell gives the expected remaining loss plus future computation cost if the named action is taken and the policy continues optimally. A dash marks an unavailable action. The lowest value gives the decision. Fitting observations are registered selection-NMSE bin labels.</p>
{"".join(_example(rec, key) for key, rec in examples)}
<p>The selection-NMSE upper bounds, starting with bin 0, are {escape(", ".join(map(str, design["selection_nmse_upper_edges"])))}. Higher scores enter the last bin. Each bin's planning risk in the same order is {escape(", ".join(map(str, design["risk_estimates"])))}.</p>
<h2>Budget, prior and accuracy sensitivity</h2>
<p>A zero cross-domain weight uses the broad prior alone. The precision conditions register basin downstream depth tolerances of 0.1, 1 and 10 mm. The counts below refer to the six fitting folds. Family rejection requires inconsistent admissible coefficient bounds. An open family can still have an ineligible fitted endpoint. These episodes reuse validation events and are descriptive comparisons within each domain.</p>
{tolerance_table}
{"".join(sensitivity)}
<p><a href="{framework}">Decision problem and worked example</a> · <a href="plan.json">Registered design</a> · <a href="choices.json">Frozen choices</a> · <a href="results.json">Per-episode results</a></p>
</main></html>"""
    report.write_text(html, encoding="utf-8")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT)
    print(render(cli.parse_args().root))
